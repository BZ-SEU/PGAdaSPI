"""Train the same-domain BSD68_481x321 prior (UNetDenoiser, v2 architecture) on BSD500 minus BSD68.

Data: assets/bsd68_prior_manifest.json train (389 images), native 321x481 Y (PIL L), portrait rotated
clockwise as in evaluation. Validation: manifest validation images minus --exclude (held-out probe images),
fixed-noise x0 MSE. Same EDM-x0 loss/weighting as scripts/train_prior.py.
--init: start from a checkpoint (e.g. the MIRFLICKR v2 prior) or 'none' for random init.
Phase 1 (< --full-after) uses random square crops with rot90/flip; afterwards full 321x481 with h/v flips.
"""
from pathlib import Path
import argparse, copy, json, math, sys, time
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from daps_spi.priors import UNetDenoiser, build_prior  # noqa: E402
from daps_spi.io import atomic_save, file_hash  # noqa: E402

SIGMAS = [.02, .05, .1, .2, .5, 1., 2., 5., 10., 20.]
H, W = 321, 481


def load(src: Path, names: list[str]) -> torch.Tensor:
    out = []
    for n in names:
        with Image.open(src / n) as im:
            im = im.convert('L')
            if im.height > im.width: im = im.rotate(-90, expand=True)
            a = np.asarray(im)
        if a.shape != (H, W): raise ValueError(f'{n}: unexpected shape {a.shape}')
        out.append(a)
    return torch.from_numpy(np.stack(out))


def batch(data, size, crop, gen, device):
    x = data[torch.randint(len(data), (size,), generator=gen)].to(device).float().div_(127.5).sub_(1).unsqueeze(1)
    if crop:
        t = int(torch.randint(H - crop + 1, (1,), generator=gen)); l = int(torch.randint(W - crop + 1, (1,), generator=gen))
        x = x[..., t:t + crop, l:l + crop]
        k = int(torch.randint(4, (1,), generator=gen))
        if k: x = torch.rot90(x, k, (-2, -1))
    elif int(torch.randint(2, (1,), generator=gen)): x = x.flip(-2)
    if int(torch.randint(2, (1,), generator=gen)): x = x.flip(-1)
    return x.contiguous()


@torch.no_grad()
def validate(net, val, device):
    net.eval(); g = torch.Generator(device=device).manual_seed(12345); rows = []
    x = val.to(device).float().div(127.5).sub(1).unsqueeze(1)
    for s in SIGMAS:
        err = 0.
        for xb in x.split(4):
            noisy = xb + s * torch.randn(xb.shape, generator=g, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                err += float((net(noisy, s).float() - xb).square().sum())
        rows.append(dict(sigma=s, mse=err / x.numel()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--init', default='outputs/prior_daps_v2/final_65k.pt', help="checkpoint path or 'none'")
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--updates', type=int, default=60000)
    ap.add_argument('--full-after', type=int, default=0, help='random-crop phase length (0: full images only)')
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--batch', type=int, default=4, help='full-image micro-batch; crop phase keeps pixels per update')
    ap.add_argument('--accum', type=int, default=2, help='gradient accumulation; effective batch = batch*accum')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr-min', type=float, default=1e-6)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--ema', type=float, default=.9995)
    ap.add_argument('--val-every', type=int, default=2000)
    ap.add_argument('--exclude', nargs='*', default=['103006.jpg', '107045.jpg'], help='held-out probe images')
    ap.add_argument('--seed', type=int, default=20261027)
    a = ap.parse_args(); device = torch.device(a.device); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.manual_seed(a.seed)
    man_path = ROOT / 'assets/bsd68_prior_manifest.json'; man = json.loads(man_path.read_text()); src = Path(man['source_root'])
    train = load(src, man['train']); val_names = [n for n in man['validation'] if n not in set(a.exclude)]; val = load(src, val_names)
    if a.init != 'none':
        base = torch.load(ROOT / a.init, map_location='cpu', weights_only=False); arch = dict(base['metadata'])
        net = build_prior(arch).to(device); net.load_state_dict(base['ema'], strict=True); init_hash = file_hash(ROOT / a.init)
    else:
        arch = dict(arch='unet_v2', channels=64, mult=[1, 2, 2, 4, 4], blocks=2, attention_levels=[3, 4], heads=4, dropout=0.,
                    sigma_data=.5, sigma_min=.002, sigma_max=20., domain='[-1,1]', parameterization='EDM-x0')
        net = UNetDenoiser(64, .5, (1, 2, 2, 4, 4), 2, (3, 4), 4, 0.).to(device); init_hash = None
    keep = ('arch', 'channels', 'mult', 'blocks', 'attention_levels', 'heads', 'dropout', 'sigma_data', 'sigma_min', 'sigma_max', 'domain', 'parameterization')
    meta = {k: arch[k] for k in keep}
    meta.update(training_split=f'assets/bsd68_prior_manifest.json train ({len(train)} BSD500 images, BSD68 excluded), native {H}x{W}',
                init=a.init, init_checkpoint_sha256=init_hash, manifest_sha256=file_hash(man_path),
                validation=f'fixed-noise x0 MSE on {len(val)} manifest validation images (excluded: {a.exclude})',
                status='candidate same-domain BSD prior; adopt only after held-out probe check',
                hyper=dict(updates=a.updates, full_after=a.full_after, crop=a.crop, batch=a.batch, accum=a.accum, lr=a.lr, lr_min=a.lr_min, ema=a.ema, seed=a.seed))
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(.9, .99), weight_decay=0.)
    a.out.mkdir(parents=True, exist_ok=True); ck = a.out / 'last.pt'; start = 0; best = math.inf; log = []
    if ck.exists():
        p = torch.load(ck, map_location='cpu', weights_only=False)
        net.load_state_dict(p['model']); ema.load_state_dict(p['ema']); opt.load_state_dict(p['optimizer'])
        start, best, log = p['step'], p['best_score'], p['log']; torch.set_rng_state(p['rng']); print(f'resumed at {start}', flush=True)
    else:
        rows = validate(ema, val, device); log.append(dict(step=0, val=rows, val_mean=float(np.mean([r['mse'] for r in rows]))))
        print(json.dumps(log[-1]), flush=True)
    print(json.dumps(dict(train=len(train), val=len(val), params=sum(p.numel() for p in net.parameters()))), flush=True)
    gen = torch.Generator().manual_seed(a.seed + start); begin = time.perf_counter(); run = 0.
    for step in range(start, a.updates):
        net.train()
        crop = a.crop if step < a.full_after else 0
        bs = max(1, a.batch * H * W // (a.crop * a.crop)) if crop else a.batch
        cos = .5 * (1 + math.cos(math.pi * step / a.updates))
        for gp in opt.param_groups: gp['lr'] = min(1., (step + 1) / a.warmup) * (a.lr_min + (a.lr - a.lr_min) * cos)
        opt.zero_grad(set_to_none=True); total = 0.
        for _ in range(a.accum):
            x = batch(train, bs, crop, gen, device)
            sigma = (torch.randn(len(x), device=device) * 1.2 - 1.2).exp().clamp(.002, 80.)
            noisy = x + sigma[:, None, None, None] * torch.randn_like(x)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = net(noisy, sigma)
            w = (sigma.square() + .25) / (sigma * .5).square()
            loss = (w[:, None, None, None] * (pred.float() - x).square()).mean() / a.accum
            loss.backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
        with torch.no_grad():
            for e, q in zip(ema.parameters(), net.parameters()): e.lerp_(q, 1 - min(a.ema, (1 + step) / (10 + step)))
        run += total
        if (step + 1) % 500 == 0:
            print(json.dumps(dict(step=step + 1, loss=run / 500, crop=crop, batch=bs * a.accum, seconds=time.perf_counter() - begin)), flush=True); run = 0.
        if (step + 1) % a.val_every == 0 or step + 1 == a.updates:
            rows = validate(ema, val, device); score = float(np.mean([r['mse'] for r in rows]))
            log.append(dict(step=step + 1, val=rows, val_mean=score)); print(json.dumps(log[-1]), flush=True)
            if score < best:
                best = score
                atomic_save(dict(ema=ema.state_dict(), metadata=dict(meta, training_updates=step + 1), validation=rows), a.out / 'best.pt')
            atomic_save(dict(model=net.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(), rng=torch.get_rng_state(),
                             step=step + 1, best_score=best, log=log, metadata=dict(meta, training_updates=step + 1)), ck)
    print('done', flush=True)


if __name__ == '__main__':
    main()
