"""Train a larger EDM-x0 DAPS prior (UNetDenoiser) on the MIRFLICKR train split only.

Dev split images are used only for a fixed-noise denoising validation; Set11/BSD68 never enter.
The checkpoint keeps the load_prior convention (metadata + 'ema' state dict) and adds 'arch'.
"""
from pathlib import Path
import argparse, copy, json, math, random, sys, time
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from daps_spi.priors import UNetDenoiser  # noqa: E402
from daps_spi.io import atomic_save, file_hash  # noqa: E402

SIGMAS = [.02, .05, .1, .2, .5, 1., 2., 5., 10., 20.]


def load_split(root: Path, names: list[str]) -> torch.Tensor:
    """uint8 [N,256,256], kept in host memory."""
    out = np.empty((len(names), 256, 256), dtype=np.uint8)
    for i, n in enumerate(names):
        with Image.open(root / n) as im:
            a = np.asarray(im.convert('L'))
        if a.shape != (256, 256): raise ValueError(f'{n}: unexpected shape {a.shape}')
        out[i] = a
    return torch.from_numpy(out)


def batch(data: torch.Tensor, size: int, crop: int, gen: torch.Generator, device) -> torch.Tensor:
    idx = torch.randint(len(data), (size,), generator=gen)
    x = data[idx].to(device, non_blocking=True).float().div_(127.5).sub_(1).unsqueeze(1)
    if crop < 256:
        t, l = (int(v) for v in torch.randint(256 - crop + 1, (2,), generator=gen))
        x = x[..., t:t + crop, l:l + crop]
    k = int(torch.randint(4, (1,), generator=gen))
    if k: x = torch.rot90(x, k, (-2, -1))
    if int(torch.randint(2, (1,), generator=gen)): x = x.flip(-1)
    return x.contiguous()


@torch.no_grad()
def validate(net, val: torch.Tensor, device, amp: bool) -> list[dict]:
    """Fixed-noise x0 MSE per sigma on dev images, model domain [-1,1]."""
    net.eval(); rows = []
    g = torch.Generator(device=device).manual_seed(12345)
    x = val.to(device).float().div(127.5).sub(1).unsqueeze(1)
    for s in SIGMAS:
        err = 0.
        for xb in x.split(8):
            noisy = xb + s * torch.randn(xb.shape, generator=g, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                pred = net(noisy, s)
            err += float((pred.float() - xb).square().sum())
        rows.append(dict(sigma=s, mse=err / x.numel()))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/prior_daps_v2')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--mult', type=int, nargs='+', default=[1, 2, 2, 4, 4])
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--attention-levels', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--crop', type=int, default=128, help='random crop size for the first phase')
    ap.add_argument('--full-after', type=int, default=80000, help='switch to full 256 images after this update (as used for final_65k)')
    ap.add_argument('--updates', type=int, default=300000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ema', type=float, default=.9999)
    ap.add_argument('--val-count', type=int, default=64)
    ap.add_argument('--val-every', type=int, default=5000)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--seed', type=int, default=20261010)
    a = ap.parse_args(); amp = not a.no_amp
    device = torch.device(a.device); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    split = json.loads((ROOT / 'assets/split.json').read_text()); root = Path(split['root'])
    t = time.perf_counter()
    train = load_split(root, split['split']['train'])
    dev_names = sorted(split['split']['dev'])[::max(1, len(split['split']['dev']) // a.val_count)][:a.val_count]
    val = load_split(root, dev_names)
    print(json.dumps(dict(train=len(train), val=len(val), load_seconds=time.perf_counter() - t)), flush=True)
    meta = dict(arch='unet_v2', channels=a.channels, mult=a.mult, blocks=a.blocks, attention_levels=a.attention_levels,
                heads=a.heads, dropout=a.dropout, sigma_data=.5, sigma_min=.002, sigma_max=20., domain='[-1,1]',
                parameterization='EDM-x0', training_split='assets/split.json train only', split_sha256=file_hash(ROOT / 'assets/split.json'),
                validation='fixed-noise x0 MSE on %d dev images' % len(val), status='candidate prior v2; adopt only after dev gate')
    net = UNetDenoiser(a.channels, .5, tuple(a.mult), a.blocks, tuple(a.attention_levels), a.heads, a.dropout).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(.9, .99), weight_decay=0.)
    a.out.mkdir(parents=True, exist_ok=True); ck = a.out / 'last.pt'; start = 0; best = math.inf; log = []
    if ck.exists():
        p = torch.load(ck, map_location='cpu', weights_only=False)
        net.load_state_dict(p['model']); ema.load_state_dict(p['ema']); opt.load_state_dict(p['optimizer'])
        start, best, log = p['step'], p['best_score'], p['log']; torch.set_rng_state(p['rng'])
        print(f'resumed at {start}', flush=True)
    gen = torch.Generator().manual_seed(a.seed + start)
    n = sum(p.numel() for p in net.parameters()); print(json.dumps(dict(params=n)), flush=True)
    begin = time.perf_counter(); run_loss = 0.
    for step in range(start, a.updates):
        net.train()
        crop = a.crop if step < a.full_after else 256
        # Keep pixels per update constant when switching from crop to full 256 images.
        bs = a.batch if crop == a.crop else max(1, a.batch * a.crop * a.crop // (256 * 256))
        x = batch(train, bs, crop, gen, device)
        sigma = (torch.randn(len(x), device=device) * 1.2 - 1.2).exp().clamp(.002, 80.)
        noisy = x + sigma[:, None, None, None] * torch.randn_like(x)
        lr = a.lr * min(1., (step + 1) / 5000)
        for gp in opt.param_groups: gp['lr'] = lr
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            pred = net(noisy, sigma)
        w = (sigma.square() + .25) / (sigma * .5).square()
        loss = (w[:, None, None, None] * (pred.float() - x).square()).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        decay = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for e, q in zip(ema.parameters(), net.parameters()): e.lerp_(q, 1 - decay)
        run_loss += float(loss.detach())
        if (step + 1) % 500 == 0:
            log.append(dict(step=step + 1, loss=run_loss / 500, crop=crop, batch=bs, seconds=time.perf_counter() - begin))
            print(json.dumps(log[-1]), flush=True); run_loss = 0.
        if (step + 1) % a.val_every == 0 or step + 1 == a.updates:
            rows = validate(ema, val, device, amp); score = float(np.mean([r['mse'] for r in rows]))
            payload = dict(model=net.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(), rng=torch.get_rng_state(),
                           step=step + 1, best_score=min(best, score), log=log, validation=rows, metadata=dict(meta, training_updates=step + 1))
            atomic_save(payload, ck)
            if score < best:
                best = score
                atomic_save(dict(ema=ema.state_dict(), metadata=dict(meta, training_updates=step + 1), validation=rows), a.out / 'best.pt')
            print(json.dumps(dict(step=step + 1, val_mean_mse=score, best=best, rows=rows)), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
