#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 8: native BSD68 qualitative comparison, matching Fig. 4.

Columns: Ground Truth, Fixed + PInv, Fixed + DiffPIR, Ours (PInv),
Ours (DiffPIR), ProxUnroll, AdaSense (when archived results are available). Full rectangular images retain their 321x481
aspect ratio, followed by two identical square crops and PSNR/SSIM.

Run: /home/baojr/.conda/envs/PIE2E/bin/python figures/fig8_bsd68_qualitative.py --image-id 253027
Redraw: same command reuses the NPZ; --plot-data PATH bypasses reconstruction.
Outputs: one 300-dpi PNG, provenance JSON and plotting NPZ per selected image.
Only archived exposures/final seeds are replayed; no acquisition is rerun.
PInv is read-only replay of each corresponding method's own exposure set.
All reconstructions must match archived metrics and measurement residuals.

Default sources: bsd68_fixed (MIRFLICKR v2 prior) and bsd68_bsdv2
(BSD-domain v2 prior). The prior difference is disclosed in the figure and
metadata; this is not a controlled same-prior acquisition-only comparison.
ProxUnroll uses its released pretrained HQS model and 1617 learned measurements.
Our/Fixed physical budget is 1544. Native images are not resized or padded.
Examples are selected by visible subject content, not performance ranking.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

def save_figure(fig, path, **kwargs):
    """Save PNG or vector EMF (requires Inkscape); intermediates are temporary."""
    import shutil
    import struct
    import subprocess
    import tempfile

    path = Path(path)
    if path.suffix.lower() != '.emf':
        fig.savefig(path, **kwargs)
        return
    executable = shutil.which('inkscape') or shutil.which('inkscape-paper')
    if executable is None:
        local = Path.home() / '.local/bin/inkscape-paper'
        executable = str(local) if local.is_file() else None
    if executable is None:
        raise RuntimeError('EMF export requires Inkscape (inkscape on PATH).')
    with tempfile.TemporaryDirectory(prefix='paper_figure_emf_') as directory:
        temporary = Path(directory)
        source = temporary / 'figure.pdf'
        fig.savefig(source, format='pdf', **kwargs)
        converted = source.with_suffix('.emf')
        result = subprocess.run(
            [executable, str(source), '--pdf-poppler', '--export-type=emf',
             '--export-filename=' + str(converted)],
            capture_output=True, text=True, timeout=180)
        if result.returncode or not converted.exists():
            raise RuntimeError('EMF conversion failed: ' + result.stdout + result.stderr)
        content = converted.read_bytes()
        if (len(content) < 88 or content[40:44] != b' EMF'
                or struct.unpack_from('<I', content, 48)[0] != len(content)):
            raise RuntimeError('Invalid EMF output: ' + str(path))
        path.write_bytes(content)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from daps_spi.compact_output import metrics
from daps_spi.data import load_image_template
from daps_spi.io import digest, file_hash
from daps_spi.priors import load_prior
from daps_spi.propagation_pinv import IntensityOptics, OpticalConfig, PseudoinverseDecoder, MeasurementFactor
from daps_spi.sampling import chain_seeds, sample
from daps_spi.types import MeasurementState

PROX_ROOT = ROOT / 'comparable_method/ProxUnroll-main'
PROX_RESULTS = PROX_ROOT / 'test_results/bsd68_cr001_hqs'
# (key, display name, production directory or None for ProxUnroll)
METHODS = [('fixed_pinv', 'Fixed + PInv', 'bsd68_fixed'),
           ('fixed', 'Fixed + DiffPIR', 'bsd68_fixed'),
           ('pinv_final', 'Ours (PInv)', 'bsd68_bsdv2'),
           ('full', 'Ours (DiffPIR)', 'bsd68_bsdv2'),
           ('proxunroll', 'ProxUnroll', None),
           ('adasense', 'AdaSense', ROOT / 'outputs/ablation/adasense_ddrm_bsd68')]
EXAMPLES = {'253027': 'zebra stripes and contours',
            '106024': 'bird silhouette and beak',
            '108082': 'tiger face and stripes',
            '351093': 'stone arch bridge and train'}
# Fixed GT-only feature locations (y, x), for 64x64 crops. No reconstruction ranking.
FEATURE_CROPS = {'253027': [(76, 152), (135, 55)],
                 '106024': [(32, 209), (155, 204)],
                 '108082': [(50, 225), (135, 90)],
                 '351093': [(203, 292), (134, 269)]}
PSNR_TOL = {'pinv': 1e-3, 'daps_mean': 1e-3, 'proxunroll': 0.02}
CROP_COLORS = ['#e8222a', '#1f77ff']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=ROOT / 'outputs/generalization')
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/bsd68_proxunroll/native')
    p.add_argument('--image-id', default='253027', help='BSD68 numeric image ID; default: zebra 253027')
    p.add_argument('--crop-size', type=int, default=64)
    p.add_argument('--crops', type=int, nargs=4, metavar=('Y1', 'X1', 'Y2', 'X2'),
                   help='Top-left corners (also overrides cached crops); default: fixed subject features or max GT gradient energy')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--plot-data', type=Path, help='Redraw from saved NPZ on CPU (no decoding)')
    p.add_argument('--regenerate', action='store_true', help='Ignore cached NPZ and decode again')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_figures/fig8')
    return p.parse_args()


def default_crops(truth, size):
    """Two non-overlapping size x size windows with the largest mean GT gradient magnitude."""
    gy, gx = np.gradient(truth.astype(np.float64))
    grad = np.hypot(gx, gy)
    integral = np.pad(grad, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = grad.shape
    energy = (integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size])
    first = np.unravel_index(np.argmax(energy), energy.shape)
    ys, xs = np.ogrid[:energy.shape[0], :energy.shape[1]]
    overlap = (np.abs(ys - first[0]) < size) & (np.abs(xs - first[1]) < size)
    second = np.unravel_index(np.argmax(np.where(overlap, -np.inf, energy)), energy.shape)
    return [tuple(int(v) for v in first), tuple(int(v) for v in second)]


def replay(source, image_path, device, nets, force_pinv=False):
    """Regenerate one archived final reconstruction from its sealed uint8 exposures."""
    meta = json.loads(source.read_text())
    settings = meta['settings']; cfg = settings['config']; recipe = settings['recipe']
    require(meta['status'] == 'complete' and meta['method'] == 'propagation-full-pseudoinverse', f'Bad source {source}')
    require(file_hash(source.with_suffix('.pt')) == meta['pt_sha256'], f'PT hash mismatch: {source}')
    require(file_hash(image_path) == settings['image_sha256'], 'Image hash mismatch')
    require(file_hash(ROOT / recipe['mean']) == settings['mean_sha256'], 'Mean hash mismatch')
    plus = torch.load(source.with_suffix('.pt'), map_location='cpu', weights_only=True)['plus_uint8']
    height, width = cfg['image_shape']
    count = cfg['final_measurements']
    require((height, width) == (321, 481), 'Expected native BSD68 321x481 dimensions')
    require(count == math.floor(.01 * height * width), 'Expected archived 1% measurement budget')
    require(plus.dtype == torch.uint8 and list(plus.shape) == [count, height, width],
            f'Expected {count} native {height}x{width} patterns')
    require(meta['final']['M'] == count and len(meta['patterns']['ids']) == count, 'Archived measurement count mismatch')
    require(digest(plus) == meta['patterns']['plus_hash'], 'Executed pattern hash mismatch')
    physics = settings['optics']['parameters']
    require(physics['image_pitch'] == physics['dmd_pitch'] == 10.8e-6, 'Incorrect physical pitch')
    truth, _ = load_image_template(image_path, height, width)
    truth = truth.to(device).double()
    optics = IntensityOptics(OpticalConfig(**physics), device)
    codes = plus.flatten(1).to(device)
    A = torch.cat([optics.effective_rows((2 * c.double() - 255) / 255) for c in codes.split(32)])
    y = optics.observe(truth, codes)
    tolerance = 1e-6 * (1 + float(y.abs().max()))
    state = MeasurementState(A, y, codes, meta['patterns']['ids'])
    mu = torch.load(ROOT / recipe['mean'], map_location=device, weights_only=True)['mu'].flatten().double()
    factor = MeasurementFactor(state,recipe['rank_rtol']) if cfg.get('shared_factor',False) else None
    decoder = 'pinv' if force_pinv else cfg['final_decoder']
    if decoder == 'pinv':
        rec = PseudoinverseDecoder(state, mu, recipe['rank_rtol'],factor).xhat
        info = dict(decoder='pinv')
    else:
        require(decoder == 'daps_mean', f'Unsupported final decoder {decoder}')
        final = settings['final_daps']
        expected_sampler = 'ddrm' if source.parent.parent.name == 'adasense_ddrm_bsd68' else 'diffpir'
        require(final.get('sampler') == expected_sampler, 'Unexpected archived final sampler')
        fcfg = dict(cfg['sampling'], **cfg['final_sampling'])
        fcfg.pop('validation_scenarios', None); fcfg['image_shape'] = [height, width]
        require(fcfg['chains'] == 32 and fcfg.get('sampler') == expected_sampler, 'Expected F32 and matching sampler')
        require(fcfg.get('diffpir_steps') == 50 if expected_sampler == 'diffpir' else fcfg.get('ddrm_steps') == 25, 'Unexpected archived sampler steps')
        decisions = math.ceil((cfg['final_measurements'] - cfg['initial_measurements']) / cfg['batch_size'])
        seeds = chain_seeds(cfg['seed'], settings['image_index'], decisions, 'F', 32)
        require(seeds == final['seeds'], 'Recomputed F seeds differ from the archive')
        prior = ROOT / recipe['final_prior']
        if prior not in nets:
            net, _, loaded = load_prior(str(prior), device)
            nets[prior] = (net, loaded)
        net, loaded = nets[prior]
        require(loaded == final['checkpoint_sha256'], 'Final prior hash mismatch')
        ensemble = sample(net, state, fcfg, seeds, 'F', loaded,factor)
        require(ensemble.diagnostics['rank'] == count, 'Final sampling projection truncated a row')
        rec = ensemble.samples.mean(0)
        info = dict(decoder='daps_mean',sampler=expected_sampler,sampling=fcfg,nfe_per_chain=ensemble.diagnostics['nfe_per_chain'], seeds=seeds, sample_seconds=ensemble.diagnostics['seconds'])
        del ensemble
    residual = float((A @ rec - y).abs().max())
    require(residual <= tolerance, f'Measurement residual {residual:.2e} exceeds {tolerance:.2e}')
    rec = rec.reshape(height, width)
    measured = metrics(rec, truth, count, 0.)
    archived = settings['final_pinv'] if force_pinv else meta['final']
    require(abs(measured['ssim'] - archived['ssim']) <= 1e-4, f'{source}: regenerated SSIM differs from archive')
    require(abs(measured['psnr'] - archived['psnr']) <= PSNR_TOL[decoder],
            f'{source}: regenerated PSNR {measured["psnr"]:.4f} vs archived {archived["psnr"]:.4f}')
    info.update(prior=str(recipe['final_prior']), forced_pinv_from_archived_exposures=force_pinv,source_json=str(source), source_json_sha256=file_hash(source), pt_sha256=meta['pt_sha256'],
                residual=residual, residual_tolerance=tolerance, archived=archived, regenerated=measured)
    return rec.cpu().numpy(), truth.cpu().numpy(), info


def run_proxunroll(image_id, device, dataset):
    """Released HQS checkpoint at CR 0.01, with the same input and clamp as evaluate_bsd68_cr001.py."""
    import os
    from PIL import Image
    meta = json.loads((PROX_RESULTS / image_id / f'{image_id}.json').read_text())
    weight = PROX_ROOT / meta['checkpoint']
    require(file_hash(weight) == meta['checkpoint_sha256'], 'ProxUnroll checkpoint hash mismatch')
    image_path = dataset / f'{image_id}.png'
    require(file_hash(image_path) == meta['input_sha256'], 'ProxUnroll input hash mismatch')
    sys.path.insert(0, str(PROX_ROOT))
    from model.proxunroll import ProxUnroll
    cwd = os.getcwd(); os.chdir(PROX_ROOT)  # the model loads measurement_matrix/*.mat relatively
    try:
        model = ProxUnroll(solver='hqs', color_channel=1, dim=48, mid_blocks=2,
                           enc_blocks=[2, 2, 2], dec_blocks=[2, 2, 2]).to(device)
    finally:
        os.chdir(cwd)
    state = torch.load(weight, map_location=device, weights_only=False)['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(not missing and not unexpected, 'ProxUnroll checkpoint mismatch')
    model.eval()
    truth = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32) / 255.0
    with torch.inference_mode():
        outs, _, _ = model(torch.from_numpy(truth.copy()).unsqueeze(0).to(device), 0.01)
        rec = outs[-1].squeeze(0).clamp(0, 1).double().cpu()
    del model
    measured = metrics(rec, torch.from_numpy(truth).double(), meta['effective_measurements'], 0.)
    archived = meta['final']
    require(abs(measured['psnr'] - archived['psnr']) <= PSNR_TOL['proxunroll'],
            f'ProxUnroll PSNR {measured["psnr"]:.4f} vs archived {archived["psnr"]:.4f}')
    require(abs(measured['ssim'] - archived['ssim']) <= 1e-4, 'ProxUnroll SSIM differs from archive')
    return rec.numpy(), dict(source_json=str(PROX_RESULTS / image_id / f'{image_id}.json'),
                             source_json_sha256=file_hash(PROX_RESULTS / image_id / f'{image_id}.json'), decoder='proxunroll-hqs', checkpoint_sha256=meta['checkpoint_sha256'],
                             measurements=meta['effective_measurements'], archived=archived, regenerated=measured)


def draw(truth, recs, report, size):
    """Rendering only: display clips to [0,1]; metrics come from the validated report.

    Explicit inch layout: each column is a native rectangular full image above two square crops whose
    combined width equals the image width, followed by two label lines and the footnote."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10, 'pdf.fonttype': 42})
    columns = [('gt', 'Ground Truth', truth)] + [(k, name, recs[k]) for k, name, _ in METHODS if k in recs]
    crops = [tuple(c) for c in report['crops']]
    scores = {k: report['methods'][k]['regenerated'] for k, _, _ in METHODS if k in recs}
    best_psnr = max(s['psnr'] for s in scores.values()); best_ssim = max(s['ssim'] for s in scores.values())
    n = len(columns); cw, gap, inner, pad = 1.9, .12, .05, .1     # inches
    cs = (cw - inner) / 2                                          # crop side
    image_height = cw * truth.shape[0] / truth.shape[1]
    title, labels, foot = .78, .62, .04
    W = 2 * pad + n * cw + (n - 1) * gap
    H = title + image_height + inner + cs + labels + foot
    fig = plt.figure(figsize=(W, H))
    def box(x, y, w, h):                                           # inches from bottom-left
        return fig.add_axes([x / W, y / H, w / W, h / H])
    y_crop = foot + labels; y_top = y_crop + cs + inner
    for j, (key, name, image) in enumerate(columns):
        x0 = pad + j * (cw + gap)
        top = box(x0, y_top, cw, image_height)
        top.imshow(np.clip(image, 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        for (cy, cx), color in zip(crops, CROP_COLORS):
            top.add_patch(Rectangle((cx - .5, cy - .5), size, size, fill=False, ec=color, lw=1.2))
        top.set_axis_off()
        for k, ((cy, cx), color) in enumerate(zip(crops, CROP_COLORS)):
            ax = box(x0 + k * (cs + inner), y_crop, cs, cs)
            ax.imshow(np.clip(image[cy:cy + size, cx:cx + size], 0, 1), cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(1.6)
        xc = (x0 + cw / 2) / W; y1 = (y_crop - .08) / H; y2 = (y_crop - .34) / H
        fig.text(xc, y1, name, ha='center', va='top', fontsize=10.5,
                 fontweight='bold' if key in {'pinv_final', 'full'} else 'normal')
        if key == 'gt':
            fig.text(xc, y2, 'PSNR (dB) / SSIM', ha='center', va='top', fontsize=9.5, color='.35')
        else:
            p, s = scores[key]['psnr'], scores[key]['ssim']
            fig.text(xc, y2, f'{p:.2f}', ha='right', va='top', fontsize=9.5,
                     fontweight='bold' if abs(p - best_psnr) < 5e-3 else 'normal')
            fig.text(xc, y2, f' / {s:.4f}', ha='left', va='top', fontsize=9.5,
                     fontweight='bold' if abs(s - best_ssim) < 5e-5 else 'normal')
    height, width = truth.shape
    count = scores['full']['M']
    prox_count = scores['proxunroll']['M']
    fig.text(.5, 1 - .04 / H, f'Fig. 8 | BSD68 qualitative comparison — {report["image_id"]} ({height}×{width}, 1% sampling)',
             ha='center', va='top', fontsize=15)
    return fig


def export(truth, recs, report, args, data_path):
    if args.crops:
        crops = [args.crops[:2], args.crops[2:]]
        require(all(0 <= y <= truth.shape[0] - args.crop_size and
                    0 <= x <= truth.shape[1] - args.crop_size for y, x in crops), 'Crop outside image')
        report = dict(report, crops=crops, crop_size=args.crop_size, crop_rule='explicit --crops')
    report = dict(report, column_order=['gt'] + [k for k, _, _ in METHODS if k in recs],
                  labels={k: n for k, n, _ in METHODS if k in recs},
                  script_sha256=file_hash(__file__), plotting_data=str(data_path.resolve()),
                  plotting_data_sha256=file_hash(data_path))
    fig = draw(truth, recs, report, report['crop_size'])
    stem = args.output / f'fig8_{report["image_id"]}'
    for extension in ['png', 'emf']:
        save_figure(fig, stem.with_suffix('.' + extension), dpi=args.dpi, facecolor='white',
                    bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Finished: {stem}.png / .json; plotting data: {data_path}', flush=True)


def read_plot_data(path):
    with np.load(path, allow_pickle=False) as data:
        require(int(data['format_version']) == 3, 'Unsupported plotting data version')
        report = json.loads(str(data['metadata']))
        truth = data['truth']
        recs = {k: data[f'rec_{k}'] for k, _, _ in METHODS if f'rec_{k}' in data}
        require(all(k in recs for k, _, _ in METHODS if k != 'adasense'), 'Missing required reconstruction')
    require(truth.shape == (321, 481) and all(r.shape == truth.shape for r in recs.values()), 'Bad plot data shape')
    require(np.isfinite(truth).all() and all(np.isfinite(r).all() for r in recs.values()), 'Nonfinite reconstruction in plot data')
    size = report['crop_size']
    require(all(0 <= y <= truth.shape[0] - size and 0 <= x <= truth.shape[1] - size
                for y, x in report['crops']), 'Cached crop outside image')
    return truth, recs, report


def save_plot_data(path, truth, recs, report):
    arrays = dict(format_version=np.array(3), metadata=np.array(json.dumps(report, allow_nan=False)), truth=truth)
    arrays.update({f'rec_{k}': v.astype(np.float64) for k, v in recs.items()})
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main():
    args = parse_args()
    args.output = args.output.resolve()
    require(not args.output.is_relative_to(args.results.resolve()), 'Output must be outside production results')
    require(not args.output.is_relative_to(PROX_RESULTS.resolve()), 'Output must be outside ProxUnroll results')
    require(8 <= args.crop_size <= 128 and args.dpi >= 72, 'Invalid crop size or dpi')
    if args.plot_data:
        truth, recs, report = read_plot_data(args.plot_data)
        print(f'Redrawing from {args.plot_data}; no decoding', flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        export(truth, recs, report, args, args.plot_data); return
    image_id = args.image_id
    require(image_id.isdigit(), 'BSD68 image ID must be numeric')
    data_path = args.output / f'fig8_{image_id}_data.npz'
    core = {p.name: file_hash(p) for p in sorted((ROOT / 'src/daps_spi').glob('*.py'))}
    if data_path.exists() and not args.regenerate:
        truth, recs, report = read_plot_data(data_path)
        # Cached pixels do not depend on the current reconstruction code. Preserve their original hashes.
        require(all(
            Path(report['methods'][k]['source_json']).resolve() == (args.results / d / image_id / f'{image_id}.json').resolve() and report['methods'][k].get('source_json_sha256') == file_hash(Path(report['methods'][k]['source_json']))
            for k, _, d in METHODS if d and k in recs), 'Cached provenance changed; use --regenerate')
        print(f'Reusing cached reconstructions from {data_path}', flush=True)
        ada_source = ROOT / 'outputs/ablation/adasense_ddrm_bsd68' / image_id / f'{image_id}.json'
        if 'adasense' not in recs and ada_source.exists():
            if torch.device(args.device).type == 'cuda':
                torch.cuda.set_device(args.device)
            torch.set_num_threads(4)
            print('Replaying only the newly available AdaSense final reconstruction', flush=True)
            recs['adasense'], gt, info = replay(ada_source, args.dataset / f'{image_id}.png', args.device, {})
            require(np.array_equal(truth, gt), 'Inconsistent ground truth for AdaSense')
            info['replay_core_hashes'] = core
            report['methods']['adasense'] = info
            report['labels'] = {k: n for k, n, _ in METHODS if k in recs}
            report['prior_comparison'] += '; AdaSense uses DDRM with BSD-domain v2'
            save_plot_data(data_path, truth, recs, report)
            print(f'AdaSense verified: {info["regenerated"]["psnr"]:.4f} dB / {info["regenerated"]["ssim"]:.4f}', flush=True)
    else:
        if torch.device(args.device).type == 'cuda':
            torch.cuda.set_device(args.device)
        torch.set_num_threads(4)
        image_path = args.dataset / f'{image_id}.png'
        recs, infos, nets, truth = {}, {}, {}, None; begin = time.perf_counter()
        for key, name, directory in METHODS:
            if key == 'adasense' and not (args.results / directory / image_id / f'{image_id}.json').exists():
                continue
            tick = time.perf_counter()
            if directory is None:
                recs[key], infos[key] = run_proxunroll(image_id, args.device, args.dataset)
            else:
                source = args.results / directory / image_id / f'{image_id}.json'
                recs[key], gt, infos[key] = replay(source, image_path, args.device, nets, force_pinv=(key in {'fixed_pinv', 'pinv_final'}))
                require(truth is None or np.array_equal(truth, gt), 'Inconsistent ground truth across sources')
                truth = gt
            infos[key]['seconds'] = time.perf_counter() - tick
            r = infos[key]['regenerated']
            print(f'{name:14s} PSNR {r["psnr"]:.4f} SSIM {r["ssim"]:.4f} '
                  f'(archived {infos[key]["archived"]["psnr"]:.4f}) {infos[key]["seconds"]:.0f}s', flush=True)
        if args.crops:
            crops, crop_rule = [args.crops[:2], args.crops[2:]], 'explicit --crops'
        elif image_id in FEATURE_CROPS and args.crop_size == 64:
            crops, crop_rule = FEATURE_CROPS[image_id], 'fixed GT feature windows, shared across methods'
        else:
            crops = [list(c) for c in default_crops(truth, args.crop_size)]
            crop_rule = 'two non-overlapping windows, max GT gradient energy'
        require(all(0 <= y <= truth.shape[0] - args.crop_size and 0 <= x <= truth.shape[1] - args.crop_size
                    for y, x in crops), 'Crop outside image')
        report = dict(image_id=image_id, figure=8,
                      selection='Subject-content example, not selected by reconstruction metrics',
                      subject=EXAMPLES.get(image_id, 'user-selected BSD68 image'),
                      prior_comparison='Fixed uses MIRFLICKR v2; Ours uses BSD-domain v2; archived settings preserved',
                      crop_size=args.crop_size, crops=crops, crop_rule=crop_rule,
                      methods=infos, labels={k: n for k, n, _ in METHODS if k in recs}, device=args.device,
                      torch_version=torch.__version__, current_core_hashes=core, seconds=time.perf_counter() - begin,
                      definitions=dict(psnr='unclipped (project metric); ProxUnroll output is clamped by the model script',
                                       ssim='clip [0,1], x255, skimage structural_similarity(data_range=255)',
                                       replay='archived sealed uint8 exposures; buckets regenerated by physical simulation; '
                                              'DiffPIR / DDRM means redraw the archived F seeds; no new acquisition'))
        args.output.mkdir(parents=True, exist_ok=True)
        save_plot_data(data_path, truth, recs, report)
    export(truth, recs, report, args, data_path)


if __name__ == '__main__':
    main()
