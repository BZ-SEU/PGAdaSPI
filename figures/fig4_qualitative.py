#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 4: qualitative comparison on one native Set11 256x256 or 512x512 image.

Run from any directory with the project's PIE2E Python:
  python figures/fig4_qualitative.py                      # default image, cuda:0
  python figures/fig4_qualitative.py --image-id Parrots --device cuda:2
  python figures/fig4_qualitative.py --image-id fingerprint --crop-size 128
  python figures/fig4_qualitative.py --image-id flinstones --crop-size 128
  python figures/fig4_qualitative.py --plot-data outputs/paper_figures/fig4/fig4_lena256_data.npz

Columns: GT, Fixed+PInv, Fixed+DiffPIR, Ours (PInv on full exposures), Ours (DiffPIR), ProxUnroll, AdaSense.
Each column shows the full image and two crops; labels give the method and PSNR/SSIM.

No method is re-run as an acquisition. Production results keep only the executed
uint8 patterns, so reconstructions are regenerated read-only:
  * PInv decoders replay the archived exposures and recompute the centred pseudo-inverse;
  * DiffPIR-mean decoders replay the exposures and redraw the archived F seeds (32 chains,
    50 configured steps, 49 NFE per chain, same prior), exactly as runner.py does after the last batch;
  * AdaSense replays archived DDRM F32 seeds with 25 NFE per chain;
  * ProxUnroll re-runs its released HQS checkpoint at CR 0.01 (676 / 2704 learned measurements for 256 / 512).
Every regenerated reconstruction is checked against its archived JSON metrics. PSNR is the
project metric (unclipped); SSIM clips to [0,1] then x255 (ProxUnroll protocol).

Default image: the native 256 image whose Ours-minus-AdaSense PSNR gap is the median over
the nine images (deterministic, never chosen by visual quality). Default crops: the two
non-overlapping windows with the largest GT gradient energy. GT is used only for display,
crop placement and offline metrics. Reconstructions are cached as NPZ for CPU redraws.
Outputs: 300-dpi PNG, provenance JSON, plotting NPZ under outputs/paper_figures/fig4.
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
PROX_RESULTS = PROX_ROOT / 'test_results/set11_cr001_hqs'
# (key, display name, production directory or None for ProxUnroll)
METHODS = [('fixed_pinv', 'Fixed + PInv', 'fixed'),
           ('fixed', 'Fixed + DiffPIR', 'fixed'),
           ('pinv_final', 'Ours (PInv)', 'full'),
           ('full', 'Ours (DiffPIR)', 'full'),
           ('proxunroll', 'ProxUnroll', None),
           ('adasense', 'AdaSense', 'adasense_ddrm')]
NATIVE_256 = ['Monarch', 'Parrots', 'barbara', 'boats', 'cameraman', 'foreman', 'house', 'lena256', 'peppers256']
NATIVE_512 = ['fingerprint', 'flinstones']
PSNR_TOL = {'pinv': 1e-3, 'daps_mean': 1e-3, 'proxunroll': 0.02}
CROP_COLORS = ['#e8222a', '#1f77ff']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=ROOT / 'outputs/ablation')
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/set11_proxunroll/native')
    p.add_argument('--image-id', choices=NATIVE_256 + NATIVE_512, help='Default: median Ours-minus-AdaSense PSNR gap')
    p.add_argument('--crop-size', type=int, default=64)
    p.add_argument('--crops', type=int, nargs=4, metavar=('Y1', 'X1', 'Y2', 'X2'),
                   help='Top-left corners of the two crops; default: max GT gradient energy')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--plot-data', type=Path, help='Redraw from saved NPZ on CPU (no decoding)')
    p.add_argument('--regenerate', action='store_true', help='Ignore cached NPZ and decode again')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_figures/fig4')
    return p.parse_args()


def per_image(directory):
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob('*/*.json'))]
    return {r['image_id']: r['final'] for r in rows if r.get('status') == 'complete'}


def default_image(results):
    """Median of the Ours-minus-AdaSense PSNR gap over the nine native 256 images."""
    ours, ada = per_image(results / 'full'), per_image(results / 'adasense_ddrm')
    gaps = sorted((ours[i]['psnr'] - ada[i]['psnr'], i) for i in NATIVE_256)
    return gaps[len(gaps) // 2][1], {i: g for g, i in gaps}


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
    require((height, width) in {(256, 256), (512, 512)}, 'Expected native Set11 dimensions')
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
        expected_sampler = 'ddrm' if source.parent.parent.name == 'adasense_ddrm' else 'diffpir'
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
    info.update(forced_pinv_from_full=force_pinv,source_json=str(source), source_json_sha256=file_hash(source), pt_sha256=meta['pt_sha256'],
                residual=residual, residual_tolerance=tolerance, archived=archived, regenerated=measured)
    return rec.cpu().numpy(), truth.cpu().numpy(), info


def run_proxunroll(image_id, device):
    """Released HQS checkpoint at CR 0.01, with the same input and clamp as evaluate_set11_cr001.py."""
    import os
    from PIL import Image
    meta = json.loads((PROX_RESULTS / image_id / f'{image_id}.json').read_text())
    weight = PROX_ROOT / meta['checkpoint']
    require(file_hash(weight) == meta['checkpoint_sha256'], 'ProxUnroll checkpoint hash mismatch')
    image_path = ROOT / 'data/set11_proxunroll/native' / f'{image_id}.png'
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
    return rec.numpy(), dict(decoder='proxunroll-hqs', checkpoint_sha256=meta['checkpoint_sha256'],
                             measurements=meta['effective_measurements'], archived=archived, regenerated=measured)


def draw(truth, recs, report, size):
    """Rendering only: display clips to [0,1]; metrics come from the validated report.

    Explicit inch layout: each column is a square full image above two square crops whose
    combined width equals the image width, followed by two label lines and the footnote."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10, 'pdf.fonttype': 42})
    columns = [('gt', 'Ground Truth', truth)] + [(k, name, recs[k]) for k, name, _ in METHODS]
    crops = [tuple(c) for c in report['crops']]
    scores = {k: report['methods'][k]['regenerated'] for k, _, _ in METHODS}
    best_psnr = max(s['psnr'] for s in scores.values()); best_ssim = max(s['ssim'] for s in scores.values())
    n = len(columns); cw, gap, inner, pad = 1.9, .12, .05, .1     # inches
    cs = (cw - inner) / 2                                          # crop side
    title, labels, foot = .78, .62, .04
    W = 2 * pad + n * cw + (n - 1) * gap
    H = title + cw + inner + cs + labels + foot
    fig = plt.figure(figsize=(W, H))
    def box(x, y, w, h):                                           # inches from bottom-left
        return fig.add_axes([x / W, y / H, w / W, h / H])
    y_crop = foot + labels; y_top = y_crop + cs + inner
    for j, (key, name, image) in enumerate(columns):
        x0 = pad + j * (cw + gap)
        top = box(x0, y_top, cw, cw)
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
                 fontweight='bold' if key == 'full' else 'normal')
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
    fig.text(.5, 1 - .04 / H, f'Fig. 4 | Qualitative comparison — {report["image_id"]} ({height}×{width}, 1% sampling)',
             ha='center', va='top', fontsize=15)
    return fig


def export(truth, recs, report, args, data_path):
    report = dict(report, script_sha256=file_hash(__file__), plotting_data=str(data_path.resolve()),
                  plotting_data_sha256=file_hash(data_path))
    fig = draw(truth, recs, report, report['crop_size'])
    stem = args.output / f'fig4_{report["image_id"]}'
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
        recs = {k: data[f'rec_{k}'] for k, _, _ in METHODS}
    require(truth.shape in {(256, 256), (512, 512)} and all(r.shape == truth.shape for r in recs.values()), 'Bad plot data shape')
    require(np.isfinite(truth).all() and all(np.isfinite(r).all() for r in recs.values()), 'Nonfinite reconstruction in plot data')
    size = report['crop_size']
    require(all(0 <= y <= truth.shape[0] - size and 0 <= x <= truth.shape[1] - size
                for y, x in report['crops']), 'Cached crop outside image')
    return truth, recs, report


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
    if args.image_id:
        image_id, gaps = args.image_id, {}
    else:
        image_id, gaps = default_image(args.results)
    data_path = args.output / f'fig4_{image_id}_data.npz'
    core = {p.name: file_hash(p) for p in sorted((ROOT / 'src/daps_spi').glob('*.py'))}
    if data_path.exists() and not args.regenerate:
        truth, recs, report = read_plot_data(data_path)
        require(report['current_core_hashes'] == core and all(
            Path(report['methods'][k]['source_json']).resolve() == (args.results / d / image_id / f'{image_id}.json').resolve() and report['methods'][k].get('source_json_sha256') == file_hash(Path(report['methods'][k]['source_json']))
            for k, _, d in METHODS if d), 'Cached provenance changed; use --regenerate')
        print(f'Reusing {data_path}; no decoding', flush=True)
    else:
        if torch.device(args.device).type == 'cuda':
            torch.cuda.set_device(args.device)
        torch.set_num_threads(4)
        image_path = args.dataset / f'{image_id}.png'
        recs, infos, nets, truth = {}, {}, {}, None; begin = time.perf_counter()
        for key, name, directory in METHODS:
            tick = time.perf_counter()
            if directory is None:
                recs[key], infos[key] = run_proxunroll(image_id, args.device)
            else:
                source = args.results / directory / image_id / f'{image_id}.json'
                recs[key], gt, infos[key] = replay(source, image_path, args.device, nets, force_pinv=(key in {'fixed_pinv', 'pinv_final'}))
                require(truth is None or np.array_equal(truth, gt), 'Inconsistent ground truth across sources')
                truth = gt
            infos[key]['seconds'] = time.perf_counter() - tick
            r = infos[key]['regenerated']
            print(f'{name:14s} PSNR {r["psnr"]:.4f} SSIM {r["ssim"]:.4f} '
                  f'(archived {infos[key]["archived"]["psnr"]:.4f}) {infos[key]["seconds"]:.0f}s', flush=True)
        crops = [list(c) for c in ([args.crops[:2], args.crops[2:]] if args.crops else default_crops(truth, args.crop_size))]
        require(all(0 <= y <= truth.shape[0] - args.crop_size and 0 <= x <= truth.shape[1] - args.crop_size
                    for y, x in crops), 'Crop outside image')
        report = dict(image_id=image_id, selection=('explicit --image-id' if args.image_id else
                      'median Ours-minus-AdaSense PSNR gap over nine native 256 images'),
                      psnr_gaps_ours_minus_adasense=gaps, crop_size=args.crop_size, crops=crops,
                      crop_rule='explicit --crops' if args.crops else 'two non-overlapping windows, max GT gradient energy',
                      methods=infos, labels={k: n for k, n, _ in METHODS}, device=args.device,
                      torch_version=torch.__version__, current_core_hashes=core, seconds=time.perf_counter() - begin,
                      definitions=dict(psnr='unclipped (project metric); ProxUnroll output is clamped by the model script',
                                       ssim='clip [0,1], x255, skimage structural_similarity(data_range=255)',
                                       replay='archived sealed uint8 exposures; buckets regenerated by physical simulation; '
                                              'DiffPIR / DDRM means redraw the archived F seeds; no new acquisition'))
        args.output.mkdir(parents=True, exist_ok=True)
        arrays = dict(format_version=np.array(3), metadata=np.array(json.dumps(report, allow_nan=False)), truth=truth)
        arrays.update({f'rec_{k}': v.astype(np.float64) for k, v in recs.items()})
        temporary = data_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(data_path)
    export(truth, recs, report, args, data_path)


if __name__ == '__main__':
    main()
