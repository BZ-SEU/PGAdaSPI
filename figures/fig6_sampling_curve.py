#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 6: GT, fixed/adaptive prefix reconstructions and sampling curves.

CPU only; no acquisition or posterior sampling is rerun. Intermediate images
are replayed from archived uint8 pattern prefixes with the original optics and
centered pseudoinverse, then checked against archived metrics. Production PTs
do not contain buckets: GT is used only in offline exposure replay/evaluation.
Default barbara matches Figs. 3--5, independently of the performance gap.
Outputs: 300-dpi PNG, provenance/metric JSON and small plotting NPZ.
Redraw: python figures/fig6_sampling_curve.py --image-id cameraman
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from daps_spi.io import digest, file_hash
from daps_spi.data import load_image_template
from daps_spi.compact_output import metrics
from daps_spi.propagation_pinv import IntensityOptics, OpticalConfig, PseudoinverseDecoder
from daps_spi.types import MeasurementState


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def collect(results: Path, image_id: str) -> dict:
    """Read scalar histories, validating physical codes and comparable settings."""
    records, settings, prefixes = {}, {}, {}
    for method, folder in [('fixed', 'fixed'), ('adaptive', 'full')]:
        source = results / folder / image_id / f'{image_id}.json'
        meta = json.loads(source.read_text())
        s = meta['settings']; cfg = s['config']
        require(meta['status'] == 'complete', f'Incomplete result: {source}')
        require(meta['method'] == 'propagation-full-pseudoinverse', 'Unexpected method')
        require(file_hash(source.with_suffix('.pt')) == meta['pt_sha256'], 'PT hash mismatch')
        pt = torch.load(source.with_suffix('.pt'), map_location='cpu', weights_only=True)
        codes = pt['plus_uint8']; K = cfg['final_measurements']
        require(codes.dtype == torch.uint8 and len(codes) == K, 'Invalid physical codes')
        require(digest(codes) == meta['patterns']['plus_hash'], 'Physical pattern hash mismatch')
        prefixes[method] = codes[:cfg['initial_measurements']].clone()
        history = pt['history']
        expected = list(range(cfg['initial_measurements'], K, cfg['batch_size'])) + [K]
        require([r['M'] for r in history] == expected, 'Incomplete measurement history')
        require(all(r.get('decoder', 'pinv') == 'pinv' for r in history[:-1]),
                'Intermediate points must be pseudoinverse reconstructions')
        require(history[-1].get('decoder') == cfg['final_decoder'] == 'daps_mean',
                'Expected archived final posterior mean')
        require(s['final_daps']['sampler'] == 'diffpir', 'Expected DiffPIR final decoder')
        final = meta['final']; endpoint = s['final_pinv']
        require(final['M'] == endpoint['M'] == K, 'Inconsistent endpoint budget')
        require(abs(history[-1]['psnr'] - final['psnr']) < 1e-10, 'Final metric mismatch')
        pinv = [dict(r) for r in history[:-1]] + [dict(endpoint)]
        for r in pinv + [final]:
            require(abs(-10 * np.log10(r['mse']) - r['psnr']) < 1e-8,
                    'PSNR must be unclipped with data_range=1')
        records[method] = {
            'source_json': str(source.resolve()), 'source_json_sha256': file_hash(source),
            'pt_sha256': meta['pt_sha256'], 'run_identity': s['run_identity'],
            'pattern_hash': meta['patterns']['plus_hash'], 'pinv_history': pinv,
            'final': final, 'final_sampling': s['final_daps'],
            'seed': s['seed'], 'image_index': s['image_index'],
        }
        settings[method] = s
        del pt, codes
    a, b = settings['fixed'], settings['adaptive']
    for key in ['image_sha256', 'mean_sha256', 'optics', 'input_region']:
        require(a[key] == b[key], f'Incomparable {key}')
    for key in ['initial_measurements', 'final_measurements', 'batch_size', 'image_shape',
                'measurement_convention', 'final_decoder', 'final_sampling', 'seed']:
        require(a['config'][key] == b['config'][key], f'Incomparable config.{key}')
    require(a['recipe']['rank_rtol'] == b['recipe']['rank_rtol'], 'Different pseudoinverse cutoff')
    require(a['final_daps'] == b['final_daps'], 'Different final sampler/prior/seeds')
    require(torch.equal(prefixes['fixed'], prefixes['adaptive']), 'Different initial physical patterns')
    require(abs(records['fixed']['pinv_history'][0]['psnr'] -
                records['adaptive']['pinv_history'][0]['psnr']) < 1e-10, 'Different initial PSNR')
    cfg = b['config']; K = cfg['final_measurements']
    return {
        'figure': 6, 'image_id': image_id, 'image_shape': cfg['image_shape'],
        'image_selection': 'Explicit default barbara, matching Figs. 3--5; no metric ranking',
        'metric': 'Unclipped PSNR, data_range=1, dB',
        'x_axis': 'M effective differential measurements; physical exposures = 2*M',
        'initial_M': cfg['initial_measurements'], 'final_M': K, 'batch_size': cfg['batch_size'],
        'effective_sampling_ratio': K / int(np.prod(cfg['image_shape'])),
        'final_physical_exposures': 2*K, 'methods': records,
        'checks': ['PT and physical pattern hashes', 'shared exact physical prefix',
                   'same image, mean, optics, budget, pseudoinverse cutoff and final sampler/seeds',
                   'complete saved histories', 'PSNR/MSE identity'],
        'interpretation': 'Solid curves/circles are centered full pseudoinverse; diamonds are final '
                          'DiffPIR ensemble means. Vertical dotted segments change decoder at the '
                          'same M, with no new measurements. No intermediate DiffPIR values are inferred.',
        'source_reuse': 'Existing PT histories and JSON final_pinv/final; no model loaded or rerun',
        'final_adaptive_minus_fixed_db': records['adaptive']['final']['psnr'] - records['fixed']['final']['psnr'],
    }


def replay_prefixes(data: dict, dataset: Path, budgets: list[int]) -> dict[str, np.ndarray]:
    """Return raw physical-domain GT [H,W] and PInv images [len(budgets),H,W]."""
    require(bool(budgets) and budgets == sorted(set(budgets)), 'Specify increasing unique budgets')
    image = dataset / f'{data["image_id"]}.png'
    h, w = data['image_shape']
    truth, _ = load_image_template(image, h, w)
    truth = truth.double()
    arrays = {'truth': truth.numpy(), 'reconstruction_M': np.array(budgets, dtype=np.int64)}
    checks = {}
    for method, record in data['methods'].items():
        source = Path(record['source_json'])
        require(file_hash(source) == record['source_json_sha256'], 'Source changed during replay')
        meta = json.loads(source.read_text()); s = meta['settings']
        require(file_hash(image) == s['image_sha256'], 'GT image hash mismatch')
        mean_path = ROOT / s['recipe']['mean']
        require(file_hash(mean_path) == s['mean_sha256'], 'Training mean hash mismatch')
        history = {r['M']: r for r in record['pinv_history']}
        require(all(m in history for m in budgets), 'Requested M is not an archived prefix')
        pt = torch.load(source.with_suffix('.pt'), map_location='cpu', weights_only=True)
        codes = pt['plus_uint8'][:max(budgets)].flatten(1)
        optics = IntensityOptics(OpticalConfig(**s['optics']['parameters']), 'cpu')
        A = torch.cat([optics.effective_rows((2*c.double()-255)/255) for c in codes.split(32)])
        y = optics.observe(truth, codes)
        forward_error = float((A@truth.flatten()-y).abs().max())
        require(forward_error < 1e-8, 'Offline exposure/matrix mismatch')
        mu = torch.load(mean_path, map_location='cpu', weights_only=True)['mu'].flatten().double()
        images, checked = [], []
        for m in budgets:
            state = MeasurementState(A[:m], y[:m], codes[:m], meta['patterns']['ids'][:m])
            decoder = PseudoinverseDecoder(state, mu, s['recipe']['rank_rtol'])
            rec = decoder.xhat.reshape(h,w)
            measured = metrics(rec, truth, m, 0.)
            residual = float((state.A@rec.flatten()-state.y).abs().max())
            require(residual < 1e-6*(1+float(y.abs().max())), 'PInv measurement residual exceeded')
            require(abs(measured['psnr']-history[m]['psnr']) < 1e-3, 'Replay PSNR differs from archive')
            require(abs(measured['ssim']-history[m]['ssim']) < 1e-4, 'Replay SSIM differs from archive')
            images.append(rec.numpy())
            checked.append({'M':m, 'archived':history[m], 'replayed':measured,
                            'max_measurement_residual':residual, 'condition':decoder.condition})
            print(f'{method} M{m}: replay PSNR={measured["psnr"]:.6f}, verified', flush=True)
            del decoder, state
        arrays[f'{method}_reconstructions'] = np.stack(images)
        checks[method] = {'forward_error':forward_error, 'prefixes':checked}
    data['reconstruction_budgets'] = budgets
    data['reconstruction_checks'] = checks
    data['display'] = 'Common grayscale [0,1]; raw untruncated arrays saved; all metrics unclipped PSNR'
    data['source_reuse'] = ('Curves/final metrics from archived PT/JSON; intermediate centered PInv '
                           'images from archived actual uint8 prefixes and offline exposure replay. '
                           'No new selection, diffusion samples, or final reconstruction.')
    return arrays


def draw(data: dict, arrays: dict[str, np.ndarray], output: Path, dpi: int) -> Path:
    """GT on the left; two rows of PInv images; sampling curves on the right."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 11,
                         'axes.spines.top': True, 'axes.spines.right': True})
    fig = plt.figure(figsize=(22, 6.3))
    # Match image slots to their displayed aspect ratio: oversized slots leave
    # blank margins even when GridSpec wspace is zero.
    h, w = arrays['truth'].shape
    image_width = (.80-.22)*6.3/22*w/h
    gt_right = .025 + image_width
    middle_left = gt_right + .014  # Only room for the two vertical row labels.
    num_columns = len(data['reconstruction_budgets'])
    middle_right = middle_left + image_width/2.24*(num_columns+(num_columns-1)*.025)
    gt_grid = fig.add_gridspec(1,1,left=.025,right=gt_right,bottom=.22,top=.80)
    grid = fig.add_gridspec(2,num_columns,left=middle_left,right=middle_right,
                            bottom=.22,top=.80,wspace=.025,hspace=.24)
    curve_grid = fig.add_gridspec(1,1,left=middle_right+.045,right=.985,
                                  bottom=.22,top=.80)
    colors = {'fixed': '#D98232', 'adaptive': '#167EAB'}
    gt_ax = fig.add_subplot(gt_grid[0])
    gt_ax.imshow(arrays['truth'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    gt_ax.set_title('Ground truth', fontsize=13, pad=10)
    gt_ax.set_xticks([]); gt_ax.set_yticks([])
    for spine in gt_ax.spines.values(): spine.set_visible(False)
    for row, method in enumerate(['fixed','adaptive']):
        for col, m in enumerate(data['reconstruction_budgets']):
            panel = fig.add_subplot(grid[row,col])
            panel.imshow(arrays[f'{method}_reconstructions'][col], cmap='gray',
                         vmin=0,vmax=1,interpolation='nearest')
            panel.set_xticks([]); panel.set_yticks([])
            for spine in panel.spines.values(): spine.set_visible(False)
            if row == 0: panel.set_title(f'M = {m}', fontsize=11, pad=7)
            if col == 0: panel.set_ylabel(method.title(), color=colors[method], fontsize=12,
                                          fontweight='bold',labelpad=5)
            measured = data['reconstruction_checks'][method]['prefixes'][col]['replayed']
            panel.set_xlabel(f'PSNR {measured["psnr"]:.2f} dB | SSIM {measured["ssim"]:.3f}',
                             fontsize=8, labelpad=3)
    ax = fig.add_subplot(curve_grid[0])
    ax.set_title('PSNR versus measurement count', fontsize=12, pad=12)
    K = data['final_M']; start = data['initial_M']; span = K-start
    values = []
    for method in ['fixed', 'adaptive']:
        record = data['methods'][method]; hist = record['pinv_history']
        ms = np.array([r['M'] for r in hist]); ys = np.array([r['psnr'] for r in hist])
        color = colors[method]; final = record['final']['psnr']
        ax.plot(ms, ys, color=color, lw=2.1, marker='o', ms=4.3,
                mfc='white', mew=1.1, label=method.title(), zorder=3)
        ax.plot([K,K], [ys[-1], final], color=color, lw=1.6, ls=':', zorder=4)
        ax.scatter([K], [final], color=color, marker='D', s=66,
                   edgecolor='white', linewidth=.8, zorder=5)
        label_position = ({'xytext': (.97, .96), 'textcoords': 'axes fraction',
                           'ha': 'right', 'va': 'top'} if method == 'adaptive' else
                          {'xytext': (6, -22), 'textcoords': 'offset points'})
        ax.annotate(f'{method.title()} final\n{final:.2f} dB', xy=(K,final),
                    **label_position,
                    fontsize=9, color=color, annotation_clip=False,
                    arrowprops={'arrowstyle':'-', 'color':color, 'lw':.8})
        values.extend(ys); values.append(final)
    ax.axvline(K, color='#999999', lw=.8, ls='--', alpha=.6, zorder=0)
    ax.set_xlim(start-span*.035, K+span*.25)
    ax.set_ylim(min(values)-.3, max(values)+.75)
    ticks = [start] + [int(v) for v in np.linspace(start,K,6)[1:-1]//32*32] + [K]
    ax.set_xticks(sorted(set(ticks)))
    ax.set_xlabel('Number of effective measurements, M', labelpad=9)
    ax.set_ylabel('PSNR (dB)')
    ax.grid(axis='y', color='#e1e5e9', lw=.7)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, edgecolor='#dddddd', fontsize=10)
    h,w = data['image_shape']
    fig.suptitle(f'Fig. 6  |  Reconstruction progress and sampling curves — {data["image_id"]}', x=.025, ha='left',
                 y=.995, fontsize=14, fontweight='bold')
    seeds = data['methods']['adaptive']['seed']
    fig.text(.025, .895, f'{h} × {w}   |   M{start} → {K}   |   B{data["batch_size"]}   |   seed {seeds}   |   Intermediate images: centered pseudoinverse',
             fontsize=10, color='#555555')
    encoding = [Line2D([],[], color='#555555', marker='o', mfc='white', lw=1.5,
                       label='Pseudoinverse (all sampled M)'),
                Line2D([],[], color='#555555', marker='D', ls='None',
                       label='Final reconstruction (DiffPIR mean)')]
    fig.legend(handles=encoding, loc='lower left', bbox_to_anchor=(.62,.065),
               frameon=False, ncol=2, fontsize=9.5, handlelength=2)
    path = output / f'fig6_{data["image_id"]}.png'
    save_figure(fig, path, dpi=dpi, facecolor='white')
    save_figure(fig, path.with_suffix('.emf'), dpi=dpi, facecolor='white')
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-id', default='barbara')
    parser.add_argument('--results', type=Path, default=ROOT/'outputs/ablation')
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/set11_proxunroll/native')
    parser.add_argument('--budgets', type=int, nargs='+', default=[160,288,416,512,655])
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/paper_figures/fig6')
    parser.add_argument('--plot-data', type=Path, help='Redraw saved NPZ without source PT/JSON')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.plot_data:
        with np.load(args.plot_data, allow_pickle=False) as saved:
            data = json.loads(str(saved['metadata_json'].item()))
            require('truth' in saved, 'Old curve-only NPZ: run once without --plot-data to add reconstructions')
            arrays = {key:saved[key].copy() for key in ['truth','reconstruction_M',
                      'fixed_reconstructions','adaptive_reconstructions']}
    else:
        torch.set_num_threads(args.cpu_threads)
        data = collect(args.results, args.image_id)
        arrays = replay_prefixes(data, args.dataset, args.budgets)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f'fig6_{data["image_id"]}'
    for method, record in data['methods'].items():
        arrays[f'{method}_M'] = np.array([r['M'] for r in record['pinv_history']], dtype=np.int64)
        arrays[f'{method}_pinv_psnr'] = np.array([r['psnr'] for r in record['pinv_history']])
        arrays[f'{method}_final_psnr'] = np.array(record['final']['psnr'])
    np.savez_compressed(str(stem)+'_data.npz', **arrays,
                        metadata_json=np.array(json.dumps(data, ensure_ascii=False)))
    stem.with_suffix('.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(draw(data, arrays, args.output, args.dpi))
    print(json.dumps({'final_adaptive_minus_fixed_db': data['final_adaptive_minus_fixed_db'],
                      'source_reuse': data['source_reuse']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
