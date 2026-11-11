#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 3: offline posterior uncertainty along an archived Set11 trajectory.

Run from any directory with the project's PIE2E Python:
  python figures/fig3_posterior_uncertainty.py
  python figures/fig3_posterior_uncertainty.py --image-id barbara --device cuda:0

Each column is one completed measurement batch. Rows: mean of all 32 O samples,
the first three O chains (never selected by GT), sample standard deviation
(ddof=1, all 32 *unshrunk, unclipped* O samples).
The mean and standard deviation use unshrunk, unclipped samples. Pseudoinverse
metrics remain internal replay checks and are not displayed in the figure.
Image panels alone use display clipping [0,1]. The standard-deviation row shares a
linear scale across all columns; values over --heatmap-max saturate only
in the rendering and their fractions are reported in JSON.

The default source is outputs/ablation/full. This is a read-only
replay, not a new adaptive experiment or training run. Saved uint8
patterns are already sealed; GT is used only for physical bucket simulation
and offline evaluation. Original O seeds/settings are reused at intermediate
stages. At the terminal budget a new O ensemble is drawn for visualization
with the next decision index, NOT the F final decoder (independent seeds). Historical
V-gated trajectory provenance is disclosed when present in the source JSON.
Completed plotting data (all O samples, mean, std, and provenance) are saved
as a compressed NPZ, without optimization state. Existing compatible plotting
data are reused automatically; --resample explicitly regenerates them.
Use --plot-data PATH to redraw directly on CPU from a saved NPZ.
Outputs: publication 300-dpi PNG, scalar/provenance JSON, and plotting NPZ. All code
specific to this figure lives here; core physics/sampling are imported unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
import torch

from daps_spi.compact_output import metrics
from daps_spi.data import load_image_template
from daps_spi.io import digest, file_hash
from daps_spi.priors import load_prior
from daps_spi.propagation_pinv import IntensityOptics, OpticalConfig, PseudoinverseDecoder, MeasurementFactor
from daps_spi.sampling import chain_seeds, sample
from daps_spi.types import MeasurementState


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_hash(path, expected):
    actual = file_hash(path)
    require(actual == expected, f'Hash mismatch: {path}')
    return actual


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=ROOT / 'outputs/ablation/full')
    p.add_argument('--dataset', type=Path, default=ROOT / 'data/set11_proxunroll/native')
    p.add_argument('--image-id', help='Default: seeded random choice among completed native 256x256 images')
    p.add_argument('--selection-seed', type=int, default=20261016)
    p.add_argument('--measurements', nargs='+', type=int, default=[64, 192, 384, 655])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--plot-data', type=Path, help='Redraw from saved NPZ without sampling or loading the prior')
    p.add_argument('--resample', action='store_true', help='Ignore existing plotting data and regenerate O samples')
    p.add_argument('--shown-samples', type=int, default=3)
    p.add_argument('--heatmap-max', type=float, default=0.08, help='Fixed physical-intensity scale for sample std')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_figures/fig3')
    return p.parse_args()


def select_source(args):
    candidates = []
    for path in sorted(args.results.glob('*/*.json')):
        meta = json.loads(path.read_text())
        if meta.get('status') != 'complete' or meta.get('patterns', {}).get('shape') != [655, 256, 256]:
            continue
        name = meta['image_id']
        filename = meta['settings']['evaluation_dataset']['image_files'][name]
        image_path = args.dataset / filename
        if image_path.exists():
            with Image.open(image_path) as im:
                if im.size == (256, 256):
                    candidates.append((name, path, image_path))
    require(candidates, 'No completed native Set11 256x256 source results found')
    if args.image_id:
        chosen = [x for x in candidates if x[0] == args.image_id]
        require(len(chosen) == 1, f'Unavailable native image: {args.image_id}')
        selected = chosen[0]
    else:
        selected = random.Random(args.selection_seed).choice(candidates)
    return selected, [x[0] for x in candidates]


def draw(columns, args, name, historical):
    rows = args.shown_samples + 2
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.titleweight': 'normal'})
    fig, axes = plt.subplots(rows, len(columns), squeeze=False,
                             figsize=(2.25 * len(columns) + 1.8, 2.12 * rows + 1.05))
    # Keep room for the title and footnote when fewer sample rows are shown.
    height = fig.get_figheight()
    fig.subplots_adjust(left=.16, right=.925, top=min(.925, 1-.9/height),
                        bottom=.035, wspace=.035, hspace=.12)
    labels = ['O-sample mean'] + [rf'$O_{{{i+1}}}$ sample' for i in range(args.shown_samples)]
    labels += [r'Sample std. $s_O$']
    for j, col in enumerate(columns):
        arrays = [col['sample_mean']] + list(col['shown']) + [col['std']]
        for i, arr in enumerate(arrays):
            ax = axes[i, j]
            ax.imshow(arr, cmap='gray' if i < rows-1 else 'magma', vmin=0,
                      vmax=1 if i < rows-1 else args.heatmap_max, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(labels[i], rotation=0, ha='right', va='center', labelpad=10)
        axes[0, j].set_title(rf'$M={col["M"]}$' + f'\n{100*col["M"]/65536:.3f}% measurements', pad=9)
        axes[-1, j].set_xlabel(f'Mean std. {col["summary"]["mean_std"]:.4f}', labelpad=2, fontsize=9)
    bottom = axes[-1, -1].get_position().y0
    top = axes[-1, -1].get_position().y1
    cax = fig.add_axes([.943, bottom, .014, top-bottom])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, args.heatmap_max), cmap='magma'), cax=cax, extend='max' if any((c['std'] > args.heatmap_max).any() for c in columns) else 'neither')
    cb.set_label('Sample std. (physical intensity)', fontsize=9)
    cb.set_ticks(np.linspace(0, args.heatmap_max, 5))
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(f'Fig. 3 | Posterior uncertainty evolution — {name}', y=.998, fontsize=15)
    provenance = 'Archived trajectory (historical V gate)' if historical else 'Archived E32 trajectory (no validation gate)'
    return fig


def export_figure(columns, report, args, data_path):
    """Rendering only: never invokes a prior, optics, or a CUDA operation."""
    for col in columns:
        col['shown'] = col['samples'][:args.shown_samples]
        col['summary']['std_saturated_fraction'] = float((col['std'] > args.heatmap_max).mean())
    report = copy.deepcopy(report)
    report.update(script_sha256=file_hash(__file__), heatmap_range=[0, args.heatmap_max],
                  shown_chain_indices=list(range(args.shown_samples)),
                  columns=[c['summary'] for c in columns],
                  plotting_data=str(data_path.resolve()), plotting_data_sha256=file_hash(data_path))
    fig = draw(columns, args, report['image_id'], report['historical_V_gate'])
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f'fig3_{report["image_id"]}'
    for extension in ['png', 'emf']:
        save_figure(fig, stem.with_suffix('.'+extension), dpi=args.dpi, facecolor='white',
                    bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f'Finished: {stem}.png / .json; plotting data: {data_path}', flush=True)


def read_plot_data(path, measurements):
    with np.load(path, allow_pickle=False) as data:
        require(int(data['format_version']) == 2, 'Unsupported plotting data version')
        report = json.loads(str(data['metadata']))
        summaries = {c['M']: c for c in report['columns']}
        require(all(m in summaries for m in measurements), 'Requested M is absent from plotting data')
        columns = []
        for m in measurements:
            samples = data[f'O_{m}']
            mean = data[f'mean_{m}']; std = data[f'std_{m}']
            require(samples.shape == (32, 256, 256) and mean.shape == std.shape == (256, 256),
                    f'Invalid plotting data dimensions at M={m}')
            require(all(np.isfinite(a).all() for a in (samples, mean, std)), 'Nonfinite plotting data')
            columns.append(dict(M=m, samples=samples, sample_mean=mean, std=std, summary=summaries[m]))
    return columns, report


def main():
    args = parse_args()
    require(1 <= args.shown_samples <= 32, 'shown-samples must be in [1,32]')
    require(math.isfinite(args.heatmap_max) and args.heatmap_max > 0, 'heatmap-max must be positive and finite')
    require(args.dpi >= 72, 'dpi must be at least 72')
    require(args.measurements == sorted(set(args.measurements)), 'Measurement columns must be unique and increasing')
    require(not (args.plot_data and args.resample), '--plot-data and --resample are mutually exclusive')
    args.output = args.output.resolve()
    require(not args.output.is_relative_to(args.results.resolve()), 'Output must be outside production results')
    require(not args.output.is_relative_to(args.dataset.resolve()), 'Output must be outside dataset')
    if args.plot_data:
        columns, report = read_plot_data(args.plot_data, args.measurements)
        require(not args.image_id or args.image_id == report['image_id'], 'Plotting data image does not match --image-id')
        print(f'Redrawing from {args.plot_data}; no sampling', flush=True)
        export_figure(columns, report, args, args.plot_data)
        return
    (name, source_path, image_path), candidates = select_source(args)
    meta = json.loads(source_path.read_text())
    settings = meta['settings']; cfg = settings['config']; recipe = settings['recipe']
    require(meta['method'] == 'propagation-full-pseudoinverse', 'Wrong source method')
    require(recipe['options']['candidate_design']['name'] == 'eigen_v1', 'Expected E32 trajectory')
    args.output = args.output.resolve()
    require(not args.output.is_relative_to(args.results.resolve()), 'Output must be outside production results')
    require(not args.output.is_relative_to(args.dataset.resolve()), 'Output must be outside dataset')
    pt_path = source_path.with_suffix('.pt')
    checked_hash(pt_path, meta['pt_sha256'])
    checked_hash(image_path, settings['image_sha256'])
    prior_path = ROOT / recipe['prior']; mean_path = ROOT / recipe['mean']
    prior_hash = checked_hash(prior_path, settings['checkpoint_sha256'])
    checked_hash(mean_path, settings['mean_sha256'])
    frozen = json.loads((ROOT / 'assets/frozen_assets.json').read_text())
    require(prior_hash == frozen['outputs/prior_daps_v2/final_65k.pt'], 'Expected frozen Set11 v2 prior')
    data_path = args.output / f'fig3_{name}_data.npz'
    core_hashes = {p.name:file_hash(p) for p in sorted((ROOT/'src/daps_spi').glob('*.py'))}
    if data_path.exists() and not args.resample:
        columns, report = read_plot_data(data_path, args.measurements)
        require(report['source_json_sha256'] == file_hash(source_path)
                and report['source_pt_sha256'] == meta['pt_sha256']
                and report['current_core_hashes'] == core_hashes,
                'Plotting data provenance changed; use --resample to regenerate')
        print(f'Reusing {data_path}; no sampling', flush=True)
        export_figure(columns, report, args, data_path)
        return
    payload = torch.load(pt_path, map_location='cpu', weights_only=True)
    plus = payload['plus_uint8']
    require(plus.dtype == torch.uint8 and list(plus.shape) == [655, 256, 256], 'Invalid physical pattern tensor')
    require(digest(plus) == meta['patterns']['plus_hash'], 'Executed pattern hash mismatch')
    ids = meta['patterns']['ids']
    require(len(ids) == len(plus) and len(set(ids)) == len(ids), 'Invalid reserved pattern IDs')
    history = {item['M']: item for item in payload['history']}
    require(all(m in history for m in args.measurements), 'Requested M is not an archived batch boundary')
    require(cfg['initial_measurements'] == 64 and cfg['batch_size'] == 32, 'Expected M64/B32')
    require(meta['patterns']['max_relative_rms'] <= cfg['optimization']['rms_bound'] + 1e-12, 'Source exceeds RMS bound')
    physics = settings['optics']['parameters']
    require(physics['image_pitch'] == physics['dmd_pitch'] == 10.8e-6, 'Incorrect physical pitch')
    sampling = copy.deepcopy(cfg['sampling'])
    sampling.pop('validation_scenarios', None)
    require(sampling['num_scenarios'] == 32 and sampling.get('sampler') == 'diffpir' and sampling['diffpir_steps'] == 50, 'Expected full O32 / DiffPIR50')
    sampling['image_shape'] = [256, 256]
    historical = any('V' in entry for entry in settings.get('chain_seeds', []))
    print(f'Selected {name}; M={args.measurements}; O32/DiffPIR50; device={args.device}; historical_V={historical}', flush=True)
    torch.set_num_threads(4)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
    truth, _ = load_image_template(image_path, 256, 256)
    truth = truth.to(args.device).double()
    mu = torch.load(mean_path, map_location=args.device, weights_only=True)['mu'].flatten().double()
    optics = IntensityOptics(OpticalConfig(**physics), args.device)
    net, _, loaded_hash = load_prior(str(prior_path), args.device)
    require(loaded_hash == prior_hash, 'Prior changed during loading')
    columns = []; begin = time.perf_counter()
    for m in args.measurements:
        print(f'M={m}: rebuilding sealed measurement prefix', flush=True)
        codes = plus[:m].flatten(1).to(args.device)
        A = torch.cat([optics.effective_rows((2*c.double()-255)/255) for c in codes.split(32)])
        y = optics.observe(truth, codes)
        tolerance = 1e-6 * (1 + float(y.abs().max()))
        require(float((A @ truth.flatten()-y).abs().max()) <= tolerance, 'Optical forward/adjoint mismatch')
        state = MeasurementState(A, y, codes, ids[:m])
        factor = MeasurementFactor(state,recipe['rank_rtol']) if cfg.get('shared_factor',False) else None
        decoder = PseudoinverseDecoder(state, mu, recipe['rank_rtol'],factor)
        pinv = decoder.xhat.reshape(256, 256)
        pinv_residual = float((A @ pinv.flatten()-y).abs().max())
        require(pinv_residual <= tolerance, 'Pseudoinverse measurement residual failed')
        measured = metrics(pinv, truth, m, 0.)
        reference = settings['final_pinv'] if m == len(plus) else history[m]
        require(abs(measured['psnr']-reference['psnr']) < 1e-3, 'Replayed pinv disagrees with archived PSNR')
        terminal = m == len(plus)
        decision = math.ceil((m-cfg['initial_measurements']) / cfg['batch_size'])
        seeds = (chain_seeds(cfg['seed'], settings['image_index'], decision, 'O', 32)
                 if terminal else settings['chain_seeds'][decision]['O'])
        require(len(seeds) == 32 and len(set(seeds)) == 32, 'Expected 32 independent O seeds')
        print(f'M={m}: sampling 32 O chains (DiffPIR50, 49 NFE/chain)', flush=True)
        ensemble = sample(net, state, sampling, seeds, 'O', prior_hash,factor)
        require(ensemble.diagnostics['rank'] == m, 'DiffPIR projection truncated a measurement direction')
        samples = ensemble.samples
        require(bool(torch.isfinite(samples).all()), 'Nonfinite posterior sample')
        residual = float((samples @ A.T-y).abs().max())
        require(residual <= tolerance, 'O sample measurement residual failed')
        std = samples.std(dim=0, unbiased=True).reshape(256, 256)
        sample_mean = samples.mean(dim=0).reshape(256, 256)
        summary = dict(M=m, decision=decision, terminal_diagnostic=terminal,
                       seeds=seeds, pinv=measured, mean_std=float(std.mean()),
                       rms_std=float(std.square().mean().sqrt()),
                       std_saturated_fraction=float((std > args.heatmap_max).double().mean()),
                       pinv_residual=pinv_residual, sample_residual=residual,
                       residual_tolerance=tolerance, condition=decoder.condition,
                       projection_rank=ensemble.diagnostics['rank'],
                       sample_seconds=ensemble.diagnostics['seconds'],
                       sampler=ensemble.diagnostics['sampler'],nfe_per_chain=ensemble.diagnostics['nfe_per_chain'])
        columns.append(dict(M=m, sample_mean=sample_mean.cpu().numpy(), samples=samples.reshape(-1,256,256).cpu().numpy(),
                            std=std.cpu().numpy(), summary=summary))
        print(f'M={m}: PSNR={measured["psnr"]:.4f}, mean std={summary["mean_std"]:.5f}, residual={residual:.3e}', flush=True)
        del ensemble, samples, decoder, state, A, y, codes
    report = dict(image_id=name, candidate_images=candidates, selection_seed=args.selection_seed,
                  explicitly_selected=bool(args.image_id), source_json=str(source_path),
                  source_json_sha256=file_hash(source_path), source_pt_sha256=meta['pt_sha256'],
                  image_sha256=settings['image_sha256'], prior_sha256=prior_hash,
                  mean_sha256=settings['mean_sha256'], source_run_identity=settings['run_identity'],
                  historical_V_gate=historical, device=args.device, torch_version=torch.__version__,
                  sampling=sampling, physics=physics, script_sha256=file_hash(__file__),
                  current_core_hashes={p.name:file_hash(p) for p in sorted((ROOT/'src/daps_spi').glob('*.py'))},
                  definitions=dict(sample_mean='mean of all 32 unshrunk/unclipped O samples; not final F reconstruction',
                                   std='unshrunk/unclipped O sample std, ddof=1, K=32',
                                   terminal='diagnostic O32/DiffPIR50, not final F32/DiffPIR50 reconstruction',
                                   replay='archived sealed patterns; buckets regenerated by physical simulation; not a new selection run',
                                   display='grayscale [0,1]; sample std heatmaps share fixed linear scale across columns'),
                  heatmap_range=[0,args.heatmap_max], shown_chain_indices=list(range(args.shown_samples)),
                  seconds=time.perf_counter()-begin, columns=[c['summary'] for c in columns])
    report['sampling_script_sha256'] = file_hash(__file__)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = dict(format_version=np.array(2), metadata=np.array(json.dumps(report, allow_nan=False)))
    for col in columns:
        m = col['M']
        arrays.update({f'O_{m}': col['samples'], f'mean_{m}': col['sample_mean'], f'std_{m}': col['std']})
    # Write only a complete, validated dataset; never expose a partial cache.
    temporary = data_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(data_path)
    export_figure(columns, report, args, data_path)


if __name__ == '__main__':
    main()
