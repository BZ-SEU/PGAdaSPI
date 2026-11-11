#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 5: executed E32 codes, Walsh reference, physical rows and spectra.

Example (from the repository root):
  /home/baojr/.conda/envs/PIE2E/bin/python figures/fig5_pattern_design.py

Read-only visualization of outputs/ablation/full; no training,
selection, posterior sampling, or new reconstruction experiment. Defaults:
barbara, batch M=384 -> 416, within-batch rows 1/9/17/25 (chosen without GT).
Walsh reference = first B unused canonical pool IDs at the SAME archived
state, NOT the reserved IDs attached to E32 and NOT a separate fixed run.
E32 positive codes are read verbatim from PT. Effective rows are recomputed
from these quantized codes with the archived incoherent intensity optics.
Frequency maps show fftshift |FFT2(a - spatial_mean(a))|^2 divided by each
row's total non-DC power, in dB relative to ONE maximum across BOTH batches.
Effective-row images are divided by their individual RMS for shape comparison;
original RMS values are printed. These are display normalizations only.
DC is removed for frequency visualization only; physical rows are unchanged.
GT feature maps are offline annotations, not design inputs.
Outputs: one PNG and scalar/provenance JSON, no arrays or scene cache.
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy.ndimage import sobel, uniform_filter
import torch

from daps_spi.data import load_image_template
from daps_spi.io import digest, file_hash
from daps_spi.patterns import pattern_batch
from daps_spi.propagation_pinv import OpticalConfig, IntensityOptics


def require(value, message):
    if not value:
        raise ValueError(message)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=ROOT/'outputs/ablation/full')
    p.add_argument('--dataset', type=Path, default=ROOT/'data/set11_proxunroll/native')
    p.add_argument('--image-id', default='barbara')
    p.add_argument('--before-m', type=int, default=384, help='Measured prefix BEFORE the displayed batch')
    p.add_argument('--rows', nargs=4, type=int, default=[1,9,17,25], help='Four distinct 1-based batch row positions')
    p.add_argument('--device', default='cpu', help='CPU suffices; no neural network is loaded')
    p.add_argument('--db-floor', type=float, default=-60.)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/paper_figures/fig5')
    p.add_argument('--plot-data', type=Path, help='Redraw saved figure NPZ')
    return p.parse_args()


def draw(args, name, before, after, truth, gradients, contrast,
         codes, rows, spectra, sensitivity, ids):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'pdf.fonttype':42})
    fig, axes = plt.subplots(7, 4, figsize=(12.4, 17.3))
    fig.subplots_adjust(left=.17, right=.9, top=.93, bottom=.035, wspace=.12, hspace=.26)
    top = [truth, gradients, contrast, sensitivity[1]]
    titles = ['GT (offline reference)', 'GT gradient magnitude', 'GT local contrast (7×7)', 'E32 batch RMS / max']
    top[-1] = sensitivity[1]/sensitivity[1].max()
    for j, (arr, title) in enumerate(zip(top, titles)):
        axes[0,j].imshow(arr, cmap='gray' if j == 0 else 'magma', vmin=0,
                         vmax=1 if j in (0,3) else float(arr.max()),
                         interpolation='nearest')
        axes[0,j].set_title(title, fontsize=10, pad=6)
    positions = np.array(args.rows)-1
    row_rms = np.sqrt(np.mean(rows**2,axis=(-2,-1),keepdims=True))
    display_rows = rows/row_rms
    row_max = float(np.abs(display_rows).max())
    extent = [-.5, .5, -.5, .5]
    row_labels = ['Image structure', 'Walsh reference\npositive code', 'Executed E32\npositive code',
                  'Walsh effective row\n(RMS normalized)', 'E32 effective row\n(RMS normalized)',
                  'Walsh row spectrum', 'E32 row spectrum']
    for method in range(2):
        for j, index in enumerate(positions):
            ax = axes[1+method,j]
            ax.imshow(codes[method,index], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            label = f'Walsh ID {ids[0][index]}' if method == 0 else f'exposure row {before+index+1}'
            ax.set_title(f'Batch row {index+1} | {label}', fontsize=9, pad=4)
            ax = axes[3+method,j]
            ax.imshow(display_rows[method,index], cmap='RdBu_r', vmin=-row_max, vmax=row_max, interpolation='nearest')
            ax.set_title(f'Original RMS: {row_rms[method,index,0,0]:.2e}', fontsize=9, pad=4)
            ax = axes[5+method,j]
            ax.imshow(spectra[method,index], cmap='magma', vmin=args.db_floor, vmax=0,
                      extent=extent, origin='lower', interpolation='nearest')
    for i in range(7):
        for j in range(4):
            ax = axes[i,j]
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([]); ax.set_yticks([])
            if i >= 5:
                ax.set_xticks([-.5,0,.5]); ax.set_yticks([-.5,0,.5])
                ax.tick_params(labelsize=7, length=2, pad=1)
                if i == 6:
                    ax.set_xlabel(r'$f_x$ (cycles/pixel)', fontsize=8, labelpad=2)
            if j == 0:
                ax.set_ylabel(row_labels[i], rotation=0, ha='right', va='center', labelpad=12, fontsize=10)
    for first, last, norm, cmap, label in [
        (1,2,Normalize(0,255),'gray','Positive exposure code (uint8)'),
        (3,4,Normalize(-row_max,row_max),'RdBu_r','Effective row / row RMS'),
        (5,6,Normalize(args.db_floor,0),'magma','Unit-energy power (dB; common reference)')]:
        y0 = axes[last,-1].get_position().y0
        y1 = axes[first,-1].get_position().y1
        cax = fig.add_axes([.92,y0,.012,y1-y0])
        cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=cax,
                          extend='min' if first == 5 else 'neither')
        cb.set_label(label, fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.suptitle(f'Fig. 5 | Pattern design and effective-row spectra — {name}\n'
                 f'Sealed E32 batch: M = {before} → {after}  |  Four fixed row positions', fontsize=15, y=.997)
    return fig


def main():
    args = parse_args()
    if args.plot_data:
        with np.load(args.plot_data, allow_pickle=False) as z:
            report = json.loads(str(z['metadata']))
            args.rows = report['shown_rows_1based']; args.db_floor = report['spectrum']['db_floor']
            fig = draw(args, report['image_id'], report['before_m'], report['after_m'],
                       *[z[k] for k in ['truth','gradients','contrast','codes','rows','spectra','sensitivity','ids']])
        stem = args.output / args.plot_data.stem.removesuffix('_data')
        for ext in ['png','emf']:
            save_figure(fig, stem.with_suffix('.'+ext), dpi=args.dpi, bbox_inches='tight', pad_inches=.15, facecolor='white')
        plt.close(fig)
        print(stem)
        return
    require(np.isfinite(args.db_floor) and args.db_floor < 0, 'db-floor must be negative and finite')
    require(args.dpi >= 72, 'dpi must be >=72')
    require(Path(args.image_id).name == args.image_id, 'image-id must be a basename')
    source = args.results/args.image_id/(args.image_id+'.json')
    meta = json.loads(source.read_text()); settings = meta['settings']; cfg = settings['config']
    require(meta['status'] == 'complete' and meta['method'] == 'propagation-full-pseudoinverse', 'Invalid source result')
    require(settings['recipe']['options']['candidate_design'] == {'name':'eigen_v1','rows':32}, 'Expected E32')
    require(not any('V' in x for x in settings['chain_seeds']), 'Expected no-gate source trajectory')
    require(meta['patterns']['shape'] == [655,256,256], 'Expected native Set11 256x256 / M655')
    require(cfg['initial_measurements'] == 64 and cfg['batch_size'] == 32, 'Expected M64/B32')
    require(64 <= args.before_m < 655 and (args.before_m-64)%32 == 0, 'before-m must precede an archived adaptive batch')
    end = min(args.before_m+32,655); count = end-args.before_m
    require(len(set(args.rows)) == 4 and all(1 <= x <= count for x in args.rows), 'rows must be distinct valid batch positions')
    for protected in [args.results,args.dataset]:
        require(not args.output.resolve().is_relative_to(protected.resolve()), 'Output must be outside source data')
    pt = source.with_suffix('.pt')
    require(file_hash(pt) == meta['pt_sha256'], 'PT hash mismatch')
    payload = torch.load(pt, map_location='cpu', weights_only=True)
    plus = payload['plus_uint8']
    require(plus.dtype == torch.uint8 and list(plus.shape) == [655,256,256], 'Invalid uint8 pattern shape')
    require(digest(plus) == meta['patterns']['plus_hash'], 'Pattern hash mismatch')
    boundaries = [x['M'] for x in payload['history']]
    require(args.before_m in boundaries and end in boundaries, 'Missing batch boundaries')
    canonical = settings['pattern_pool']['canonical_ids']
    require(digest(canonical) == settings['pattern_pool']['sha256'], 'Pool hash mismatch')
    used = set(meta['patterns']['ids'][:args.before_m])
    reference_ids = [i for i in canonical if f'walsh:{i}' not in used][:count]
    require(len(reference_ids) == count, 'Insufficient unused Walsh IDs')
    filename = settings['evaluation_dataset']['image_files'][args.image_id]
    image_path = args.dataset/filename
    require(file_hash(image_path) == settings['image_sha256'], 'GT identity mismatch')
    truth, _ = load_image_template(image_path,256,256)
    physics = settings['optics']['parameters']
    require(physics['image_pitch'] == physics['dmd_pitch'] == 10.8e-6, 'Incorrect physical pitch')
    torch.set_num_threads(4)
    optics = IntensityOptics(OpticalConfig(**physics),args.device)
    reference = pattern_batch(reference_ids,256,256,args.device,cfg.get('pattern_pool'))
    actual = plus[args.before_m:end].flatten(1).to(args.device)
    # Reserved IDs define original centers for the physical RMS bound, not reference selection.
    reserved_ids = [int(x.split(':')[1]) for x in meta['patterns']['ids'][args.before_m:end]]
    centers = pattern_batch(reserved_ids,256,256,args.device,cfg.get('pattern_pool')).continuous
    signed = (2*actual.double()-255)/255
    rms = ((signed-centers).square().mean(1)/centers.square().mean(1)).sqrt()
    require(float(rms.max()) <= cfg['optimization']['rms_bound']+1e-12, 'Executed codes exceed configured RMS bound')
    code_batches = [reference.plus,actual]
    effective = [optics.effective_rows((2*c.double()-255)/255) for c in code_batches]
    # Independent forward calculation checks both visualization operators against the physics.
    residuals = []
    for codes, a in zip(code_batches,effective):
        y = optics.observe(truth.to(args.device).double(),codes)
        residual = float((a@truth.to(args.device).double().flatten()-y).abs().max())
        require(residual <= 1e-6*(1+float(y.abs().max())), 'Physical forward/adjoint mismatch')
        residuals.append(residual)
    rows = torch.stack(effective).reshape(2,count,256,256).cpu().numpy()
    codes = torch.stack(code_batches).reshape(2,count,256,256).cpu().numpy()
    centered = rows-rows.mean(axis=(-2,-1),keepdims=True)
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered,norm='ortho'),axes=(-2,-1)))**2
    total_power = power.sum(axis=(-2,-1),keepdims=True)
    require(np.isfinite(power).all() and bool((total_power > 0).all()), 'Invalid spectrum')
    normalized_power = power/total_power
    peak = float(normalized_power.max())
    spectra = 10*np.log10(np.maximum(normalized_power/peak,10**(args.db_floor/10)))
    sensitivity = np.sqrt(np.mean(rows**2,axis=1))
    gt = truth.numpy().astype(np.float64)
    gradients = np.hypot(sobel(gt,axis=0,mode='reflect')/8,sobel(gt,axis=1,mode='reflect')/8)
    contrast = np.sqrt(np.maximum(uniform_filter(gt**2,size=7)-uniform_filter(gt,size=7)**2,0))
    fig = draw(args,args.image_id,args.before_m,end,gt,gradients,contrast,
               codes,rows,spectra,sensitivity,[reference_ids,reserved_ids])
    args.output.mkdir(parents=True,exist_ok=True)
    stem = args.output/f'fig5_{args.image_id}_M{args.before_m}_{end}'
    for ext in ['png', 'emf']:
        save_figure(fig, stem.with_suffix('.'+ext),dpi=args.dpi,bbox_inches='tight',pad_inches=.15,facecolor='white')
    plt.close(fig)
    report = dict(image_id=args.image_id,source_json=str(source.resolve()),source_json_sha256=file_hash(source),
                  source_pt_sha256=meta['pt_sha256'],image_sha256=settings['image_sha256'],
                  script_sha256=file_hash(__file__),physics=physics,device=args.device,
                  core_hashes={p.name:file_hash(p) for p in sorted((ROOT/'src/daps_spi').glob('*.py'))},
                  before_m=args.before_m,after_m=end,batch_rows=count,shown_rows_1based=args.rows,
                  reference_ids=reference_ids,e32_reserved_ids=reserved_ids,
                  reference_definition='First B unused canonical Walsh IDs at archived state; synthetic reference, not executed fixed baseline',
                  e32_definition='Verbatim saved positive uint8 exposures; negative code = 255 - positive',
                  effective_row_definition=meta['patterns']['effective_row'],
                  max_relative_rms=float(rms.max()),rms_bound=cfg['optimization']['rms_bound'],
                  forward_residuals=dict(zip(['walsh','e32'],residuals)),
                  spectrum=dict(transform='orthonormal FFT2 after spatial mean removal, fftshift',
                                power_reference=peak,db_floor=args.db_floor,
                                normalization='unit total non-DC energy per row, then single peak across all rows of both batches',
                                original_non_dc_power=total_power[:,:,0,0].tolist(),
                                frequency_unit='cycles/pixel',pixel_pitch_um=10.8,
                                zero_frequency_removed=True),
                  structure=dict(gradient='Sobel magnitude /8, reflect boundary',
                                 local_contrast='7x7 local population standard deviation, reflect boundary',
                                 sensitivity='sqrt(mean over whole batch of effective row squared)); no DC removal; displayed divided by E32 maximum',
                                 scope='Qualitative spatial correspondence; GT is not used to choose image, batch, rows or patterns'),
                  display=dict(code_range=[0,255],effective_row_abs_max=float(np.abs(rows).max()),
                               effective_row_normalization='individual row RMS, common diverging color scale',
                               original_row_rms=np.sqrt(np.mean(rows**2,axis=(-2,-1))).tolist(),
                               spectrum_db_range=[args.db_floor,0],
                               top_feature_scaling='gradient, local contrast and E32 sensitivity each use their own maximum'))
    np.savez_compressed(stem.with_name(stem.name+'_data').with_suffix('.npz'), metadata=np.array(json.dumps(report)),
                        truth=gt, gradients=gradients, contrast=contrast, codes=codes, rows=rows,
                        spectra=spectra, sensitivity=sensitivity, ids=np.array([reference_ids,reserved_ids]))
    stem.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(f'Complete: {stem}.png / .json',flush=True)
    print(f'Forward residuals (Walsh, E32): {residuals}',flush=True)


if __name__ == '__main__':
    main()
