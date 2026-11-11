#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. S2: actual 1-bit alignment DMD patterns and their physical realization.

This is a read-only visualization of sealed ``abl_dmd_1bit_alignment``
outputs. The compact experimental artifact deliberately does not retain O
scenarios or per-row threshold-search traces, so this figure makes no claim to
show unavailable target directions. It shows the exact emitted positive uint8
codes, their physical propagated effective rows, their change from the
bookkeeping Walsh centre, and the effective-row spectra.

No sampling, selection, optimization, GT image, or future bucket value is
loaded. Only saved codes, source metadata, and the frozen physical operator
are used. The output is a PNG plus a scalar/provenance JSON.
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch

from daps_spi.io import digest, file_hash
from daps_spi.patterns import pattern_batch
from daps_spi.propagation_pinv import OpticalConfig, IntensityOptics


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT/'outputs/ablation/dmd_1bit_alignment',
                        help='Sealed 1-bit alignment result directory')
    parser.add_argument('--image-id', default='cameraman', help='Completed result image ID')
    parser.add_argument('--before-m', type=int, default=384,
                        help='Measurement count immediately before the displayed batch')
    parser.add_argument('--rows', nargs=4, type=int, default=[1, 9, 17, 25],
                        help='Four distinct 1-based rows inside that batch')
    parser.add_argument('--device', default='cpu', help='CPU is sufficient; no neural network is loaded')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/paper_figures/fig_s2')
    parser.add_argument('--plot-data', type=Path, help='Redraw either saved S2 variant')
    return parser.parse_args()


def normalized_spectrum(rows: np.ndarray, floor_db: float = -55.) -> tuple[np.ndarray, float]:
    """Per-row non-DC unit-energy spectra, expressed against one common peak."""
    centred = rows - rows.mean(axis=(-2, -1), keepdims=True)
    power = np.abs(np.fft.fftshift(np.fft.fft2(centred, norm='ortho'), axes=(-2, -1)))**2
    total = power.sum(axis=(-2, -1), keepdims=True)
    require(bool((total > 0).all()), 'effective row has zero non-DC spectral energy')
    normalized = power / total
    peak = float(normalized.max())
    db = 10*np.log10(np.maximum(normalized/peak, 10**(floor_db/10)))
    return db, peak


def draw(name: str, before: int, after: int, positions: list[int], codes: np.ndarray,
         signed: np.ndarray, centers: np.ndarray, physical_rows: np.ndarray,
         spectra: np.ndarray, physical_limit: float) -> plt.Figure:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(4, 4, figsize=(12.2, 12.0))
    fig.subplots_adjust(left=.16, right=.88, top=.90, bottom=.035, wspace=.08, hspace=.19)
    difference = (signed-centers)/2.  # {-1,0,+1}; display only, not a physical row.
    labels = ['Emitted 1-bit\npositive code $P^+$', 'Physical effective row $a_j$\n(RMS normalized)',
              'Signed-code change from\nreserved Walsh centre $(p_j-w_j)/2$',
              'Physical-row spectrum\n(non-DC unit energy, dB)']
    row_rms = np.sqrt(np.mean(physical_rows**2, axis=(-2, -1), keepdims=True))
    display_rows = physical_rows/row_rms
    for col, pos in enumerate(positions):
        maps = [codes[pos], display_rows[pos], difference[pos], spectra[pos]]
        settings = [('gray', 0, 255), ('RdBu_r', -physical_limit, physical_limit),
                    ('RdBu_r', -1, 1), ('magma', -55, 0)]
        for row, (array, (cmap, vmin, vmax)) in enumerate(zip(maps, settings)):
            ax = axes[row, col]
            ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f'Batch row {pos+1}', fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(labels[row], rotation=0, ha='right', va='center', labelpad=15, fontsize=9)
    colorbars = [
        (0, Normalize(0, 255), 'gray', 'Positive DMD code (uint8)'),
        (1, Normalize(-physical_limit, physical_limit), 'RdBu_r', 'Effective row / row RMS'),
        (2, Normalize(-1, 1), 'RdBu_r', 'Signed-code change / 2'),
        (3, Normalize(-55, 0), 'magma', 'Unit-energy power (dB)'),
    ]
    for row, norm, cmap, label in colorbars:
        first = axes[row, 0].get_position()
        cax = fig.add_axes([.90, first.y0, .012, first.height])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle('Fig. S2 | Actual 1-bit alignment patterns and physical realization\n'
                 f'{name}  |  sealed batch: $M={before} \\rightarrow {after}$  |  four fixed within-batch rows',
                 fontsize=14, y=.995)
    return fig


def draw_target_directions(data, meta):
    """Original 8-bit target-direction figure from its preserved plotting cache."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    positions = np.array(meta['rows']) - 1
    u, a = data['directions'][positions], data['effective_rows'][positions]
    u = u / np.sqrt(np.mean(u*u, axis=(-2,-1), keepdims=True))
    a = a / np.sqrt(np.mean(a*a, axis=(-2,-1), keepdims=True))
    limit = float(np.quantile(np.abs(np.stack([u,a])), .99))
    patterns = 2*data['executed_codes'][positions].astype(float)/255-1
    fig, axes = plt.subplots(3,4,figsize=(12.4,9.1))
    fig.subplots_adjust(left=.16,right=.88,top=.87,bottom=.035,wspace=.09,hspace=.16)
    labels = [r'Target direction $u_j$',r'Quantized pattern $p_j$',r'Effective row $a_j$']
    for i, (arrays, cmap, bound) in enumerate([(u,'RdBu_r',limit),(patterns,'gray',1),(a,'RdBu_r',limit)]):
        for j, array in enumerate(arrays):
            ax=axes[i,j];ax.imshow(array,cmap=cmap,vmin=-bound,vmax=bound,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if i==0:ax.set_title(f'$j={positions[j]+1}$',pad=8)
            if j==0:ax.set_ylabel(labels[i],rotation=0,ha='right',va='center',labelpad=12)
        box=axes[i,-1].get_position()
        cax=fig.add_axes([.90,box.y0,.012,box.height])
        cb=fig.colorbar(ScalarMappable(norm=Normalize(-bound,bound),cmap=cmap),cax=cax,
                       extend='neither' if i==1 else 'both')
        cb.set_label('Signed quantized pattern' if i==1 else 'Amplitude / row RMS')
    fig.suptitle('Fig. S2 | Target directions and physical realization\n'
                 f'{meta["image_id"]} | Round $t={meta["round_t"]}$ | $M={meta["before_m"]}\\rightarrow {meta["after_m"]}$',
                 y=.995,fontsize=15)
    return fig


def main() -> None:
    args = parse_args()
    if args.plot_data:
        with np.load(args.plot_data, allow_pickle=False) as z:
            report=json.loads(str(z['metadata']))
            if 'directions' in z:
                fig=draw_target_directions(z,report)
            else:
                fig=draw(report['image_id'],report['before_m'],report['after_m'],
                         [x-1 for x in report['shown_rows_1based']],
                         *[z[k] for k in ['codes','signed','centres','physical_rows','spectra']],float(z['physical_limit']))
        stem=args.output/args.plot_data.stem.removesuffix('_data')
        for ext in ['png','emf']:
            save_figure(fig, stem.with_suffix('.'+ext),dpi=args.dpi,facecolor='white',bbox_inches='tight',pad_inches=.12)
        plt.close(fig)
        report.update(script_sha256=file_hash(__file__),plotting_data_sha256=file_hash(args.plot_data))
        stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        print(stem)
        return
    torch.set_num_threads(4)
    require(args.dpi >= 72, 'dpi must be >= 72')
    require(Path(args.image_id).name == args.image_id, 'image-id must be a basename')
    source_json = args.results/args.image_id/(args.image_id+'.json')
    source_pt = source_json.with_suffix('.pt')
    require(source_json.is_file() and source_pt.is_file(), 'completed source JSON/PT are required')
    require(not args.output.resolve().is_relative_to(args.results.resolve()), 'output must not be inside result artifacts')
    source = json.loads(source_json.read_text())
    require(source.get('status') == 'complete', 'source result is not complete')
    checks = source['patterns']['physical_checks']
    require(checks.get('dmd_bit_depth') == 1 and checks.get('binary_codes_only') is True,
            'source is not a verified 1-bit DMD result')
    cfg = source['settings']['config']; height, width = cfg['image_shape']
    before, batch_size = args.before_m, int(cfg['batch_size'])
    after = min(before+batch_size, int(source['final']['M']))
    require(after-before == batch_size, 'Fig. S2 requires a full archived batch')
    require(before >= int(cfg['initial_measurements']) and (before-int(cfg['initial_measurements'])) % batch_size == 0,
            'before-m must be an adaptive batch boundary')
    require(len(set(args.rows)) == 4 and all(1 <= x <= batch_size for x in args.rows),
            'rows must be four distinct valid 1-based batch positions')
    positions = [x-1 for x in args.rows]
    payload = torch.load(source_pt, map_location='cpu', weights_only=True)
    plus = payload['plus_uint8']
    require(plus.dtype == torch.uint8 and tuple(plus.shape) == (source['final']['M'], height, width),
            'saved positive-code tensor shape/dtype mismatch')
    require(file_hash(source_pt) == source['pt_sha256'], 'PT hash mismatch')
    require(digest(plus) == source['patterns']['plus_hash'], 'saved positive-code hash mismatch')
    selected = plus[before:after]
    require(bool(torch.all((selected == 0) | (selected == 255))), 'selected batch is not binary')
    ids = source['patterns']['ids'][before:after]
    require(len(ids) == batch_size and len(set(ids)) == batch_size, 'invalid reserved IDs')
    center_indices = [int(item.split(':', 1)[1]) for item in ids]
    centres = pattern_batch(center_indices, height, width, args.device, cfg.get('pattern_pool'), bit_depth=1).continuous
    physics = source['settings']['optics']['parameters']
    optics = IntensityOptics(OpticalConfig(**physics), args.device)
    signed = (2*selected.to(args.device).double().flatten(1)-255)/255
    effective = optics.effective_rows(signed)
    codes_np = selected.numpy().astype(np.float64)
    signed_np = signed.reshape(batch_size, height, width).cpu().numpy()
    centres_np = centres.reshape(batch_size, height, width).cpu().numpy()
    effective_np = effective.reshape(batch_size, height, width).cpu().numpy()
    spectra, spectrum_peak = normalized_spectrum(effective_np)
    row_rms = np.sqrt(np.mean(effective_np**2, axis=(1, 2)))
    display_rows = effective_np/row_rms[:, None, None]
    physical_limit = float(np.quantile(np.abs(display_rows), .99))
    duty = (codes_np/255).mean(axis=(1, 2))
    signed_flat = signed_np.reshape(batch_size, -1); center_flat = centres_np.reshape(batch_size, -1)
    centre_corr = (signed_flat*center_flat).sum(1)/np.linalg.norm(signed_flat, axis=1)/np.linalg.norm(center_flat, axis=1)
    fig = draw(args.image_id, before, after, positions, codes_np, signed_np, centres_np,
               effective_np, spectra, physical_limit)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output/'fig_s2_1bit_alignment'
    save_figure(fig, stem.with_suffix('.png'), dpi=args.dpi, facecolor='white')
    save_figure(fig, stem.with_suffix('.emf'), dpi=args.dpi, facecolor='white')
    plt.close(fig)
    report = dict(schema='fig-s2-1bit-alignment-v1', image_id=args.image_id, before_m=before, after_m=after,
                  shown_rows_1based=args.rows, source_json=str(source_json.resolve()),
                  source_json_sha256=file_hash(source_json), source_pt_sha256=source['pt_sha256'],
                  script_sha256=file_hash(__file__), dmd=dict(bit_depth=1, positive_levels=[0,255],
                  complement='P_minus = 255 - P_plus', source_binary_verified=True),
                  pattern_definition='p=(2*P_plus-255)/255, directly from sealed positive uint8 codes',
                  physical_row_definition=source['patterns']['effective_row'], physics=physics,
                  reservation_centres=dict(ids=ids, definition='canonical Walsh centres tied to the sealed action IDs; display reference only'),
                  batch_statistics=dict(duty_cycle=dict(mean=float(duty.mean()), minimum=float(duty.min()), maximum=float(duty.max()), values=duty.tolist()),
                                        signed_correlation_to_reservation_walsh=dict(mean=float(centre_corr.mean()), minimum=float(centre_corr.min()), maximum=float(centre_corr.max()), values=centre_corr.tolist()),
                                        effective_row_rms=dict(mean=float(row_rms.mean()), minimum=float(row_rms.min()), maximum=float(row_rms.max()), values=row_rms.tolist())),
                  spectrum=dict(non_dc_per_row_unit_energy=True, common_peak=spectrum_peak, display_floor_db=-55.),
                  provenance_note='No GT, O samples, threshold traces, continuous eigen directions, selection scores, or bucket measurements are retained/read. This figure therefore documents actual deployed patterns and propagation only.')
    np.savez_compressed(stem.with_name(stem.name+'_data').with_suffix('.npz'), metadata=np.array(json.dumps(report)),
                        codes=codes_np,signed=signed_np,centres=centres_np,physical_rows=effective_np,
                        spectra=spectra,physical_limit=np.array(physical_limit))
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
