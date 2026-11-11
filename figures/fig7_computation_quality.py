#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. 7: selection computation versus final reconstruction quality.

Read-only figure generation: reuses completed Set11 ablation outputs and never
runs acquisition, action selection, or diffusion sampling.  The x-coordinate
is the mean (over the 11 native Set11 images) cumulative O-set DiffPIR NFE:

  (diffpir_steps - 1) * num_scenarios * ceil((M_final-M_initial)/batch_size).

It excludes the final F-set reconstruction NFE.  This makes the horizontal
axis a selection-cost measure even for the existing T ablation, whose archived
configuration also changes final F sampling steps.

Outputs a PNG, provenance JSON, and compressed NPZ.  Example:
  /home/baojr/.conda/envs/PIE2E/bin/python figures/fig7_computation_quality.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import numpy as np

from daps_spi.io import file_hash

EXPECTED = frozenset({
    'Monarch', 'Parrots', 'barbara', 'boats', 'cameraman', 'foreman', 'house',
    'lena256', 'peppers256', 'fingerprint', 'flinstones',
})
# Existing output folder, parameter label, and one line family.
FAMILIES = {
    'T': [('steps10', 10), ('steps20', 20), ('full', 50), ('steps100', 100), ('steps200', 200)],
    'O': [('O8', 8), ('O16', 16), ('full', 32), ('O64', 64)],
    'b': [('B16', 16), ('full', 32), ('B64', 64)],
}
DISPLAY = {
    'T': dict(label='DiffPIR steps $T$', color='#D55E00', marker='o', prefix='$T$ = '),
    'O': dict(label='Design scenarios $L_O$', color='#0072B2', marker='s', prefix='$L_O$ = '),
    'b': dict(label='Batch size $b$', color='#009E73', marker='^', prefix='$b$ = '),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_run(root: Path, folder: str) -> tuple[list[dict], list[dict]]:
    """Load and validate the completed, canonical 11-image output set."""
    run = root / folder
    require(run.is_dir(), f'Missing ablation directory: {run}')
    rows, provenance = [], []
    seen = set()
    for image_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        image_id = image_dir.name
        marker = image_dir / f'{image_id}.json'
        pt = image_dir / f'{image_id}.pt'
        png = image_dir / f'{image_id}.png'
        require(marker.is_file() and pt.is_file() and png.is_file(), f'Incomplete three-file output: {image_dir}')
        meta = json.loads(marker.read_text(encoding='utf-8'))
        require(meta.get('status') == 'complete', f'Incomplete result marker: {marker}')
        require(meta.get('image_id') == image_id, f'Image ID mismatch: {marker}')
        require(file_hash(pt) == meta.get('pt_sha256'), f'PT hash mismatch: {pt}')
        cfg = meta['settings']['config']
        sampling = cfg['sampling']
        require(sampling['sampler'] == 'diffpir', f'{marker}: selection sampler is not DiffPIR')
        require(int(meta['final']['M']) == int(cfg['final_measurements']), f'{marker}: terminal M mismatch')
        require(int(cfg['final_measurements']) > int(cfg['initial_measurements']), f'{marker}: invalid budget')
        require(image_id not in seen, f'Duplicate result: {image_id}')
        seen.add(image_id)
        batches = math.ceil((int(cfg['final_measurements']) - int(cfg['initial_measurements'])) /
                            int(cfg['batch_size']))
        nfe_per_chain = int(sampling['diffpir_steps']) - 1  # DiffPIR final update is projection only.
        require(nfe_per_chain >= 1, f'{marker}: invalid DiffPIR step count')
        selection_nfe = nfe_per_chain * int(sampling['num_scenarios']) * batches
        rows.append(dict(
            image_id=image_id,
            psnr=float(meta['final']['psnr']),
            ssim=float(meta['final']['ssim']),
            nfe_selection=int(selection_nfe),
            nfe_per_chain=int(nfe_per_chain),
            O=int(sampling['num_scenarios']),
            T=int(sampling['diffpir_steps']),
            batch_size=int(cfg['batch_size']),
            selection_batches=int(batches),
            M_initial=int(cfg['initial_measurements']),
            M_final=int(cfg['final_measurements']),
            image_shape=list(cfg['image_shape']),
        ))
        provenance.append(dict(image_id=image_id, json=str(marker.resolve()), json_sha256=file_hash(marker),
                               pt=str(pt.resolve()), pt_sha256=meta['pt_sha256'],
                               run_identity=meta['settings']['run_identity']))
    require(seen == EXPECTED, f'{folder}: expected exactly canonical Set11 IDs; missing={sorted(EXPECTED-seen)}, extra={sorted(seen-EXPECTED)}')
    return rows, provenance


def aggregate(root: Path) -> tuple[dict, dict]:
    point_data, source_data = {}, {}
    for family, runs in FAMILIES.items():
        values, sources = [], {}
        for folder, parameter in runs:
            rows, provenance = read_run(root, folder)
            # Every image in a particular run must have a common setting.  This
            # catches mixed configs in one output directory before plotting.
            for key in ['T', 'O', 'batch_size']:
                require(len({row[key] for row in rows}) == 1, f'{folder}: mixed {key}')
            values.append(dict(
                folder=folder,
                parameter=int(parameter),
                psnr=float(np.mean([row['psnr'] for row in rows])),
                psnr_std=float(np.std([row['psnr'] for row in rows], ddof=1)),
                ssim=float(np.mean([row['ssim'] for row in rows])),
                nfe_selection_mean=float(np.mean([row['nfe_selection'] for row in rows])),
                nfe_selection_values=[int(row['nfe_selection']) for row in rows],
                T=rows[0]['T'], O=rows[0]['O'], batch_size=rows[0]['batch_size'],
                images=rows,
            ))
            sources[folder] = provenance
        point_data[family] = values
        source_data[family] = sources
    defaults = {family: next(x for x in values if x['folder'] == 'full')
                for family, values in point_data.items()}
    # All three source views of the default must be exactly the same archived run.
    default_psnrs = [x['psnr'] for x in defaults.values()]
    default_nfes = [x['nfe_selection_mean'] for x in defaults.values()]
    require(np.ptp(default_psnrs) < 1e-12 and np.ptp(default_nfes) < 1e-12,
            'Default full point disagrees between curve families')
    data = dict(families=point_data, default=defaults['T'],
                n_images=len(EXPECTED), metric='Final PSNR (unclipped, data_range=1; dB)',
                nfe_definition=('Mean per-image cumulative selection NFE across all adaptive batches: '
                                '(DiffPIR steps - 1) × O scenarios × ceil((M_final-M_initial)/batch_size). '
                                'Final F decoding NFE is excluded.'),
                important_scope=('All values are equal-weight means over 9 native 256×256 and 2 native 512×512 '
                                 'Set11 images. T ablations change both O and F DiffPIR steps in the archived '
                                 'experiments; the x-axis nevertheless includes O selection NFE only.'))
    return data, source_data


def draw(data: dict, output: Path, dpi: int) -> Path:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 11,
                         'axes.spines.top': True, 'axes.spines.right': True})
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    fig.subplots_adjust(left=.115, right=.985, bottom=.16, top=.84)
    for family, points in data['families'].items():
        spec = DISPLAY[family]
        ordered = sorted(points, key=lambda x: x['nfe_selection_mean'])
        x = np.array([p['nfe_selection_mean'] for p in ordered])
        y = np.array([p['psnr'] for p in ordered])
        ax.plot(x, y, color=spec['color'], lw=2.1, marker=spec['marker'], ms=7.2,
                mfc='white', mew=1.7, label=spec['label'], zorder=3)
        for p in ordered:
            if p['folder'] == 'full':
                continue
            # The b=16 and L_O=64 results are deliberately close in NFE; use
            # deterministic family/parameter-specific offsets so labels remain readable.
            offsets = {('T', 100): (-17, -23), ('O', 64): (25, 19), ('b', 16): (-12, 10), ('b', 64): (22, -8)}
            dx, dy = offsets.get((family, p['parameter']), (0, 10))
            ax.annotate(f"{spec['prefix']}{p['parameter']}", (p['nfe_selection_mean'], p['psnr']),
                        textcoords='offset points', xytext=(dx, dy), ha='center',
                        va='bottom' if dy >= 0 else 'top', color=spec['color'], fontsize=9)
    default = data['default']
    ax.scatter([default['nfe_selection_mean']], [default['psnr']], marker='*', s=275,
               color='#F0A202', edgecolor='#333333', linewidth=0.75, zorder=6,
               label='Default: $T$=50, $L_O$=32, $b$=32')
    ax.annotate('Default\n$T$=50, $L_O$=32, $b$=32',
                (default['nfe_selection_mean'], default['psnr']),
                textcoords='offset points', xytext=(-16, 18), ha='right', va='bottom',
                fontsize=9.5, color='#333333',
                arrowprops={'arrowstyle': '-', 'lw': 0.9, 'color': '#555555'})
    ax.set_xscale('log')
    ax.set_xlabel('Selection NFE per image (log scale)')
    ax.set_ylabel('Final PSNR (dB)')
    ax.set_title('Fig. 7  |  Computation versus reconstruction quality', loc='left', pad=24,
                 fontsize=14, fontweight='bold')
    ax.grid(True, which='major', color='#e1e5e9', lw=.75)
    ax.grid(True, which='minor', axis='x', color='#eff1f3', lw=.55)
    ax.set_axisbelow(True)
    all_y = [p['psnr'] for group in data['families'].values() for p in group]
    ax.set_ylim(min(all_y)-.24, max(all_y)+.32)
    ax.legend(loc='lower right', frameon=True, edgecolor='#d8d8d8', fontsize=9.5)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'fig7_computation_quality.png'
    save_figure(fig, path, dpi=dpi, facecolor='white')
    save_figure(fig, path.with_suffix('.emf'), dpi=dpi, facecolor='white')
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'outputs' / 'ablation')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'paper_figures' / 'fig7')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--plot-data', type=Path, help='Redraw from saved figure JSON without rereading experiment artifacts')
    args = parser.parse_args()
    if args.plot_data:
        data = json.loads(args.plot_data.read_text(encoding='utf-8'))
        require(data.get('figure') == 7, 'Expected Fig. 7 plotting JSON')
        require(data['data_sha256'] == sha256_text(json.dumps(data['families'], sort_keys=True)),
                'Cached plotting data hash mismatch')
        print(draw(data, args.output, args.dpi))
        return
    data, sources = aggregate(args.results)
    figure = draw(data, args.output, args.dpi)
    payload = dict(figure=7, figure_png=str(figure.resolve()), source_root=str(args.results.resolve()),
                   **data, source_provenance=sources,
                   redraw_command=f'/home/baojr/.conda/envs/PIE2E/bin/python {Path(__file__).resolve()}')
    payload['data_sha256'] = sha256_text(json.dumps(payload['families'], sort_keys=True))
    json_path = args.output / 'fig7_computation_quality.json'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    arrays = {}
    for family, points in data['families'].items():
        arrays[f'{family}_parameter'] = np.array([p['parameter'] for p in points], dtype=np.int64)
        arrays[f'{family}_nfe'] = np.array([p['nfe_selection_mean'] for p in points], dtype=np.float64)
        arrays[f'{family}_psnr'] = np.array([p['psnr'] for p in points], dtype=np.float64)
        arrays[f'{family}_psnr_std'] = np.array([p['psnr_std'] for p in points], dtype=np.float64)
    arrays['default_nfe'] = np.array([data['default']['nfe_selection_mean']], dtype=np.float64)
    arrays['default_psnr'] = np.array([data['default']['psnr']], dtype=np.float64)
    np.savez_compressed(args.output / 'fig7_computation_quality_data.npz', **arrays)
    print(json.dumps(dict(png=str(figure), json=str(json_path), default=data['default'],
                          curves={k:[dict(parameter=p['parameter'], nfe=p['nfe_selection_mean'], psnr=p['psnr'])
                                     for p in v] for k,v in data['families'].items()}),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
