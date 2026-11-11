#!/home/baojr/.conda/envs/PIE2E/bin/python
"""Fig. S3: original dev denoising and checkpoint-selection evaluations.

Top: all 60 logged fixed-noise validations (64 dev images, ten noise levels).
Bottom: matched historical DAPS/200 outer-step/K32 evaluations on eight dev images.
Reads archived JSON members in memory; never extracts the historical experiments.
Saves plotting NPZ/JSON so later redraws do not require the external archive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
ARCHIVE = ROOT.parent/'adaptive_SPI_backups/finalize_20261012_034713/summaries_and_source.tar.gz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(archive):
    validation, sources = {}, []
    for name in ('train_phase1_crop128.log', 'train_phase2.log'):
        path = ROOT/'outputs/prior_daps_v2'/name
        sources.append(dict(path=str(path), sha256=sha(path)))
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if 'val_mean_mse' in row:
                step, value = int(row['step']), float(row['val_mean_mse'])
                assert np.isclose(value, np.mean([r['mse'] for r in row['rows']]), atol=1e-12)
                if step in validation:
                    assert value == validation[step]
                validation[step] = value
    assert sorted(validation) == list(range(5000, 300001, 5000))
    split = json.loads((ROOT/'assets/split.json').read_text())
    dev = {Path(p).stem for p in split['split']['dev']}
    steps = [50000, 65000, 95000, 165000]
    gains, members = [], []
    baseline, image_ids, protocol = None, None, None
    with tarfile.open(archive, 'r:gz') as tar:
        for step in steps:
            member = f'outputs/analysis/dev_daps_mean_v2_{step//1000}k_outer200.json'
            raw = tar.extractfile(member).read()
            result = json.loads(raw)
            sampling = result['sampling']
            assert sampling.get('sampler', 'daps') == 'daps'
            assert sampling['outer_steps'] == 200 and sampling['num_scenarios'] == 32
            comparable = {k: v for k, v in sampling.items() if k != 'gpu_batch_size'}
            assert protocol is None or comparable == protocol
            protocol = comparable
            rows = sorted(result['images'], key=lambda r: r['image_id'])
            ids = [r['image_id'] for r in rows]
            assert len(set(ids)) == result['n'] == 8 and set(ids) <= dev
            assert all(r['M'] == 655 for r in rows)
            pinv = np.array([r['pinv']['psnr'] for r in rows])
            if baseline is not None:
                np.testing.assert_array_equal(pinv, baseline)
                assert ids == image_ids
            baseline, image_ids = pinv, ids
            mean_psnr = np.array([r['mean_K32']['psnr'] for r in rows])
            np.testing.assert_allclose(mean_psnr.mean(), result['mean']['mean_K32']['psnr'], atol=1e-12)
            assert all(r['mean_K32']['max_residual'] < 1e-6 for r in rows)
            gains.append(mean_psnr-pinv)
            if step == 65000:
                assert result['prior_sha256'] == sha(ROOT/'outputs/prior_daps_v2/final_65k.pt')
            members.append(dict(member=member, sha256=hashlib.sha256(raw).hexdigest(),
                                prior_sha256=result['prior_sha256'], sampling=sampling))
    data = dict(denoising_steps=np.array(sorted(validation)),
                denoising_mse=np.array([validation[s] for s in sorted(validation)]),
                posterior_steps=np.array(steps), posterior_gain_per_image=np.stack(gains),
                image_ids=np.array(image_ids), pinv_psnr=baseline)
    meta = dict(sources=sources, archive=str(archive.resolve()), archive_sha256=sha(archive),
                members=members, selected_step=65000, denoising_dev_images=64,
                denoising_noise_levels=10, denoising_domain='[-1,1]', posterior_dev_images=8,
                posterior_protocol='Historical checkpoint selection: DAPS, 200 outer steps, 32 chains, M=655',
                gain_definition='per-image unclamped PSNR(posterior mean) minus PSNR(pseudoinverse), then image average',
                scope='original checkpoint-selection experiment; not a DiffPIR checkpoint sweep')
    return data, meta


def draw(data, meta):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.0))
    fig.subplots_adjust(left=.14, right=.97, bottom=.08, top=.89, hspace=.40)
    blue, orange = '#0072B2', '#D55E00'
    x, y = data['denoising_steps']/1000, data['denoising_mse']
    axes[0].plot(x, y, '-o', color=blue, lw=1.8, ms=3, label='Dev denoising MSE')
    selected = int(np.flatnonzero(data['denoising_steps'] == meta['selected_step'])[0])
    best = int(np.argmin(y))
    axes[0].scatter(x[best], y[best], marker='D', s=42, facecolors='white', edgecolors=blue, zorder=4)
    axes[0].annotate(f'Min. MSE: {int(x[best])}k', (x[best], y[best]), xytext=(15, 25),
                     textcoords='offset points', fontsize=9, arrowprops=dict(arrowstyle='-', color='.45'))
    axes[0].set(ylabel='Dev denoising MSE', xlim=(0, 305), title='(a) Denoising validation')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0].set_xticks([0, 65, 100, 150, 200, 250, 300])
    px = data['posterior_steps']/1000
    py = data['posterior_gain_per_image'].mean(axis=1)
    axes[1].plot(px, py, '-o', color=blue, lw=1.8, ms=6)
    pselected = int(np.flatnonzero(data['posterior_steps'] == meta['selected_step'])[0])
    axes[1].annotate(f'+{py[pselected]:.3f} dB', (65, py[pselected]), xytext=(24, 10),
                     textcoords='offset points', fontsize=10, color=orange)
    axes[1].set(ylabel='Dev posterior-mean PSNR gain (dB)', xlim=(40, 175), ylim=(0, .82),
                title='(b) Posterior-mean validation')
    axes[1].set_xticks(px)
    for ax, sx, sy in ((axes[0], x[selected], y[selected]), (axes[1], 65, py[pselected])):
        ax.axvline(65, color=orange, ls='--', lw=1.2, alpha=.8)
        ax.scatter(sx, sy, color=orange, marker='*', s=150, edgecolors='white', linewidths=.5,
                   zorder=5, label='Selected checkpoint: 65k')
        ax.set_xlabel('Training step (k)')
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)
        ax.grid(True, alpha=.2, lw=.6)
        ax.set_axisbelow(True)
    axes[1].legend(loc='lower left', frameon=False, fontsize=9)
    fig.suptitle('Fig. S3 | Prior checkpoint selection on the dev split', fontsize=15, y=.99)
    meta.update(selected_denoising_mse=float(y[selected]), minimum_mse_step=int(data['denoising_steps'][best]),
                posterior_gain_mean=py.tolist(), posterior_steps=data['posterior_steps'].tolist())
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/paper_figures/fig_s3')
    parser.add_argument('--refresh', action='store_true', help='re-read original logs and archive')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    assert args.dpi >= 72
    for protected in (ROOT/'data', ROOT/'outputs/ablation', ROOT/'outputs/generalization', ROOT/'outputs/prior_daps_v2'):
        assert not args.output.resolve().is_relative_to(protected.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output/'fig_s3_checkpoint_selection'
    cache = stem.with_name(stem.name+'_data').with_suffix('.npz')
    if cache.exists() and not args.refresh:
        with np.load(cache, allow_pickle=False) as f:
            meta = json.loads(str(f['metadata']))
            assert meta['archive'] == str(args.archive.resolve()), 'Archive changed; use --refresh'
            data = {k: f[k] for k in f.files if k != 'metadata'}
    else:
        data, meta = prepare(args.archive)
        np.savez_compressed(cache, metadata=np.array(json.dumps(meta)), **data)
    fig = draw(data, meta)
    save_figure(fig, stem.with_suffix('.png'), dpi=args.dpi, facecolor='white')
    save_figure(fig, stem.with_suffix('.emf'), dpi=args.dpi, facecolor='white')
    plt.close(fig)
    meta.update(script_sha256=sha(__file__), plotting_data_sha256=sha(cache))
    stem.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(stem.with_suffix('.png'))
    print(json.dumps({k: meta[k] for k in ('selected_denoising_mse', 'posterior_steps', 'posterior_gain_mean')}))


if __name__ == '__main__':
    main()
