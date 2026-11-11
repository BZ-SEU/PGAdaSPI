"""Read completed three-file outputs; never run reconstruction or selection.

One directory: JSON with per-image rows and means.
Several directories (e.g. one per ablation): one Markdown table row per directory,
restricted to the image IDs completed in every directory so rows are comparable.
With --write-summary, each directory also receives summarize.json for all its completed images,
independently of the common-image subset used in the comparison table.
"""
import argparse
import json
from pathlib import Path
import statistics


def scalar_summary(values: list[float]) -> dict:
    """Descriptive statistics for a per-image offline scalar, never a CI."""
    return dict(count=len(values), mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.,
                median=statistics.median(values), minimum=min(values), maximum=max(values))


def load(output: Path) -> tuple[dict, list[str]]:
    rows, skipped = {}, []
    for directory in sorted(output.iterdir()):
        if not directory.is_dir():
            continue
        marker = directory / (directory.name + '.json')
        if not marker.exists():
            skipped.append(directory.name)
            continue
        result = json.loads(marker.read_text())
        if result.get('status') != 'complete' or any(
            not (directory / (directory.name + suffix)).exists() for suffix in ('.pt', '.png')
        ):
            skipped.append(directory.name)
            continue
        pinv = result['settings'].get('final_pinv', result['final'])
        capture = result.get('information_capture', {})
        eta = capture.get('eta')
        # Older OOD archives saved capture at M640; the paper reports final M655.
        # Use the same exact full-canvas centered-PInv identity as the producer.
        if capture and capture.get('M') != pinv.get('M'):
            total = capture.get('total_centered_energy', 0.)
            if ('full physical measurement canvas' not in capture.get('domain', '')
                    or total <= 0 or 'full_dim' not in capture):
                raise ValueError(f'{marker}: cannot derive final-budget information capture')
            eta = 1. - capture['full_dim'] * pinv['mse'] / total
        rows[result['image_id']] = dict(image_id=result['image_id'], **result['final'],
                                        pinv_psnr=pinv['psnr'], pinv_ssim=pinv['ssim'],
                                        seconds=result['runtime']['adaptive_wall_seconds'],
                                        overrides=result['settings']['recipe'].get('overrides', []),
                                        captured_energy_ratio=eta)
    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs', type=Path, nargs='+')
    parser.add_argument('--write-summary', action='store_true',
                        help='Also save summarize.json; default only prints results')
    args = parser.parse_args()
    keys = ['psnr', 'psnr_clamped', 'mse', 'ssim', 'pinv_psnr', 'pinv_ssim', 'seconds']
    loaded = [load(o) for o in args.outputs]
    summaries = []
    for output, (rows, skipped) in zip(args.outputs, loaded):
        eta_rows = {image_id: float(row['captured_energy_ratio']) for image_id, row in rows.items()
                    if row.get('captured_energy_ratio') is not None}
        capture_summary = dict(
            definition='eta = ||P_A(x-mu)||^2 / ||x-mu||^2; P_A is the actual effective-row space projector',
            association='offline GT-only centered PInv quantity; not a final DiffPIR reconstruction metric and never used for selection',
            available_images=len(eta_rows), missing_images=sorted(set(rows) - set(eta_rows)),
            eta=scalar_summary(list(eta_rows.values())) if eta_rows else None,
            per_image=eta_rows)
        summary = dict(completed_images=len(rows), incomplete_directories=skipped,
                       mean={k: statistics.mean(float(r[k]) for r in rows.values()) for k in keys} if rows else {},
                       information_capture=capture_summary, per_image=list(rows.values()),
                       scope='Only completed image folders in this directory; no claim of dataset completeness.')
        serialized = json.dumps(summary, indent=2, ensure_ascii=False)
        if args.write_summary:
            (output / 'summarize.json').write_text(serialized + '\n', encoding='utf-8')
        summaries.append(serialized)
    if len(args.outputs) == 1:
        print(summaries[0])
        return
    common = sorted(set.intersection(*(set(r) for r, _ in loaded)))
    print(f'common completed images: {len(common)}')
    print('| run | N | PSNR | SSIM | pinv PSNR | pinv SSIM | min/img |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for output, (rows, _) in zip(args.outputs, loaded):
        if not common:
            break
        m = {k: statistics.mean(float(rows[i][k]) for i in common) for k in keys}
        print(f"| {output.name} | {len(common)} | {m['psnr']:.2f} | {m['ssim']:.4f} | "
              f"{m['pinv_psnr']:.2f} | {m['pinv_ssim']:.4f} | {m['seconds'] / 60:.1f} |")


if __name__ == '__main__':
    main()
