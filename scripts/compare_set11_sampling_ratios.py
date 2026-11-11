"""Read-only, paired 256x256 Set11 comparison at one or more sampling ratios.

The adaptive result has two reconstruction endpoints on exactly the same
adaptively acquired uint8 physical patterns: its centered pseudoinverse
(``settings.final_pinv``) and its final DiffPIR posterior mean (``final``).
HATNet and ProxUnroll have their released learned sensing models, so their
numbers are reported alongside—not treated as identical physical acquisitions.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RATIOS = {
    "0.04": {
        "adaptive": ROOT / "outputs/generalization/set11_0_04_256",
        "hatnet": ROOT / "comparable_method/HATNet-SPI-master/test_results/set11_cr004_hatnet",
        "proxunroll": ROOT / "comparable_method/ProxUnroll-main/test_results/set11_cr004_hqs",
    },
    "0.1": {
        "adaptive": ROOT / "outputs/generalization/set11_0_1_256",
        "hatnet": ROOT / "comparable_method/HATNet-SPI-master/test_results/set11_cr01_hatnet",
        "proxunroll": ROOT / "comparable_method/ProxUnroll-main/test_results/set11_cr01_hqs",
    },
}


def json_rows(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in root.glob("*/*.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "complete" and value.get("image_shape", [256, 256]) == [256, 256]:
            rows[str(value["image_id"]).lower()] = value
        elif value.get("status") == "complete" and value.get("evaluation_region", {}).get("height") == 256:
            rows[str(value["image_id"]).lower()] = value
    return rows


def record(image_id: str, ratio: str, method: str, data: dict, endpoint: str) -> dict:
    if endpoint == "pinv":
        value = data["settings"]["final_pinv"]
        seconds = data["runtime"]["adaptive_wall_seconds"]
        reconstruction = "centered_full_pseudoinverse"
    else:
        value = data["final"]
        seconds = data.get("runtime", {}).get("adaptive_wall_seconds", value.get("seconds"))
        reconstruction = "final_DiffPIR_posterior_mean" if method == "adaptive_diffpir" else method
    return {
        "sampling_ratio_nominal": float(ratio), "image_id": image_id, "method": method,
        "reconstruction": reconstruction, "M": int(value["M"]), "mse": float(value["mse"]),
        "psnr": float(value.get("psnr_clamped", value["psnr"])), "ssim": float(value["ssim"]),
        "seconds": None if seconds is None else float(seconds),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for ratio in sorted({row["sampling_ratio_nominal"] for row in rows}):
        for method in ("adaptive_pinv", "adaptive_diffpir", "HATNet", "ProxUnroll-HQS"):
            subset = [row for row in rows if row["sampling_ratio_nominal"] == ratio and row["method"] == method]
            if not subset:
                continue
            output.append({
                "sampling_ratio_nominal": ratio, "method": method, "images": len(subset),
                "mean_M": statistics.mean(row["M"] for row in subset),
                "mean_mse": statistics.mean(row["mse"] for row in subset),
                "mean_psnr": statistics.mean(row["psnr"] for row in subset),
                "mean_ssim": statistics.mean(row["ssim"] for row in subset),
                "mean_seconds": statistics.mean(row["seconds"] for row in subset if row["seconds"] is not None),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/generalization/set11_256_sampling_comparison")
    args = parser.parse_args()
    all_rows: list[dict] = []
    coverage: dict[str, dict] = {}
    for ratio, paths in RATIOS.items():
        loaded = {name: json_rows(path) for name, path in paths.items()}
        common = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
        if not common:
            raise RuntimeError(f"no common completed 256x256 images at CR={ratio}")
        coverage[ratio] = {"common_images": common, "counts": {name: len(rows) for name, rows in loaded.items()}}
        for image_id in common:
            adaptive = loaded["adaptive"][image_id]
            if "final_pinv" not in adaptive.get("settings", {}):
                raise RuntimeError(f"adaptive output lacks final_pinv: {image_id}, CR={ratio}")
            all_rows.extend((record(image_id, ratio, "adaptive_pinv", adaptive, "pinv"),
                             record(image_id, ratio, "adaptive_diffpir", adaptive, "diffpir"),
                             record(image_id, ratio, "HATNet", loaded["hatnet"][image_id], "other"),
                             record(image_id, ratio, "ProxUnroll-HQS", loaded["proxunroll"][image_id], "other")))
    summary = aggregate(all_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    report = {
        "scope": "completed common native 256x256 Set11 images only; 512x512 fingerprint/flinstones excluded",
        "comparison_boundary": "adaptive uses uint8 complementary DMD patterns and incoherent propagation; HATNet/ProxUnroll use their released learned sensing matrices, so metrics are outcome comparisons rather than matched-physics comparisons",
        "adaptive_endpoints": "adaptive_pinv and adaptive_diffpir share each image's identical completed adaptive acquisition; selection always uses centered PInv, then DiffPIR is the final decoder",
        "coverage": coverage, "summary": summary, "per_image_csv": "per_image.csv",
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("| CR | method | N | mean M | PSNR | SSIM | MSE | mean seconds |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        print(f"| {row['sampling_ratio_nominal']:.2f} | {row['method']} | {row['images']} | {row['mean_M']:.0f} | "
              f"{row['mean_psnr']:.4f} | {row['mean_ssim']:.5f} | {row['mean_mse']:.8f} | {row['mean_seconds']:.3f} |")


if __name__ == "__main__":
    main()
