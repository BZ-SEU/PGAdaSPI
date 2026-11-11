"""Create a lower-budget endpoint from a sealed higher-budget adaptive run.

Only the first ``floor(ratio*H*W)`` already executed uint8 DMD codes are used.
No policy, O scenes, future buckets, or GT-derived selection signal is invoked.
The physical buckets are replayed from those sealed codes, PInv is checked against
the saved prefix trajectory, and one new final DiffPIR posterior mean is sampled.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from daps_spi.compact_output import metrics, save_result
from daps_spi.data import load_image
from daps_spi.io import digest, file_hash
from daps_spi.priors import load_prior
from daps_spi.propagation_pinv import IntensityOptics, MeasurementFactor, OpticalConfig, PseudoinverseDecoder, SensorDarkNoise
from daps_spi.sampling import chain_seeds, sample, sampler_steps
from daps_spi.types import MeasurementState


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0. < args.ratio < 1.: parser.error("ratio must be in (0,1)")
    for marker in sorted(args.source.glob("*/*.json")):
        source = json.loads(marker.read_text())
        name = source["image_id"]
        if source.get("status") != "complete": raise RuntimeError(f"incomplete source: {name}")
        source_pt = marker.with_suffix(".pt")
        if file_hash(source_pt) != source["pt_sha256"]: raise RuntimeError(f"{name}: source PT hash mismatch")
        payload = torch.load(source_pt, map_location="cpu", weights_only=False)
        plus_all = payload["plus_uint8"]
        if digest(plus_all) != source["patterns"]["plus_hash"]: raise RuntimeError(f"{name}: source pattern hash mismatch")
        settings = copy.deepcopy(source["settings"]); cfg = settings["config"]; recipe = settings["recipe"]
        h, w = int(cfg["image_height"]), int(cfg["image_width"])
        target_m = math.floor(args.ratio * h * w)
        if target_m < cfg["initial_measurements"] or target_m >= len(plus_all):
            raise RuntimeError(f"{name}: target M={target_m} is not an interior sealed prefix of M={len(plus_all)}")
        prefix_history = [copy.deepcopy(row) for row in payload["history"] if int(row["M"]) <= target_m]
        if not prefix_history:
            raise RuntimeError(f"{name}: source has no history before M={target_m}")
        # A floor(ratio*d) endpoint need not lie on the source B32 grid.  The
        # sealed code prefix remains valid; only this final PInv/DiffPIR state
        # is newly evaluated.  Do not claim it was an original decision node.
        stored_exact_pinv = int(prefix_history[-1]["M"]) == target_m
        target = args.output / name / f"{name}.json"
        identity = digest("adaptive-prefix-redecode-v1", source["pt_sha256"], target_m, args.ratio,
                          recipe["final_prior"], recipe["config"]["final_sampling"])
        if target.exists():
            old = json.loads(target.read_text())
            if old.get("settings", {}).get("run_identity") != identity: raise RuntimeError(f"incompatible existing {target}")
            if old.get("status") == "complete" and file_hash(target.with_suffix(".pt")) == old["pt_sha256"] and target.with_suffix(".png").exists():
                print(f"{name}: verified complete, skip", flush=True); continue
            raise RuntimeError(f"corrupt incomplete output: {target}")
        plus = plus_all[:target_m].to(args.device); ids = source["patterns"]["ids"][:target_m]
        dataset = settings["evaluation_dataset"]; image_path = Path(dataset["root"]) / dataset["image_files"][name]
        if file_hash(image_path) != settings["image_sha256"]: raise RuntimeError(f"{name}: image identity mismatch")
        truth = load_image(image_path, cfg["image_size"], native=dataset.get("native", False)).to(args.device).double()
        mu = torch.load(ROOT / recipe["mean"], map_location=args.device, weights_only=False)["mu"]
        optics = IntensityOptics(OpticalConfig(**recipe["physics"]), args.device)
        sync(args.device); begin = time.perf_counter()
        rows = torch.empty((target_m, h * w), dtype=torch.float64, device=args.device); buckets = []
        for start in range(0, target_m, 32):
            codes = plus[start:start + 32]
            rows[start:start + len(codes)] = optics.effective_rows((2 * codes.double() - 255) / 255)
            buckets.append(optics.observe(truth, codes))
        clean = torch.cat(buckets); noise = SensorDarkNoise(cfg.get("sensor_noise"), optics, cfg["seed"], settings["image_index"])
        state = MeasurementState(rows, clean + noise(0, target_m, args.device), plus.flatten(1), ids)
        replay_seconds = time.perf_counter() - begin
        factor = MeasurementFactor(state, recipe["rank_rtol"]) if cfg.get("shared_factor", False) else None
        decoder = PseudoinverseDecoder(state, mu, recipe["rank_rtol"], factor)
        pinv = decoder.xhat.reshape_as(truth); pinv_metrics = metrics(pinv, truth, target_m, 0.)
        if stored_exact_pinv:
            saved_pinv = prefix_history[-1]
            for key in ("mse", "psnr", "psnr_clamped", "ssim"):
                if abs(pinv_metrics[key] - saved_pinv[key]) > 1e-7: raise RuntimeError(f"{name}: prefix PInv mismatch at {key}")
        info = decoder.captured_energy_ratio(truth)
        prior_path = ROOT / recipe["final_prior"]
        net, meta, checkpoint_hash = load_prior(str(prior_path), args.device)
        fcfg = dict(cfg["sampling"], **cfg["final_sampling"])
        if any(fcfg[key] != meta[key] for key in ("sigma_min", "sigma_max")): raise RuntimeError("final prior/sampler sigma mismatch")
        decisions = math.ceil((target_m - cfg["initial_measurements"]) / cfg["batch_size"])
        seeds = chain_seeds(cfg["seed"], settings["image_index"], decisions, "F", fcfg["chains"])
        sync(args.device); tick = time.perf_counter()
        ensemble = sample(net, state, fcfg, seeds, "F", checkpoint_hash, factor)
        sync(args.device); final_seconds = time.perf_counter() - tick
        reconstruction = ensemble.samples.mean(0).reshape_as(truth)
        residual = float((rows @ reconstruction.flatten() - state.y).abs().max())
        if residual > 1e-6 * (1. + float(state.y.abs().max())): raise RuntimeError(f"{name}: final measurement residual {residual}")
        sampler, steps = sampler_steps(fcfg)
        prefix_history[-1] = dict(metrics(reconstruction, truth, target_m, final_seconds), decoder="daps_mean")
        target_cfg = copy.deepcopy(cfg); target_recipe = copy.deepcopy(recipe)
        target_cfg["final_measurements"] = target_m; target_cfg["sampling_ratio"] = args.ratio; target_cfg["device"] = args.device
        target_recipe["config"] = target_cfg
        settings.update(config=target_cfg, recipe=target_recipe, final_pinv=pinv_metrics,
                        final_daps=dict(sampler=sampler, chains=fcfg["chains"], steps=steps, outer_steps=fcfg["outer_steps"],
                                        seeds=seeds, checkpoint_sha256=checkpoint_hash, nfe_per_chain=ensemble.diagnostics["nfe_per_chain"]),
                        run_identity=identity,
                        derived_from=dict(source_json=str(marker.resolve()), source_pt_sha256=source["pt_sha256"],
                                          source_pattern_hash=source["patterns"]["plus_hash"], prefix_m=target_m,
                                          source_last_logged_m=int(payload["history"][-1]["M"]),
                                          prefix_was_logged_decision_node=stored_exact_pinv,
                                          rule="Exact prefix of sealed adaptive uint8 codes; physical buckets replayed; no selection rerun"))
        settings["numerics"].update(final_decoder="daps_mean", max_measurement_residual=residual, final_condition=decoder.condition)
        source_final_time = source["runtime"]["components"]["final_seconds"]
        acquisition_to_prefix = sum(float(row.get("seconds", 0.)) for row in prefix_history[1:-1])
        runtime = dict(adaptive_wall_seconds=acquisition_to_prefix + final_seconds, model_load_seconds=0., device=args.device,
                       GPU=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                       components=dict(reused_prefix_acquisition_seconds=acquisition_to_prefix, final_seconds=final_seconds),
                       offline_replay_seconds=replay_seconds, offline_pinv_check_seconds=0., final_redecode_seconds=final_seconds,
                       peak_gpu_allocated_bytes=None,
                       timing_definition="Prefix trajectory timing reused from source history; one new final DiffPIR decode measured separately. No new closed-loop selection.")
        save_result(args.output / name, name, plus, ids, prefix_history, truth, reconstruction, settings, runtime,
                    source["evaluation_region"], information_capture=info)
        print(json.dumps({"image_id": name, "M": target_m, "pinv": pinv_metrics, "final": prefix_history[-1], "final_seconds": final_seconds}), flush=True)
        del ensemble, net, state, rows, clean, plus, decoder, factor, optics
        if args.device.startswith("cuda"): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
