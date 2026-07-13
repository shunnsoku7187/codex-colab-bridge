"""Stress-test dual-sided early exit on reliability-oriented scenarios.

The target task is ordinary CIFAR-10 classification, not binary inspection.
Only reliable labels are wanted: a policy may either emit a class label or
reject the image as not reliably classifiable.

This experiment moves toward the intended use case by mixing clean images with
low-quality variants. Thresholds are selected on a validation split, then
reported on a separate evaluation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ImageEnhance, ImageFilter, ImageDraw
from torch.utils.data import DataLoader

from scripts.branchynet_cifar_sweep import (
    BranchyHookedModel,
    CIFAR_STATS,
    DEFAULT_EXITS,
    collect_outputs,
    custom_collate,
    estimate_costs,
    load_cifar_model,
    make_datasets,
    parse_csv_ints,
    transform_eval,
)
from scripts.compare_final_threshold_vs_dual_exit import (
    calibrate_final_threshold,
    choose_best,
    evaluate_policy,
    thresholds_from_quantiles,
)
from src.checkpointing import checkpoint_dir, latest_checkpoint
from src.experiment_paths import RESULTS_DIR, ensure_dirs


class ScenarioTransform:
    def __init__(self, base_transform, scenario: str):
        self.base_transform = base_transform
        self.scenario = scenario

    def __call__(self, image):
        image = image.convert("RGB")
        if self.scenario == "clean":
            pass
        elif self.scenario == "blur_r1":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif self.scenario == "blur_r2":
            image = image.filter(ImageFilter.GaussianBlur(radius=2.0))
        elif self.scenario == "occlude_8":
            image = occlude_center(image, 8)
        elif self.scenario == "occlude_16":
            image = occlude_center(image, 16)
        elif self.scenario == "dark_50":
            image = ImageEnhance.Brightness(image).enhance(0.5)
        elif self.scenario == "low_contrast_50":
            image = ImageEnhance.Contrast(image).enhance(0.5)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        return self.base_transform(image)


def occlude_center(image, size: int):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    draw.rectangle([left, top, left + size - 1, top + size - 1], fill=(0, 0, 0))
    return image


def load_branchynet(output_name: str, arch: str, dataset: str, exit_modules: list[str], branch_depths: list[int], device):
    base_model = load_cifar_model(arch, dataset, device)
    model = BranchyHookedModel(base_model, exit_modules, CIFAR_STATS[dataset]["classes"], branch_depths).to(device)
    ckpt_dir = checkpoint_dir(output_name)
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"Missing checkpoint under {ckpt_dir}. Run scripts.branchynet_cifar_sweep for {output_name} first."
        )
    try:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state"])
    print(f"Loaded BranchyNet checkpoint: {ckpt}", flush=True)
    return model, payload


def make_loader(dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate)


def collect_scenario(model, loader, base_transform, scenario: str, device, exit_names, costs):
    print(f"Collecting scenario={scenario}", flush=True)
    return collect_outputs(
        model,
        loader,
        ScenarioTransform(base_transform, scenario),
        device,
        exit_names,
        costs,
    )


def subset_data(data: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    count = min(count, len(data["labels"]))
    out = {}
    for key, value in data.items():
        if key in {"exit_names", "exit_costs"}:
            out[key] = value
        else:
            out[key] = value[:count]
    return out


def concat_data(parts: list[dict[str, np.ndarray]], weights: list[float]) -> dict[str, np.ndarray]:
    if len(parts) != len(weights):
        raise ValueError("parts and weights must have the same length")
    min_n = min(len(part["labels"]) for part in parts)
    counts = [int(round(min_n * weight)) for weight in weights]
    if sum(counts) == 0:
        raise ValueError("empty mixture")
    arrays: dict[str, Any] = {
        "exit_names": parts[0]["exit_names"],
        "exit_costs": parts[0]["exit_costs"],
    }
    for key in ["labels", "pred", "correct", "entropy", "confidence"]:
        arrays[key] = np.concatenate([subset_data(part, count)[key] for part, count in zip(parts, counts)], axis=0)
    return arrays


def sweep_policies(val_data, eval_data, target_accuracy: float, grid_quantiles: list[float]) -> dict[str, Any]:
    val_correct = np.asarray(val_data["correct"], dtype=bool)
    val_conf = np.asarray(val_data["confidence"], dtype=float)
    eval_correct = np.asarray(eval_data["correct"], dtype=bool)
    eval_conf = np.asarray(eval_data["confidence"], dtype=float)
    costs = np.asarray(eval_data["exit_costs"], dtype=float)

    final_threshold = calibrate_final_threshold(val_correct[:, -1], val_conf[:, -1], target_accuracy)
    eval_final_reliable = eval_correct[:, -1] & (eval_conf[:, -1] >= final_threshold["threshold"])

    final_only = evaluate_policy(
        eval_correct,
        eval_conf,
        costs,
        eval_final_reliable,
        final_threshold["threshold"],
        lower0=-1.0,
        upper0=2.0,
        lower1=-1.0,
        upper1=2.0,
    )

    def eval_from_val_thresholds(row):
        t = row["thresholds"]
        return evaluate_policy(
            eval_correct,
            eval_conf,
            costs,
            eval_final_reliable,
            final_threshold["threshold"],
            lower0=t["lower0"],
            upper0=t["upper0"],
            lower1=t["lower1"],
            upper1=t["upper1"],
        )

    val_final_reliable = val_correct[:, -1] & (val_conf[:, -1] >= final_threshold["threshold"])
    upper0_values = thresholds_from_quantiles(val_conf[:, 0], grid_quantiles)
    upper1_values = thresholds_from_quantiles(val_conf[:, 1], grid_quantiles)
    lower0_values = thresholds_from_quantiles(val_conf[:, 0], grid_quantiles)
    lower1_values = thresholds_from_quantiles(val_conf[:, 1], grid_quantiles)

    upper_rows = []
    for upper0 in upper0_values:
        for upper1 in upper1_values:
            upper_rows.append(
                evaluate_policy(
                    val_correct,
                    val_conf,
                    costs,
                    val_final_reliable,
                    final_threshold["threshold"],
                    lower0=-1.0,
                    upper0=upper0,
                    lower1=-1.0,
                    upper1=upper1,
                )
            )

    dual_rows = []
    for lower0 in lower0_values:
        for upper0 in upper0_values:
            if lower0 >= upper0:
                continue
            for lower1 in lower1_values:
                for upper1 in upper1_values:
                    if lower1 >= upper1:
                        continue
                    dual_rows.append(
                        evaluate_policy(
                            val_correct,
                            val_conf,
                            costs,
                            val_final_reliable,
                            final_threshold["threshold"],
                            lower0=lower0,
                            upper0=upper0,
                            lower1=lower1,
                            upper1=upper1,
                        )
                    )

    selected = {
        "upper_only_best_cost": choose_best(upper_rows, target_accuracy),
        "dual_side_best_cost_lost_final_reliable_le_1pct": choose_best(dual_rows, target_accuracy, max_lost=0.01),
        "dual_side_best_cost_lost_final_reliable_le_2pct": choose_best(dual_rows, target_accuracy, max_lost=0.02),
        "dual_side_best_cost_lost_final_reliable_le_5pct": choose_best(dual_rows, target_accuracy, max_lost=0.05),
    }
    evaluated = {key: None if row is None else eval_from_val_thresholds(row) for key, row in selected.items()}
    return {
        "validation_final_threshold": final_threshold,
        "eval_final_accuracy_without_reject": float(eval_correct[:, -1].mean()),
        "eval_final_only": final_only,
        "selected_on_validation": selected,
        "evaluated_on_heldout": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/dual_exit_reliability_shift_001_summary.json")
    parser.add_argument("--model-output-name", default="0000b_branchynet_reproduce_resnet56_cifar10")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--arch", default="resnet56", choices=["resnet56"])
    parser.add_argument("--exit-modules", default="")
    parser.add_argument("--branch-depths", default="3,2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-val-size", type=int, default=5000)
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--scenarios", nargs="*", default=["clean", "blur_r1", "blur_r2", "occlude_8", "occlude_16", "dark_50", "low_contrast_50"])
    parser.add_argument("--grid-quantiles", nargs="*", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99])
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    exit_modules = [item.strip() for item in (args.exit_modules or DEFAULT_EXITS[args.arch]).split(",") if item.strip()]
    branch_depths = parse_csv_ints(args.branch_depths)
    costs = np.asarray(estimate_costs(args.arch, exit_modules), dtype=np.float32)
    model, checkpoint_payload = load_branchynet(args.model_output_name, args.arch, args.dataset, exit_modules, branch_depths, device)
    _, val_set, eval_set = make_datasets(args.dataset, args.threshold_val_size, seed=123)
    val_loader = make_loader(val_set, args.batch_size)
    eval_loader = make_loader(eval_set, args.batch_size)
    base_transform = transform_eval(args.dataset)
    exit_names = model.exit_names

    val_by_scenario = {}
    eval_by_scenario = {}
    for scenario in args.scenarios:
        val_by_scenario[scenario] = collect_scenario(model, val_loader, base_transform, scenario, device, exit_names, costs)
        eval_by_scenario[scenario] = collect_scenario(model, eval_loader, base_transform, scenario, device, exit_names, costs)

    mixture_specs = {
        "clean_only": [("clean", 1.0)],
        "clean75_blur_r2_25": [("clean", 0.75), ("blur_r2", 0.25)],
        "clean75_occlude16_25": [("clean", 0.75), ("occlude_16", 0.25)],
        "clean60_mixed_low_quality40": [("clean", 0.60), ("blur_r2", 0.15), ("occlude_16", 0.15), ("dark_50", 0.10)],
        "clean40_mixed_low_quality60": [("clean", 0.40), ("blur_r2", 0.25), ("occlude_16", 0.25), ("low_contrast_50", 0.10)],
    }

    scenario_results = {}
    for name, spec in mixture_specs.items():
        val_mix = concat_data([val_by_scenario[key] for key, _ in spec], [weight for _, weight in spec])
        eval_mix = concat_data([eval_by_scenario[key] for key, _ in spec], [weight for _, weight in spec])
        print(f"Sweeping policy mixture={name} samples={len(eval_mix['labels'])}", flush=True)
        scenario_results[name] = {
            "mixture": spec,
            **sweep_policies(val_mix, eval_mix, args.target_accuracy, args.grid_quantiles),
        }

    payload = {
        "purpose": "Strengthen the dual-sided early-exit direction under reliability-oriented low-quality input mixtures.",
        "claim_tested": (
            "When only reliable image labels are wanted, low-quality inputs that are unlikely to receive a reliable final label "
            "can be rejected early, reducing computation beyond final-only confidence filtering and upper-only early exit."
        ),
        "target_accuracy": args.target_accuracy,
        "model": {
            "checkpoint_output_name": args.model_output_name,
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "dataset": args.dataset,
            "arch": args.arch,
            "exit_modules": exit_modules,
            "branch_depths": branch_depths,
            "exit_names": exit_names,
            "exit_costs": [float(x) for x in costs],
        },
        "definitions": {
            "final_only": "Run every image to final, then emit a label only if final confidence passes the validation-calibrated 99%-accuracy threshold.",
            "upper_only": "Accept high-confidence early exits, otherwise continue; final still uses the calibrated final threshold.",
            "dual_side": "Upper-only plus low-confidence early rejection; accepted labels must still meet the same target accuracy on validation.",
            "lost_final_reliable_rate": "Fraction of all samples that final-only would have accepted correctly but the dual-side policy rejected early.",
        },
        "scenario_results": scenario_results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    model.close()


if __name__ == "__main__":
    main()
