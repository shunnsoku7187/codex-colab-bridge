"""Cost credibility audit for profiled minimal PatchCore designs.

The previous holdout experiment reported very small relative NN-search ops.
This script checks what that number really means by decomposing it into
patch-count, memory-bank, and feature-dimension ratios, then measuring the
online feature-extraction and nearest-neighbor scoring time for the baseline
and the selected category-profiled configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.mvtec_ad_parquet_anomaly_probe import (
    collect_features,
    find_materialized_samples,
    image_scores_from_patch_scores,
    make_backbone,
    patchcore_scores,
    sample_normal_patch_bank,
)
from scripts.mvtec_patchcore_lightweight_sweep import normal_train_and_test
from scripts.mvtec_patchcore_profiled_minimal_search import approx_nn_ops, config_name, parse_feature_profile
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


def parse_profile_from_config(config_name_text: str) -> tuple[dict, int, float]:
    parts = config_name_text.split("_")
    if len(parts) < 4 or not parts[-2].startswith("b") or not parts[-1].startswith("topk"):
        raise ValueError(f"unexpected config name: {config_name_text}")
    profile_name = "_".join(parts[:-2])
    bank = int(parts[-2][1:])
    topk = float(parts[-1][4:].replace("p", "."))

    if profile_name.startswith("wrn_l23_g"):
        grid = int(profile_name.split("g")[-1])
        profile = parse_feature_profile(f"name={profile_name},backbone=wide_resnet50_2,out=1:2,grid={grid}")
    elif profile_name.startswith("wrn_l3_g"):
        grid = int(profile_name.split("g")[-1])
        profile = parse_feature_profile(f"name={profile_name},backbone=wide_resnet50_2,out=2,grid={grid}")
    elif profile_name.startswith("wrn_l2_g"):
        grid = int(profile_name.split("g")[-1])
        profile = parse_feature_profile(f"name={profile_name},backbone=wide_resnet50_2,out=1,grid={grid}")
    elif profile_name.startswith("res18_l23_g"):
        grid = int(profile_name.split("g")[-1])
        profile = parse_feature_profile(f"name={profile_name},backbone=resnet18,out=1:2,grid={grid}")
    else:
        raise ValueError(f"unknown profile name: {profile_name}")
    return profile, bank, topk


def selected_config_by_category(holdout_summary: dict, target: float, tolerance: float) -> dict[str, str]:
    rows = [
        row
        for row in holdout_summary["validation_rows"]
        if abs(row["target_false_pass_rate_defect"] - target) < 1e-12
        and abs(row["allowed_good_pass_drop"] - tolerance) < 1e-12
    ]
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row["selected_config"])
    return {category: Counter(configs).most_common(1)[0][0] for category, configs in grouped.items()}


def timed_collect_features(model, samples, image_size, batch_size, patch_grid, device, desc: str) -> tuple[np.ndarray, np.ndarray, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    features, labels = collect_features(model, samples, image_size, batch_size, patch_grid, device, desc)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return features, labels, elapsed


def timed_patchcore_scores(features: np.ndarray, bank: np.ndarray, chunk_size: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    scores = patchcore_scores(features, bank, chunk_size)
    elapsed = time.perf_counter() - start
    return scores, elapsed


def run_category_config(
    args: argparse.Namespace,
    category: str,
    profile: dict,
    bank_patches: int,
    topk_fraction: float,
    materialized_root: Path,
    device: torch.device,
) -> dict:
    samples = find_materialized_samples(materialized_root, category)
    train, test = normal_train_and_test(samples)
    if not train or not test:
        raise RuntimeError(f"missing train/test samples for {category}")

    model = make_backbone(profile["backbone"], tuple(profile["out_indices"]), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (profile["patch_grid"], profile["patch_grid"])
    train_features, _train_labels_unused, train_feature_sec = timed_collect_features(
        model, train, image_size, args.batch_size, patch_grid, device, f"{profile['name']} {category} train"
    )
    test_features, test_labels, test_feature_sec = timed_collect_features(
        model, test, image_size, args.batch_size, patch_grid, device, f"{profile['name']} {category} test"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_labels = np.zeros(len(train_features), dtype=np.int64)
    bank_start = time.perf_counter()
    bank = sample_normal_patch_bank(train_features, train_labels, bank_patches, args.seed)
    bank_select_sec = time.perf_counter() - bank_start
    patch_scores, nn_search_sec = timed_patchcore_scores(test_features, bank, args.nn_chunk_size)
    score_start = time.perf_counter()
    image_scores = image_scores_from_patch_scores(patch_scores, topk_fraction)
    score_aggregate_sec = time.perf_counter() - score_start

    feature_dim = int(test_features.shape[-1])
    patch_count = int(test_features.shape[1])
    test_count = int(len(test_features))
    online_sec = test_feature_sec + nn_search_sec + score_aggregate_sec
    return {
        "category": category,
        "config": config_name(profile, bank_patches, topk_fraction),
        "profile": profile,
        "sample_counts": {
            "train_normal": int(len(train_features)),
            "test": test_count,
            "test_good": int((test_labels == 0).sum()),
            "test_defect": int((test_labels == 1).sum()),
        },
        "footprint": {
            "patch_count": patch_count,
            "feature_dim": feature_dim,
            "bank_patches": int(len(bank)),
            "bank_bytes_int8": int(len(bank) * feature_dim),
            "bank_bytes_fp32": int(len(bank) * feature_dim * 4),
            "approx_nn_ops": approx_nn_ops(test_count, patch_count, int(len(bank)), feature_dim),
            "per_image_nn_ops": int(patch_count * int(len(bank)) * feature_dim),
            "per_image_feature_values": int(patch_count * feature_dim),
        },
        "timing_sec": {
            "offline_train_feature_extract": round_float(train_feature_sec),
            "offline_bank_select": round_float(bank_select_sec),
            "online_test_feature_extract": round_float(test_feature_sec),
            "online_nn_search": round_float(nn_search_sec),
            "online_score_aggregate": round_float(score_aggregate_sec),
            "online_total": round_float(online_sec),
            "online_total_per_image_ms": round_float(1000.0 * online_sec / max(1, test_count)),
            "online_nn_per_image_ms": round_float(1000.0 * nn_search_sec / max(1, test_count)),
        },
        "score_sanity": {
            "topk_score_mean": round_float(float(np.mean(image_scores["topk_score"]))),
            "topk_score_std": round_float(float(np.std(image_scores["topk_score"]))),
        },
    }


def build_rows(baseline_rows: dict[str, dict], selected_rows: dict[str, dict]) -> list[dict]:
    rows = []
    for category in sorted(selected_rows):
        base = baseline_rows[category]
        sel = selected_rows[category]
        base_fp = base["footprint"]
        sel_fp = sel["footprint"]
        base_time = base["timing_sec"]
        sel_time = sel["timing_sec"]
        patch_ratio = sel_fp["patch_count"] / base_fp["patch_count"]
        bank_ratio = sel_fp["bank_patches"] / base_fp["bank_patches"]
        dim_ratio = sel_fp["feature_dim"] / base_fp["feature_dim"]
        nn_ratio_formula = patch_ratio * bank_ratio * dim_ratio
        nn_ratio_ops = sel_fp["approx_nn_ops"] / base_fp["approx_nn_ops"]
        rows.append(
            {
                "category": category,
                "baseline_config": base["config"],
                "selected_config": sel["config"],
                "patch_ratio": round_float(patch_ratio),
                "bank_ratio": round_float(bank_ratio),
                "feature_dim_ratio": round_float(dim_ratio),
                "nn_ratio_formula": round_float(nn_ratio_formula),
                "relative_nn_ops": round_float(nn_ratio_ops),
                "nn_ops_reduction": round_float(1.0 - nn_ratio_ops),
                "relative_bank_int8": round_float(sel_fp["bank_bytes_int8"] / base_fp["bank_bytes_int8"]),
                "relative_online_nn_time": round_float(sel_time["online_nn_search"] / base_time["online_nn_search"])
                if base_time["online_nn_search"] > 0
                else None,
                "relative_online_total_time": round_float(sel_time["online_total"] / base_time["online_total"])
                if base_time["online_total"] > 0
                else None,
                "baseline_online_total_per_image_ms": base_time["online_total_per_image_ms"],
                "selected_online_total_per_image_ms": sel_time["online_total_per_image_ms"],
                "baseline_nn_per_image_ms": base_time["online_nn_per_image_ms"],
                "selected_nn_per_image_ms": sel_time["online_nn_per_image_ms"],
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    keys = [
        "relative_nn_ops",
        "relative_bank_int8",
        "relative_online_nn_time",
        "relative_online_total_time",
    ]
    summary = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        summary[f"mean_{key}"] = round_float(float(np.mean(values))) if values else None
        summary[f"median_{key}"] = round_float(float(np.median(values))) if values else None
    summary["mean_nn_ops_reduction"] = round_float(float(np.mean([row["nn_ops_reduction"] for row in rows])))
    return summary


def plot_summary(payload: dict, path: Path) -> None:
    rows = payload["rows"]
    labels = [row["category"] for row in rows]
    x = np.arange(len(labels))
    rel_ops = [100.0 * row["relative_nn_ops"] for row in rows]
    rel_nn_time = [100.0 * row["relative_online_nn_time"] if row["relative_online_nn_time"] is not None else 0.0 for row in rows]
    rel_total = [100.0 * row["relative_online_total_time"] if row["relative_online_total_time"] is not None else 0.0 for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    axes[0].bar(x - 0.2, rel_ops, width=0.4, label="formula NN ops", color="#2878b5")
    axes[0].bar(x + 0.2, rel_nn_time, width=0.4, label="measured NN time", color="#d95f02")
    axes[0].set_ylabel("selected / baseline [%]")
    axes[0].set_title("Nearest-neighbor search reduction: formula vs measured runtime")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x, rel_total, width=0.55, color="#5b8c5a", label="measured online total time")
    axes[1].set_ylabel("selected / baseline [%]")
    axes[1].set_title("Online total time includes feature extraction + NN search + score aggregation")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# PatchCore cost credibility audit",
        "",
        "Purpose: verify whether the reported ~98% NN-search reduction is a transparent formula result and whether it also appears in measured runtime.",
        "",
        "## What is counted",
        "",
        "The NN-search cost is counted as:",
        "",
        "`test images x patches per image x memory-bank patches x feature dimension`",
        "",
        "For one image, this reduces to:",
        "",
        "`patches per image x memory-bank patches x feature dimension`",
        "",
        "This is not yet total FPGA system power.  The measured online total adds feature extraction, nearest-neighbor search, and score aggregation.",
        "",
        "## Aggregate",
        "",
        f"- mean relative NN ops: `{payload['aggregate']['mean_relative_nn_ops']:.4f}x`",
        f"- mean NN ops reduction: `{pct(payload['aggregate']['mean_nn_ops_reduction'])}`",
        f"- mean measured NN time: `{payload['aggregate']['mean_relative_online_nn_time']:.4f}x`",
        f"- mean measured online total time: `{payload['aggregate']['mean_relative_online_total_time']:.4f}x`",
        f"- median measured online total time: `{payload['aggregate']['median_relative_online_total_time']:.4f}x`",
        "",
        "## Decomposition by category",
        "",
        "| category | selected config | patch ratio | bank ratio | feature-dim ratio | formula NN ratio | measured NN time | measured total time |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        measured_nn_time = "" if row["relative_online_nn_time"] is None else f"{row['relative_online_nn_time']:.4f}x"
        measured_total_time = (
            "" if row["relative_online_total_time"] is None else f"{row['relative_online_total_time']:.4f}x"
        )
        lines.append(
            f"| {row['category']} | `{row['selected_config']}` | {row['patch_ratio']:.4f}x | "
            f"{row['bank_ratio']:.4f}x | {row['feature_dim_ratio']:.4f}x | "
            f"{row['relative_nn_ops']:.4f}x | {measured_nn_time} | {measured_total_time} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If formula NN ratio and measured NN time ratio move together, the ~98% reduction is not a table artifact.",
        "- If measured total time is much larger than the NN ratio, the remaining cost is mainly feature extraction and framework overhead.",
        "- For FPGA claims, the next step is to replace Python/GPU/CPU wall time with hardware-estimated CNN MAC, memory bandwidth, and distance-engine throughput.",
        "",
        f"Figure: `{payload['figure']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-summary", default="results/mvtec_patchcore_profiled_holdout_validation_001_summary.json")
    parser.add_argument("--output", default="results/mvtec_patchcore_cost_credibility_audit_001_summary.json")
    parser.add_argument("--markdown", default="docs/mvtec_patchcore_cost_credibility_audit_001.md")
    parser.add_argument("--figure", default="results/mvtec_patchcore_cost_credibility_audit_001.png")
    parser.add_argument("--materialized-root", default="/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2")
    parser.add_argument("--baseline-config", default="wrn_l23_g14_b12000_topk0p01")
    parser.add_argument("--report-false-pass-target", type=float, default=0.01)
    parser.add_argument("--report-tolerance", type=float, default=0.02)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    holdout = json.loads(Path(args.holdout_summary).read_text(encoding="utf-8"))
    selected = selected_config_by_category(holdout, args.report_false_pass_target, args.report_tolerance)
    baseline_profile, baseline_bank, baseline_topk = parse_profile_from_config(args.baseline_config)
    materialized_root = Path(args.materialized_root)

    baseline_rows = {}
    selected_rows = {}
    for category in sorted(selected):
        print(f"== {category} baseline ==", flush=True)
        baseline_rows[category] = run_category_config(
            args, category, baseline_profile, baseline_bank, baseline_topk, materialized_root, device
        )
        selected_profile, selected_bank, selected_topk = parse_profile_from_config(selected[category])
        print(f"== {category} selected {selected[category]} ==", flush=True)
        selected_rows[category] = run_category_config(
            args, category, selected_profile, selected_bank, selected_topk, materialized_root, device
        )

    rows = build_rows(baseline_rows, selected_rows)
    payload = {
        "purpose": "Cost credibility audit for category-profiled minimal PatchCore designs.",
        "config": vars(args),
        "selected_config_by_category": selected,
        "aggregate": summarize_rows(rows),
        "rows": rows,
        "baseline_details": baseline_rows,
        "selected_details": selected_rows,
        "figure": args.figure,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(payload, Path(args.figure))
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "figure": args.figure}, indent=2), flush=True)


if __name__ == "__main__":
    main()
