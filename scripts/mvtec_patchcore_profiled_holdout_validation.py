"""Holdout validation for category-profiled minimal PatchCore designs.

The previous profiled search selected the smallest configuration directly from
the same test split used for reporting.  This script removes that selection
bias: each MVTec AD category's official test split is divided into validation
and holdout subsets.  The category-specific minimal configuration and anomaly
threshold are chosen only on validation, then evaluated once on holdout.
"""

from __future__ import annotations

import argparse
import json
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
    score_auc,
)
from scripts.mvtec_patchcore_lightweight_sweep import normal_train_and_test
from scripts.mvtec_patchcore_profiled_minimal_search import (
    ALL_MVTEC_CATEGORIES,
    approx_nn_ops,
    config_name,
    default_feature_profiles,
    parse_feature_profile,
)
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


def stratified_holdout_indices(labels: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    val_parts = []
    test_parts = []
    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == label)
        idx = rng.permutation(idx)
        if len(idx) <= 1:
            n_val = len(idx)
        else:
            n_val = int(round(len(idx) * val_fraction))
            n_val = min(max(1, n_val), len(idx) - 1)
        val_parts.append(idx[:n_val])
        test_parts.append(idx[n_val:])
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=np.int64)
    return val_idx.astype(np.int64), test_idx.astype(np.int64)


def select_threshold(labels: np.ndarray, scores: np.ndarray, target_false_pass: float, points: int) -> dict | None:
    good = labels == 0
    defect = labels == 1
    if not good.any() or not defect.any():
        return None
    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, points))
    best = None
    for threshold in thresholds:
        predicted_defect = scores >= threshold
        false_pass = float((~predicted_defect[defect]).mean())
        if false_pass > target_false_pass:
            continue
        good_loss = float(predicted_defect[good].mean())
        row = {
            "threshold": round_float(float(threshold)),
            "false_pass_rate_defect": round_float(false_pass),
            "defect_reject_rate_defect": round_float(1.0 - false_pass),
            "good_loss_rate_good": round_float(good_loss),
            "good_pass_rate_good": round_float(1.0 - good_loss),
        }
        if best is None or row["good_pass_rate_good"] > best["good_pass_rate_good"]:
            best = row
    return best


def evaluate_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    good = labels == 0
    defect = labels == 1
    predicted_defect = scores >= threshold
    false_pass = float((~predicted_defect[defect]).mean()) if defect.any() else None
    good_loss = float(predicted_defect[good].mean()) if good.any() else None
    return {
        "false_pass_rate_defect": round_float(false_pass) if false_pass is not None else None,
        "defect_reject_rate_defect": round_float(1.0 - false_pass) if false_pass is not None else None,
        "good_loss_rate_good": round_float(good_loss) if good_loss is not None else None,
        "good_pass_rate_good": round_float(1.0 - good_loss) if good_loss is not None else None,
    }


def run_profile(
    args: argparse.Namespace,
    profile: dict,
    categories: list[str],
    materialized_root: Path,
    device: torch.device,
) -> list[dict]:
    model = make_backbone(profile["backbone"], tuple(profile["out_indices"]), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (profile["patch_grid"], profile["patch_grid"])
    profile_results: list[dict] = []
    for category in categories:
        samples = find_materialized_samples(materialized_root, category)
        train, test = normal_train_and_test(samples)
        if not train or not test or len({sample.label for sample in test}) < 2:
            profile_results.append({"category": category, "profile": profile, "status": "skipped"})
            continue
        train_features, _ = collect_features(
            model, train, image_size, args.batch_size, patch_grid, device, f"{profile['name']} {category} train"
        )
        test_features, test_labels = collect_features(
            model, test, image_size, args.batch_size, patch_grid, device, f"{profile['name']} {category} test"
        )
        train_labels = np.zeros(len(train_features), dtype=np.int64)
        feature_dim = int(train_features.shape[-1])
        patch_count = int(train_features.shape[1])
        variants = []
        for requested_bank in args.bank_patches:
            bank = sample_normal_patch_bank(train_features, train_labels, requested_bank, args.seed)
            patch_scores = patchcore_scores(test_features, bank, args.nn_chunk_size)
            for topk_fraction in args.topk_fractions:
                image_scores = image_scores_from_patch_scores(patch_scores, topk_fraction)
                scores = image_scores[args.score_name].astype(np.float32)
                variants.append(
                    {
                        "config": config_name(profile, requested_bank, topk_fraction),
                        "feature_profile": profile["name"],
                        "backbone": profile["backbone"],
                        "out_indices": profile["out_indices"],
                        "patch_grid": profile["patch_grid"],
                        "actual_bank_patches": int(len(bank)),
                        "topk_fraction": float(topk_fraction),
                        "scores": scores,
                        "auc": score_auc(test_labels, scores),
                        "footprint": {
                            "patch_count": patch_count,
                            "feature_dim": feature_dim,
                            "bank_patches": int(len(bank)),
                            "approx_nn_ops": approx_nn_ops(len(test), patch_count, len(bank), feature_dim),
                            "bank_bytes_int8": int(len(bank) * feature_dim),
                        },
                    }
                )
        profile_results.append(
            {
                "category": category,
                "profile": profile,
                "status": "done",
                "test_labels": test_labels,
                "sample_counts": {
                    "train_normal": len(train),
                    "official_test": len(test),
                    "official_test_good": int((test_labels == 0).sum()),
                    "official_test_defect": int((test_labels == 1).sum()),
                },
                "variants": variants,
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return profile_results


def flatten_profile_results(profile_results: list[dict]) -> dict[str, dict]:
    by_category: dict[str, dict] = {}
    for result in profile_results:
        if result.get("status") != "done":
            continue
        category = result["category"]
        entry = by_category.setdefault(
            category,
            {"category": category, "test_labels": result["test_labels"], "sample_counts": result["sample_counts"], "variants": []},
        )
        entry["variants"].extend(result["variants"])
    return by_category


def add_relative_costs(by_category: dict[str, dict], baseline_config: str) -> None:
    for category, entry in by_category.items():
        baseline = next((variant for variant in entry["variants"] if variant["config"] == baseline_config), None)
        if baseline is None:
            raise RuntimeError(f"baseline config {baseline_config} not found for {category}")
        base_ops = baseline["footprint"]["approx_nn_ops"]
        base_bank = baseline["footprint"]["bank_bytes_int8"]
        for variant in entry["variants"]:
            variant["relative_nn_ops"] = round_float(variant["footprint"]["approx_nn_ops"] / base_ops)
            variant["relative_bank_int8"] = round_float(variant["footprint"]["bank_bytes_int8"] / base_bank)


def evaluate_seed_category(args: argparse.Namespace, entry: dict, split_seed: int, target: float, tolerance: float) -> dict:
    labels = entry["test_labels"]
    val_idx, holdout_idx = stratified_holdout_indices(labels, split_seed, args.val_fraction)
    if len(holdout_idx) == 0:
        raise RuntimeError(f"empty holdout split for {entry['category']} seed={split_seed}")
    baseline = next(variant for variant in entry["variants"] if variant["config"] == args.baseline_config)

    baseline_val = select_threshold(labels[val_idx], baseline["scores"][val_idx], target, args.threshold_points)
    if baseline_val is None:
        raise RuntimeError(f"invalid baseline validation split for {entry['category']} seed={split_seed}")
    baseline_holdout = evaluate_threshold(labels[holdout_idx], baseline["scores"][holdout_idx], baseline_val["threshold"])
    min_val_good = max(0.0, baseline_val["good_pass_rate_good"] - tolerance)

    feasible = []
    for variant in entry["variants"]:
        val = select_threshold(labels[val_idx], variant["scores"][val_idx], target, args.threshold_points)
        if val is None or val["good_pass_rate_good"] < min_val_good:
            continue
        holdout = evaluate_threshold(labels[holdout_idx], variant["scores"][holdout_idx], val["threshold"])
        feasible.append((variant, val, holdout))
    if not feasible:
        selected = (baseline, baseline_val, baseline_holdout)
    else:
        selected = min(
            feasible,
            key=lambda item: (
                item[0]["relative_nn_ops"],
                item[0]["relative_bank_int8"],
                -item[1]["good_pass_rate_good"],
            ),
        )
    selected_variant, selected_val, selected_holdout = selected
    return {
        "split_seed": split_seed,
        "category": entry["category"],
        "target_false_pass_rate_defect": target,
        "allowed_good_pass_drop": tolerance,
        "val_size": int(len(val_idx)),
        "holdout_size": int(len(holdout_idx)),
        "sample_counts": entry["sample_counts"],
        "baseline_config": baseline["config"],
        "baseline_val": baseline_val,
        "baseline_holdout": baseline_holdout,
        "selected_config": selected_variant["config"],
        "selected_val": selected_val,
        "selected_holdout": selected_holdout,
        "holdout_good_pass_drop": round_float(
            baseline_holdout["good_pass_rate_good"] - selected_holdout["good_pass_rate_good"]
        ),
        "holdout_false_pass_delta": round_float(
            selected_holdout["false_pass_rate_defect"] - baseline_holdout["false_pass_rate_defect"]
        ),
        "relative_nn_ops": selected_variant["relative_nn_ops"],
        "nn_ops_reduction": round_float(1.0 - selected_variant["relative_nn_ops"]),
        "relative_bank_int8": selected_variant["relative_bank_int8"],
        "patch_grid": selected_variant["patch_grid"],
        "out_indices": selected_variant["out_indices"],
        "backbone": selected_variant["backbone"],
        "bank_patches": selected_variant["actual_bank_patches"],
        "topk_fraction": selected_variant["topk_fraction"],
        "feature_dim": selected_variant["footprint"]["feature_dim"],
        "auc_official_test": selected_variant["auc"]["image_auroc"],
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["target_false_pass_rate_defect"], row["allowed_good_pass_drop"]), []).append(row)
    summary = []
    for (target, tolerance), group in sorted(groups.items()):
        summary.append(
            {
                "target_false_pass_rate_defect": target,
                "allowed_good_pass_drop": tolerance,
                "rows": len(group),
                "categories": len({row["category"] for row in group}),
                "split_seeds": len({row["split_seed"] for row in group}),
                "baseline_holdout_good_pass_mean": round_float(float(np.mean([row["baseline_holdout"]["good_pass_rate_good"] for row in group]))),
                "selected_holdout_good_pass_mean": round_float(float(np.mean([row["selected_holdout"]["good_pass_rate_good"] for row in group]))),
                "holdout_good_pass_drop_mean": round_float(float(np.mean([row["holdout_good_pass_drop"] for row in group]))),
                "baseline_holdout_false_pass_mean": round_float(float(np.mean([row["baseline_holdout"]["false_pass_rate_defect"] for row in group]))),
                "selected_holdout_false_pass_mean": round_float(float(np.mean([row["selected_holdout"]["false_pass_rate_defect"] for row in group]))),
                "holdout_false_pass_delta_mean": round_float(float(np.mean([row["holdout_false_pass_delta"] for row in group]))),
                "relative_nn_ops_mean": round_float(float(np.mean([row["relative_nn_ops"] for row in group]))),
                "relative_nn_ops_median": round_float(float(np.median([row["relative_nn_ops"] for row in group]))),
                "nn_ops_reduction_mean": round_float(float(np.mean([row["nn_ops_reduction"] for row in group]))),
                "relative_bank_int8_mean": round_float(float(np.mean([row["relative_bank_int8"] for row in group]))),
                "selected_holdout_constraint_violations": int(
                    sum(row["selected_holdout"]["false_pass_rate_defect"] > target for row in group)
                ),
            }
        )
    return summary


def category_summary(rows: list[dict], report_target: float, report_tolerance: float) -> list[dict]:
    selected = [
        row
        for row in rows
        if abs(row["target_false_pass_rate_defect"] - report_target) < 1e-12
        and abs(row["allowed_good_pass_drop"] - report_tolerance) < 1e-12
    ]
    output = []
    for category in sorted({row["category"] for row in selected}):
        group = [row for row in selected if row["category"] == category]
        output.append(
            {
                "category": category,
                "runs": len(group),
                "baseline_holdout_good_pass_mean": round_float(float(np.mean([row["baseline_holdout"]["good_pass_rate_good"] for row in group]))),
                "selected_holdout_good_pass_mean": round_float(float(np.mean([row["selected_holdout"]["good_pass_rate_good"] for row in group]))),
                "holdout_good_pass_drop_mean": round_float(float(np.mean([row["holdout_good_pass_drop"] for row in group]))),
                "selected_holdout_false_pass_mean": round_float(float(np.mean([row["selected_holdout"]["false_pass_rate_defect"] for row in group]))),
                "relative_nn_ops_mean": round_float(float(np.mean([row["relative_nn_ops"] for row in group]))),
                "relative_nn_ops_median": round_float(float(np.median([row["relative_nn_ops"] for row in group]))),
                "relative_bank_int8_mean": round_float(float(np.mean([row["relative_bank_int8"] for row in group]))),
                "most_common_selected_config": max(
                    sorted({row["selected_config"] for row in group}),
                    key=lambda config: sum(row["selected_config"] == config for row in group),
                ),
            }
        )
    return output


def plot_category_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    labels = [row["category"] for row in rows]
    baseline = [100.0 * row["baseline_holdout_good_pass_mean"] for row in rows]
    selected = [100.0 * row["selected_holdout_good_pass_mean"] for row in rows]
    rel_ops = [100.0 * row["relative_nn_ops_mean"] for row in rows]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.3), sharex=True)
    ax1.plot(x, baseline, marker="o", label="baseline holdout good pass")
    ax1.plot(x, selected, marker="o", label="profiled holdout good pass")
    ax1.set_ylabel("holdout good pass [%]")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.bar(x, rel_ops, color="#d95f02")
    ax2.set_ylabel("relative NN ops [%]")
    ax2.set_ylim(0, max(105, max(rel_ops) * 1.15))
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_title("holdout-validated category-profiled minimal configs")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Holdout-validated profiled PatchCore minimal search",
        "",
        "Purpose: confirm whether category-profiled minimal PatchCore configs survive a validation/test split.",
        "",
        "## Protocol",
        "",
        f"- baseline config: `{payload['baseline_config']}`",
        f"- validation fraction from official test split: `{100*payload['config']['val_fraction']:.1f}%`",
        f"- split seeds: `{payload['config']['split_seeds']}`",
        f"- report false-pass target: `{100*payload['report_false_pass_target']:.1f}%`",
        f"- report allowed validation good-pass drop: `{100*payload['report_tolerance']:.1f}%`",
        "",
        "Configuration and threshold are selected on validation only.  The table reports holdout evaluation.",
        "",
        "## Aggregate result",
        "",
        "| false-pass target | allowed val drop | rows | categories | seeds | baseline holdout good pass | selected holdout good pass | holdout good-pass drop | selected holdout false pass | relative NN ops | NN ops reduction | constraint violations |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_summary"]:
        lines.append(
            f"| {100*row['target_false_pass_rate_defect']:.1f}% | {100*row['allowed_good_pass_drop']:.1f}% | "
            f"{row['rows']} | {row['categories']} | {row['split_seeds']} | "
            f"{pct(row['baseline_holdout_good_pass_mean'])} | {pct(row['selected_holdout_good_pass_mean'])} | "
            f"{pct(row['holdout_good_pass_drop_mean'])} | {pct(row['selected_holdout_false_pass_mean'])} | "
            f"{row['relative_nn_ops_mean']:.4f}x | {pct(row['nn_ops_reduction_mean'])} | "
            f"{row['selected_holdout_constraint_violations']} |"
        )
    lines += [
        "",
        "## Category result at report setting",
        "",
        "| category | runs | baseline holdout good pass | selected holdout good pass | holdout good-pass drop | selected holdout false pass | relative NN ops | relative bank | common selected config |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["category_summary"]:
        lines.append(
            f"| {row['category']} | {row['runs']} | {pct(row['baseline_holdout_good_pass_mean'])} | "
            f"{pct(row['selected_holdout_good_pass_mean'])} | {pct(row['holdout_good_pass_drop_mean'])} | "
            f"{pct(row['selected_holdout_false_pass_mean'])} | {row['relative_nn_ops_mean']:.4f}x | "
            f"{row['relative_bank_int8_mean']:.4f}x | `{row['most_common_selected_config']}` |"
        )
    lines += [
        "",
        "## Interpretation guide",
        "",
        "- If holdout good-pass stays close to baseline while relative NN ops remains tiny, the category-profiled FPGA theme is credible.",
        "- Constraint violations show how often a validation-selected threshold fails the target on holdout.",
        "- Large gaps between validation-selected and holdout performance indicate overfitting to the small MVTec test split.",
        "",
        f"Figure: `{payload['figure']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", default="/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2")
    parser.add_argument("--categories", nargs="*", default=ALL_MVTEC_CATEGORIES)
    parser.add_argument("--feature-profile", action="append", default=[])
    parser.add_argument("--bank-patches", nargs="*", type=int, default=[12000, 9000, 6000, 4000, 3000, 2000, 1500, 1000, 750, 500, 250, 125])
    parser.add_argument("--topk-fractions", nargs="*", type=float, default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--split-seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument("--val-fraction", type=float, default=0.5)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--threshold-points", type=int, default=180)
    parser.add_argument("--false-pass-targets", nargs="*", type=float, default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--tolerances", nargs="*", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--score-name", default="topk_score", choices=["max_score", "topk_score"])
    parser.add_argument("--baseline-config", default="wrn_l23_g14_b12000_topk0p01")
    parser.add_argument("--report-false-pass-target", type=float, default=0.01)
    parser.add_argument("--report-tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="results/mvtec_patchcore_profiled_holdout_validation_001_summary.json")
    parser.add_argument("--markdown", default="docs/mvtec_patchcore_profiled_holdout_validation_001.md")
    parser.add_argument("--figure", default="results/mvtec_patchcore_profiled_holdout_validation_001.png")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    profiles = [parse_feature_profile(text) for text in args.feature_profile] or default_feature_profiles()
    profile_results: list[dict] = []
    materialized_root = Path(args.materialized_root)
    for profile in profiles:
        profile_results.extend(run_profile(args, profile, args.categories, materialized_root, device))

    by_category = flatten_profile_results(profile_results)
    add_relative_costs(by_category, args.baseline_config)
    validation_rows = []
    for split_seed in args.split_seeds:
        for target in args.false_pass_targets:
            for tolerance in args.tolerances:
                for entry in by_category.values():
                    validation_rows.append(evaluate_seed_category(args, entry, split_seed, target, tolerance))

    aggregate_summary = summarize(validation_rows)
    cat_summary = category_summary(validation_rows, args.report_false_pass_target, args.report_tolerance)
    payload = {
        "purpose": "Holdout validation for category-profiled minimal PatchCore-like FPGA designs.",
        "config": vars(args),
        "feature_profiles": profiles,
        "baseline_config": args.baseline_config,
        "validation_rows": validation_rows,
        "aggregate_summary": aggregate_summary,
        "category_summary": cat_summary,
        "report_false_pass_target": args.report_false_pass_target,
        "report_tolerance": args.report_tolerance,
        "figure": args.figure,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    plot_category_summary(cat_summary, Path(args.figure))
    print(
        json.dumps(
            {
                "wrote": args.output,
                "markdown": args.markdown,
                "figure": args.figure,
                "validation_rows": len(validation_rows),
                "categories": len(by_category),
                "feature_profiles": len(profiles),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
