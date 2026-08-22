"""Large profiled PatchCore-lite search for FPGA-oriented minimal designs.

This script is intentionally wider than the earlier lightweight sweeps.  It
computes feature tensors once per feature profile/category, then evaluates many
memory-bank and scoring variants.  The thesis-facing output is a category table:
for each MVTec AD category, how small can the PatchCore-like design become while
matching the full baseline under a defect false-pass constraint?
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
    curve_rows,
    find_materialized_samples,
    image_scores_from_patch_scores,
    make_backbone,
    patchcore_scores,
    sample_normal_patch_bank,
    score_auc,
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass, normal_train_and_test
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


ALL_MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def parse_feature_profile(text: str) -> dict:
    values: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return {
        "name": values["name"],
        "backbone": values.get("backbone", "wide_resnet50_2"),
        "out_indices": [int(v) for v in values.get("out", "1:2").split(":") if v != ""],
        "patch_grid": int(values.get("grid", "14")),
    }


def default_feature_profiles() -> list[dict]:
    specs: list[str] = []
    for grid in [16, 14, 12, 10, 8, 7, 6, 5, 4]:
        specs.append(f"name=wrn_l23_g{grid},backbone=wide_resnet50_2,out=1:2,grid={grid}")
    for grid in [16, 14, 12, 10, 8, 7, 6, 5, 4]:
        specs.append(f"name=wrn_l3_g{grid},backbone=wide_resnet50_2,out=2,grid={grid}")
    for grid in [14, 10, 7, 5]:
        specs.append(f"name=wrn_l2_g{grid},backbone=wide_resnet50_2,out=1,grid={grid}")
    for grid in [14, 10, 7, 5]:
        specs.append(f"name=res18_l23_g{grid},backbone=resnet18,out=1:2,grid={grid}")
    return [parse_feature_profile(spec) for spec in specs]


def approx_nn_ops(test_samples: int, patch_count: int, bank_patches: int, feature_dim: int) -> int:
    return int(test_samples * patch_count * bank_patches * feature_dim)


def config_name(profile: dict, bank_patches: int, topk_fraction: float) -> str:
    topk_text = f"{topk_fraction:g}".replace(".", "p")
    return f"{profile['name']}_b{bank_patches}_topk{topk_text}"


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
            profile_results.append(
                {
                    "category": category,
                    "profile": profile,
                    "status": "skipped",
                    "reason": "insufficient train/test labels",
                }
            )
            continue

        train_features, _ = collect_features(
            model,
            train,
            image_size,
            args.batch_size,
            patch_grid,
            device,
            f"{profile['name']} {category} train",
        )
        test_features, test_labels = collect_features(
            model,
            test,
            image_size,
            args.batch_size,
            patch_grid,
            device,
            f"{profile['name']} {category} test",
        )
        train_labels = np.zeros(len(train_features), dtype=np.int64)
        feature_dim = int(train_features.shape[-1])
        patch_count = int(train_features.shape[1])
        variants: list[dict] = []
        for requested_bank in args.bank_patches:
            bank = sample_normal_patch_bank(train_features, train_labels, requested_bank, args.seed)
            patch_scores = patchcore_scores(test_features, bank, args.nn_chunk_size)
            for topk_fraction in args.topk_fractions:
                image_scores = image_scores_from_patch_scores(patch_scores, topk_fraction)
                selected_scores = image_scores[args.score_name]
                rows = curve_rows(test_labels, selected_scores, args.curve_points)
                variants.append(
                    {
                        "config": config_name(profile, requested_bank, topk_fraction),
                        "feature_profile": profile["name"],
                        "backbone": profile["backbone"],
                        "out_indices": profile["out_indices"],
                        "patch_grid": profile["patch_grid"],
                        "requested_bank_patches": int(requested_bank),
                        "actual_bank_patches": int(len(bank)),
                        "topk_fraction": float(topk_fraction),
                        "selected_score": args.score_name,
                        "auc": {name: score_auc(test_labels, scores) for name, scores in image_scores.items()},
                        "best_rows": [best_under_false_pass(rows, target) for target in args.false_pass_targets],
                        "footprint": {
                            "patch_count": patch_count,
                            "feature_dim": feature_dim,
                            "bank_patches": int(len(bank)),
                            "approx_nn_ops": approx_nn_ops(len(test), patch_count, len(bank), feature_dim),
                            "feature_values": int(patch_count * feature_dim),
                            "bank_feature_values": int(len(bank) * feature_dim),
                            "bank_bytes_fp32": int(len(bank) * feature_dim * 4),
                            "bank_bytes_int8": int(len(bank) * feature_dim),
                            "test_samples": int(len(test)),
                        },
                    }
                )
        profile_results.append(
            {
                "category": category,
                "profile": profile,
                "status": "done",
                "sample_counts": {
                    "train_normal": len(train),
                    "test": len(test),
                    "test_good": int((test_labels == 0).sum()),
                    "test_defect": int((test_labels == 1).sum()),
                },
                "variants": variants,
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return profile_results


def flatten_variants(profile_results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for result in profile_results:
        if result.get("status") != "done":
            continue
        category = result["category"]
        for variant in result["variants"]:
            row = dict(variant)
            row["category"] = category
            row["sample_counts"] = result["sample_counts"]
            rows.append(row)
    return rows


def add_relative_costs(rows: list[dict], baseline_config: str) -> None:
    by_category: dict[str, dict] = {}
    for row in rows:
        if row["config"] == baseline_config:
            by_category[row["category"]] = row
    for row in rows:
        baseline = by_category.get(row["category"])
        if not baseline:
            row["relative_nn_ops"] = None
            row["relative_bank_int8"] = None
            continue
        base_ops = baseline["footprint"]["approx_nn_ops"]
        base_bank = baseline["footprint"]["bank_bytes_int8"]
        row["relative_nn_ops"] = round_float(row["footprint"]["approx_nn_ops"] / base_ops) if base_ops else None
        row["relative_bank_int8"] = round_float(row["footprint"]["bank_bytes_int8"] / base_bank) if base_bank else None


def best_for_target(row: dict, target: float) -> dict | None:
    for best in row["best_rows"]:
        if abs(best["target"] - target) < 1e-12:
            return best
    return None


def build_minimal_table(rows: list[dict], baseline_config: str, false_pass_targets: list[float], tolerances: list[float]) -> list[dict]:
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    table: list[dict] = []
    for category, category_rows in sorted(by_category.items()):
        baseline = next((row for row in category_rows if row["config"] == baseline_config), None)
        if baseline is None:
            continue
        for target in false_pass_targets:
            baseline_best = best_for_target(baseline, target)
            baseline_good = None if baseline_best is None else baseline_best["good_pass_rate_good"]
            for tolerance in tolerances:
                selected = None
                if baseline_good is not None:
                    min_good = max(0.0, baseline_good - tolerance)
                    feasible = []
                    for row in category_rows:
                        best = best_for_target(row, target)
                        if best is None or best["good_pass_rate_good"] is None:
                            continue
                        if best["good_pass_rate_good"] >= min_good:
                            feasible.append((row, best))
                    if feasible:
                        selected = min(
                            feasible,
                            key=lambda pair: (
                                pair[0]["relative_nn_ops"] if pair[0]["relative_nn_ops"] is not None else float("inf"),
                                pair[0]["relative_bank_int8"] if pair[0]["relative_bank_int8"] is not None else float("inf"),
                                -pair[1]["good_pass_rate_good"],
                            ),
                        )
                if selected is None:
                    table.append(
                        {
                            "category": category,
                            "max_false_pass_rate_defect": target,
                            "allowed_good_pass_drop": tolerance,
                            "baseline_config": baseline_config,
                            "baseline_good_pass": baseline_good,
                            "selected_config": None,
                        }
                    )
                    continue
                row, best = selected
                table.append(
                    {
                        "category": category,
                        "max_false_pass_rate_defect": target,
                        "allowed_good_pass_drop": tolerance,
                        "baseline_config": baseline_config,
                        "baseline_good_pass": baseline_good,
                        "selected_config": row["config"],
                        "selected_good_pass": best["good_pass_rate_good"],
                        "good_pass_drop": round_float(baseline_good - best["good_pass_rate_good"]) if baseline_good is not None else None,
                        "relative_nn_ops": row["relative_nn_ops"],
                        "nn_ops_reduction": round_float(1.0 - row["relative_nn_ops"]) if row["relative_nn_ops"] is not None else None,
                        "relative_bank_int8": row["relative_bank_int8"],
                        "bank_int8_reduction": round_float(1.0 - row["relative_bank_int8"]) if row["relative_bank_int8"] is not None else None,
                        "patch_grid": row["patch_grid"],
                        "backbone": row["backbone"],
                        "out_indices": row["out_indices"],
                        "bank_patches": row["actual_bank_patches"],
                        "feature_dim": row["footprint"]["feature_dim"],
                        "patch_count": row["footprint"]["patch_count"],
                        "threshold": best["threshold"],
                        "auc": row["auc"][row["selected_score"]]["image_auroc"],
                    }
                )
    return table


def aggregate_table(minimal_table: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for row in minimal_table:
        if row.get("selected_config") is None:
            continue
        key = (row["max_false_pass_rate_defect"], row["allowed_good_pass_drop"])
        groups.setdefault(key, []).append(row)
    aggregate = []
    for (target, tolerance), rows in sorted(groups.items()):
        aggregate.append(
            {
                "max_false_pass_rate_defect": target,
                "allowed_good_pass_drop": tolerance,
                "categories_solved": len(rows),
                "mean_baseline_good_pass": round_float(float(np.mean([row["baseline_good_pass"] for row in rows]))),
                "mean_selected_good_pass": round_float(float(np.mean([row["selected_good_pass"] for row in rows]))),
                "mean_good_pass_drop": round_float(float(np.mean([row["good_pass_drop"] for row in rows]))),
                "mean_relative_nn_ops": round_float(float(np.mean([row["relative_nn_ops"] for row in rows]))),
                "mean_nn_ops_reduction": round_float(float(np.mean([row["nn_ops_reduction"] for row in rows]))),
                "median_relative_nn_ops": round_float(float(np.median([row["relative_nn_ops"] for row in rows]))),
                "mean_relative_bank_int8": round_float(float(np.mean([row["relative_bank_int8"] for row in rows]))),
            }
        )
    return aggregate


def pareto_rows(rows: list[dict], false_pass_targets: list[float]) -> list[dict]:
    output: list[dict] = []
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        for target in false_pass_targets:
            candidates = []
            for row in category_rows:
                best = best_for_target(row, target)
                if best is None or best["good_pass_rate_good"] is None:
                    continue
                candidates.append((row, best))
            for row, best in candidates:
                dominated = False
                for other, other_best in candidates:
                    if other is row:
                        continue
                    if (
                        other["relative_nn_ops"] <= row["relative_nn_ops"]
                        and other_best["good_pass_rate_good"] >= best["good_pass_rate_good"]
                        and (
                            other["relative_nn_ops"] < row["relative_nn_ops"]
                            or other_best["good_pass_rate_good"] > best["good_pass_rate_good"]
                        )
                    ):
                        dominated = True
                        break
                if not dominated:
                    output.append(
                        {
                            "category": category,
                            "max_false_pass_rate_defect": target,
                            "config": row["config"],
                            "good_pass_rate_good": best["good_pass_rate_good"],
                            "relative_nn_ops": row["relative_nn_ops"],
                            "relative_bank_int8": row["relative_bank_int8"],
                            "patch_grid": row["patch_grid"],
                            "out_indices": row["out_indices"],
                            "bank_patches": row["actual_bank_patches"],
                            "feature_dim": row["footprint"]["feature_dim"],
                            "auc": row["auc"][row["selected_score"]]["image_auroc"],
                        }
                    )
    return output


def plot_minimal_table(minimal_table: list[dict], false_pass_target: float, tolerance: float, path: Path) -> None:
    rows = [
        row
        for row in minimal_table
        if row["selected_config"] is not None
        and abs(row["max_false_pass_rate_defect"] - false_pass_target) < 1e-12
        and abs(row["allowed_good_pass_drop"] - tolerance) < 1e-12
    ]
    rows.sort(key=lambda row: row["category"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    labels = [row["category"] for row in rows]
    baseline = [100.0 * row["baseline_good_pass"] for row in rows]
    selected = [100.0 * row["selected_good_pass"] for row in rows]
    rel_ops = [100.0 * row["relative_nn_ops"] for row in rows]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    ax1.plot(x, baseline, marker="o", linewidth=1.6, label="baseline good pass")
    ax1.plot(x, selected, marker="o", linewidth=1.6, label="selected minimal good pass")
    ax1.set_ylabel("good pass [%]")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.bar(x, rel_ops, color="#d95f02")
    ax2.set_ylabel("relative NN ops [%]")
    ax2.set_ylim(0, max(105, max(rel_ops) * 1.15))
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_title(f"minimal config at defect false-pass <= {100*false_pass_target:.1f}%, good-pass drop <= {100*tolerance:.1f}%")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    target = payload["report_false_pass_target"]
    tolerance = payload["report_tolerance"]
    rows = [
        row
        for row in payload["minimal_table"]
        if row["selected_config"] is not None
        and abs(row["max_false_pass_rate_defect"] - target) < 1e-12
        and abs(row["allowed_good_pass_drop"] - tolerance) < 1e-12
    ]
    rows.sort(key=lambda row: row["category"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MVTec PatchCore profiled minimal search",
        "",
        "Purpose: identify the smallest PatchCore-like configuration per category under inspection false-pass constraints.",
        "",
        "## Report setting",
        "",
        f"- baseline config: `{payload['baseline_config']}`",
        f"- defect false-pass constraint: `{100*target:.1f}%`",
        f"- allowed good-pass drop from baseline: `{100*tolerance:.1f}%`",
        "",
        "## Category minimal configuration table",
        "",
        "| category | baseline good pass | selected config | selected good pass | good-pass drop | relative NN ops | NN ops reduction | relative bank | grid | layers | bank | feature dim |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {pct(row['baseline_good_pass'])} | `{row['selected_config']}` | "
            f"{pct(row['selected_good_pass'])} | {pct(row['good_pass_drop'])} | "
            f"{row['relative_nn_ops']:.4f}x | {pct(row['nn_ops_reduction'])} | "
            f"{row['relative_bank_int8']:.4f}x | {row['patch_grid']} | {row['out_indices']} | "
            f"{row['bank_patches']} | {row['feature_dim']} |"
        )
    lines += [
        "",
        "## Aggregate minimal-design summary",
        "",
        "| max defect false-pass | allowed good-pass drop | categories | mean baseline good pass | mean selected good pass | mean drop | mean relative NN ops | mean NN ops reduction | median relative NN ops | mean relative bank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_minimal_table"]:
        lines.append(
            f"| {100*row['max_false_pass_rate_defect']:.1f}% | {100*row['allowed_good_pass_drop']:.1f}% | "
            f"{row['categories_solved']} | {pct(row['mean_baseline_good_pass'])} | "
            f"{pct(row['mean_selected_good_pass'])} | {pct(row['mean_good_pass_drop'])} | "
            f"{row['mean_relative_nn_ops']:.4f}x | {pct(row['mean_nn_ops_reduction'])} | "
            f"{row['median_relative_nn_ops']:.4f}x | {row['mean_relative_bank_int8']:.4f}x |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If a category keeps baseline good-pass with very small relative NN ops, it is a strong FPGA specialization candidate.",
        "- If a category requires the baseline or near-baseline config, it should use a larger bank or a different algorithmic treatment.",
        "- The thesis claim should compare uniform compression against this category-profiled selection.",
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
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--curve-points", type=int, default=180)
    parser.add_argument("--false-pass-targets", nargs="*", type=float, default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--tolerances", nargs="*", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--score-name", default="topk_score", choices=["max_score", "topk_score"])
    parser.add_argument("--baseline-config", default="wrn_l23_g14_b12000_topk0p01")
    parser.add_argument("--report-false-pass-target", type=float, default=0.01)
    parser.add_argument("--report-tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="results/mvtec_patchcore_profiled_minimal_search_001_summary.json")
    parser.add_argument("--markdown", default="docs/mvtec_patchcore_profiled_minimal_search_001.md")
    parser.add_argument("--figure", default="results/mvtec_patchcore_profiled_minimal_search_001.png")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    profiles = [parse_feature_profile(text) for text in args.feature_profile] or default_feature_profiles()
    materialized_root = Path(args.materialized_root)
    profile_results: list[dict] = []
    for profile in profiles:
        profile_results.extend(run_profile(args, profile, args.categories, materialized_root, device))

    variant_rows = flatten_variants(profile_results)
    add_relative_costs(variant_rows, args.baseline_config)
    minimal_table = build_minimal_table(variant_rows, args.baseline_config, args.false_pass_targets, args.tolerances)
    aggregate_minimal = aggregate_table(minimal_table)
    pareto = pareto_rows(variant_rows, args.false_pass_targets)
    payload = {
        "purpose": "Profile category-specific minimal PatchCore-like configs for FPGA-oriented inspection.",
        "config": vars(args),
        "feature_profiles": profiles,
        "baseline_config": args.baseline_config,
        "variant_rows": variant_rows,
        "minimal_table": minimal_table,
        "aggregate_minimal_table": aggregate_minimal,
        "pareto_rows": pareto,
        "report_false_pass_target": args.report_false_pass_target,
        "report_tolerance": args.report_tolerance,
        "figure": args.figure,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    plot_minimal_table(minimal_table, args.report_false_pass_target, args.report_tolerance, Path(args.figure))
    print(
        json.dumps(
            {
                "wrote": args.output,
                "markdown": args.markdown,
                "figure": args.figure,
                "feature_profiles": len(profiles),
                "variant_rows": len(variant_rows),
                "minimal_rows": len(minimal_table),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
