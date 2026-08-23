"""Probe why PatchCore-lite can or cannot be reduced per category.

This experiment turns the latest profiled sweep into explanation-facing
quantities.  The goal is not another leaderboard table; it is to support a
plausible hypothesis with measurable terms and formulas:

* Coverage: how well a reduced normal memory bank covers normal patch features.
* Margin: how much score-space separation remains between normal and defect
  images after reducing backbone/layers/grid/bank.
* Collision: how often defect images fall inside a high-good-pass normal region.
* Cost product: why the nearest-neighbor cost reduction follows
  Patches x Bank x FeatureDim.
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
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass, normal_train_and_test
from scripts.mvtec_patchcore_profiled_minimal_search import parse_feature_profile
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


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def parse_topk(config_name: str) -> float:
    suffix = config_name.split("_")[-1]
    if not suffix.startswith("topk"):
        raise ValueError(f"config has no topk suffix: {config_name}")
    return float(suffix[4:].replace("p", "."))


def profile_from_row(row: dict) -> dict:
    config = row["selected_config"]
    profile_name = config.rsplit("_b", 1)[0]
    out = ":".join(str(index) for index in row["out_indices"])
    return parse_feature_profile(
        f"name={profile_name},backbone={row['backbone']},out={out},grid={row['patch_grid']}"
    )


def select_minimal_rows(summary: dict, target: float, tolerance: float) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for row in summary["minimal_table"]:
        if abs(row["max_false_pass_rate_defect"] - target) < 1e-12 and abs(row["allowed_good_pass_drop"] - tolerance) < 1e-12:
            selected[row["category"]] = row
    return selected


def cdf_collision(defect_scores: np.ndarray, normal_scores: np.ndarray, normal_accept_quantile: float) -> float:
    if len(defect_scores) == 0 or len(normal_scores) == 0:
        return float("nan")
    threshold = float(np.quantile(normal_scores, normal_accept_quantile))
    return float((defect_scores <= threshold).mean())


def score_margin(normal_scores: np.ndarray, defect_scores: np.ndarray, false_pass_target: float) -> dict:
    if len(normal_scores) == 0 or len(defect_scores) == 0:
        return {
            "threshold_from_defect_quantile": None,
            "good_pass_at_threshold": None,
            "margin_vs_good_q95": None,
            "margin_vs_good_q99": None,
            "overlap_defect_in_good95_region": None,
        }
    threshold = float(np.quantile(defect_scores, false_pass_target))
    good_pass = float((normal_scores < threshold).mean())
    good_q95 = float(np.quantile(normal_scores, 0.95))
    good_q99 = float(np.quantile(normal_scores, 0.99))
    return {
        "threshold_from_defect_quantile": round_float(threshold),
        "good_pass_at_threshold": round_float(good_pass),
        "margin_vs_good_q95": round_float(threshold - good_q95),
        "margin_vs_good_q99": round_float(threshold - good_q99),
        "overlap_defect_in_good95_region": round_float(cdf_collision(defect_scores, normal_scores, 0.95)),
    }


def summarize_distribution(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"mean": None, "q50": None, "q90": None, "q95": None, "q99": None, "max": None}
    return {
        "mean": round_float(float(np.mean(values))),
        "q50": round_float(float(np.quantile(values, 0.50))),
        "q90": round_float(float(np.quantile(values, 0.90))),
        "q95": round_float(float(np.quantile(values, 0.95))),
        "q99": round_float(float(np.quantile(values, 0.99))),
        "max": round_float(float(np.max(values))),
    }


def evaluate_config(args: argparse.Namespace, category: str, label: str, profile: dict, bank_patches: int, topk_fraction: float, materialized_root: Path, device: torch.device) -> dict:
    samples = find_materialized_samples(materialized_root, category)
    train, test = normal_train_and_test(samples)
    if not train or not test:
        raise RuntimeError(f"insufficient samples for {category}")

    model = make_backbone(profile["backbone"], tuple(profile["out_indices"]), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (profile["patch_grid"], profile["patch_grid"])
    train_features, train_labels = collect_features(
        model,
        train,
        image_size,
        args.batch_size,
        patch_grid,
        device,
        f"{label} {category} train",
    )
    test_features, test_labels = collect_features(
        model,
        test,
        image_size,
        args.batch_size,
        patch_grid,
        device,
        f"{label} {category} test",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    bank = sample_normal_patch_bank(train_features, train_labels, bank_patches, args.seed)
    train_patch_scores = patchcore_scores(train_features, bank, args.nn_chunk_size)
    test_patch_scores = patchcore_scores(test_features, bank, args.nn_chunk_size)
    image_scores = image_scores_from_patch_scores(test_patch_scores, topk_fraction)
    selected_scores = image_scores[args.score_name]
    normal_scores = selected_scores[test_labels == 0]
    defect_scores = selected_scores[test_labels == 1]
    curve = curve_rows(test_labels, selected_scores, args.curve_points)
    best = best_under_false_pass(curve, args.false_pass_target)

    patch_count = int(test_features.shape[1])
    feature_dim = int(test_features.shape[2])
    per_image_nn_ops = int(patch_count * int(len(bank)) * feature_dim)
    train_patch_flat = train_patch_scores.reshape(-1)
    defect_patch_flat = test_patch_scores[test_labels == 1].reshape(-1)
    train_q95 = float(np.quantile(train_patch_flat, 0.95)) if len(train_patch_flat) else float("nan")
    patch_collision = float((defect_patch_flat <= train_q95).mean()) if len(defect_patch_flat) else float("nan")

    return {
        "label": label,
        "category": category,
        "config": {
            "profile_name": profile["name"],
            "backbone": profile["backbone"],
            "out_indices": profile["out_indices"],
            "patch_grid": profile["patch_grid"],
            "bank_patches": int(len(bank)),
            "topk_fraction": topk_fraction,
        },
        "sample_counts": {
            "train_normal_images": int(len(train)),
            "test_images": int(len(test)),
            "test_good_images": int((test_labels == 0).sum()),
            "test_defect_images": int((test_labels == 1).sum()),
        },
        "footprint": {
            "patch_count": patch_count,
            "feature_dim": feature_dim,
            "bank_patches": int(len(bank)),
            "per_image_nn_ops": per_image_nn_ops,
            "bank_feature_values": int(len(bank) * feature_dim),
        },
        "coverage": {
            "normal_train_patch_to_bank_distance": summarize_distribution(train_patch_flat),
            "defect_patch_collision_under_train_q95": round_float(patch_collision),
        },
        "image_score_distribution": {
            "good": summarize_distribution(normal_scores),
            "defect": summarize_distribution(defect_scores),
        },
        "margin": score_margin(normal_scores, defect_scores, args.false_pass_target),
        "constraint_result": {
            "false_pass_target": args.false_pass_target,
            "good_pass_rate_good": best["good_pass_rate_good"],
            "good_loss_rate_good": best["good_loss_rate_good"],
            "threshold": best["threshold"],
        },
    }


def classify_reason(base: dict, reduced: dict, args: argparse.Namespace) -> str:
    best_good = reduced["constraint_result"]["good_pass_rate_good"]
    margin = reduced["margin"]["margin_vs_good_q95"]
    rel_ops = reduced["relative_to_baseline"]["per_image_nn_ops"]
    if best_good is None:
        return "threshold separation failed"
    if best_good >= args.strong_good_pass and margin is not None and margin >= 0:
        return "reducible: positive margin remains"
    if best_good >= args.strong_good_pass:
        return "reducible but margin is tight"
    if margin is not None and margin < 0:
        return "limit: normal/defect score overlap"
    if rel_ops is not None and rel_ops > 0.05:
        return "limit: still needs large profile"
    return "weak baseline or category-specific issue"


def build_rows(args: argparse.Namespace, source_summary: dict, materialized_root: Path, device: torch.device) -> list[dict]:
    selected_by_category = select_minimal_rows(source_summary, args.false_pass_target, args.tolerance)
    baseline_profile = parse_feature_profile("name=wrn_l23_g14,backbone=wide_resnet50_2,out=1:2,grid=14")
    rows = []
    for category in args.categories:
        selected_row = selected_by_category[category]
        reduced_profile = profile_from_row(selected_row)
        baseline = evaluate_config(
            args,
            category,
            "baseline",
            baseline_profile,
            args.baseline_bank_patches,
            args.baseline_topk_fraction,
            materialized_root,
            device,
        )
        reduced = evaluate_config(
            args,
            category,
            "reduced",
            reduced_profile,
            selected_row["bank_patches"],
            parse_topk(selected_row["selected_config"]),
            materialized_root,
            device,
        )

        relative = {}
        for key in ["patch_count", "feature_dim", "bank_patches", "per_image_nn_ops", "bank_feature_values"]:
            base_value = baseline["footprint"][key]
            reduced_value = reduced["footprint"][key]
            relative[key] = round_float(reduced_value / base_value) if base_value else None
        reduced["relative_to_baseline"] = relative
        baseline["relative_to_baseline"] = {
            "patch_count": 1.0,
            "feature_dim": 1.0,
            "bank_patches": 1.0,
            "per_image_nn_ops": 1.0,
            "bank_feature_values": 1.0,
        }

        rows.append(
            {
                "category": category,
                "source_selected_config": selected_row["selected_config"],
                "baseline": baseline,
                "reduced": reduced,
                "hypothesis_reason": classify_reason(baseline, reduced, args),
            }
        )
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PatchCore-lite reduction-limit probe",
        "",
        "## Purpose",
        "",
        "This experiment explains why a category-specific PatchCore-lite profile can or cannot be reduced.",
        "It supports the profile-selection hypothesis with measurable quantities instead of only reporting that a sweep happened to work.",
        "",
        "## Formulas used",
        "",
        "- Nearest-neighbor cost proxy: `C_NN = P * B * D`",
        "  - `P`: number of test patches per image",
        "  - `B`: number of memory-bank patches",
        "  - `D`: feature dimension per patch",
        "- Normal-bank coverage radius: `R_q = quantile_q min_{b in Bank} ||z_normal - b||_2`",
        "- Defect-safe threshold at false-pass target `alpha`: `tau_alpha = quantile_alpha(S_defect)`",
        "- Good pass predicted by the score distributions: `GP_alpha = Pr[S_good < tau_alpha]`",
        "- Margin for accepting 95% of good samples: `M_95 = tau_alpha - quantile_0.95(S_good)`",
        "  - `M_95 > 0`: enough score-space margin remains.",
        "  - `M_95 < 0`: normal and defect score distributions overlap under that constraint.",
        "",
        "## Category summary",
        "",
        "| category | selected profile | good pass | NN ops ratio | patch ratio | bank ratio | dim ratio | margin M95 | defect collision | interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["category_rows"]:
        reduced = row["reduced"]
        rel = reduced["relative_to_baseline"]
        lines.append(
            f"| {row['category']} | `{row['source_selected_config']}` | "
            f"{pct(reduced['constraint_result']['good_pass_rate_good'])} | "
            f"{rel['per_image_nn_ops']:.6f}x | {rel['patch_count']:.4f}x | "
            f"{rel['bank_patches']:.4f}x | {rel['feature_dim']:.4f}x | "
            f"{reduced['margin']['margin_vs_good_q95']} | "
            f"{pct(reduced['coverage']['defect_patch_collision_under_train_q95'])} | "
            f"{row['hypothesis_reason']} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- A category is a strong reduction candidate when the reduced profile keeps good pass high and `M_95` remains positive while `C_NN` is much smaller.",
        "- A category is near its reduction limit when `M_95` becomes negative: there is no threshold that both accepts most good images and rejects almost all defects under the reduced feature space.",
        "- If `C_NN` remains large even in the selected profile, the category likely needs either stronger features, finer grids, or a larger bank.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(payload: dict, path: Path) -> None:
    rows = payload["category_rows"]
    labels = [row["category"] for row in rows]
    good_pass = [100.0 * (row["reduced"]["constraint_result"]["good_pass_rate_good"] or 0.0) for row in rows]
    rel_ops = [100.0 * row["reduced"]["relative_to_baseline"]["per_image_nn_ops"] for row in rows]
    margins = [row["reduced"]["margin"]["margin_vs_good_q95"] or 0.0 for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].bar(x, good_pass, color="#2f7ed8")
    axes[0].set_ylabel("good pass [%]")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, rel_ops, color="#f28e2b")
    axes[1].set_ylabel("NN ops ratio [%]")
    axes[1].grid(True, axis="y", alpha=0.3)

    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in margins]
    axes[2].bar(x, margins, color=colors)
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_ylabel("M95 margin")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle("PatchCore-lite reduction-limit indicators")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, default=Path("results/mvtec_patchcore_backbone_floor_probe_001_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/mvtec_patchcore_reduction_limit_probe_001_summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/mvtec_patchcore_reduction_limit_probe_001.md"))
    parser.add_argument("--figure", type=Path, default=Path("results/mvtec_patchcore_reduction_limit_probe_001.png"))
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=ALL_MVTEC_CATEGORIES)
    parser.add_argument("--false-pass-target", type=float, default=0.03)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--strong-good-pass", type=float, default=0.80)
    parser.add_argument("--baseline-bank-patches", type=int, default=12000)
    parser.add_argument("--baseline-topk-fraction", type=float, default=0.01)
    parser.add_argument("--score-name", default="topk_score")
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--curve-points", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this reduction-limit probe.")

    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    category_rows = build_rows(args, source_summary, args.materialized_root, device)
    payload = {
        "purpose": "explain category-specific PatchCore-lite reduction limits with coverage, margin, collision, and cost-product formulas",
        "config": vars(args) | {"device": str(device)},
        "category_rows": category_rows,
        "outputs": {
            "markdown": str(args.markdown),
            "figure": str(args.figure),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.markdown)
    plot(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "markdown": str(args.markdown), "figure": str(args.figure), "categories": len(category_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
