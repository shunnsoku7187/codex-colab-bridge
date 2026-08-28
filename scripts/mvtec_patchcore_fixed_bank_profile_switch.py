"""Compare PatchCore profile switching with a fixed backbone and fixed bank count.

This experiment removes the easiest objection to category-profile switching:
"maybe it only wins because each category was allowed to use a smaller memory
bank."  The bank count is fixed as follows.

For an AB task:
* common bank system stores and searches 2K bank vectors: {A+B}
* switched-bank system stores K for A and K for B: {A}+{B}

The total stored bank count is therefore equal.  The remaining variables are
feature layer, patch grid, top-k aggregation, threshold, and whether the bank is
searched as a common pool or the category-specific pool.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.mvtec_ad_parquet_anomaly_probe import curve_rows, image_scores_from_patch_scores, score_auc
from scripts.mvtec_patchcore_fixed_coreset_profile_switch import (
    FeatureCache,
    baseline_rows,
    kcenter_bank,
    patchcore_scores_gpu,
    reservoir_rows,
    row_config,
    rows_by_category,
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass
from scripts.train_kolektor_strong_final import round_float, set_seed


DEFAULT_SOURCE = Path("results/mvtec_patchcore_backbone_floor_probe_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_fixed_bank_profile_switch_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_fixed_bank_profile_switch_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_fixed_bank_profile_switch_001.png")


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def best_for_target(row: dict, target: float) -> dict | None:
    for best in row.get("best_rows", []):
        if abs(float(best["target"]) - target) < 1e-12:
            return best
    return None


def row_good(row: dict, target: float) -> float | None:
    best = best_for_target(row, target)
    if not best:
        return None
    value = best.get("good_pass_rate_good")
    return None if value is None else float(value)


def bank_from_config(config: str) -> int | None:
    match = re.search(r"_b(\d+)_", config)
    return int(match.group(1)) if match else None


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def make_standard_cfg() -> dict:
    return {
        "config": "standard_wrn_l23_g14_topk0p01",
        "backbone": "wide_resnet50_2",
        "out_indices": (1, 2),
        "patch_grid": 14,
        "feature_dim": 768,
        "topk_fraction": 0.01,
    }


def choose_category_profile(
    category: str,
    by_category: dict[str, list[dict]],
    baseline_good: float,
    args: argparse.Namespace,
) -> dict:
    candidates = []
    for row in by_category[category]:
        if row.get("backbone") != args.backbone:
            continue
        if bank_from_config(str(row.get("config", ""))) != args.profile_selection_bank:
            continue
        good = row_good(row, args.false_pass_target)
        if good is None:
            continue
        if good + 1e-12 < baseline_good - args.allowed_good_pass_drop:
            continue
        cfg = row_config(row)
        footprint = row["footprint"]
        ops = int(footprint["patch_count"] * footprint["feature_dim"])
        candidates.append((ops, -good, cfg))
    if not candidates:
        return make_standard_cfg()
    return min(candidates, key=lambda item: item[:2])[2]


def build_bank(
    category_features: dict[str, np.ndarray],
    categories: list[str],
    bank_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    pools = []
    per_category_pool = max(args.coreset_candidate_pool, bank_size)
    for category in categories:
        pools.append(reservoir_rows(category_features[category], per_category_pool, args.seed))
    merged = np.concatenate(pools, axis=0)
    return kcenter_bank(merged, 1.0, bank_size, bank_size, args.seed, device, args.distance_batch_size)


def evaluate_category(
    category: str,
    cfg: dict,
    bank: np.ndarray,
    cache: FeatureCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    _train_features, test_features, test_labels, _train_labels, meta = cache.get(category, cfg)
    patch_scores = patchcore_scores_gpu(test_features, bank, args.nn_chunk_size, device)
    scores = image_scores_from_patch_scores(patch_scores, cfg["topk_fraction"])[args.score_name]
    rows = curve_rows(test_labels, scores, args.curve_points)
    best = best_under_false_pass(rows, args.false_pass_target)
    patch_count = int(test_features.shape[1])
    feature_dim = int(test_features.shape[2])
    return {
        "category": category,
        "config": cfg["config"],
        "backbone": cfg["backbone"],
        "out_indices": list(cfg["out_indices"]),
        "patch_grid": cfg["patch_grid"],
        "patch_count": patch_count,
        "feature_dim": feature_dim,
        "topk_fraction": cfg["topk_fraction"],
        "bank_patches_searched": int(len(bank)),
        "nn_ops_per_image": int(patch_count * feature_dim * len(bank)),
        "auc": score_auc(test_labels, scores),
        "good_pass": round_float(best["good_pass_rate_good"]),
        "good_loss": round_float(best["good_loss_rate_good"]),
        "threshold": round_float(best["threshold"]),
        "false_pass_target": args.false_pass_target,
        "sample_counts": meta,
    }


def summarize_system(name: str, rows: list[dict], baseline_ops: list[int], total_stored_bank: int) -> dict:
    goods = [row["good_pass"] for row in rows if row["good_pass"] is not None]
    ops = [row["nn_ops_per_image"] for row in rows]
    rel = [op / base for op, base in zip(ops, baseline_ops)]
    return {
        "name": name,
        "mean_good_pass": round_float(mean(goods)) if goods else None,
        "min_good_pass": round_float(min(goods)) if goods else None,
        "mean_nn_ops_per_image": round_float(mean(ops)) if ops else None,
        "mean_relative_nn_ops_to_common_standard": round_float(mean(rel)) if rel else None,
        "median_relative_nn_ops_to_common_standard": round_float(median(rel)) if rel else None,
        "max_relative_nn_ops_to_common_standard": round_float(max(rel)) if rel else None,
        "total_stored_bank_patches": int(total_stored_bank),
        "category_rows": rows,
    }


def feature_values(cfg: dict, bank_patches: int) -> int:
    return int(cfg["patch_grid"] * cfg["patch_grid"] * cfg["feature_dim"] * bank_patches)


def evaluate_subset(
    subset: tuple[str, ...],
    selected_profiles: dict[str, dict],
    standard_cfg: dict,
    cache: FeatureCache,
    args: argparse.Namespace,
    device: torch.device,
    bank_per_category: int,
) -> dict:
    subset_list = list(subset)
    total_bank = bank_per_category * len(subset_list)

    # Common standard profile + common merged bank.
    std_train = {category: cache.get(category, standard_cfg)[0] for category in subset_list}
    standard_common_bank = build_bank(std_train, subset_list, total_bank, args, device)
    standard_common_rows = [
        evaluate_category(category, standard_cfg, standard_common_bank, cache, args, device)
        for category in subset_list
    ]
    baseline_ops = [row["nn_ops_per_image"] for row in standard_common_rows]
    systems = [summarize_system("common standard profile + common bank", standard_common_rows, baseline_ops, total_bank)]

    # Common standard profile + category bank switching.
    standard_switch_rows = []
    for category in subset_list:
        bank = build_bank(std_train, [category], bank_per_category, args, device)
        standard_switch_rows.append(evaluate_category(category, standard_cfg, bank, cache, args, device))
    systems.append(summarize_system("common standard profile + category bank switch", standard_switch_rows, baseline_ops, total_bank))

    # One category's profile is fixed and reused for every category.
    for owner in subset_list:
        fixed_cfg = selected_profiles[owner]
        fixed_train = {category: cache.get(category, fixed_cfg)[0] for category in subset_list}
        fixed_rows = []
        for category in subset_list:
            bank = build_bank(fixed_train, [category], bank_per_category, args, device)
            fixed_rows.append(evaluate_category(category, fixed_cfg, bank, cache, args, device))
        systems.append(summarize_system(f"fixed profile={owner} + category bank switch", fixed_rows, baseline_ops, total_bank))

    # Proposed category profile switching.
    proposed_rows = []
    stored_feature_values = 0
    for category in subset_list:
        cfg = selected_profiles[category]
        train_features = cache.get(category, cfg)[0]
        bank = build_bank({category: train_features}, [category], bank_per_category, args, device)
        proposed_rows.append(evaluate_category(category, cfg, bank, cache, args, device))
        stored_feature_values += feature_values(cfg, bank_per_category)
    proposed = summarize_system("proposed profile switch + category bank switch", proposed_rows, baseline_ops, total_bank)
    proposed["total_stored_feature_values"] = int(stored_feature_values)
    systems.append(proposed)

    bank_only = systems[1]
    best_fixed = max(systems[2:-1], key=lambda item: item["min_good_pass"] or -1.0)
    return {
        "subset": subset_list,
        "bank_per_category": int(bank_per_category),
        "total_stored_bank_patches": int(total_bank),
        "systems": systems,
        "claim_metrics": {
            "proposed_vs_common_ops_reduction": round_float(1.0 - systems[-1]["mean_relative_nn_ops_to_common_standard"]),
            "proposed_vs_bank_only_ops_reduction": round_float(
                1.0 - systems[-1]["mean_relative_nn_ops_to_common_standard"] / bank_only["mean_relative_nn_ops_to_common_standard"]
            ),
            "proposed_min_good_minus_bank_only": round_float((systems[-1]["min_good_pass"] or 0.0) - (bank_only["min_good_pass"] or 0.0)),
            "proposed_min_good_minus_best_fixed_profile": round_float((systems[-1]["min_good_pass"] or 0.0) - (best_fixed["min_good_pass"] or 0.0)),
        },
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# 固定bank数・固定backboneでのprofile切替比較",
        "",
        "## 目的",
        "",
        "bank数を最適化変数から外し，総bank数を揃えた状態で，特徴層・grid・top-k・閾値のカテゴリ別切替だけに効果が残るかを確認する。",
        "",
        "## 条件",
        "",
        f"- backbone: `{payload['config']['backbone']}` 固定",
        f"- 欠陥誤通過率上限: {pct(payload['config']['false_pass_target'])}",
        f"- profile選択時のbank数: {payload['config']['profile_selection_bank']}",
        "- ABでは `{A+B}` と `{A}+{B}` の総bank数を同じにする。",
        "- ABCでは `{A+B+C}` と `{A}+{B}+{C}` の総bank数を同じにする。",
        "",
    ]
    for bank_size, rows in payload["results_by_bank_per_category"].items():
        lines += [
            f"## bank/category = {bank_size}",
            "",
            "| subset | system | min good-pass | mean good-pass | mean NN ops | relative ops | total bank |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for item in rows[: payload["config"]["markdown_top_rows"]]:
            for system in item["systems"]:
                lines.append(
                    f"| {' + '.join(item['subset'])} | {system['name']} | {pct(system['min_good_pass'])} | "
                    f"{pct(system['mean_good_pass'])} | {system['mean_nn_ops_per_image']} | "
                    f"{system['mean_relative_nn_ops_to_common_standard']:.6f}x | {system['total_stored_bank_patches']} |"
                )
            metric = item["claim_metrics"]
            lines.append(
                f"| {' + '.join(item['subset'])} | difference |  |  | "
                f"vs common削減 {pct(metric['proposed_vs_common_ops_reduction'])} / "
                f"vs bank-only追加削減 {pct(metric['proposed_vs_bank_only_ops_reduction'])} | "
                f"min good差 vs bank-only {metric['proposed_min_good_minus_bank_only']:+.4f} |  |"
            )
    lines += [
        "",
        "## 読み取り方",
        "",
        "- `common standard profile + common bank` は，対象カテゴリ全体のbankを毎回探索する基準方式。",
        "- `common standard profile + category bank switch` は，総保存bank数は同じだが，検品対象カテゴリのbankだけを探索する方式。",
        "- `fixed profile=X` は，X向けprofileを他カテゴリにも流用した場合の劣化を確認する対照実験。",
        "- `proposed` が `bank-only` よりさらに軽ければ，bank数ではなくprofile切替自体に効果がある。",
        "- `proposed` が `fixed profile=X` より良品通過率で勝てば，単一軽量profileの使い回しではなくカテゴリ別profileが必要である。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    bank_size = sorted(payload["results_by_bank_per_category"], key=lambda x: int(x))[0]
    rows = payload["results_by_bank_per_category"][bank_size][:10]
    labels = ["+".join(row["subset"]) for row in rows]
    bank_only_ops = []
    proposed_ops = []
    bank_only_good = []
    proposed_good = []
    for row in rows:
        systems = row["systems"]
        bank_only_ops.append(systems[1]["mean_relative_nn_ops_to_common_standard"])
        proposed_ops.append(systems[-1]["mean_relative_nn_ops_to_common_standard"])
        bank_only_good.append(systems[1]["min_good_pass"])
        proposed_good.append(systems[-1]["min_good_pass"])

    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 0.65), 7.2), sharex=True)
    width = 0.36
    axes[0].bar(x - width / 2, bank_only_ops, width, label="bank switch only", color="#4e79a7")
    axes[0].bar(x + width / 2, proposed_ops, width, label="profile + bank switch", color="#f28e2b")
    axes[0].set_ylabel("relative NN ops")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(x - width / 2, bank_only_good, width, label="bank switch only", color="#4e79a7")
    axes[1].bar(x + width / 2, proposed_good, width, label="profile + bank switch", color="#f28e2b")
    axes[1].set_ylabel("minimum good-pass")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Fixed backbone and fixed total bank count: K={bank_size} per category")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--profile-selection-bank", type=int, default=12000)
    parser.add_argument("--bank-per-category", default="500,1500,3000")
    parser.add_argument("--false-pass-target", type=float, default=0.03)
    parser.add_argument("--allowed-good-pass-drop", type=float, default=0.02)
    parser.add_argument("--pair-limit", type=int, default=105)
    parser.add_argument("--triple-limit", type=int, default=120)
    parser.add_argument("--coreset-candidate-pool", type=int, default=12000)
    parser.add_argument("--score-name", default="topk_score", choices=["topk_score", "max_score"])
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nn-chunk-size", type=int, default=8192)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--markdown-top-rows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    args.bank_per_category = parse_ints(args.bank_per_category)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this job.")
    summary = json.loads(args.source.read_text(encoding="utf-8"))
    by_category = rows_by_category(summary["variant_rows"])
    standard = baseline_rows(summary, args.false_pass_target)
    categories = [category for category in summary["config"]["categories"] if category in standard]
    baseline_good = {category: row_good(standard[category], args.false_pass_target) or 0.0 for category in categories}
    selected_profiles = {
        category: choose_category_profile(category, by_category, baseline_good[category], args)
        for category in categories
    }
    standard_cfg = make_standard_cfg()
    cache = FeatureCache(args.materialized_root, args, device)

    subsets = []
    pairs = list(itertools.combinations(categories, 2))[: args.pair_limit]
    triples = list(itertools.combinations(categories, 3))[: args.triple_limit]
    subsets.extend(pairs)
    subsets.extend(triples)

    results_by_bank: dict[str, list[dict]] = {}
    for bank_size in args.bank_per_category:
        rows = []
        for subset in subsets:
            rows.append(evaluate_subset(subset, selected_profiles, standard_cfg, cache, args, device, bank_size))
        rows.sort(
            key=lambda item: (
                item["claim_metrics"]["proposed_min_good_minus_bank_only"],
                item["claim_metrics"]["proposed_vs_bank_only_ops_reduction"],
            ),
            reverse=True,
        )
        results_by_bank[str(bank_size)] = rows

    payload = {
        "purpose": "Fixed-backbone fixed-total-bank comparison for category-wise PatchCore profile switching.",
        "config": vars(args) | {"device": str(device), "categories": categories},
        "selected_profiles": selected_profiles,
        "standard_profile": standard_cfg,
        "results_by_bank_per_category": results_by_bank,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "device": str(device), "subsets": len(subsets)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
