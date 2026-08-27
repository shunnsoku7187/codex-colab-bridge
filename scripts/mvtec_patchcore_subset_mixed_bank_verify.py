"""Verify top AB/ABC PatchCore switching candidates with true mixed banks.

The subset screening script can compare costs from existing per-category runs,
but it cannot know the accuracy of a standard profile with a merged A+B memory
bank.  This script recomputes that mixed-bank baseline for selected subsets and
joins it with the already screened bank/profile switching systems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

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
from scripts.train_kolektor_strong_final import round_float, set_seed


DEFAULT_CANDIDATES = Path("results/mvtec_patchcore_subset_switch_search_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_subset_mixed_bank_verify_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_subset_mixed_bank_verify_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_subset_mixed_bank_verify_001.png")


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def pick_subsets(payload: dict, top_pairs: int, top_triples: int) -> list[list[str]]:
    subsets: list[list[str]] = []
    for item in payload.get("top_pairs", [])[:top_pairs]:
        subsets.append(item["subset"])
    for item in payload.get("top_triples", [])[:top_triples]:
        subsets.append(item["subset"])
    seen = set()
    unique = []
    for subset in subsets:
        key = tuple(subset)
        if key not in seen:
            seen.add(key)
            unique.append(subset)
    return unique


def candidate_by_subset(payload: dict) -> dict[tuple[str, ...], dict]:
    out = {}
    for item in payload.get("all_results", []):
        out[tuple(item["subset"])] = item
    return out


def best_row_for_target(system: dict, category: str, target: float) -> dict | None:
    for row in system["category_rows"]:
        if row["category"] == category:
            return row
    return None


def evaluate_mixed_standard(
    subset: list[str],
    args: argparse.Namespace,
    materialized_root: Path,
    device: torch.device,
) -> dict:
    model = make_backbone("wide_resnet50_2", (1, 2), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (14, 14)
    train_features_by_cat = {}
    test_features_by_cat = {}
    test_labels_by_cat = {}
    sample_counts = {}
    banks = []
    for category in subset:
        samples = find_materialized_samples(materialized_root, category)
        train, test = normal_train_and_test(samples)
        train_features, _ = collect_features(
            model,
            train,
            image_size,
            args.batch_size,
            patch_grid,
            device,
            f"mixed standard {category} train",
        )
        test_features, test_labels = collect_features(
            model,
            test,
            image_size,
            args.batch_size,
            patch_grid,
            device,
            f"mixed standard {category} test",
        )
        train_labels = np.zeros(len(train_features), dtype=np.int64)
        bank = sample_normal_patch_bank(train_features, train_labels, args.bank_patches_per_category, args.seed)
        banks.append(bank)
        train_features_by_cat[category] = train_features
        test_features_by_cat[category] = test_features
        test_labels_by_cat[category] = test_labels
        sample_counts[category] = {
            "train_normal": int(len(train)),
            "test": int(len(test)),
            "test_good": int((test_labels == 0).sum()),
            "test_defect": int((test_labels == 1).sum()),
            "bank_patches": int(len(bank)),
        }
    merged_bank = np.concatenate(banks, axis=0)
    category_rows = []
    goods = []
    patch_count = int(next(iter(test_features_by_cat.values())).shape[1])
    feature_dim = int(next(iter(test_features_by_cat.values())).shape[2])
    denom = patch_count * feature_dim * int(len(merged_bank))
    for category in subset:
        patch_scores = patchcore_scores(test_features_by_cat[category], merged_bank, args.nn_chunk_size)
        image_scores = image_scores_from_patch_scores(patch_scores, args.topk_fraction)
        scores = image_scores[args.score_name]
        rows = curve_rows(test_labels_by_cat[category], scores, args.curve_points)
        best = best_under_false_pass(rows, args.false_pass_target)
        good = float(best["good_pass_rate_good"])
        goods.append(good)
        category_rows.append(
            {
                "category": category,
                "good_pass": round_float(good),
                "threshold": round_float(best["threshold"]),
                "false_pass_target": args.false_pass_target,
                "patch_grid": 14,
                "patch_count": patch_count,
                "feature_dim": feature_dim,
                "merged_bank_patches": int(len(merged_bank)),
                "relative_nn_ops_to_subset_standard": 1.0,
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": "subset_standard_true_merged_bank",
        "subset": subset,
        "bank_patches_per_category": args.bank_patches_per_category,
        "merged_bank_patches": int(len(merged_bank)),
        "patch_count": patch_count,
        "feature_dim": feature_dim,
        "denominator_per_image_nn_ops": int(denom),
        "mean_good_pass": round_float(mean(goods)),
        "min_good_pass": round_float(min(goods)),
        "category_rows": category_rows,
        "sample_counts": sample_counts,
    }


def compact_system(system: dict) -> dict:
    return {
        "name": system["name"],
        "mean_good_pass": system["mean_good_pass"],
        "min_good_pass": system["min_good_pass"],
        "mean_relative_nn_ops_to_subset_standard": system["mean_relative_nn_ops_to_subset_standard"],
        "max_relative_nn_ops_to_subset_standard": system["max_relative_nn_ops_to_subset_standard"],
        "mean_relative_bank_to_subset_standard": system["mean_relative_bank_to_subset_standard"],
        "category_rows": system["category_rows"],
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# AB/ABC混合bank標準構成の実測検証",
        "",
        "## 目的",
        "",
        "候補探索で未実測だった `subset_standard_merged_bank` を実際に構築し，提案切替と同じ対象カテゴリ集合で比較する。",
        "",
        "| subset | 標準混合bank 平均良品通過 | 標準混合bank 最低良品通過 | 提案 平均良品通過 | 提案 最低良品通過 | 提案NN | vs標準NN削減 | vs bank-only追加削減 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["verified_subsets"]:
        subset = " + ".join(item["subset"])
        mixed = item["true_mixed_standard"]
        proposed = item["proposed_profile_and_bank_switch"]
        claim = item["screening_claim_metrics"]
        lines.append(
            f"| {subset} | {pct(mixed['mean_good_pass'])} | {pct(mixed['min_good_pass'])} | "
            f"{pct(proposed['mean_good_pass'])} | {pct(proposed['min_good_pass'])} | "
            f"{proposed['mean_relative_nn_ops_to_subset_standard']:.6f}x | "
            f"{pct(claim['proposed_vs_subset_standard_nn_reduction'])} | "
            f"{pct(claim['proposed_vs_bank_only_nn_reduction'])} |"
        )
    lines += [
        "",
        "## 読み取り",
        "",
        "- ここでの標準構成は，対象カテゴリ集合だけの正常bankを結合したフェアな比較対象である。",
        "- 提案のNN計算量は，その集合の標準混合bank構成を1.0倍とした相対値である。",
        "- この表で，標準混合bankに対する計算量削減と，片方profile固定に対する品質差を同時に確認する。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["+".join(item["subset"]) for item in payload["verified_subsets"]]
    nn = [item["proposed_profile_and_bank_switch"]["mean_relative_nn_ops_to_subset_standard"] for item in payload["verified_subsets"]]
    std_good = [item["true_mixed_standard"]["min_good_pass"] for item in payload["verified_subsets"]]
    prop_good = [item["proposed_profile_and_bank_switch"]["min_good_pass"] for item in payload["verified_subsets"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 1.2), 7), constrained_layout=True)
    axes[0].bar(x, nn, color="#f28e2b")
    axes[0].set_ylabel("Relative NN ops")
    axes[0].set_title("Proposed cost relative to true subset standard")
    axes[0].set_ylim(0, max(0.02, max(nn) * 1.25))
    axes[1].plot(x, std_good, marker="o", label="standard merged bank", color="#4e79a7")
    axes[1].plot(x, prop_good, marker="o", label="proposed switch", color="#59a14f")
    axes[1].set_ylabel("Min good-pass")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--materialized-root", type=Path, default=Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2"))
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--top-triples", type=int, default=5)
    parser.add_argument("--false-pass-target", type=float, default=0.03)
    parser.add_argument("--bank-patches-per-category", type=int, default=12000)
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--score-name", default="topk_score")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nn-chunk-size", type=int, default=16384)
    parser.add_argument("--curve-points", type=int, default=180)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but not available.")

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    by_subset = candidate_by_subset(candidates)
    subsets = pick_subsets(candidates, args.top_pairs, args.top_triples)
    verified = []
    for subset in subsets:
        key = tuple(subset)
        candidate = by_subset[key]
        true_mixed = evaluate_mixed_standard(subset, args, args.materialized_root, device)
        proposed = next(system for system in candidate["systems"] if system["name"] == "proposed_profile_and_bank_switch")
        bank_only = next(system for system in candidate["systems"] if system["name"] == "bank_only_switch")
        fixed = [system for system in candidate["systems"] if system["name"].startswith("profile_fixed_to_")]
        verified.append(
            {
                "subset": subset,
                "true_mixed_standard": true_mixed,
                "bank_only_switch": compact_system(bank_only),
                "fixed_profile_systems": [compact_system(system) for system in fixed],
                "proposed_profile_and_bank_switch": compact_system(proposed),
                "screening_claim_metrics": candidate["claim_metrics"],
            }
        )
    payload = {
        "purpose": "verify fair AB/ABC comparison candidates with true merged standard banks",
        "config": vars(args) | {"device": str(device)},
        "verified_subsets": verified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "verified": len(verified), "device": str(device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
