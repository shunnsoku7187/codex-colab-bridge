"""Find accuracy-preserving PatchCore bank sizes and test additivity.

The research claim should not depend on a hand-picked bank size.  For each
profile, this script asks the operational question directly:

* What is the smallest k-center memory bank that preserves full-bank accuracy?
* For A+B/ABC, is the required merged bank close to N(A)+N(B)+... or can one
  merged bank share representatives across categories?
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

from scripts.mvtec_ad_parquet_anomaly_probe import curve_rows, image_scores_from_patch_scores, score_auc
from scripts.mvtec_patchcore_fixed_coreset_profile_switch import (
    FeatureCache,
    baseline_rows,
    kcenter_bank,
    patchcore_scores_gpu,
    reservoir_rows,
    row_config,
    rows_by_category,
    selected_minimal,
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass
from scripts.train_kolektor_strong_final import round_float, set_seed


DEFAULT_SOURCE = Path("results/mvtec_patchcore_backbone_floor_probe_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_min_bank_additivity_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_min_bank_additivity_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_min_bank_additivity_001.png")


def parse_sizes(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def make_profile_id(cfg: dict) -> str:
    outs = ":".join(str(i) for i in cfg["out_indices"])
    return f"{cfg['backbone']}_out{outs}_g{cfg['patch_grid']}_topk{cfg['topk_fraction']}"


def full_reference_size(train_features: np.ndarray, args: argparse.Namespace) -> int:
    flat_count = int(np.prod(train_features.shape[:2]))
    return min(args.max_bank_patches, flat_count, args.coreset_candidate_pool)


def build_bank(train_features_by_cat: dict[str, np.ndarray], categories: list[str], size: int, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    pools = [reservoir_rows(train_features_by_cat[category], args.coreset_candidate_pool, args.seed) for category in categories]
    merged = np.concatenate(pools, axis=0)
    return kcenter_bank(merged, 1.0, size, size, args.seed, device, args.distance_batch_size)


def evaluate_category_with_bank(
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
    return {
        "category": category,
        "good_pass": round_float(best["good_pass_rate_good"]),
        "good_loss": round_float(best["good_loss_rate_good"]),
        "threshold": round_float(best["threshold"]),
        "auc": score_auc(test_labels, scores),
        "sample_counts": meta,
    }


def meets_reference(candidate: dict, reference: dict, args: argparse.Namespace) -> bool:
    if candidate["good_pass"] is None or reference["good_pass"] is None:
        return False
    return float(candidate["good_pass"]) + 1e-12 >= float(reference["good_pass"]) - args.allowed_good_pass_drop


def evaluate_single_minimum(
    category: str,
    cfg: dict,
    cache: FeatureCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    train_features, _test_features, _test_labels, _train_labels, _meta = cache.get(category, cfg)
    ref_size = full_reference_size(train_features, args)
    reference_bank = build_bank({category: train_features}, [category], ref_size, args, device)
    reference = evaluate_category_with_bank(category, cfg, reference_bank, cache, args, device)
    rows = []
    required = None
    for size in args.bank_sizes:
        if size > ref_size:
            continue
        bank = build_bank({category: train_features}, [category], size, args, device)
        result = evaluate_category_with_bank(category, cfg, bank, cache, args, device)
        result["bank_size"] = int(size)
        result["relative_to_reference_bank"] = round_float(size / ref_size)
        rows.append(result)
        if required is None and meets_reference(result, reference, args):
            required = int(size)
    return {
        "category": category,
        "profile_id": make_profile_id(cfg),
        "profile": {
            "config": cfg["config"],
            "backbone": cfg["backbone"],
            "out_indices": list(cfg["out_indices"]),
            "patch_grid": cfg["patch_grid"],
            "feature_dim": cfg["feature_dim"],
            "topk_fraction": cfg["topk_fraction"],
        },
        "reference_bank_size": int(ref_size),
        "reference": reference,
        "required_bank_size": required,
        "rows": rows,
    }


def evaluate_merged_minimum(
    subset: tuple[str, ...],
    cfg: dict,
    single_lookup: dict[tuple[str, str], dict],
    cache: FeatureCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    train_features_by_cat = {}
    for category in subset:
        train_features, _test_features, _test_labels, _train_labels, _meta = cache.get(category, cfg)
        train_features_by_cat[category] = train_features
    ref_size = min(args.max_bank_patches, args.coreset_candidate_pool * len(subset))
    reference_bank = build_bank(train_features_by_cat, list(subset), ref_size, args, device)
    reference_rows = [evaluate_category_with_bank(category, cfg, reference_bank, cache, args, device) for category in subset]
    reference_min_good = min(row["good_pass"] for row in reference_rows if row["good_pass"] is not None)
    rows = []
    required = None
    for size in args.bank_sizes:
        if size > ref_size:
            continue
        bank = build_bank(train_features_by_cat, list(subset), size, args, device)
        category_rows = [evaluate_category_with_bank(category, cfg, bank, cache, args, device) for category in subset]
        goods = [row["good_pass"] for row in category_rows if row["good_pass"] is not None]
        row = {
            "bank_size": int(size),
            "mean_good_pass": round_float(mean(goods)) if goods else None,
            "min_good_pass": round_float(min(goods)) if goods else None,
            "category_rows": category_rows,
        }
        rows.append(row)
        if required is None and row["min_good_pass"] is not None and row["min_good_pass"] + 1e-12 >= reference_min_good - args.allowed_good_pass_drop:
            required = int(size)
    profile_id = make_profile_id(cfg)
    single_sum = 0
    single_required_by_category = {}
    for category in subset:
        single = single_lookup.get((profile_id, category))
        if single is None:
            single = evaluate_single_minimum(category, cfg, cache, args, device)
            single_lookup[(profile_id, category)] = single
        if single["required_bank_size"] is not None:
            single_sum += int(single["required_bank_size"])
            single_required_by_category[category] = int(single["required_bank_size"])
        else:
            single_required_by_category[category] = None
            single_sum = 0
    return {
        "subset": list(subset),
        "profile_id": profile_id,
        "reference_bank_size": int(ref_size),
        "reference_min_good": round_float(reference_min_good),
        "reference_category_rows": reference_rows,
        "required_merged_bank_size": required,
        "single_required_sum": single_sum or None,
        "single_required_by_category": single_required_by_category,
        "merged_over_single_sum": round_float(required / single_sum) if required is not None and single_sum else None,
        "rows": rows,
    }


def profile_rows(summary: dict, target: float, tolerance: float) -> tuple[dict[str, dict], dict[str, dict], dict[str, list[dict]]]:
    by_category = rows_by_category(summary["variant_rows"])
    selected = selected_minimal(summary, target, tolerance)
    standard = baseline_rows(summary, target)
    return selected, standard, by_category


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# 精度維持最小bank数とカテゴリ加法性",
        "",
        "## 目的",
        "",
        "bank数を固定比率で決めるのではなく，full bank時の良品通過率を保てる最小bank数を求める。さらに同一profileで `N(A+B)` が `N(A)+N(B)` に近いかを確認し，bank切替の意義を評価する。",
        "",
        "## 単独カテゴリ",
        "",
        "| profile | category | reference bank | reference good-pass | required bank | required/reference |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["single_results"]:
        rel = None if row["required_bank_size"] is None else row["required_bank_size"] / row["reference_bank_size"]
        lines.append(
            f"| {row['profile_group']} | {row['category']} | {row['reference_bank_size']} | "
            f"{pct(row['reference']['good_pass'])} | {row['required_bank_size']} | {pct(rel)} |"
        )
    lines += [
        "",
        "## 混合カテゴリでの加法性",
        "",
        "| profile | subset | N(A)+N(B/...) | N(A+B/...) | merged/sum | reference min good-pass |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["merged_results"]:
        lines.append(
            f"| {row['profile_group']} | {' + '.join(row['subset'])} | {row['single_required_sum']} | "
            f"{row['required_merged_bank_size']} | {row['merged_over_single_sum']} | {pct(row['reference_min_good'])} |"
        )
    lines += [
        "",
        "## 読み取り",
        "",
        "- `merged/sum` が1に近い場合，混合bankはカテゴリごとの必要bank数をほぼ足し合わせる必要があり，カテゴリ別bank切替の意義が強い。",
        "- `merged/sum` が大きく1を下回る場合，カテゴリ間で代表点を共有でき，bank切替だけの寄与は弱くなる。",
        "- 提案の中心は，bank削減ではなく，この最小bank数を含むprofileをカテゴリごとに切り替えることである。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    rows = payload["merged_results"][:12]
    labels = [f"{row['profile_group']}:{'+'.join(row['subset'])}" for row in rows]
    ratios = [row["merged_over_single_sum"] or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.72), 4.8))
    x = np.arange(len(labels))
    ax.bar(x, ratios, color="#4e79a7")
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="additive")
    ax.set_ylabel("N(merged) / sum N(single)")
    ax.set_title("PatchCore bank additivity under accuracy-preserving minimum")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
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
    parser.add_argument("--false-pass-target", type=float, default=0.05)
    parser.add_argument("--allowed-good-pass-drop", type=float, default=0.0)
    parser.add_argument("--selected-profile-tolerance", type=float, default=0.05)
    parser.add_argument("--bank-sizes", default="25,50,100,125,250,500,750,1000,1500,2000,3000,4000,6000,9000,12000")
    parser.add_argument("--categories", default="")
    parser.add_argument("--pair-limit", type=int, default=20)
    parser.add_argument("--triple-limit", type=int, default=20)
    parser.add_argument("--coreset-candidate-pool", type=int, default=12000)
    parser.add_argument("--max-bank-patches", type=int, default=12000)
    parser.add_argument("--score-name", default="topk_score", choices=["topk_score", "max_score"])
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nn-chunk-size", type=int, default=8192)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    args.bank_sizes = parse_sizes(args.bank_sizes)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this job.")
    summary = json.loads(args.source.read_text(encoding="utf-8"))
    selected, standard, _by_category = profile_rows(summary, args.false_pass_target, args.selected_profile_tolerance)
    categories = [cat for cat in (args.categories.split(",") if args.categories else summary["config"]["categories"]) if cat in selected and cat in standard]
    cache = FeatureCache(args.materialized_root, args, device)

    single_results = []
    single_lookup = {}
    for profile_group, source_rows in [("standard", standard), ("selected", selected)]:
        for category in categories:
            cfg = row_config(source_rows[category])
            result = evaluate_single_minimum(category, cfg, cache, args, device)
            result["profile_group"] = profile_group
            single_results.append(result)
            single_lookup[(result["profile_id"], category)] = result

    subsets = []
    pair_count = 0
    triple_count = 0
    for i, a in enumerate(categories):
        for b in categories[i + 1 :]:
            if pair_count < args.pair_limit:
                subsets.append((a, b))
                pair_count += 1
            for c in categories[categories.index(b) + 1 :]:
                if triple_count < args.triple_limit:
                    subsets.append((a, b, c))
                    triple_count += 1
        if pair_count >= args.pair_limit and triple_count >= args.triple_limit:
            break

    merged_results = []
    for subset in subsets:
        cfgs = [("standard", row_config(standard[subset[0]]))]
        for category in subset:
            cfgs.append((f"selected:{category}", row_config(selected[category])))
        seen = set()
        for profile_group, cfg in cfgs:
            key = (profile_group, make_profile_id(cfg), subset)
            if key in seen:
                continue
            seen.add(key)
            try:
                result = evaluate_merged_minimum(subset, cfg, single_lookup, cache, args, device)
            except Exception as exc:
                result = {"subset": list(subset), "profile_group": profile_group, "profile_id": make_profile_id(cfg), "error": str(exc)}
            result["profile_group"] = profile_group
            merged_results.append(result)

    payload = {
        "purpose": "Find accuracy-preserving minimal PatchCore bank sizes and compare merged-bank additivity.",
        "config": vars(args) | {"device": str(device), "categories": categories},
        "single_results": single_results,
        "merged_results": merged_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "singles": len(single_results), "merged": len(merged_results), "device": str(device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
