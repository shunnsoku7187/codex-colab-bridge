"""Coreset coverage audit for PatchCore memory-bank reduction.

The goal is to avoid claiming that a tiny bank works only because of a lucky
random sample.  This script uses farthest-first k-center coreset selection and
asks how many representative normal patch features are needed to cover normal
patch distributions for A, B, and A+B/ABC subsets.
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

from scripts.mvtec_ad_parquet_anomaly_probe import collect_features, find_materialized_samples, make_backbone
from scripts.mvtec_patchcore_lightweight_sweep import normal_train_and_test
from scripts.train_kolektor_strong_final import round_float, set_seed


DEFAULT_CANDIDATES = Path("results/mvtec_patchcore_subset_mixed_bank_verify_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_bank_coverage_coreset_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_bank_coverage_coreset_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_bank_coverage_coreset_001.png")


def parse_sizes(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def subset_key(subset: list[str]) -> str:
    return "+".join(subset)


def select_subsets(payload: dict, top_pairs: int, top_triples: int) -> list[list[str]]:
    subsets = []
    for item in payload.get("verified_subsets", []):
        if len(item["subset"]) == 2 and len([s for s in subsets if len(s) == 2]) < top_pairs:
            subsets.append(item["subset"])
        if len(item["subset"]) == 3 and len([s for s in subsets if len(s) == 3]) < top_triples:
            subsets.append(item["subset"])
    return subsets


def reservoir_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(x) <= max_rows:
        return x.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx].astype(np.float32, copy=False)


def update_min_dist(points: torch.Tensor, selected: torch.Tensor, current: torch.Tensor, batch_size: int) -> torch.Tensor:
    for start in range(0, len(points), batch_size):
        chunk = points[start : start + batch_size]
        dist = torch.sum((chunk - selected[None, :]) ** 2, dim=1)
        current[start : start + len(chunk)] = torch.minimum(current[start : start + len(chunk)], dist)
    return current


@torch.no_grad()
def kcenter_coverage_curve(
    candidate: np.ndarray,
    eval_points: np.ndarray,
    sizes: list[int],
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    candidate_t = torch.from_numpy(candidate).to(device)
    eval_t = torch.from_numpy(eval_points).to(device)
    max_size = min(max(sizes), len(candidate))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(candidate)))
    selected_indices = []
    candidate_min = torch.full((len(candidate_t),), float("inf"), device=device)
    eval_min = torch.full((len(eval_t),), float("inf"), device=device)
    rows = []
    target_sizes = sorted({size for size in sizes if size <= max_size})

    for step in range(1, max_size + 1):
        if step == 1:
            idx = first
        else:
            idx = int(torch.argmax(candidate_min).item())
        selected_indices.append(idx)
        selected = candidate_t[idx]
        candidate_min = update_min_dist(candidate_t, selected, candidate_min, batch_size)
        eval_min = update_min_dist(eval_t, selected, eval_min, batch_size)
        candidate_min[idx] = 0.0
        if step in target_sizes:
            dist = torch.sqrt(eval_min).detach().cpu().numpy()
            rows.append(
                {
                    "bank_size": int(step),
                    "coverage_q50": round_float(float(np.quantile(dist, 0.50))),
                    "coverage_q90": round_float(float(np.quantile(dist, 0.90))),
                    "coverage_q95": round_float(float(np.quantile(dist, 0.95))),
                    "coverage_q99": round_float(float(np.quantile(dist, 0.99))),
                    "coverage_max": round_float(float(np.max(dist))),
                }
            )
    return {"rows": rows, "selected_count": int(len(selected_indices))}


def required_bank_size(rows: list[dict], reference_q95: float, slack: float) -> int | None:
    threshold = reference_q95 * (1.0 + slack)
    for row in rows:
        if row["coverage_q95"] <= threshold:
            return int(row["bank_size"])
    return None


def feature_profile_from_system(item: dict, category: str) -> dict:
    proposed = item["proposed_profile_and_bank_switch"]
    row = next(row for row in proposed["category_rows"] if row["category"] == category)
    return {
        "name": f"{row['backbone']}_out{':'.join(str(i) for i in row['out_indices'])}_g{row['patch_grid']}",
        "backbone": row["backbone"],
        "out_indices": tuple(int(i) for i in row["out_indices"]),
        "patch_grid": int(row["patch_grid"]),
        "selected_bank_patches": int(row["bank_patches"]),
    }


def collect_normal_patch_pool(
    category: str,
    profile: dict,
    materialized_root: Path,
    args: argparse.Namespace,
    device: torch.device,
    cache: dict[tuple[str, str, tuple[int, ...], int], np.ndarray],
) -> np.ndarray:
    key = (category, profile["backbone"], tuple(profile["out_indices"]), int(profile["patch_grid"]))
    if key in cache:
        return cache[key]
    samples = find_materialized_samples(materialized_root, category)
    train, _test = normal_train_and_test(samples)
    model = make_backbone(profile["backbone"], tuple(profile["out_indices"]), device)
    features, _ = collect_features(
        model,
        train,
        (args.image_height, args.image_width),
        args.batch_size,
        (profile["patch_grid"], profile["patch_grid"]),
        device,
        f"{category} {profile['name']} normal features",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pool = features.reshape(-1, features.shape[-1]).astype(np.float32)
    cache[key] = pool
    return pool


def standard_profile() -> dict:
    return {
        "name": "standard_wrn_l23_g14",
        "backbone": "wide_resnet50_2",
        "out_indices": (1, 2),
        "patch_grid": 14,
        "selected_bank_patches": 12000,
    }


def evaluate_distribution(
    label: str,
    pools: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    merged = np.concatenate(pools, axis=0)
    candidate = reservoir_rows(merged, args.candidate_pool, args.seed)
    eval_points = reservoir_rows(merged, args.eval_pool, args.seed + 1009)
    rows = kcenter_coverage_curve(candidate, eval_points, args.bank_sizes, args.seed, device, args.distance_batch_size)["rows"]
    reference = rows[-1]["coverage_q95"]
    required = required_bank_size(rows, reference, args.coverage_slack)
    return {
        "label": label,
        "normal_patch_pool": int(len(merged)),
        "candidate_pool": int(len(candidate)),
        "eval_pool": int(len(eval_points)),
        "feature_dim": int(merged.shape[-1]),
        "reference_bank_size": int(rows[-1]["bank_size"]),
        "reference_q95": reference,
        "coverage_slack": args.coverage_slack,
        "required_bank_size_by_q95": required,
        "rows": rows,
    }


def evaluate_subset(
    item: dict,
    args: argparse.Namespace,
    materialized_root: Path,
    device: torch.device,
    feature_cache: dict[tuple[str, str, tuple[int, ...], int], np.ndarray],
) -> dict:
    subset = item["subset"]
    std = standard_profile()
    std_pools = {
        category: collect_normal_patch_pool(category, std, materialized_root, args, device, feature_cache)
        for category in subset
    }
    standard_single = {
        category: evaluate_distribution(f"standard:{category}", [pool], args, device)
        for category, pool in std_pools.items()
    }
    standard_merged = evaluate_distribution(f"standard:{subset_key(subset)}", list(std_pools.values()), args, device)

    proposed_single = {}
    for category in subset:
        profile = feature_profile_from_system(item, category)
        pool = collect_normal_patch_pool(category, profile, materialized_root, args, device, feature_cache)
        proposed_single[category] = {
            "profile": {
                "name": profile["name"],
                "backbone": profile["backbone"],
                "out_indices": list(profile["out_indices"]),
                "patch_grid": profile["patch_grid"],
                "selected_bank_patches_from_detection": profile["selected_bank_patches"],
            },
            **evaluate_distribution(f"proposed:{category}", [pool], args, device),
        }

    single_sum = sum(v["required_bank_size_by_q95"] or 0 for v in standard_single.values())
    merged_required = standard_merged["required_bank_size_by_q95"]
    return {
        "subset": subset,
        "standard_profile": std | {"out_indices": list(std["out_indices"])},
        "standard_single": standard_single,
        "standard_merged": standard_merged,
        "standard_required_sum_single": single_sum,
        "standard_required_merged": merged_required,
        "merged_to_sum_required_ratio": round_float(merged_required / single_sum) if single_sum and merged_required else None,
        "proposed_single": proposed_single,
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# PatchCore bank削減のcoresetカバー率検証",
        "",
        "## 目的",
        "",
        "bank削減をランダムな当たり外れではなく，既存のk-center coreset選択で「正常分布を覆う代表点数」として評価する。",
        "また，A単独，B単独，A+B/ABC混合集合で必要bank数が単純加算になるかを確認する。",
        "",
        "## 定義",
        "",
        "- 代表点選択: farthest-first k-center greedy。",
        "- カバー半径: 全正常パッチから最も近いbank代表点までの距離。",
        "- 必要bank数: 最大bank候補での95%カバー半径に対し，指定slack以内に入る最小bank数。",
        "",
        "| subset | 標準A単独+B単独の必要数合計 | 標準merged必要数 | merged/sum | 提案側カテゴリ別bank数 | 読み取り |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in payload["subset_results"]:
        proposed_banks = [str(v["profile"]["selected_bank_patches_from_detection"]) for v in item["proposed_single"].values()]
        ratio = item["merged_to_sum_required_ratio"]
        note = "mergedが単純和より小さい" if ratio is not None and ratio < 0.95 else "ほぼ単純和または追加検証が必要"
        lines.append(
            f"| {' + '.join(item['subset'])} | {item['standard_required_sum_single']} | "
            f"{item['standard_required_merged']} | {ratio if ratio is not None else '-'} | "
            f"{' / '.join(proposed_banks)} | {note} |"
        )
    lines += [
        "",
        "## 解釈",
        "",
        "- bankのみ切替の0.5x/0.333xは，対象カテゴリごとに標準bankを分ける効果である。",
        "- profile切替でさらに小さくなるかは，そのカテゴリの正常特徴分布を少数代表点で覆えるかに依存する。",
        "- merged/sumが1に近い場合，A+Bの正常分布を覆う代表点数はほぼ加法的であり，カテゴリ別bank切替の意義が出やすい。",
        "- merged/sumが大きく1を下回る場合，カテゴリ間で正常特徴の共通部分があり，混合bankでも代表点を共有できる可能性がある。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["+".join(item["subset"]) for item in payload["subset_results"]]
    sums = [item["standard_required_sum_single"] for item in payload["subset_results"]]
    merged = [item["standard_required_merged"] or 0 for item in payload["subset_results"]]
    ratios = [item["merged_to_sum_required_ratio"] or 0 for item in payload["subset_results"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(labels) * 1.4), 7), constrained_layout=True)
    w = 0.38
    axes[0].bar(x - w / 2, sums, width=w, label="sum of single-category banks", color="#4e79a7")
    axes[0].bar(x + w / 2, merged, width=w, label="merged-category bank", color="#f28e2b")
    axes[0].set_ylabel("Required coreset size")
    axes[0].legend()
    axes[1].bar(x, ratios, color="#59a14f")
    axes[1].axhline(1.0, color="#333333", linewidth=1.0)
    axes[1].set_ylabel("merged / sum")
    axes[1].set_ylim(0, max(1.2, max(ratios) * 1.15 if ratios else 1.2))
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Coreset size needed to cover normal PatchCore features")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--materialized-root", type=Path, default=Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2"))
    parser.add_argument("--top-pairs", type=int, default=3)
    parser.add_argument("--top-triples", type=int, default=3)
    parser.add_argument("--bank-sizes", type=parse_sizes, default=parse_sizes("25,50,100,125,250,500,750,1000,1500,2000,3000"))
    parser.add_argument("--coverage-slack", type=float, default=0.05)
    parser.add_argument("--candidate-pool", type=int, default=8000)
    parser.add_argument("--eval-pool", type=int, default=8000)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but not available.")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    subsets = select_subsets(candidates, args.top_pairs, args.top_triples)
    feature_cache: dict[tuple[str, str, tuple[int, ...], int], np.ndarray] = {}
    results = [
        evaluate_subset(item, args, args.materialized_root, device, feature_cache)
        for item in candidates["verified_subsets"]
        if item["subset"] in subsets
    ]
    payload = {
        "purpose": "Use k-center coreset coverage to explain PatchCore bank-size reduction and A/B/ABC merged-bank scaling.",
        "config": {
            "candidates": str(args.candidates),
            "bank_sizes": args.bank_sizes,
            "coverage_slack": args.coverage_slack,
            "candidate_pool": args.candidate_pool,
            "eval_pool": args.eval_pool,
            "seed": args.seed,
            "device": str(device),
        },
        "subset_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "subsets": len(results), "device": str(device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
