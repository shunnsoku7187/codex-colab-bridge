"""Audit why aggressive PatchCore memory-bank pruning can work.

This experiment separates two different questions:

1. Does a small bank cover the whole normal feature distribution?
2. Does a small bank keep the normal points that actually affect PatchCore
   image scores and threshold decisions?

The first question was tested by the coreset coverage audit.  This script tests
the second question by measuring nearest-neighbor activity and by comparing
random, k-center, and normal-calibration based bank pruning.
"""

from __future__ import annotations

import argparse
import json
import re
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
    sample_normal_patch_bank,
    score_auc,
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass, normal_train_and_test
from scripts.train_kolektor_strong_final import round_float, set_seed


DEFAULT_CANDIDATES = Path("results/mvtec_patchcore_subset_mixed_bank_verify_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_bank_pruning_mechanism_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_bank_pruning_mechanism_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_bank_pruning_mechanism_001.png")


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def parse_profile_label(row: dict) -> str:
    outs = ":".join(str(i) for i in row["out_indices"])
    return f"{row['backbone']}_out{outs}_g{row['patch_grid']}_b{row['bank_patches']}"


def parse_topk_fraction(row: dict) -> float:
    value = row.get("topk_fraction")
    if value is not None:
        return float(value)
    match = re.search(r"topk([0-9]+)p([0-9]+)", str(row.get("config", "")))
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return 0.01


def pick_category_profiles(payload: dict, max_categories: int) -> list[dict]:
    profiles: dict[str, dict] = {}
    for item in payload.get("verified_subsets", []):
        proposed = item.get("proposed_profile_and_bank_switch", {})
        for row in proposed.get("category_rows", []):
            category = row["category"]
            if category not in profiles:
                profiles[category] = {
                    "category": category,
                    "profile": {
                        "name": parse_profile_label(row),
                        "backbone": row["backbone"],
                        "out_indices": tuple(int(i) for i in row["out_indices"]),
                        "patch_grid": int(row["patch_grid"]),
                        "target_bank_patches": int(row["bank_patches"]),
                        "topk_fraction": parse_topk_fraction(row),
                    },
                }
    return list(profiles.values())[:max_categories]


def split_train_calibration(samples: list, calibration_fraction: float, seed: int) -> tuple[list, list, list]:
    train, test = normal_train_and_test(samples)
    rng = np.random.default_rng(seed)
    order = np.arange(len(train))
    rng.shuffle(order)
    n_calib = max(1, int(round(len(train) * calibration_fraction)))
    calib_idx = set(int(i) for i in order[:n_calib])
    base = [sample for i, sample in enumerate(train) if i not in calib_idx]
    calib = [sample for i, sample in enumerate(train) if i in calib_idx]
    if not base:
        base, calib = train, train
    return base, calib, test


def flatten_features(features: np.ndarray) -> np.ndarray:
    return features.reshape(-1, features.shape[-1]).astype(np.float32, copy=False)


def random_bank(features: np.ndarray, size: int, seed: int) -> np.ndarray:
    flat = flatten_features(features)
    rng = np.random.default_rng(seed)
    if len(flat) <= size:
        return flat.copy()
    idx = rng.choice(len(flat), size=size, replace=False)
    return flat[idx].astype(np.float32, copy=False)


@torch.no_grad()
def nearest_min_and_idx(features: np.ndarray, bank: np.ndarray, chunk_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    shape = features.shape[:2]
    flat = flatten_features(features)
    bank_t = torch.from_numpy(bank.astype(np.float32, copy=False)).to(device)
    mins = np.empty(len(flat), dtype=np.float32)
    idxs = np.empty(len(flat), dtype=np.int64)
    for start in range(0, len(flat), chunk_size):
        chunk = torch.from_numpy(flat[start : start + chunk_size]).to(device)
        distances = torch.cdist(chunk, bank_t)
        values, indices = distances.min(dim=1)
        mins[start : start + len(chunk)] = values.detach().cpu().numpy()
        idxs[start : start + len(chunk)] = indices.detach().cpu().numpy()
    return mins.reshape(shape), idxs.reshape(shape)


@torch.no_grad()
def kcenter_indices(points: np.ndarray, size: int, seed: int, device: torch.device, batch_size: int) -> np.ndarray:
    if len(points) <= size:
        return np.arange(len(points), dtype=np.int64)
    points_t = torch.from_numpy(points.astype(np.float32, copy=False)).to(device)
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(points)))
    selected: list[int] = []
    min_dist = torch.full((len(points_t),), float("inf"), device=device)
    for _ in range(size):
        selected.append(idx)
        chosen = points_t[idx]
        for start in range(0, len(points_t), batch_size):
            chunk = points_t[start : start + batch_size]
            dist = torch.sum((chunk - chosen[None, :]) ** 2, dim=1)
            min_dist[start : start + len(chunk)] = torch.minimum(min_dist[start : start + len(chunk)], dist)
        min_dist[idx] = 0.0
        idx = int(torch.argmax(min_dist).item())
    return np.array(selected, dtype=np.int64)


def fill_indices(selected: list[int], bank: np.ndarray, size: int, seed: int, device: torch.device, batch_size: int) -> np.ndarray:
    seen = set(selected)
    out = list(selected)
    if len(out) < size:
        fallback = kcenter_indices(bank, min(size * 2, len(bank)), seed, device, batch_size)
        for idx in fallback:
            value = int(idx)
            if value not in seen:
                out.append(value)
                seen.add(value)
            if len(out) >= size:
                break
    if len(out) < size:
        rng = np.random.default_rng(seed)
        for idx in rng.permutation(len(bank)):
            value = int(idx)
            if value not in seen:
                out.append(value)
                seen.add(value)
            if len(out) >= size:
                break
    return np.array(out[:size], dtype=np.int64)


def select_active_normal_bank(
    calib_features: np.ndarray,
    full_bank: np.ndarray,
    size: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    _mins, idxs = nearest_min_and_idx(calib_features, full_bank, chunk_size, device)
    counts = np.bincount(idxs.reshape(-1), minlength=len(full_bank))
    ranked = np.argsort(-counts)
    selected = [int(i) for i in ranked if counts[int(i)] > 0][:size]
    filled = fill_indices(selected, full_bank, size, seed, device, batch_size)
    return full_bank[filled], {
        "calib_active_bank_points": int(np.count_nonzero(counts)),
        "calib_active_fraction": round_float(float(np.count_nonzero(counts) / len(full_bank))),
        "selected_from_activity": int(min(len(selected), size)),
    }


def select_kcenter_bank(full_bank: np.ndarray, size: int, seed: int, device: torch.device, batch_size: int) -> np.ndarray:
    return full_bank[kcenter_indices(full_bank, size, seed, device, batch_size)]


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float | None:
    if len(a) < 2:
        return None
    x = rankdata(a) if rank else a.astype(np.float64)
    y = rankdata(b) if rank else b.astype(np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return round_float(float(np.corrcoef(x, y)[0, 1]))


def evaluate_bank(
    name: str,
    bank: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
    topk_fraction: float,
    device: torch.device,
    full_scores: np.ndarray | None,
    full_bank_size: int,
) -> dict:
    patch_scores, _idx = nearest_min_and_idx(test_features, bank, args.nn_chunk_size, device)
    image_scores = image_scores_from_patch_scores(patch_scores, topk_fraction)[args.score_name]
    rows = curve_rows(test_labels, image_scores, args.curve_points)
    best = best_under_false_pass(rows, args.false_pass_target)
    return {
        "name": name,
        "bank_patches": int(len(bank)),
        "relative_bank_size": round_float(float(len(bank) / full_bank_size)),
        "relative_nn_ops": round_float(float(len(bank) / full_bank_size)),
        "auc": score_auc(test_labels, image_scores),
        "best_at_false_pass_target": best,
        "score_pearson_to_full": corr(full_scores, image_scores) if full_scores is not None else None,
        "score_spearman_to_full": corr(full_scores, image_scores, rank=True) if full_scores is not None else None,
    }


def active_test_stats(
    test_features: np.ndarray,
    full_bank: np.ndarray,
    topk_fraction: float,
    chunk_size: int,
    device: torch.device,
) -> dict:
    patch_scores, idxs = nearest_min_and_idx(test_features, full_bank, chunk_size, device)
    patch_count = patch_scores.shape[1]
    k = max(1, int(round(patch_count * topk_fraction)))
    topk_idx = np.argpartition(patch_scores, kth=patch_count - k, axis=1)[:, -k:]
    active_all = np.unique(idxs.reshape(-1))
    active_topk = np.unique(np.take_along_axis(idxs, topk_idx, axis=1).reshape(-1))
    return {
        "active_all_bank_points": int(len(active_all)),
        "active_all_fraction": round_float(float(len(active_all) / len(full_bank))),
        "active_score_bank_points": int(len(active_topk)),
        "active_score_fraction": round_float(float(len(active_topk) / len(full_bank))),
        "topk_patches_per_image": int(k),
        "full_patch_scores": patch_scores,
    }


def run_category(item: dict, args: argparse.Namespace, materialized_root: Path, device: torch.device) -> dict:
    category = item["category"]
    profile = item["profile"]
    samples = find_materialized_samples(materialized_root, category)
    base, calib, test = split_train_calibration(samples, args.calibration_fraction, args.seed)
    model = make_backbone(profile["backbone"], tuple(profile["out_indices"]), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (profile["patch_grid"], profile["patch_grid"])
    base_features, _ = collect_features(model, base, image_size, args.batch_size, patch_grid, device, f"{category} base normal")
    calib_features, _ = collect_features(model, calib, image_size, args.batch_size, patch_grid, device, f"{category} calibration normal")
    test_features, test_labels = collect_features(model, test, image_size, args.batch_size, patch_grid, device, f"{category} test")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_labels = np.zeros(len(base_features), dtype=np.int64)
    full_size = min(args.full_bank_patches, len(flatten_features(base_features)))
    full_bank = sample_normal_patch_bank(base_features, base_labels, full_size, args.seed)
    target_size = min(int(profile["target_bank_patches"]), len(full_bank))
    topk_fraction = float(profile["topk_fraction"])

    active = active_test_stats(test_features, full_bank, topk_fraction, args.nn_chunk_size, device)
    full_image_scores = image_scores_from_patch_scores(active.pop("full_patch_scores"), topk_fraction)[args.score_name]

    banks = [
        ("full_bank", full_bank, {}),
        ("random_target_bank", random_bank(base_features, target_size, args.seed + 17), {}),
        ("kcenter_target_bank", select_kcenter_bank(full_bank, target_size, args.seed + 29, device, args.distance_batch_size), {}),
    ]
    active_bank, active_extra = select_active_normal_bank(
        calib_features, full_bank, target_size, args.nn_chunk_size, device, args.seed + 41, args.distance_batch_size
    )
    banks.append(("normal_activity_target_bank", active_bank, active_extra))

    evaluations = []
    for name, bank, extra in banks:
        full_scores = None if name == "full_bank" else full_image_scores
        result = evaluate_bank(name, bank, test_features, test_labels, args, topk_fraction, device, full_scores, len(full_bank))
        result.update(extra)
        evaluations.append(result)

    return {
        "category": category,
        "profile": profile,
        "sample_counts": {
            "train_base_normal": int(len(base)),
            "train_calibration_normal": int(len(calib)),
            "test": int(len(test)),
            "test_good": int((test_labels == 0).sum()),
            "test_defect": int((test_labels == 1).sum()),
        },
        "full_bank_patches": int(len(full_bank)),
        "target_bank_patches": int(target_size),
        "target_bank_fraction": round_float(float(target_size / len(full_bank))),
        "active_test_stats": active,
        "evaluations": evaluations,
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# PatchCore bank削減メカニズム監査",
        "",
        "## 目的",
        "",
        "bank削減が単なるランダムな当たりではなく，PatchCoreの判定に効く代表点を残せているために成立するのかを確認する。",
        "",
        "PatchCoreの異常スコアは「正常bankへの最短距離」である。したがってbankを削ると最短距離は同じか大きくなるだけで，固定閾値では欠陥を良品扱いしやすくなる方向には動かない。問題は正常品まで異常扱いするかである。",
        "",
        "| category | target bank | testで参照されたbank | score算出patchで参照されたbank | random good-pass | k-center good-pass | normal-activity good-pass | normal-activity Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["category_results"]:
        evals = {item["name"]: item for item in row["evaluations"]}
        active = row["active_test_stats"]
        lines.append(
            f"| {row['category']} | {row['target_bank_patches']} / {row['full_bank_patches']} "
            f"({pct(row['target_bank_fraction'])}) | "
            f"{active['active_all_bank_points']} ({pct(active['active_all_fraction'])}) | "
            f"{active['active_score_bank_points']} ({pct(active['active_score_fraction'])}) | "
            f"{pct(evals['random_target_bank']['best_at_false_pass_target']['good_pass_rate_good'])} | "
            f"{pct(evals['kcenter_target_bank']['best_at_false_pass_target']['good_pass_rate_good'])} | "
            f"{pct(evals['normal_activity_target_bank']['best_at_false_pass_target']['good_pass_rate_good'])} | "
            f"{evals['normal_activity_target_bank']['score_spearman_to_full']} |"
        )
    lines += [
        "",
        "## 読み取り方",
        "",
        "- `testで参照されたbank` は，全test patchの最近傍として一度でも使われたbank点数である。",
        "- `score算出patchで参照されたbank` は，画像スコアに使われる上位patchだけに限定した最近傍bank点数である。",
        "- `normal-activity` は，正常校正画像で頻繁に最近傍になるbank点を優先して残す削減である。",
        "- normal-activityがrandomより安定して良ければ，bank削減は運ではなく，正常分布の実運用上よく使う代表点を残す設計問題として扱える。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    rows = payload["category_results"]
    labels = [row["category"] for row in rows]
    target = [100.0 * row["target_bank_fraction"] for row in rows]
    active_score = [100.0 * row["active_test_stats"]["active_score_fraction"] for row in rows]
    random_gp = []
    activity_gp = []
    for row in rows:
        evals = {item["name"]: item for item in row["evaluations"]}
        random_gp.append(100.0 * (evals["random_target_bank"]["best_at_false_pass_target"]["good_pass_rate_good"] or 0.0))
        activity_gp.append(100.0 * (evals["normal_activity_target_bank"]["best_at_false_pass_target"]["good_pass_rate_good"] or 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, target, width=0.36, label="target bank")
    axes[0].bar(x + 0.18, active_score, width=0.36, label="score-active bank")
    axes[0].set_ylabel("bank fraction [%]")
    axes[0].set_title("Bank points needed by score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - 0.18, random_gp, width=0.36, label="random")
    axes[1].bar(x + 0.18, activity_gp, width=0.36, label="normal-activity")
    axes[1].set_ylabel("good pass at constraint [%]")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Pruning method comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--max-categories", type=int, default=5)
    parser.add_argument("--full-bank-patches", type=int, default=12000)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--false-pass-target", type=float, default=0.05)
    parser.add_argument("--score-name", default="topk_score", choices=["topk_score", "max_score"])
    parser.add_argument("--curve-points", type=int, default=101)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this job.")
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    category_profiles = pick_category_profiles(payload, args.max_categories)
    results = [run_category(item, args, args.materialized_root, device) for item in category_profiles]
    out = {
        "purpose": "Explain intentional PatchCore bank pruning through nearest-neighbor activity and pruning-method comparison.",
        "config": {
            "candidates": str(args.candidates),
            "false_pass_target": args.false_pass_target,
            "full_bank_patches": args.full_bank_patches,
            "calibration_fraction": args.calibration_fraction,
            "device": str(device),
        },
        "category_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out, args.markdown)
    write_figure(out, args.figure)
    print(json.dumps({"wrote": str(args.output), "categories": len(results), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
