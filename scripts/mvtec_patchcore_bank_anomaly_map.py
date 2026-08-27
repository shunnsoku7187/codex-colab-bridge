"""Inspect surprising PatchCore minimum-bank additivity cases.

This audit focuses on cases such as standard bottle+carpet where a merged bank
appears much smaller than the sum of the category-wise minimum banks.  It checks
whether the result is stable across seeds and visualizes the feature geometry of
the selected memory-bank points.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from scripts.mvtec_ad_parquet_anomaly_probe import (
    curve_rows,
    image_scores_from_patch_scores,
    score_auc,
)
from scripts.mvtec_patchcore_fixed_coreset_profile_switch import (
    FeatureCache,
    baseline_rows,
    patchcore_scores_gpu,
    row_config,
    rows_by_category,
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass
from scripts.train_kolektor_strong_final import round_float, set_seed


DEFAULT_SOURCE = Path("results/mvtec_patchcore_backbone_floor_probe_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_bank_anomaly_map_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_bank_anomaly_map_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_bank_anomaly_map_001.png")


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def deterministic_reservoir_with_labels(
    arrays_by_category: dict[str, np.ndarray],
    max_rows_per_category: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    labels = []
    for category, features in arrays_by_category.items():
        flat = features.reshape(-1, features.shape[-1]).astype(np.float32, copy=False)
        if len(flat) > max_rows_per_category:
            idx = np.linspace(0, len(flat) - 1, num=max_rows_per_category, dtype=np.int64)
            flat = flat[idx]
        arrays.append(flat)
        labels.extend([category] * len(flat))
    return np.concatenate(arrays, axis=0), np.array(labels)


@torch.no_grad()
def kcenter_indices(points: np.ndarray, size: int, device: torch.device, batch_size: int) -> list[int]:
    size = min(size, len(points))
    points = points.astype(np.float32, copy=False)
    points_t = torch.from_numpy(points).to(device)
    center = points.mean(axis=0, keepdims=True)
    idx = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
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
    return selected


@torch.no_grad()
def random_bank(points: np.ndarray, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=min(size, len(points)), replace=False)
    return points[idx].astype(np.float32, copy=False)


def nested_kcenter_banks(
    points: np.ndarray,
    sizes: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    max_size = max(size for size in sizes if size <= len(points))
    indices = kcenter_indices(points, max_size, device, batch_size)
    return {size: points[np.array(indices[:size], dtype=np.int64)].astype(np.float32, copy=False) for size in sizes if size <= len(points)}


def evaluate_category(category: str, cfg: dict, bank: np.ndarray, cache: FeatureCache, args: argparse.Namespace, device: torch.device) -> dict:
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
        "score_good_q50": round_float(float(np.quantile(scores[test_labels == 0], 0.50))),
        "score_good_q90": round_float(float(np.quantile(scores[test_labels == 0], 0.90))),
        "score_defect_q10": round_float(float(np.quantile(scores[test_labels == 1], 0.10))),
        "score_defect_q50": round_float(float(np.quantile(scores[test_labels == 1], 0.50))),
    }


def evaluate_bank(label: str, subset: list[str], cfg: dict, bank: np.ndarray, cache: FeatureCache, args: argparse.Namespace, device: torch.device) -> dict:
    rows = [evaluate_category(category, cfg, bank, cache, args, device) for category in subset]
    good_values = [row["good_pass"] for row in rows if row["good_pass"] is not None]
    return {
        "label": label,
        "bank_size": int(len(bank)),
        "category_rows": rows,
        "min_good_pass": round_float(min(good_values)) if good_values else None,
        "mean_good_pass": round_float(mean(good_values)) if good_values else None,
    }


def summarize_trials(values: list[dict]) -> dict:
    grouped: dict[int, list[float]] = {}
    for row in values:
        if row["min_good_pass"] is not None:
            grouped.setdefault(int(row["bank_size"]), []).append(float(row["min_good_pass"]))
    return {
        str(size): {
            "trials": len(vals),
            "mean_min_good_pass": round_float(mean(vals)),
            "std_min_good_pass": round_float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "min_min_good_pass": round_float(min(vals)),
            "max_min_good_pass": round_float(max(vals)),
        }
        for size, vals in sorted(grouped.items())
    }


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# bottle+carpet bank size anomaly audit",
        "",
        "## Question",
        "",
        "`standard bottle+carpet` previously reported `N(A)+N(B)=3025` and `N(A+B)=125`. This audit checks whether that very small merged-bank result is a stable geometric effect or a fragile artifact of bank selection and thresholding.",
        "",
        "## Main results",
        "",
        "| bank | selection | min good-pass | bottle good-pass | carpet good-pass | selected bottle | selected carpet |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["bank_evaluations"]:
        counts = row.get("selected_category_counts", {})
        cat = {r["category"]: r for r in row["category_rows"]}
        lines.append(
            f"| {row['bank_size']} | {row['selection']} | {pct(row['min_good_pass'])} | "
            f"{pct(cat['bottle']['good_pass'])} | {pct(cat['carpet']['good_pass'])} | "
            f"{counts.get('bottle', '-')} | {counts.get('carpet', '-')} |"
        )
    lines += [
        "",
        "## Random-bank stability",
        "",
        "| bank | trials | mean min good-pass | std | min | max |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size, row in payload["random_trial_summary"].items():
        lines.append(
            f"| {size} | {row['trials']} | {pct(row['mean_min_good_pass'])} | "
            f"{pct(row['std_min_good_pass'])} | {pct(row['min_min_good_pass'])} | {pct(row['max_min_good_pass'])} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        payload["interpretation"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, points: np.ndarray, labels: np.ndarray, selected_by_size: dict[int, list[int]], path: Path) -> None:
    sample_limit = min(len(points), 5000)
    sample_idx = np.linspace(0, len(points) - 1, sample_limit, dtype=np.int64)
    selected_all = sorted(set(i for indices in selected_by_size.values() for i in indices))
    fit_idx = np.unique(np.concatenate([sample_idx, np.array(selected_all, dtype=np.int64)]))
    pca = PCA(n_components=2, random_state=0)
    pca.fit(points[fit_idx])
    sample_xy = pca.transform(points[sample_idx])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    colors = {"bottle": "#1f77b4", "carpet": "#d62728"}
    for ax, size in zip(axes, sorted(selected_by_size)):
        for category in ["bottle", "carpet"]:
            mask = labels[sample_idx] == category
            ax.scatter(sample_xy[mask, 0], sample_xy[mask, 1], s=4, alpha=0.18, color=colors[category], label=f"{category} normal patches")
        sel_idx = np.array(selected_by_size[size], dtype=np.int64)
        sel_xy = pca.transform(points[sel_idx])
        sel_labels = labels[sel_idx]
        for category in ["bottle", "carpet"]:
            mask = sel_labels == category
            ax.scatter(sel_xy[mask, 0], sel_xy[mask, 1], s=28, alpha=0.95, color=colors[category], marker="x", linewidths=1.2)
        ax.set_title(f"k-center bank={size}")
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.grid(True, alpha=0.25)
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
    parser.add_argument("--subset", default="bottle,carpet")
    parser.add_argument("--bank-sizes", default="100,125,250,3000")
    parser.add_argument("--random-trials", type=int, default=8)
    parser.add_argument("--coreset-candidate-pool", type=int, default=12000)
    parser.add_argument("--false-pass-target", type=float, default=0.05)
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
    args.bank_sizes = parse_ints(args.bank_sizes)
    args.subset = [part.strip() for part in args.subset.split(",") if part.strip()]
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this job.")

    summary = json.loads(args.source.read_text(encoding="utf-8"))
    standard = baseline_rows(summary, args.false_pass_target)
    cfg = row_config(standard[args.subset[0]])
    cache = FeatureCache(args.materialized_root, args, device)

    train_by_category = {}
    for category in args.subset:
        train_features, _test_features, _test_labels, _train_labels, _meta = cache.get(category, cfg)
        train_by_category[category] = train_features
    candidate, candidate_labels = deterministic_reservoir_with_labels(train_by_category, args.coreset_candidate_pool)

    nested_banks = nested_kcenter_banks(candidate, args.bank_sizes, device, args.distance_batch_size)
    selected_indices = {size: kcenter_indices(candidate, size, device, args.distance_batch_size) for size in args.bank_sizes if size <= len(candidate)}

    evaluations = []
    for size in sorted(nested_banks):
        bank = nested_banks[size]
        selected_labels = candidate_labels[selected_indices[size]]
        counts = {category: int(np.sum(selected_labels == category)) for category in args.subset}
        row = evaluate_bank(f"kcenter_{size}", args.subset, cfg, bank, cache, args, device)
        row["selection"] = "nested k-center"
        row["selected_category_counts"] = counts
        evaluations.append(row)

    random_rows = []
    for size in args.bank_sizes:
        for trial in range(args.random_trials):
            bank = random_bank(candidate, size, args.seed + 1000 + trial)
            row = evaluate_bank(f"random_{size}_{trial}", args.subset, cfg, bank, cache, args, device)
            row["selection"] = "random"
            random_rows.append(row)

    stable_125 = [row for row in random_rows if row["bank_size"] == 125 and row["min_good_pass"] is not None]
    if stable_125:
        lo = min(float(row["min_good_pass"]) for row in stable_125)
        hi = max(float(row["min_good_pass"]) for row in stable_125)
        interpretation = (
            f"For bank=125, random banks produced min good-pass from {pct(lo)} to {pct(hi)}. "
            "Large spread means the previous 125-bank result should be treated as selection-sensitive, not as a proof that the merged distribution is intrinsically easy."
        )
    else:
        interpretation = "No stable bank=125 random baseline was available."

    payload = {
        "purpose": "Inspect why standard bottle+carpet reported a surprisingly tiny merged minimum bank.",
        "config": vars(args) | {"device": str(device), "profile": cfg},
        "candidate_bank_points": int(len(candidate)),
        "candidate_category_counts": {category: int(np.sum(candidate_labels == category)) for category in args.subset},
        "bank_evaluations": evaluations,
        "random_trial_summary": summarize_trials(random_rows),
        "random_trials": random_rows,
        "interpretation": interpretation,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, args.markdown)
    figure_indices = {size: selected_indices[size] for size in sorted(selected_indices) if size in {125, 250, 3000}}
    write_figure(payload, candidate, candidate_labels, figure_indices, args.figure)
    print(json.dumps({"wrote": str(args.output), "figure": str(args.figure), "device": str(device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
