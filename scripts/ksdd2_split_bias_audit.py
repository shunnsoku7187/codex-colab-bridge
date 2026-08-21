"""Audit KSDD2 split sensitivity after the U-Net/ResNet50 recheck.

This script does not retrain models.  It reads the saved score files from the
foundation recheck and asks whether one seed is unusually hard under the same
inspection operating constraint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def round_float(value, ndigits: int = 6):
    if value is None:
        return None
    return round(float(value), ndigits)


def load_scores(scores_dir: Path, seed: int) -> dict[str, np.ndarray]:
    path = scores_dir / f"seed_{seed}_scores.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path))


def metric(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    good = labels == 0
    defect = labels == 1
    reject = scores >= threshold
    false_pass = defect & ~reject
    good_loss = good & reject
    return {
        "threshold": round_float(threshold),
        "good_pass_rate": round_float(float((good & ~reject).sum() / max(good.sum(), 1))),
        "good_loss_rate": round_float(float(good_loss.sum() / max(good.sum(), 1))),
        "false_pass_rate": round_float(float(false_pass.sum() / max(defect.sum(), 1))),
        "false_pass_count": int(false_pass.sum()),
        "defect_count": int(defect.sum()),
        "good_count": int(good.sum()),
    }


def curve(labels: np.ndarray, scores: np.ndarray, points: int) -> list[dict]:
    qs = np.linspace(0, 1, points)
    thresholds = np.unique(np.quantile(scores, qs))
    return [metric(labels, scores, float(t)) for t in thresholds]


def best_safe(rows: list[dict], max_false_pass: float) -> dict | None:
    safe = [row for row in rows if float(row["false_pass_rate"]) <= max_false_pass + 1e-12]
    if not safe:
        return None
    return max(safe, key=lambda row: (float(row["good_pass_rate"]), -float(row["false_pass_rate"])))


def near_good_pass(rows: list[dict], target: float) -> dict:
    return min(rows, key=lambda row: abs(float(row["good_pass_rate"]) - target))


def auc(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "aupr": None}

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(ranks[pos].sum())
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)

    desc = np.argsort(-scores)
    y = labels[desc]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    aupr = float(np.sum((recall - prev_recall) * precision))
    return {
        "auroc": round_float(auroc),
        "aupr": round_float(aupr),
    }


def summarize_seed(data: dict[str, np.ndarray], seed: int, score_name: str, points: int) -> dict:
    labels = data["test_labels"].astype(np.int64)
    scores = data[f"test_{score_name}"].astype(np.float32)
    defect_scores = scores[labels == 1]
    good_scores = scores[labels == 0]
    rows = curve(labels, scores, points)
    safe5 = best_safe(rows, 0.05)
    safe3 = best_safe(rows, 0.03)
    gp90 = near_good_pass(rows, 0.90)
    return {
        "seed": seed,
        "score_name": score_name,
        **auc(labels, scores),
        "test_counts": {"samples": int(len(labels)), "good": int((labels == 0).sum()), "defect": int((labels == 1).sum())},
        "defect_score_quantiles": {str(q): round_float(np.quantile(defect_scores, q)) for q in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]},
        "good_score_quantiles": {str(q): round_float(np.quantile(good_scores, q)) for q in [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]},
        "best_under_false_pass_5": safe5,
        "best_under_false_pass_3": safe3,
        "near_good_pass_90": gp90,
    }


def plot_seed_distributions(rows: list[dict], score_arrays: dict[tuple[int, str], dict[str, np.ndarray]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
    for ax, score_name in zip(axes, ["max_score", "topk_score"]):
        for (seed, name), arrays in score_arrays.items():
            if name != score_name:
                continue
            labels = arrays["labels"]
            scores = arrays["scores"]
            defect = scores[labels == 1]
            ax.hist(defect, bins=40, alpha=0.35, density=True, label=f"seed {seed} defect")
        ax.set_title(f"Defect score distribution: {score_name}")
        ax.set_xlabel("defect score")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KSDD2 split bias audit",
        "",
        "Purpose: check whether the weak seed in the foundation recheck looks like a data/split difficulty issue.",
        "",
        "| seed | score | AUROC | AUPR | good pass at <=5% false-pass | false-pass near 90% good-pass | defect score median | defect score 10% quantile |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        safe5 = row["best_under_false_pass_5"]
        gp90 = row["near_good_pass_90"]
        lines.append(
            f"| {row['seed']} | {row['score_name']} | {row['auroc']} | {row['aupr']} | "
            f"{100*safe5['good_pass_rate']:.2f}% | {100*gp90['false_pass_rate']:.2f}% | "
            f"{row['defect_score_quantiles']['0.5']} | {row['defect_score_quantiles']['0.1']} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- If one seed has much lower defect-score lower quantiles, the test defects for that seed are harder for the trained model or threshold calibration.",
        "- If AUROC remains high but the low defect-score tail grows, the issue is not complete model failure; it is safety-threshold stability.",
        "- This audit still cannot prove file-series bias by itself.  The next step is to join these score tails with image names and mask-area statistics.",
        "",
        f"Distribution figure: `{payload['figure']}` if matplotlib is available.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="results/ksdd2_unet_resnet50_foundation_recheck_caviar9_001_scores")
    parser.add_argument("--output", default="results/ksdd2_split_bias_audit_001.json")
    parser.add_argument("--markdown", default="docs/ksdd2_split_bias_audit_001.md")
    parser.add_argument("--figure", default="results/ksdd2_split_bias_audit_001_defect_score_distributions.png")
    parser.add_argument("--seeds", nargs="*", type=int, default=[123, 456, 789])
    parser.add_argument("--curve-points", type=int, default=300)
    args = parser.parse_args()

    score_arrays = {}
    rows = []
    for seed in args.seeds:
        data = load_scores(Path(args.scores_dir), seed)
        for score_name in ["max_score", "topk_score"]:
            labels = data["test_labels"].astype(np.int64)
            scores = data[f"test_{score_name}"].astype(np.float32)
            score_arrays[(seed, score_name)] = {"labels": labels, "scores": scores}
            rows.append(summarize_seed(data, seed, score_name, args.curve_points))
    payload = {
        "purpose": "Audit split/seed sensitivity using saved U-Net/ResNet50 foundation recheck scores.",
        "scores_dir": args.scores_dir,
        "rows": rows,
        "figure": args.figure,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    plot_seed_distributions(rows, score_arrays, Path(args.figure))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "figure": args.figure}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
