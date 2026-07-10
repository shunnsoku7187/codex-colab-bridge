"""Collect ResNet56-BranchyNet exit confidence traces for binary inspection framing.

The experiment keeps the raw per-exit behavior reusable:

* predicted class, correctness, self-confidence, and entropy at each exit
* binary yes/no mapping for several CIFAR-10 positive-class candidates
* whether an early low-confidence sample later recovers to a reliable yes

Here "reliable yes" means that the final exit predicts yes and its final
self-confidence is above a threshold calibrated to a target accepted accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


DEFAULT_POSITIVE_SETS = {
    "vehicle": [0, 1, 8, 9],
    "animal": [2, 3, 4, 5, 6, 7],
    "pet": [3, 5],
    "vehicle_vs_animal_balanced": [0, 1, 8, 9],
    "single_ship": [8],
}


def round_float(value: float | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def calibrate_yes_threshold(
    final_yes_pred: np.ndarray,
    final_binary_correct: np.ndarray,
    final_confidence: np.ndarray,
    target_precision: float,
) -> dict[str, Any] | None:
    """Find the lowest threshold that maximizes coverage under yes precision."""

    rows = []
    for threshold in sorted({float(x) for x in final_confidence[final_yes_pred]}):
        accept = final_yes_pred & (final_confidence >= threshold)
        if not accept.any():
            continue
        precision = float(final_binary_correct[accept].mean())
        rows.append({
            "threshold": threshold,
            "yes_accept_rate": float(accept.mean()),
            "yes_precision": precision,
            "false_yes_rate_among_yes": 1.0 - precision,
        })

    valid = [row for row in rows if row["yes_precision"] >= target_precision]
    if not valid:
        return None
    best = max(valid, key=lambda row: (row["yes_accept_rate"], -row["threshold"], row["yes_precision"]))
    return {
        "target_yes_precision": target_precision,
        "threshold": round_float(best["threshold"]),
        "yes_accept_rate": round_float(best["yes_accept_rate"]),
        "yes_precision": round_float(best["yes_precision"]),
        "false_yes_rate_among_yes": round_float(best["false_yes_rate_among_yes"]),
    }


def confidence_band_rows(
    source_confidence: np.ndarray,
    target_reliable_yes: np.ndarray,
    target_final_yes: np.ndarray,
    quantile_edges: list[float],
) -> list[dict[str, Any]]:
    edges = np.quantile(source_confidence, quantile_edges)
    rows = []
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            mask = (source_confidence >= lo) & (source_confidence <= hi)
        else:
            mask = (source_confidence >= lo) & (source_confidence < hi)
        count = int(mask.sum())
        rows.append({
            "band": f"q{quantile_edges[i]:.2f}_to_q{quantile_edges[i + 1]:.2f}",
            "confidence_min": round_float(lo),
            "confidence_max": round_float(hi),
            "count": count,
            "rate": round_float(mask.mean()),
            "final_yes_rate": None if count == 0 else round_float(target_final_yes[mask].mean()),
            "final_reliable_yes_rate": None if count == 0 else round_float(target_reliable_yes[mask].mean()),
        })
    return rows


def low_conf_recovery_rows(
    source_confidence: np.ndarray,
    source_yes_pred: np.ndarray,
    final_yes_pred: np.ndarray,
    final_reliable_yes: np.ndarray,
    low_quantiles: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for quantile in low_quantiles:
        threshold = float(np.quantile(source_confidence, quantile))
        low = source_confidence <= threshold
        low_not_yes = low & ~source_yes_pred
        rows.append({
            "low_conf_quantile": quantile,
            "source_low_conf_threshold": round_float(threshold),
            "low_conf_count": int(low.sum()),
            "low_conf_rate": round_float(low.mean()),
            "final_yes_rate_among_low_conf": round_float(final_yes_pred[low].mean()) if low.any() else None,
            "final_reliable_yes_rate_among_low_conf": round_float(final_reliable_yes[low].mean()) if low.any() else None,
            "low_conf_not_yes_count": int(low_not_yes.sum()),
            "final_reliable_yes_rate_among_low_conf_not_yes": (
                round_float(final_reliable_yes[low_not_yes].mean()) if low_not_yes.any() else None
            ),
        })
    return rows


def confusion(actual_yes: np.ndarray, predicted_yes: np.ndarray) -> dict[str, Any]:
    tp = int((actual_yes & predicted_yes).sum())
    fp = int((~actual_yes & predicted_yes).sum())
    fn = int((actual_yes & ~predicted_yes).sum())
    tn = int((~actual_yes & ~predicted_yes).sum())
    total = tp + fp + fn + tn
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    specificity = None if tn + fp == 0 else tn / (tn + fp)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    false_yes_rate = None if tp + fp == 0 else fp / (tp + fp)
    good_loss_rate = None if tp + fn == 0 else fn / (tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "accuracy": round_float((tp + tn) / total if total else None),
        "yes_precision": round_float(precision),
        "yes_recall": round_float(recall),
        "specificity": round_float(specificity),
        "f1": round_float(f1),
        "false_yes_rate_among_yes": round_float(false_yes_rate),
        "yes_loss_rate": round_float(good_loss_rate),
    }


def write_sample_csv(
    output_csv: Path,
    labels: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    correct: np.ndarray,
    exit_names: list[str],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_index", "true_label", "true_class"]
        for exit_name in exit_names:
            header.extend([
                f"{exit_name}_pred_label",
                f"{exit_name}_pred_class",
                f"{exit_name}_correct",
                f"{exit_name}_confidence",
                f"{exit_name}_entropy",
            ])
        writer.writerow(header)
        for idx in range(labels.shape[0]):
            row: list[Any] = [idx, int(labels[idx]), CIFAR10_CLASSES[int(labels[idx])]]
            for exit_idx in range(len(exit_names)):
                pred_label = int(pred[idx, exit_idx])
                row.extend([
                    pred_label,
                    CIFAR10_CLASSES[pred_label],
                    bool(correct[idx, exit_idx]),
                    f"{float(confidence[idx, exit_idx]):.8f}",
                    f"{float(entropy[idx, exit_idx]):.8f}",
                ])
            writer.writerow(row)


def plot_reliable_yes_curves(positive_set_results: list[dict[str, Any]], plot_dir: Path) -> list[str]:
    """Save slide-friendly PNGs for reliable-yes recovery behavior."""

    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for item in positive_set_results:
        target_key = "target_yes_precision_0.980"
        target = item["reliable_yes_recovery"].get(target_key)
        if target is None:
            target_key = next((key for key, value in item["reliable_yes_recovery"].items() if value is not None), "")
            target = item["reliable_yes_recovery"].get(target_key) if target_key else None
        if target is None:
            continue

        fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=160)
        plotted_y: list[float] = []
        for exit_label, rows_key in [
            ("出口0", "exit0_confidence_bands_to_final"),
            ("出口1", "exit1_confidence_bands_to_final"),
        ]:
            rows = target[rows_key]
            x = [(row["confidence_min"] + row["confidence_max"]) / 2 for row in rows if row["final_reliable_yes_rate"] is not None]
            y = [100 * row["final_reliable_yes_rate"] for row in rows if row["final_reliable_yes_rate"] is not None]
            ax.plot(x, y, marker="o", linewidth=2, label=exit_label)
            plotted_y.extend(y)
        ax.set_title(f"{item['name']}: 早期出口の自己確信度と最終的な「信頼あるyes」率")
        ax.set_xlabel("早期出口の自己確信度")
        ax.set_ylabel("最終的に信頼あるyesとなる割合 [%]")
        if plotted_y:
            y_max = max(plotted_y)
            ax.set_ylim(0.0, max(5.0, y_max * 1.25 + 1.0))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = plot_dir / f"{item['name']}_confidence_to_reliable_yes.png"
        fig.savefig(out)
        plt.close(fig)
        written.append(str(out))

        fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=160)
        plotted_y: list[float] = []
        for exit_label, rows_key in [
            ("出口0", "source_exit_0_low_conf_recovery"),
            ("出口1", "source_exit_1_low_conf_recovery"),
        ]:
            rows = target[rows_key]
            x = [100 * row["low_conf_rate"] for row in rows]
            y = [100 * row["final_reliable_yes_rate_among_low_conf"] for row in rows]
            ax.plot(x, y, marker="o", linewidth=2.2, label=f"{exit_label}: 低信頼すべて")
            plotted_y.extend(y)
            y_not_yes = [100 * row["final_reliable_yes_rate_among_low_conf_not_yes"] for row in rows]
            ax.plot(x, y_not_yes, marker="s", linestyle="--", linewidth=2.2, label=f"{exit_label}: 低信頼かつnot-yes")
            plotted_y.extend(y_not_yes)
        ax.set_title(f"{item['name']}: 早期に棄却した候補が最終的に「信頼あるyes」へ復帰する割合")
        ax.set_xlabel("低信頼として早期棄却する割合 [%]")
        ax.set_ylabel("最終的に信頼あるyesへ復帰した割合 [%]")
        if plotted_y:
            y_max = max(plotted_y)
            ax.set_ylim(0.0, max(5.0, y_max * 1.25 + 1.0))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = plot_dir / f"{item['name']}_low_conf_reliable_yes_recovery.png"
        fig.savefig(out)
        plt.close(fig)
        written.append(str(out))

    return written


def analyze_positive_set(
    name: str,
    positive_classes: list[int],
    labels: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    correct: np.ndarray,
    target_yes_precisions: list[float],
    low_quantiles: list[float],
    quantile_edges: list[float],
) -> dict[str, Any]:
    positive = np.array(positive_classes, dtype=int)
    actual_yes = np.isin(labels, positive)
    pred_yes = np.isin(pred, positive)
    binary_correct = pred_yes == actual_yes[:, None]
    final_yes_pred = pred_yes[:, -1]
    final_binary_correct = binary_correct[:, -1]

    final_calibration = {
        f"target_yes_precision_{target:.3f}": calibrate_yes_threshold(
            final_yes_pred,
            final_binary_correct,
            confidence[:, -1],
            target,
        )
        for target in target_yes_precisions
    }

    exit_summaries = []
    for exit_idx in range(pred.shape[1]):
        exit_summaries.append({
            "exit_index": exit_idx,
            "multiclass_accuracy": round_float(correct[:, exit_idx].mean()),
            "binary_confusion": confusion(actual_yes, pred_yes[:, exit_idx]),
            "yes_prediction_rate": round_float(pred_yes[:, exit_idx].mean()),
            "confidence_mean": round_float(confidence[:, exit_idx].mean()),
            "confidence_median": round_float(np.median(confidence[:, exit_idx])),
        })

    target_rows = {}
    for key, calibration in final_calibration.items():
        if calibration is None:
            target_rows[key] = None
            continue
        reliable_yes = final_yes_pred & final_binary_correct & (confidence[:, -1] >= calibration["threshold"])
        target_rows[key] = {
            "reliable_yes_rate": round_float(reliable_yes.mean()),
            "reliable_yes_count": int(reliable_yes.sum()),
            "source_exit_0_low_conf_recovery": low_conf_recovery_rows(
                confidence[:, 0],
                pred_yes[:, 0],
                final_yes_pred,
                reliable_yes,
                low_quantiles,
            ),
            "source_exit_1_low_conf_recovery": low_conf_recovery_rows(
                confidence[:, 1],
                pred_yes[:, 1],
                final_yes_pred,
                reliable_yes,
                low_quantiles,
            ),
            "exit0_confidence_bands_to_final": confidence_band_rows(
                confidence[:, 0],
                reliable_yes,
                final_yes_pred,
                quantile_edges,
            ),
            "exit1_confidence_bands_to_final": confidence_band_rows(
                confidence[:, 1],
                reliable_yes,
                final_yes_pred,
                quantile_edges,
            ),
        }

    return {
        "name": name,
        "positive_classes": positive_classes,
        "positive_class_names": [CIFAR10_CLASSES[i] for i in positive_classes],
        "positive_rate": round_float(actual_yes.mean()),
        "final_calibration": final_calibration,
        "exit_summaries": exit_summaries,
        "reliable_yes_recovery": target_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="results/0000b_branchynet_reproduce_resnet56_cifar10.npz")
    parser.add_argument("--output", default="results/resnet_exit_confidence_trace_binary_001_summary.json")
    parser.add_argument("--sample-csv", default="results/resnet_exit_confidence_trace_binary_001_samples.csv")
    parser.add_argument("--plot-dir", default="results/resnet_exit_confidence_trace_binary_001_plots")
    parser.add_argument("--target-yes-precisions", nargs="*", type=float, default=[0.95, 0.98, 0.99])
    parser.add_argument("--low-quantiles", nargs="*", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--band-quantiles", nargs="*", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 1.0])
    args = parser.parse_args()

    data = np.load(Path(args.trace), allow_pickle=True)
    labels = np.asarray(data["labels"], dtype=int)
    pred = np.asarray(data["pred"], dtype=int)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    entropy = np.asarray(data["entropy"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]
    exit_costs = np.asarray(data["exit_costs"], dtype=float)

    positive_set_results = [
        analyze_positive_set(
            name,
            classes,
            labels,
            pred,
            confidence,
            correct,
            args.target_yes_precisions,
            args.low_quantiles,
            args.band_quantiles,
        )
        for name, classes in DEFAULT_POSITIVE_SETS.items()
    ]
    plot_files = plot_reliable_yes_curves(positive_set_results, Path(args.plot_dir))

    payload = {
        "purpose": "Collect per-exit confidence and binary inspection metrics for ResNet56-BranchyNet.",
        "definitions": {
            "self_confidence": "maximum softmax probability at each exit",
            "yes": "the class group treated as acceptable/pass/positive for a candidate binary inspection task",
            "reliable_yes": "final exit predicts yes, binary yes/no prediction is correct, and final self-confidence meets a target yes precision threshold",
            "low_conf_reliable_yes_recovery": "among samples with low self-confidence at an early exit, the fraction that later become reliable yes at final",
        },
        "trace": args.trace,
        "n": int(labels.shape[0]),
        "exit_names": exit_names,
        "exit_costs": [round_float(x) for x in exit_costs],
        "multiclass_exit_accuracy": [round_float(correct[:, i].mean()) for i in range(correct.shape[1])],
        "positive_set_results": positive_set_results,
        "plot_files": plot_files,
        "dataset_note": {
            "cifar10_use": "Useful for controlled first evidence because the existing ResNet56-BranchyNet trace is ready, but it is not an inspection dataset.",
            "next_dataset_need": "Move to a genuine binary or anomaly/defect dataset before making the final inspection-task claim.",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_sample_csv(Path(args.sample_csv), labels, pred, confidence, entropy, correct, exit_names)

    compact = {
        "purpose": payload["purpose"],
        "trace": args.trace,
        "n": payload["n"],
        "multiclass_exit_accuracy": payload["multiclass_exit_accuracy"],
        "positive_set_summary": [
            {
                "name": item["name"],
                "positive_class_names": item["positive_class_names"],
                "positive_rate": item["positive_rate"],
                "exit_binary_metrics": item["exit_summaries"],
                "final_calibration": item["final_calibration"],
            }
            for item in positive_set_results
        ],
        "wrote": {
            "summary": str(out),
            "sample_csv": args.sample_csv,
            "plots": plot_files,
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
