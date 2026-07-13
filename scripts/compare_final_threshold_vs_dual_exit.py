"""Compare final-only confidence filtering with dual-sided early exit.

This ignores binary inspection framing and treats the task as ordinary CIFAR-10
image classification where only reliable labels are wanted.

Policies:
* final_only: run every sample to the final exit, then accept only if final
  confidence passes the calibrated target-accuracy threshold.
* upper_only: accept high-confidence samples early, otherwise continue; at the
  final exit, use the same calibrated final threshold.
* dual_side: upper_only plus early rejection for very low-confidence samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def round_float(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def thresholds_from_quantiles(values: np.ndarray, quantiles: list[float]) -> list[float]:
    return sorted({float(x) for x in np.quantile(values, quantiles)})


def calibrate_final_threshold(final_correct: np.ndarray, final_confidence: np.ndarray, target_accuracy: float) -> dict[str, Any]:
    rows = []
    for threshold in sorted({float(x) for x in final_confidence}):
        accept = final_confidence >= threshold
        if not accept.any():
            continue
        rows.append(
            {
                "threshold": threshold,
                "accept_rate": float(accept.mean()),
                "accepted_accuracy": float(final_correct[accept].mean()),
                "false_accept_rate": float((~final_correct[accept]).mean()),
            }
        )
    valid = [row for row in rows if row["accepted_accuracy"] >= target_accuracy]
    if not valid:
        raise RuntimeError(f"No final confidence threshold reaches target accuracy {target_accuracy}")
    best = max(valid, key=lambda row: (row["accept_rate"], -row["threshold"], row["accepted_accuracy"]))
    return {key: round_float(value) for key, value in best.items()}


def evaluate_policy(
    correct: np.ndarray,
    confidence: np.ndarray,
    costs: np.ndarray,
    final_reliable: np.ndarray,
    final_threshold: float,
    lower0: float,
    upper0: float,
    lower1: float,
    upper1: float,
) -> dict[str, Any]:
    n = correct.shape[0]
    final_idx = correct.shape[1] - 1

    terminal = np.full(n, "final", dtype=object)
    terminal_exit = np.full(n, final_idx, dtype=np.int16)
    accepted = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)
    accepted_correct = np.zeros(n, dtype=bool)

    accept0 = confidence[:, 0] >= upper0
    reject0 = (confidence[:, 0] <= lower0) & ~accept0
    terminal[accept0] = "accept0"
    terminal_exit[accept0] = 0
    accepted[accept0] = True
    accepted_correct[accept0] = correct[accept0, 0]
    terminal[reject0] = "reject0"
    terminal_exit[reject0] = 0
    rejected[reject0] = True

    unresolved = ~(accept0 | reject0)
    accept1 = unresolved & (confidence[:, 1] >= upper1)
    reject1 = unresolved & (confidence[:, 1] <= lower1) & ~accept1
    terminal[accept1] = "accept1"
    terminal_exit[accept1] = 1
    accepted[accept1] = True
    accepted_correct[accept1] = correct[accept1, 1]
    terminal[reject1] = "reject1"
    terminal_exit[reject1] = 1
    rejected[reject1] = True

    final_mask = terminal == "final"
    final_accept = final_mask & (confidence[:, final_idx] >= final_threshold)
    final_reject = final_mask & ~final_accept
    accepted[final_accept] = True
    accepted_correct[final_accept] = correct[final_accept, final_idx]
    rejected[final_reject] = True

    accept_count = int(accepted.sum())
    accepted_accuracy = None if accept_count == 0 else float(accepted_correct[accepted].mean())
    false_accept_rate = None if accept_count == 0 else float((~accepted_correct[accepted]).mean())
    early_rejected = (terminal == "reject0") | (terminal == "reject1")
    lost_final_reliable = early_rejected & final_reliable
    cost = costs[terminal_exit]

    counts = {name: int((terminal == name).sum()) for name in ["accept0", "reject0", "accept1", "reject1", "final"]}
    return {
        "thresholds": {
            "lower0": round_float(lower0),
            "upper0": round_float(upper0),
            "lower1": round_float(lower1),
            "upper1": round_float(upper1),
        },
        "counts": counts,
        "rates": {key: round_float(value / n) for key, value in counts.items()},
        "accept_rate": round_float(float(accepted.mean())),
        "reject_rate": round_float(float(rejected.mean())),
        "final_rate": round_float(float(final_mask.mean())),
        "early_accept_rate": round_float(float(((terminal == "accept0") | (terminal == "accept1")).mean())),
        "early_reject_rate": round_float(float(early_rejected.mean())),
        "avg_cost": round_float(float(cost.mean())),
        "accepted_accuracy": round_float(accepted_accuracy),
        "false_accept_rate": round_float(false_accept_rate),
        "lost_final_reliable_rate": round_float(float(lost_final_reliable.mean())),
        "lost_final_reliable_among_early_reject": round_float(float(final_reliable[early_rejected].mean())) if early_rejected.any() else None,
    }


def choose_best(rows: list[dict[str, Any]], target_accuracy: float, max_lost: float | None = None) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row["accepted_accuracy"] is not None
        and row["accepted_accuracy"] >= target_accuracy
        and (max_lost is None or row["lost_final_reliable_rate"] <= max_lost)
    ]
    if not valid:
        return None
    return min(valid, key=lambda row: (row["avg_cost"], -row["accept_rate"], row["lost_final_reliable_rate"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="results/0000b_branchynet_reproduce_resnet56_cifar10.npz")
    parser.add_argument("--output", default="results/final_threshold_vs_dual_exit_99_001_summary.json")
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument(
        "--grid-quantiles",
        nargs="*",
        type=float,
        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99],
    )
    args = parser.parse_args()

    data = np.load(Path(args.trace), allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]

    final_idx = correct.shape[1] - 1
    final_correct = correct[:, final_idx]
    final_confidence = confidence[:, final_idx]
    final_threshold = calibrate_final_threshold(final_correct, final_confidence, args.target_accuracy)
    final_reliable = final_correct & (final_confidence >= final_threshold["threshold"])

    final_only = evaluate_policy(
        correct,
        confidence,
        costs,
        final_reliable,
        final_threshold["threshold"],
        lower0=-1.0,
        upper0=2.0,
        lower1=-1.0,
        upper1=2.0,
    )

    upper0_values = thresholds_from_quantiles(confidence[:, 0], args.grid_quantiles)
    upper1_values = thresholds_from_quantiles(confidence[:, 1], args.grid_quantiles)
    lower0_values = thresholds_from_quantiles(confidence[:, 0], args.grid_quantiles)
    lower1_values = thresholds_from_quantiles(confidence[:, 1], args.grid_quantiles)

    upper_only_rows = []
    for upper0 in upper0_values:
        for upper1 in upper1_values:
            upper_only_rows.append(
                evaluate_policy(
                    correct,
                    confidence,
                    costs,
                    final_reliable,
                    final_threshold["threshold"],
                    lower0=-1.0,
                    upper0=upper0,
                    lower1=-1.0,
                    upper1=upper1,
                )
            )

    dual_rows = []
    for lower0 in lower0_values:
        for upper0 in upper0_values:
            if lower0 >= upper0:
                continue
            for lower1 in lower1_values:
                for upper1 in upper1_values:
                    if lower1 >= upper1:
                        continue
                    dual_rows.append(
                        evaluate_policy(
                            correct,
                            confidence,
                            costs,
                            final_reliable,
                            final_threshold["threshold"],
                            lower0=lower0,
                            upper0=upper0,
                            lower1=lower1,
                            upper1=upper1,
                        )
                    )

    payload = {
        "purpose": "Compare final-only 99%-accurate confidence filtering with upper-only and dual-sided early exit.",
        "trace": args.trace,
        "exit_names": exit_names,
        "exit_costs": [round_float(x) for x in costs],
        "target_accuracy": args.target_accuracy,
        "final_accuracy_without_reject": round_float(float(final_correct.mean())),
        "final_confidence_threshold": final_threshold,
        "definitions": {
            "accepted_accuracy": "accuracy among samples that receive a class label",
            "reject": "no reliable class label is emitted",
            "final_reliable": "sample accepted by the final-only calibrated threshold and correctly classified at the final exit",
            "lost_final_reliable_rate": "fraction of all samples that would be accepted correctly by final-only but are early-rejected by the policy",
            "avg_cost": "mean normalized exit cost; final-only is 1.0",
        },
        "policies": {
            "final_only": final_only,
            "upper_only_best_cost": choose_best(upper_only_rows, args.target_accuracy),
            "dual_side_best_cost": choose_best(dual_rows, args.target_accuracy),
            "dual_side_best_cost_lost_final_reliable_le_1pct": choose_best(dual_rows, args.target_accuracy, max_lost=0.01),
            "dual_side_best_cost_lost_final_reliable_le_2pct": choose_best(dual_rows, args.target_accuracy, max_lost=0.02),
            "dual_side_best_cost_lost_final_reliable_le_5pct": choose_best(dual_rows, args.target_accuracy, max_lost=0.05),
        },
        "sweep_counts": {
            "upper_only": len(upper_only_rows),
            "dual_side": len(dual_rows),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
