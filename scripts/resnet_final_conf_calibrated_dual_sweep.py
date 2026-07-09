"""Calibrate final self-confidence, then rerun dual-threshold sweep.

The previous dual-threshold sweep used a quantile-based definition of
"high-confidence final correct". This script replaces that with a requirement:

* choose the lowest final-exit self-confidence threshold whose accepted samples
  meet a target accuracy such as 95%, 98%, or 99%
* treat a rejected sample as "lost reliable good" only if the final exit would
  have been correct and above that calibrated threshold
* sweep exit0/exit1 lower and upper thresholds under those calibrated meanings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_thresholds(values: np.ndarray, quantiles: list[float]) -> list[float]:
    return sorted({float(x) for x in np.quantile(values, quantiles)})


def calibrate_final_thresholds(final_correct: np.ndarray, final_confidence: np.ndarray, targets: list[float]) -> dict[str, Any]:
    thresholds = sorted({float(x) for x in final_confidence})
    rows = []
    for threshold in thresholds:
        accept = final_confidence >= threshold
        if not accept.any():
            continue
        rows.append({
            "threshold": threshold,
            "accept_rate": float(accept.mean()),
            "accepted_accuracy": float(final_correct[accept].mean()),
            "false_accept_rate": float((~final_correct[accept]).mean()),
        })

    calibrated: dict[str, Any] = {}
    for target in targets:
        valid = [row for row in rows if row["accepted_accuracy"] >= target]
        key = f"target_accuracy_{target:.3f}"
        if valid:
            # Maximize coverage under the accuracy requirement. Ties prefer a
            # lower threshold and then higher observed accuracy.
            best = max(valid, key=lambda row: (row["accept_rate"], -row["threshold"], row["accepted_accuracy"]))
            calibrated[key] = {
                "target_accuracy": target,
                "threshold": round(float(best["threshold"]), 6),
                "accept_rate": round(float(best["accept_rate"]), 6),
                "accepted_accuracy": round(float(best["accepted_accuracy"]), 6),
                "false_accept_rate": round(float(best["false_accept_rate"]), 6),
            }
        else:
            calibrated[key] = None
    return {
        "curve_sample": [
            {
                "threshold": round(float(row["threshold"]), 6),
                "accept_rate": round(float(row["accept_rate"]), 6),
                "accepted_accuracy": round(float(row["accepted_accuracy"]), 6),
                "false_accept_rate": round(float(row["false_accept_rate"]), 6),
            }
            for row in rows[:: max(1, len(rows) // 40)]
        ],
        "calibrated_thresholds": calibrated,
    }


def evaluate_policy(
    correct: np.ndarray,
    confidence: np.ndarray,
    costs: np.ndarray,
    lower0: float,
    upper0: float,
    lower1: float,
    upper1: float,
    reliable_final_threshold: float,
) -> dict[str, Any]:
    n = correct.shape[0]
    final_idx = correct.shape[1] - 1
    terminal = np.full(n, "final", dtype=object)
    terminal_exit = np.full(n, final_idx, dtype=np.int16)
    accepted_correct = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)

    accept0 = confidence[:, 0] >= upper0
    reject0 = confidence[:, 0] <= lower0
    terminal[accept0] = "accept0"
    terminal_exit[accept0] = 0
    accepted_correct[accept0] = correct[accept0, 0]
    reject0_only = reject0 & ~accept0
    terminal[reject0_only] = "reject0"
    terminal_exit[reject0_only] = 0
    rejected[reject0_only] = True

    unresolved = ~(accept0 | reject0)
    accept1 = unresolved & (confidence[:, 1] >= upper1)
    reject1 = unresolved & (confidence[:, 1] <= lower1)
    terminal[accept1] = "accept1"
    terminal_exit[accept1] = 1
    accepted_correct[accept1] = correct[accept1, 1]
    terminal[reject1] = "reject1"
    terminal_exit[reject1] = 1
    rejected[reject1] = True

    final_mask = terminal == "final"
    accepted_correct[final_mask] = correct[final_mask, final_idx]

    accepted = ~rejected
    cost = costs[terminal_exit]
    reliable_final_correct = correct[:, final_idx] & (confidence[:, final_idx] >= reliable_final_threshold)

    counts = {name: int((terminal == name).sum()) for name in ["accept0", "reject0", "accept1", "reject1", "final"]}
    accept_count = int(accepted.sum())
    reject_count = int(rejected.sum())
    if accept_count:
        accepted_accuracy = float(accepted_correct[accepted].mean())
        false_accept_rate = float((~accepted_correct[accepted]).mean())
    else:
        accepted_accuracy = None
        false_accept_rate = None
    if reject_count:
        rejected_final_correct_rate = float(correct[rejected, final_idx].mean())
        rejected_reliable_final_correct_rate = float(reliable_final_correct[rejected].mean())
    else:
        rejected_final_correct_rate = None
        rejected_reliable_final_correct_rate = None
    lost_reliable_final_correct_rate = float((rejected & reliable_final_correct).mean())

    return {
        "thresholds": {
            "lower0": round(float(lower0), 6),
            "upper0": round(float(upper0), 6),
            "lower1": round(float(lower1), 6),
            "upper1": round(float(upper1), 6),
        },
        "counts": counts,
        "rates": {name: round(count / n, 6) for name, count in counts.items()},
        "accept_rate": round(float(accepted.mean()), 6),
        "reject_rate": round(float(rejected.mean()), 6),
        "final_rate": round(float(final_mask.mean()), 6),
        "early_terminal_rate": round(float((~final_mask).mean()), 6),
        "avg_cost": round(float(cost.mean()), 6),
        "accepted_accuracy": None if accepted_accuracy is None else round(accepted_accuracy, 6),
        "false_accept_rate": None if false_accept_rate is None else round(false_accept_rate, 6),
        "rejected_final_correct_rate": None if rejected_final_correct_rate is None else round(rejected_final_correct_rate, 6),
        "rejected_reliable_final_correct_rate": None if rejected_reliable_final_correct_rate is None else round(rejected_reliable_final_correct_rate, 6),
        "lost_reliable_final_correct_rate": round(lost_reliable_final_correct_rate, 6),
    }


def sweep(
    correct: np.ndarray,
    confidence: np.ndarray,
    costs: np.ndarray,
    grid_quantiles: list[float],
    reliable_final_threshold: float,
) -> list[dict[str, Any]]:
    lower0_values = make_thresholds(confidence[:, 0], grid_quantiles)
    upper0_values = make_thresholds(confidence[:, 0], grid_quantiles)
    lower1_values = make_thresholds(confidence[:, 1], grid_quantiles)
    upper1_values = make_thresholds(confidence[:, 1], grid_quantiles)
    rows = []
    for lower0 in lower0_values:
        for upper0 in upper0_values:
            if lower0 >= upper0:
                continue
            for lower1 in lower1_values:
                for upper1 in upper1_values:
                    if lower1 >= upper1:
                        continue
                    rows.append(evaluate_policy(correct, confidence, costs, lower0, upper0, lower1, upper1, reliable_final_threshold))
    return rows


def high_conf_only_baseline(correct: np.ndarray, confidence: np.ndarray, costs: np.ndarray) -> dict[str, Any]:
    final_acc = float(correct[:, -1].mean())
    rows = []
    for upper0 in make_thresholds(confidence[:, 0], [0.6, 0.7, 0.8, 0.9, 0.95]):
        for upper1 in make_thresholds(confidence[:, 1], [0.6, 0.7, 0.8, 0.9, 0.95]):
            row = evaluate_policy(correct, confidence, costs, -1.0, upper0, -1.0, upper1, 2.0)
            row["accuracy_drop_vs_final"] = None if row["accepted_accuracy"] is None else round(final_acc - row["accepted_accuracy"], 6)
            rows.append(row)
    valid = [row for row in rows if row["accepted_accuracy"] is not None and row["accepted_accuracy"] >= final_acc - 0.01]
    return min(valid, key=lambda row: (row["avg_cost"], row["final_rate"])) if valid else max(rows, key=lambda row: row["accepted_accuracy"] or 0.0)


def compact_rows(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    keys = [
        "thresholds",
        "rates",
        "accept_rate",
        "reject_rate",
        "final_rate",
        "early_terminal_rate",
        "avg_cost",
        "accepted_accuracy",
        "false_accept_rate",
        "rejected_reliable_final_correct_rate",
        "lost_reliable_final_correct_rate",
    ]
    return [{key: row[key] for key in keys} for row in rows[:limit]]


def best_under_constraints(rows: list[dict[str, Any]], false_accept_targets: list[float], lost_reliable_targets: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fa in false_accept_targets:
        for lost in lost_reliable_targets:
            key = f"false_accept_le_{fa:.2f}_lost_reliable_final_correct_overall_le_{lost:.2f}"
            valid = [
                row for row in rows
                if row["false_accept_rate"] is not None
                and row["false_accept_rate"] <= fa
                and row["lost_reliable_final_correct_rate"] <= lost
            ]
            out[key] = None if not valid else max(valid, key=lambda row: (row["early_terminal_rate"], -row["avg_cost"], row["accept_rate"]))
    return out


def diagnostic_rankings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["false_accept_rate"] is not None]
    return {
        "lowest_false_accept": compact_rows(sorted(valid, key=lambda row: (row["false_accept_rate"], row["avg_cost"]))),
        "lowest_lost_reliable_good": compact_rows(sorted(valid, key=lambda row: (row["lost_reliable_final_correct_rate"], row["avg_cost"]))),
        "lowest_cost_under_false_accept_5pct": compact_rows(
            sorted(
                [row for row in valid if row["false_accept_rate"] <= 0.05],
                key=lambda row: (row["avg_cost"], -row["early_terminal_rate"]),
            )
        ),
        "highest_early_terminal_under_false_accept_5pct": compact_rows(
            sorted(
                [row for row in valid if row["false_accept_rate"] <= 0.05],
                key=lambda row: (-row["early_terminal_rate"], row["avg_cost"]),
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="results/0000b_branchynet_reproduce_resnet56_cifar10.npz")
    parser.add_argument("--output", default="results/resnet_final_conf_calibrated_dual_sweep_001_summary.json")
    parser.add_argument("--target-accuracies", nargs="*", type=float, default=[0.95, 0.98, 0.99])
    parser.add_argument("--grid-quantiles", nargs="*", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    args = parser.parse_args()

    data = np.load(Path(args.trace), allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]

    final_correct = correct[:, -1]
    final_confidence = confidence[:, -1]
    calibration = calibrate_final_thresholds(final_correct, final_confidence, args.target_accuracies)
    baseline = high_conf_only_baseline(correct, confidence, costs)

    sweeps: dict[str, Any] = {}
    for key, threshold_info in calibration["calibrated_thresholds"].items():
        if threshold_info is None:
            sweeps[key] = None
            continue
        rows = sweep(correct, confidence, costs, args.grid_quantiles, threshold_info["threshold"])
        sweeps[key] = {
            "reliable_final_threshold": threshold_info,
            "sweep_count": len(rows),
            "diagnostic_rankings": diagnostic_rankings(rows),
            "best_under_constraints": best_under_constraints(
                rows,
                false_accept_targets=[0.02, 0.05, 0.10],
                lost_reliable_targets=[0.02, 0.05, 0.10, 0.20, 0.30],
            ),
        }

    payload = {
        "purpose": "Calibrate final self-confidence from required accuracy, then evaluate dual-threshold early accept/reject.",
        "trace": args.trace,
        "exit_names": exit_names,
        "exit_costs": [round(float(x), 6) for x in costs],
        "final_accuracy": round(float(final_correct.mean()), 6),
        "calibration": calibration,
        "high_conf_only_baseline": baseline,
        "calibrated_dual_threshold_sweeps": sweeps,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stdout_sweeps = {}
    for key, sweep_info in sweeps.items():
        if sweep_info is None:
            stdout_sweeps[key] = None
            continue
        stdout_sweeps[key] = {
            "reliable_final_threshold": sweep_info["reliable_final_threshold"],
            "best_under_constraints": sweep_info["best_under_constraints"],
        }
    print(json.dumps({
        "purpose": payload["purpose"],
        "final_accuracy": payload["final_accuracy"],
        "calibrated_thresholds": calibration["calibrated_thresholds"],
        "high_conf_only_baseline": baseline,
        "calibrated_dual_threshold_sweeps": stdout_sweeps,
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
