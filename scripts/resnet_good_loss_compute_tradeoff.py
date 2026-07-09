"""Evaluate the tradeoff between good-product loss and compute cost.

Definitions used in this experiment:

* self-confidence: max softmax probability at an exit.
* reliable good proxy: a sample that the final exit classifies correctly and
  whose final self-confidence exceeds a calibrated threshold for a required
  accepted accuracy such as 95%, 98%, or 99%.
* good loss rate: fraction of all samples that are reliable-good by the above
  proxy but are rejected by an early lower-threshold decision.
* false accept rate: fraction of accepted samples whose terminal prediction is
  wrong. This is reported both per accepted sample and over all samples.

The point is not to call reject "re-inspection". Reject is treated as a real
loss/cost. The script asks whether dual-threshold early exit can lower compute
enough to compensate for that good-product loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def thresholds(values: np.ndarray, quantiles: list[float]) -> list[float]:
    return sorted({float(x) for x in np.quantile(values, quantiles)})


def calibrate_reliable_good(final_correct: np.ndarray, final_conf: np.ndarray, target_accuracy: float) -> dict[str, Any]:
    rows = []
    for threshold in sorted({float(x) for x in final_conf}):
        accept = final_conf >= threshold
        if not accept.any():
            continue
        accuracy = float(final_correct[accept].mean())
        rows.append({
            "threshold": threshold,
            "accept_rate": float(accept.mean()),
            "accepted_accuracy": accuracy,
            "false_accept_rate": 1.0 - accuracy,
        })
    valid = [row for row in rows if row["accepted_accuracy"] >= target_accuracy]
    if not valid:
        raise ValueError(f"No final self-confidence threshold satisfies target accuracy {target_accuracy}")
    best = max(valid, key=lambda row: (row["accept_rate"], -row["threshold"]))
    return {
        "target_accuracy": target_accuracy,
        "threshold": round(float(best["threshold"]), 6),
        "reliable_good_rate": round(float(best["accept_rate"]), 6),
        "accepted_accuracy": round(float(best["accepted_accuracy"]), 6),
        "false_accept_rate": round(float(best["false_accept_rate"]), 6),
    }


def evaluate(
    correct: np.ndarray,
    conf: np.ndarray,
    costs: np.ndarray,
    reliable_good: np.ndarray,
    lower0: float,
    upper0: float,
    lower1: float,
    upper1: float,
) -> dict[str, Any]:
    n = correct.shape[0]
    final_idx = correct.shape[1] - 1
    terminal = np.full(n, "final", dtype=object)
    terminal_exit = np.full(n, final_idx, dtype=np.int16)
    terminal_correct = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)

    accept0 = conf[:, 0] >= upper0
    reject0 = conf[:, 0] <= lower0
    terminal[accept0] = "accept0"
    terminal_exit[accept0] = 0
    terminal_correct[accept0] = correct[accept0, 0]

    reject0_only = reject0 & ~accept0
    terminal[reject0_only] = "reject0"
    terminal_exit[reject0_only] = 0
    rejected[reject0_only] = True

    unresolved = ~(accept0 | reject0)
    accept1 = unresolved & (conf[:, 1] >= upper1)
    reject1 = unresolved & (conf[:, 1] <= lower1)
    terminal[accept1] = "accept1"
    terminal_exit[accept1] = 1
    terminal_correct[accept1] = correct[accept1, 1]

    terminal[reject1] = "reject1"
    terminal_exit[reject1] = 1
    rejected[reject1] = True

    final = terminal == "final"
    terminal_correct[final] = correct[final, final_idx]
    accepted = ~rejected
    false_accept = accepted & ~terminal_correct
    good_loss = rejected & reliable_good

    counts = {name: int((terminal == name).sum()) for name in ["accept0", "reject0", "accept1", "reject1", "final"]}
    accepted_count = int(accepted.sum())
    return {
        "thresholds": {
            "lower0": round(float(lower0), 6),
            "upper0": round(float(upper0), 6),
            "lower1": round(float(lower1), 6),
            "upper1": round(float(upper1), 6),
        },
        "counts": counts,
        "rates": {key: round(value / n, 6) for key, value in counts.items()},
        "avg_compute_cost": round(float(costs[terminal_exit].mean()), 6),
        "accept_rate": round(float(accepted.mean()), 6),
        "reject_rate": round(float(rejected.mean()), 6),
        "final_rate": round(float(final.mean()), 6),
        "good_loss_rate": round(float(good_loss.mean()), 6),
        "false_accept_overall_rate": round(float(false_accept.mean()), 6),
        "false_accept_among_accepted_rate": None if accepted_count == 0 else round(float(false_accept[accepted].mean()), 6),
        "accepted_accuracy": None if accepted_count == 0 else round(float(terminal_correct[accepted].mean()), 6),
    }


def high_conf_only_baseline(correct: np.ndarray, conf: np.ndarray, costs: np.ndarray, reliable_good: np.ndarray) -> dict[str, Any]:
    rows = []
    for upper0 in thresholds(conf[:, 0], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
        for upper1 in thresholds(conf[:, 1], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
            rows.append(evaluate(correct, conf, costs, reliable_good, -1.0, upper0, -1.0, upper1))
    return min(rows, key=lambda row: (row["avg_compute_cost"], row["false_accept_overall_rate"]))


def sweep(correct: np.ndarray, conf: np.ndarray, costs: np.ndarray, reliable_good: np.ndarray, grid: list[float]) -> list[dict[str, Any]]:
    lower0_values = thresholds(conf[:, 0], grid)
    upper0_values = thresholds(conf[:, 0], grid)
    lower1_values = thresholds(conf[:, 1], grid)
    upper1_values = thresholds(conf[:, 1], grid)
    rows = []
    for lower0 in lower0_values:
        for upper0 in upper0_values:
            if lower0 >= upper0:
                continue
            for lower1 in lower1_values:
                for upper1 in upper1_values:
                    if lower1 >= upper1:
                        continue
                    rows.append(evaluate(correct, conf, costs, reliable_good, lower0, upper0, lower1, upper1))
    return rows


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for row in sorted(rows, key=lambda x: (x["avg_compute_cost"], x["good_loss_rate"], x["false_accept_overall_rate"])):
        dominated = False
        for other in kept:
            if (
                other["avg_compute_cost"] <= row["avg_compute_cost"]
                and other["good_loss_rate"] <= row["good_loss_rate"]
                and other["false_accept_overall_rate"] <= row["false_accept_overall_rate"]
            ):
                dominated = True
                break
        if not dominated:
            kept.append(row)
    return kept


def best_by_loss_budget(rows: list[dict[str, Any]], loss_budgets: list[float], false_accept_budgets: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for loss in loss_budgets:
        for fa in false_accept_budgets:
            key = f"good_loss_le_{loss:.3f}_false_accept_overall_le_{fa:.3f}"
            valid = [
                row for row in rows
                if row["good_loss_rate"] <= loss and row["false_accept_overall_rate"] <= fa
            ]
            out[key] = None if not valid else min(valid, key=lambda row: (row["avg_compute_cost"], -row["accept_rate"]))
    return out


def best_by_weighted_cost(rows: list[dict[str, Any]], good_loss_weights: list[float], false_accept_weight: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for weight in good_loss_weights:
        key = f"compute_plus_{weight:g}x_good_loss_plus_{false_accept_weight:g}x_false_accept"
        best = min(
            rows,
            key=lambda row: (
                row["avg_compute_cost"]
                + weight * row["good_loss_rate"]
                + false_accept_weight * row["false_accept_overall_rate"]
            ),
        )
        out[key] = {
            "total_cost": round(
                float(best["avg_compute_cost"] + weight * best["good_loss_rate"] + false_accept_weight * best["false_accept_overall_rate"]),
                6,
            ),
            "policy": best,
        }
    return out


def compact(rows: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="results/0000b_branchynet_reproduce_resnet56_cifar10.npz")
    parser.add_argument("--output", default="results/resnet_good_loss_compute_tradeoff_001_summary.json")
    parser.add_argument("--target-accuracies", nargs="*", type=float, default=[0.95, 0.98, 0.99])
    parser.add_argument("--grid-quantiles", nargs="*", type=float, default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    args = parser.parse_args()

    data = np.load(Path(args.trace), allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    conf = np.asarray(data["confidence"], dtype=float)
    costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]
    final_correct = correct[:, -1]
    final_conf = conf[:, -1]

    scenarios: dict[str, Any] = {}
    for target in args.target_accuracies:
        calibration = calibrate_reliable_good(final_correct, final_conf, target)
        reliable_good = final_correct & (final_conf >= calibration["threshold"])
        rows = sweep(correct, conf, costs, reliable_good, args.grid_quantiles)
        front = pareto(rows)
        scenarios[f"target_accuracy_{target:.3f}"] = {
            "reliable_good_definition": calibration,
            "high_conf_only_baseline": high_conf_only_baseline(correct, conf, costs, reliable_good),
            "best_by_loss_budget": best_by_loss_budget(
                rows,
                loss_budgets=[0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30],
                false_accept_budgets=[0.02, 0.05, 0.10],
            ),
            "best_by_weighted_cost": best_by_weighted_cost(
                rows,
                good_loss_weights=[0, 1, 2, 5, 10, 20, 50],
                false_accept_weight=10,
            ),
            "pareto_front_count": len(front),
            "pareto_front_sample": compact(front, 60),
        }

    payload = {
        "purpose": "Evaluate good-product loss versus compute cost for dual-threshold ResNet early exit.",
        "definitions": {
            "self_confidence": "maximum softmax probability at each exit",
            "reliable_good_proxy": "final exit is correct and final self-confidence meets the calibrated target-accuracy threshold",
            "good_loss_rate": "fraction of all samples that are reliable-good but rejected early",
            "false_accept_overall_rate": "fraction of all samples that are accepted with a wrong terminal prediction",
            "avg_compute_cost": "mean normalized exit cost of the terminal decision",
        },
        "trace": args.trace,
        "exit_names": exit_names,
        "exit_costs": [round(float(x), 6) for x in costs],
        "final_accuracy": round(float(final_correct.mean()), 6),
        "grid_quantiles": args.grid_quantiles,
        "scenarios": scenarios,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "purpose": payload["purpose"],
        "final_accuracy": payload["final_accuracy"],
        "scenarios": {
            key: {
                "reliable_good_definition": value["reliable_good_definition"],
                "best_by_loss_budget": value["best_by_loss_budget"],
            }
            for key, value in scenarios.items()
        },
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
