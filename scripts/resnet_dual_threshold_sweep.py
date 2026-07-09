"""Sweep dual-threshold early exits for ResNet BranchyNet.

At exit0 and exit1:

* confidence >= upper threshold: accept/classify early
* confidence <= lower threshold: reject early as unsafe / inspection-fail side
* otherwise: continue to the next exit

The final exit is treated as a forced terminal classification in this first
simulation. The key question is whether early low-confidence rejection reduces
later-stage load without discarding many samples that the final exit would have
rescued into high-confidence correct decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_thresholds(values: np.ndarray, quantiles: list[float]) -> list[float]:
    return sorted({float(x) for x in np.quantile(values, quantiles)})


def evaluate_policy(
    correct: np.ndarray,
    confidence: np.ndarray,
    costs: np.ndarray,
    lower0: float,
    upper0: float,
    lower1: float,
    upper1: float,
    final_high_threshold: float,
) -> dict[str, Any]:
    n = correct.shape[0]
    final_idx = correct.shape[1] - 1
    terminal = np.full(n, "final", dtype=object)
    terminal_exit = np.full(n, final_idx, dtype=np.int16)
    accepted_correct = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)

    # exit0 decision
    accept0 = confidence[:, 0] >= upper0
    reject0 = confidence[:, 0] <= lower0
    terminal[accept0] = "accept0"
    terminal_exit[accept0] = 0
    accepted_correct[accept0] = correct[accept0, 0]
    rejected[reject0 & ~accept0] = True
    terminal[reject0 & ~accept0] = "reject0"
    terminal_exit[reject0 & ~accept0] = 0

    unresolved = ~(accept0 | reject0)
    # exit1 decision
    accept1 = unresolved & (confidence[:, 1] >= upper1)
    reject1 = unresolved & (confidence[:, 1] <= lower1)
    terminal[accept1] = "accept1"
    terminal_exit[accept1] = 1
    accepted_correct[accept1] = correct[accept1, 1]
    rejected[reject1] = True
    terminal[reject1] = "reject1"
    terminal_exit[reject1] = 1

    final_mask = terminal == "final"
    accepted_correct[final_mask] = correct[final_mask, final_idx]

    accepted = ~rejected
    cost = costs[terminal_exit]
    final_high_correct = correct[:, final_idx] & (confidence[:, final_idx] >= final_high_threshold)

    reject_count = int(rejected.sum())
    accept_count = int(accepted.sum())
    if reject_count:
        rejected_final_correct = float(correct[rejected, final_idx].mean())
        rejected_final_high_correct = float(final_high_correct[rejected].mean())
    else:
        rejected_final_correct = None
        rejected_final_high_correct = None

    if accept_count:
        accepted_accuracy = float(accepted_correct[accepted].mean())
        false_accept_rate = float((~accepted_correct[accepted]).mean())
    else:
        accepted_accuracy = None
        false_accept_rate = None

    counts = {name: int((terminal == name).sum()) for name in ["accept0", "reject0", "accept1", "reject1", "final"]}
    return {
        "thresholds": {
            "lower0": round(lower0, 6),
            "upper0": round(upper0, 6),
            "lower1": round(lower1, 6),
            "upper1": round(upper1, 6),
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
        "rejected_final_correct_rate": None if rejected_final_correct is None else round(rejected_final_correct, 6),
        "rejected_final_high_conf_correct_rate": None if rejected_final_high_correct is None else round(rejected_final_high_correct, 6),
    }


def baseline_branchynet_like(correct: np.ndarray, confidence: np.ndarray, costs: np.ndarray) -> dict[str, Any]:
    """Simple high-confidence-only baseline tuned with the same upper grid."""
    final_acc = float(correct[:, -1].mean())
    upper0_values = make_thresholds(confidence[:, 0], [0.6, 0.7, 0.8, 0.9, 0.95])
    upper1_values = make_thresholds(confidence[:, 1], [0.6, 0.7, 0.8, 0.9, 0.95])
    rows = []
    for upper0 in upper0_values:
        for upper1 in upper1_values:
            # Lower thresholds below zero disable reject.
            row = evaluate_policy(correct, confidence, costs, -1.0, upper0, -1.0, upper1, 2.0)
            if row["accepted_accuracy"] is not None:
                row["accuracy_drop_vs_final"] = round(final_acc - row["accepted_accuracy"], 6)
            rows.append(row)
    valid = [row for row in rows if row["accepted_accuracy"] is not None and row["accepted_accuracy"] >= final_acc - 0.01]
    best = min(valid, key=lambda row: (row["avg_cost"], row["final_rate"])) if valid else max(rows, key=lambda row: row["accepted_accuracy"] or 0)
    return best


def sweep(correct: np.ndarray, confidence: np.ndarray, costs: np.ndarray, grid_quantiles: list[float], final_high_quantile: float) -> list[dict[str, Any]]:
    lower0_values = make_thresholds(confidence[:, 0], grid_quantiles)
    upper0_values = make_thresholds(confidence[:, 0], grid_quantiles)
    lower1_values = make_thresholds(confidence[:, 1], grid_quantiles)
    upper1_values = make_thresholds(confidence[:, 1], grid_quantiles)
    final_high_threshold = float(np.quantile(confidence[:, -1], final_high_quantile))

    rows = []
    for lower0 in lower0_values:
        for upper0 in upper0_values:
            if lower0 >= upper0:
                continue
            for lower1 in lower1_values:
                for upper1 in upper1_values:
                    if lower1 >= upper1:
                        continue
                    rows.append(evaluate_policy(
                        correct,
                        confidence,
                        costs,
                        lower0,
                        upper0,
                        lower1,
                        upper1,
                        final_high_threshold,
                    ))
    return rows


def best_under_constraints(rows: list[dict[str, Any]], false_accept_targets: list[float], lost_good_targets: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fa in false_accept_targets:
        for lost in lost_good_targets:
            valid = [
                row for row in rows
                if row["false_accept_rate"] is not None
                and row["rejected_final_high_conf_correct_rate"] is not None
                and row["false_accept_rate"] <= fa
                and row["rejected_final_high_conf_correct_rate"] <= lost
            ]
            key = f"false_accept_le_{fa:.2f}_rejected_final_high_correct_le_{lost:.2f}"
            if valid:
                out[key] = max(valid, key=lambda row: (row["early_terminal_rate"], -row["avg_cost"], row["accept_rate"]))
            else:
                out[key] = None
    return out


def pareto_front(rows: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["false_accept_rate"] is not None and row["rejected_final_high_conf_correct_rate"] is not None]
    # Keep concise candidates: high early-terminal, low false accept, low lost-good, low cost.
    scored = sorted(
        valid,
        key=lambda row: (
            -row["early_terminal_rate"],
            row["false_accept_rate"],
            row["rejected_final_high_conf_correct_rate"],
            row["avg_cost"],
        ),
    )
    kept = []
    for row in scored:
        dominated = False
        for other in kept:
            if (
                other["early_terminal_rate"] >= row["early_terminal_rate"]
                and other["false_accept_rate"] <= row["false_accept_rate"]
                and other["rejected_final_high_conf_correct_rate"] <= row["rejected_final_high_conf_correct_rate"]
                and other["avg_cost"] <= row["avg_cost"]
            ):
                dominated = True
                break
        if not dominated:
            kept.append(row)
        if len(kept) >= limit:
            break
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="results/0000b_branchynet_reproduce_resnet56_cifar10.npz")
    parser.add_argument("--output", default="results/resnet_dual_threshold_sweep_001_summary.json")
    parser.add_argument("--grid-quantiles", nargs="*", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--final-high-quantile", type=float, default=0.7)
    args = parser.parse_args()

    data = np.load(Path(args.trace), allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]
    final_acc = float(correct[:, -1].mean())

    rows = sweep(correct, confidence, costs, args.grid_quantiles, args.final_high_quantile)
    baseline = baseline_branchynet_like(correct, confidence, costs)
    best = best_under_constraints(rows, false_accept_targets=[0.02, 0.05, 0.10], lost_good_targets=[0.02, 0.05, 0.10])
    front = pareto_front(rows, limit=50)

    payload = {
        "purpose": "Sweep dual-threshold early exit: high self-confidence accept, low self-confidence reject, middle continues.",
        "interpretation": {
            "reject_is_safe_when": "rejected_final_high_conf_correct_rate is low; few rejected samples would have become high-confidence correct at final.",
            "useful_when": "early_terminal_rate rises and final_rate/avg_cost falls while false_accept_rate and rejected_final_high_conf_correct_rate stay within acceptable bounds.",
        },
        "trace": args.trace,
        "exit_names": exit_names,
        "exit_costs": [round(float(x), 6) for x in costs],
        "final_accuracy": round(final_acc, 6),
        "final_high_conf_quantile": args.final_high_quantile,
        "final_high_conf_threshold": round(float(np.quantile(confidence[:, -1], args.final_high_quantile)), 6),
        "grid_quantiles": args.grid_quantiles,
        "sweep_count": len(rows),
        "high_conf_only_baseline": baseline,
        "best_under_constraints": best,
        "pareto_front": front,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "purpose": payload["purpose"],
        "final_accuracy": payload["final_accuracy"],
        "sweep_count": payload["sweep_count"],
        "high_conf_only_baseline": baseline,
        "best_under_constraints": best,
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
