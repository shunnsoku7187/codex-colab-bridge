"""Probe CIFAR-10 early-exit traces for inspection-oriented research value.

The goal is not to beat BranchyNet on its own average accuracy/cost metric.
Instead, this script searches for evidence that an inspection-style system can
claim different value axes:

* deadline-aware accept/reinspect behavior
* final-risk rejection of samples that even the final classifier may fail
* FPGA/profile concerns such as tail cost and distribution-dependent exit rates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TRACES = [
    "results/0000b_branchynet_reproduce_resnet56_cifar10.npz",
    "results/0001_branchynet_mobilenet_cifar10_retry.npz",
]


def quantile_thresholds(values: np.ndarray, steps: int = 21) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, steps)))


def selected_exit(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    n, exits = correct.shape
    selected = np.full(n, exits - 1, dtype=int)
    for exit_idx, threshold in enumerate(thresholds):
        mask = (selected == exits - 1) & (confidence[:, exit_idx] >= threshold)
        selected[mask] = exit_idx
    rows = np.arange(n)
    selected_correct = correct[rows, selected]
    selected_cost = exit_costs[selected]
    counts = {str(i): int((selected == i).sum()) for i in range(exits)}
    return {
        "accuracy": round(float(selected_correct.mean()), 6),
        "avg_cost": round(float(selected_cost.mean()), 6),
        "p90_cost": round(float(np.quantile(selected_cost, 0.9)), 6),
        "p99_cost": round(float(np.quantile(selected_cost, 0.99)), 6),
        "max_cost": round(float(selected_cost.max()), 6),
        "exit_counts": counts,
        "exit_rates": {k: round(v / n, 6) for k, v in counts.items()},
        "thresholds": [round(float(x), 6) for x in thresholds],
    }


def branchynet_curve(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray) -> list[dict[str, Any]]:
    final_acc = float(correct[:, -1].mean())
    thresholds0 = quantile_thresholds(confidence[:, 0], 25)
    thresholds1 = quantile_thresholds(confidence[:, 1], 25)
    tolerances = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05]
    candidates = []
    for t0 in thresholds0:
        for t1 in thresholds1:
            metrics = selected_exit(correct, confidence, exit_costs, [float(t0), float(t1)])
            metrics["accuracy_drop"] = round(final_acc - metrics["accuracy"], 6)
            candidates.append(metrics)
    curve = []
    for tolerance in tolerances:
        valid = [m for m in candidates if m["accuracy"] >= final_acc - tolerance]
        if not valid:
            curve.append({"max_accuracy_drop": tolerance, "found": False})
            continue
        best = min(valid, key=lambda m: (m["avg_cost"], m["p99_cost"], -m["accuracy"]))
        curve.append({"max_accuracy_drop": tolerance, "found": True, **best})
    return curve


def deadline_accept_reinspect(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray, deadline_exit: int) -> list[dict[str, Any]]:
    thresholds = quantile_thresholds(confidence[:, deadline_exit], 21)
    rows = []
    for threshold in thresholds:
        accept = confidence[:, deadline_exit] >= threshold
        accepted = int(accept.sum())
        if accepted:
            accepted_accuracy = float(correct[accept, deadline_exit].mean())
            false_accept_rate = float((~correct[accept, deadline_exit]).mean())
        else:
            accepted_accuracy = None
            false_accept_rate = None
        rows.append({
            "deadline_exit": int(deadline_exit),
            "deadline_cost": round(float(exit_costs[deadline_exit]), 6),
            "threshold": round(float(threshold), 6),
            "accept_rate": round(float(accept.mean()), 6),
            "reinspect_rate": round(float((~accept).mean()), 6),
            "accepted_accuracy": None if accepted_accuracy is None else round(accepted_accuracy, 6),
            "false_accept_rate": None if false_accept_rate is None else round(false_accept_rate, 6),
        })
    return rows


def best_deadline_points(curve: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [0.01, 0.02, 0.05, 0.10]
    out: dict[str, Any] = {}
    for target in targets:
        valid = [
            row for row in curve
            if row["false_accept_rate"] is not None and row["false_accept_rate"] <= target
        ]
        if valid:
            out[f"false_accept_le_{target:.2f}"] = max(valid, key=lambda row: row["accept_rate"])
        else:
            out[f"false_accept_le_{target:.2f}"] = None
    return out


def final_risk_reject(correct: np.ndarray, confidence: np.ndarray) -> list[dict[str, Any]]:
    final_correct = correct[:, -1]
    final_conf = confidence[:, -1]
    thresholds = quantile_thresholds(final_conf, 21)
    rows = []
    for threshold in thresholds:
        accept = final_conf >= threshold
        accepted = int(accept.sum())
        if accepted:
            accepted_accuracy = float(final_correct[accept].mean())
            false_accept_rate = float((~final_correct[accept]).mean())
        else:
            accepted_accuracy = None
            false_accept_rate = None
        rows.append({
            "threshold": round(float(threshold), 6),
            "accept_rate": round(float(accept.mean()), 6),
            "reinspect_rate": round(float((~accept).mean()), 6),
            "accepted_accuracy": None if accepted_accuracy is None else round(accepted_accuracy, 6),
            "false_accept_rate": None if false_accept_rate is None else round(false_accept_rate, 6),
        })
    return rows


def profile_windows(selected: np.ndarray, exit_costs: np.ndarray, window: int = 100) -> dict[str, Any]:
    n = selected.size
    usable = (n // window) * window
    if usable == 0:
        return {}
    reshaped = selected[:usable].reshape(-1, window)
    final_idx = len(exit_costs) - 1
    final_rates = (reshaped == final_idx).mean(axis=1)
    mean_costs = exit_costs[reshaped].mean(axis=1)
    return {
        "window": window,
        "num_windows": int(reshaped.shape[0]),
        "final_rate_mean": round(float(final_rates.mean()), 6),
        "final_rate_p90": round(float(np.quantile(final_rates, 0.9)), 6),
        "final_rate_p99": round(float(np.quantile(final_rates, 0.99)), 6),
        "mean_cost_p90": round(float(np.quantile(mean_costs, 0.9)), 6),
        "mean_cost_p99": round(float(np.quantile(mean_costs, 0.99)), 6),
    }


def profile_drift(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    n = correct.shape[0]
    hardness_score = confidence[:, -1]
    order = np.argsort(hardness_score)
    groups = {
        "hardest_20pct_by_final_confidence": order[: n // 5],
        "middle_20pct_by_final_confidence": order[(2 * n) // 5: (3 * n) // 5],
        "easiest_20pct_by_final_confidence": order[-(n // 5):],
    }
    out = {}
    for name, idx in groups.items():
        out[name] = selected_exit(correct[idx], confidence[idx], exit_costs, thresholds)
    return out


def analyze_trace(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    exit_costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]
    final_acc = float(correct[:, -1].mean())

    standard_curve = branchynet_curve(correct, confidence, exit_costs)
    one_pt = next(row for row in standard_curve if row.get("found") and row["max_accuracy_drop"] == 0.01)
    selected = np.full(correct.shape[0], correct.shape[1] - 1, dtype=int)
    for exit_idx, threshold in enumerate(one_pt["thresholds"]):
        mask = (selected == correct.shape[1] - 1) & (confidence[:, exit_idx] >= threshold)
        selected[mask] = exit_idx

    deadline0 = deadline_accept_reinspect(correct, confidence, exit_costs, 0)
    deadline1 = deadline_accept_reinspect(correct, confidence, exit_costs, 1)
    final_reject = final_risk_reject(correct, confidence)

    return {
        "trace": str(path),
        "n": int(correct.shape[0]),
        "exit_names": exit_names,
        "exit_costs": [round(float(x), 6) for x in exit_costs],
        "final_accuracy": round(final_acc, 6),
        "exit_accuracy": [round(float(correct[:, i].mean()), 6) for i in range(correct.shape[1])],
        "branchynet_accuracy_cost_curve": standard_curve,
        "inspection_deadline": {
            "deadline_at_exit0": deadline0,
            "deadline_at_exit1": deadline1,
            "best_points_exit0": best_deadline_points(deadline0),
            "best_points_exit1": best_deadline_points(deadline1),
        },
        "final_risk_reinspect": {
            "curve": final_reject,
            "best_points": best_deadline_points(final_reject),
        },
        "fpga_profile": {
            "branchynet_1pt_policy": one_pt,
            "window_pressure": profile_windows(selected, exit_costs, window=100),
            "profile_drift_by_final_confidence": profile_drift(correct, confidence, exit_costs, one_pt["thresholds"]),
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        trace = Path(result["trace"]).stem
        one_pt = result["fpga_profile"]["branchynet_1pt_policy"]
        deadline_exit1_5 = result["inspection_deadline"]["best_points_exit1"]["false_accept_le_0.05"]
        final_reject_2 = result["final_risk_reinspect"]["best_points"]["false_accept_le_0.02"]
        drift = result["fpga_profile"]["profile_drift_by_final_confidence"]
        rows.append({
            "trace": trace,
            "standard_branchynet_1pt": {
                "accuracy": one_pt["accuracy"],
                "avg_cost": one_pt["avg_cost"],
                "final_rate": one_pt["exit_rates"][str(len(result["exit_names"]) - 1)],
                "p99_cost": one_pt["p99_cost"],
            },
            "deadline_exit1_false_accept_le_5pct": None if deadline_exit1_5 is None else {
                "accept_rate": deadline_exit1_5["accept_rate"],
                "reinspect_rate": deadline_exit1_5["reinspect_rate"],
                "accepted_accuracy": deadline_exit1_5["accepted_accuracy"],
                "deadline_cost": deadline_exit1_5["deadline_cost"],
            },
            "final_reject_false_accept_le_2pct": None if final_reject_2 is None else {
                "accept_rate": final_reject_2["accept_rate"],
                "reinspect_rate": final_reject_2["reinspect_rate"],
                "accepted_accuracy": final_reject_2["accepted_accuracy"],
            },
            "profile_drift_final_rate": {
                group: metrics["exit_rates"][str(len(result["exit_names"]) - 1)]
                for group, metrics in drift.items()
            },
            "profile_drift_avg_cost": {
                group: metrics["avg_cost"]
                for group, metrics in drift.items()
            },
        })
    return {
        "purpose": "Find CIFAR-10 evidence that an inspection-oriented early-exit system can claim value on axes different from standard BranchyNet accuracy/average-cost.",
        "claim_axes": {
            "deadline": "Can make safe decisions before a fixed deadline and send the rest to reinspection.",
            "final_risk": "Can reject samples that even the final classifier is likely to mishandle.",
            "fpga_profile": "Exit distributions and tail/final pressure change with input profile, which matters for FPGA resource planning.",
        },
        "comparison_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="*", default=DEFAULT_TRACES)
    parser.add_argument("--output", default="results/cifar10_inspection_theme_probe_001_summary.json")
    args = parser.parse_args()

    paths = [Path(p) for p in args.traces]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trace files: {missing}")

    results = [analyze_trace(path) for path in paths]
    payload = {
        "summary": summarize(results),
        "per_trace": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
