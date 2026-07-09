"""Analyze BranchyNet traces to choose the next research direction.

This script does not train models. It reads saved per-sample early-exit traces
and estimates which follow-up theme has the strongest evidence:

* lookahead early-exit: predict samples that final layers will not rescue
* deadline/reject inspection: trade coverage for safe decisions
* FPGA-oriented implementation: quantify exit-rate and tail-final pressure
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
    "results/0002b_branchynet_resnet56_cifar100.npz",
    "results/0003_branchynet_mobilenet_cifar100_retry.npz",
]


def as_float(value: Any) -> float:
    return float(np.asarray(value).item() if np.asarray(value).shape == () else value)


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute AUROC with average ranks. Returns None for one-class targets."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)

    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_auc(labels: np.ndarray, feature: np.ndarray) -> dict[str, Any]:
    direct = auc_score(labels, feature)
    inverse = auc_score(labels, -feature)
    values = [v for v in [direct, inverse] if v is not None]
    if not values:
        return {"auc": None, "direction": None}
    if direct is not None and direct >= (inverse if inverse is not None else -1):
        return {"auc": round(float(direct), 6), "direction": "higher"}
    return {"auc": round(float(inverse), 6), "direction": "lower"}


def threshold_grid(confidence: np.ndarray, n: int = 101) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n)
    return np.unique(np.quantile(confidence, quantiles))


def selected_exit_metrics(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    n, exits = correct.shape
    selected = np.full(n, exits - 1, dtype=int)
    for exit_idx, threshold in enumerate(thresholds):
        mask = (selected == exits - 1) & (confidence[:, exit_idx] >= threshold)
        selected[mask] = exit_idx

    rows = np.arange(n)
    acc = float(correct[rows, selected].mean())
    cost = float(exit_costs[selected].mean())
    counts = {str(i): int((selected == i).sum()) for i in range(exits)}
    rates = {str(i): round(counts[str(i)] / n, 6) for i in range(exits)}
    return {
        "accuracy": round(acc, 6),
        "avg_cost": round(cost, 6),
        "exit_counts": counts,
        "exit_rates": rates,
        "thresholds": [round(float(x), 6) for x in thresholds],
    }


def search_early_exit_curve(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray) -> list[dict[str, Any]]:
    final_acc = float(correct[:, -1].mean())
    thresholds0 = threshold_grid(confidence[:, 0], 25)
    thresholds1 = threshold_grid(confidence[:, 1], 25)
    tolerances = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
    candidates = []
    for t0 in thresholds0:
        for t1 in thresholds1:
            metrics = selected_exit_metrics(correct, confidence, exit_costs, [float(t0), float(t1)])
            metrics["accuracy_drop"] = round(final_acc - metrics["accuracy"], 6)
            candidates.append(metrics)

    curve = []
    for tol in tolerances:
        valid = [m for m in candidates if m["accuracy"] >= final_acc - tol]
        if not valid:
            curve.append({"max_drop": tol, "found": False})
            continue
        best = min(valid, key=lambda m: (m["avg_cost"], -m["accuracy"]))
        curve.append({"max_drop": tol, "found": True, **best})
    return curve


def reject_curve_at_exit(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray, exit_idx: int) -> list[dict[str, Any]]:
    thresholds = np.unique(np.quantile(confidence[:, exit_idx], np.linspace(0.0, 1.0, 11)))
    rows = []
    for threshold in thresholds:
        accept = confidence[:, exit_idx] >= threshold
        coverage = float(accept.mean())
        accepted = int(accept.sum())
        if accepted:
            accepted_acc = float(correct[accept, exit_idx].mean())
            false_accept_rate = float((~correct[accept, exit_idx]).mean())
        else:
            accepted_acc = None
            false_accept_rate = None
        rows.append({
            "exit": int(exit_idx),
            "threshold": round(float(threshold), 6),
            "coverage": round(coverage, 6),
            "reject_rate": round(1.0 - coverage, 6),
            "accepted_accuracy": None if accepted_acc is None else round(accepted_acc, 6),
            "false_accept_rate": None if false_accept_rate is None else round(false_accept_rate, 6),
            "deadline_cost": round(float(exit_costs[exit_idx]), 6),
        })
    return rows


def analyze_trace(path: Path) -> dict[str, Any]:
    trace = np.load(path, allow_pickle=True)
    correct = np.asarray(trace["correct"], dtype=bool)
    confidence = np.asarray(trace["confidence"], dtype=float)
    entropy = np.asarray(trace["entropy"], dtype=float)
    exit_costs = np.asarray(trace["exit_costs"], dtype=float)
    exit_names = [str(x) for x in trace["exit_names"].tolist()]
    n, exits = correct.shape

    final_correct = correct[:, -1]
    exit_accuracy = [round(float(correct[:, i].mean()), 6) for i in range(exits)]

    pattern_counts = {
        "exit0_correct_final_correct": int((correct[:, 0] & final_correct).sum()),
        "exit0_wrong_final_correct": int((~correct[:, 0] & final_correct).sum()),
        "exit0_wrong_final_wrong": int((~correct[:, 0] & ~final_correct).sum()),
        "exit0_correct_final_wrong": int((correct[:, 0] & ~final_correct).sum()),
        "exit1_correct_final_correct": int((correct[:, 1] & final_correct).sum()),
        "exit1_wrong_final_correct": int((~correct[:, 1] & final_correct).sum()),
        "exit1_wrong_final_wrong": int((~correct[:, 1] & ~final_correct).sum()),
        "exit1_correct_final_wrong": int((correct[:, 1] & ~final_correct).sum()),
    }
    pattern_rates = {k: round(v / n, 6) for k, v in pattern_counts.items()}

    earliest = np.full(n, exits, dtype=int)
    for i in range(exits):
        mask = (earliest == exits) & correct[:, i]
        earliest[mask] = i
    earliest_counts = {str(i): int((earliest == i).sum()) for i in range(exits)}
    earliest_counts["never"] = int((earliest == exits).sum())
    earliest_rates = {k: round(v / n, 6) for k, v in earliest_counts.items()}

    lookahead = []
    for exit_idx in range(exits - 1):
        current_wrong = ~correct[:, exit_idx]
        rescued_by_final = current_wrong & final_correct
        not_rescued_by_final = current_wrong & ~final_correct
        target_final_wrong = ~final_correct
        target_no_rescue_among_current_wrong = not_rescued_by_final[current_wrong]

        predictors = {
            "confidence": confidence[:, exit_idx],
            "entropy": entropy[:, exit_idx],
        }
        if exit_idx > 0:
            predictors["confidence_delta_from_prev"] = confidence[:, exit_idx] - confidence[:, exit_idx - 1]
            predictors["entropy_delta_from_prev"] = entropy[:, exit_idx] - entropy[:, exit_idx - 1]

        aucs_final_wrong = {name: best_auc(target_final_wrong, values) for name, values in predictors.items()}
        aucs_no_rescue_current_wrong = {}
        for name, values in predictors.items():
            aucs_no_rescue_current_wrong[name] = best_auc(target_no_rescue_among_current_wrong, values[current_wrong])

        no_rescue_pool = int(not_rescued_by_final.sum())
        rescue_pool = int(rescued_by_final.sum())
        ideal_stop_saving = float((exit_costs[-1] - exit_costs[exit_idx]) * no_rescue_pool / n)
        lookahead.append({
            "exit": int(exit_idx),
            "exit_name": exit_names[exit_idx],
            "current_wrong_rate": round(float(current_wrong.mean()), 6),
            "rescued_by_final_rate": round(rescue_pool / n, 6),
            "not_rescued_by_final_rate": round(no_rescue_pool / n, 6),
            "ideal_no_rescue_stop_cost_saving": round(ideal_stop_saving, 6),
            "auc_predict_final_wrong_from_exit_info": aucs_final_wrong,
            "auc_predict_no_rescue_among_current_wrong": aucs_no_rescue_current_wrong,
        })

    reject = {
        "deadline_at_exit0": reject_curve_at_exit(correct, confidence, exit_costs, 0),
        "deadline_at_exit1": reject_curve_at_exit(correct, confidence, exit_costs, 1),
    }
    early_exit_curve = search_early_exit_curve(correct, confidence, exit_costs)

    final_rate_at_strict = next((row["exit_rates"].get(str(exits - 1), None) for row in early_exit_curve if row.get("found") and row["max_drop"] == 0.0), None)
    final_rate_at_1pt = next((row["exit_rates"].get(str(exits - 1), None) for row in early_exit_curve if row.get("found") and row["max_drop"] == 0.01), None)

    theme_signals = {
        "lookahead_early_exit": {
            "support": "higher if not_rescued_by_final_rate is large and AUC is high",
            "max_ideal_cost_saving": max(row["ideal_no_rescue_stop_cost_saving"] for row in lookahead),
            "best_auc_final_wrong": max(
                [v["auc"] for row in lookahead for v in row["auc_predict_final_wrong_from_exit_info"].values() if v["auc"] is not None],
                default=None,
            ),
        },
        "deadline_reject_inspection": {
            "support": "higher if reject allows low false-accept rate at useful coverage",
            "example_exit1_50pct_threshold": reject["deadline_at_exit1"][5] if len(reject["deadline_at_exit1"]) > 5 else None,
        },
        "fpga_early_exit_implementation": {
            "support": "higher if useful accuracy-cost curve exists but final-rate/profile pressure remains important",
            "final_rate_at_zero_drop": final_rate_at_strict,
            "final_rate_at_1pt_drop": final_rate_at_1pt,
        },
    }

    return {
        "trace": str(path),
        "n": int(n),
        "exit_names": exit_names,
        "exit_costs": [round(float(x), 6) for x in exit_costs],
        "exit_accuracy": exit_accuracy,
        "final_accuracy": round(float(final_correct.mean()), 6),
        "pattern_counts": pattern_counts,
        "pattern_rates": pattern_rates,
        "earliest_correct_counts": earliest_counts,
        "earliest_correct_rates": earliest_rates,
        "early_exit_accuracy_cost_curve": early_exit_curve,
        "lookahead_no_rescue_analysis": lookahead,
        "deadline_reject_curves": reject,
        "theme_signals": theme_signals,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        name = Path(result["trace"]).stem
        curve_1pt = next((row for row in result["early_exit_accuracy_cost_curve"] if row.get("found") and row["max_drop"] == 0.01), None)
        lookahead = result["theme_signals"]["lookahead_early_exit"]
        rows.append({
            "trace": name,
            "final_accuracy": result["final_accuracy"],
            "exit_accuracy": result["exit_accuracy"],
            "cost_at_1pt_drop": None if curve_1pt is None else curve_1pt["avg_cost"],
            "final_rate_at_1pt_drop": result["theme_signals"]["fpga_early_exit_implementation"]["final_rate_at_1pt_drop"],
            "final_wrong_rate": round(1.0 - result["final_accuracy"], 6),
            "max_ideal_no_rescue_cost_saving": lookahead["max_ideal_cost_saving"],
            "best_auc_predict_final_wrong": lookahead["best_auc_final_wrong"],
        })
    return {
        "purpose": "Decide which next research theme has the strongest preliminary evidence: lookahead early-exit, deadline/reject inspection, or FPGA-focused early-exit implementation.",
        "interpretation_rules": {
            "lookahead": "Promising only when final-wrong/no-rescue samples are common and predictable from early exit signals.",
            "deadline_reject": "Promising when useful coverage can be accepted with low false-accept rate before final.",
            "fpga": "Promising when average-cost gains exist but final-rate/profile/tail behavior creates implementation-specific design issues.",
        },
        "comparison_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="*", default=DEFAULT_TRACES)
    parser.add_argument("--output", default="results/early_exit_theme_feasibility_001.json")
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
