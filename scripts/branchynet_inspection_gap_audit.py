"""Audit gaps when vanilla BranchyNet is used as an inspection system.

This script intentionally treats saved CIFAR-10 BranchyNet traces as if they
were deployed without inspection-specific redesign. It reports the weaknesses
that motivate a task-specific approach:

1. forced-decision risk: every sample receives a label, including low-confidence
   final decisions
2. deadline mismatch: standard early-exit still sends a tail of samples past a
   chosen deadline
3. profile sensitivity: final-stage load changes sharply when the input stream
   becomes easier or harder
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


def quantile_thresholds(values: np.ndarray, steps: int = 25) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, steps)))


def select_standard_branchynet(
    correct: np.ndarray,
    confidence: np.ndarray,
    exit_costs: np.ndarray,
    max_accuracy_drop: float = 0.01,
) -> dict[str, Any]:
    final_acc = float(correct[:, -1].mean())
    n, exits = correct.shape
    candidates = []
    for t0 in quantile_thresholds(confidence[:, 0]):
        for t1 in quantile_thresholds(confidence[:, 1]):
            selected = np.full(n, exits - 1, dtype=int)
            for exit_idx, threshold in enumerate([float(t0), float(t1)]):
                mask = (selected == exits - 1) & (confidence[:, exit_idx] >= threshold)
                selected[mask] = exit_idx
            rows = np.arange(n)
            selected_correct = correct[rows, selected]
            selected_cost = exit_costs[selected]
            accuracy = float(selected_correct.mean())
            if accuracy < final_acc - max_accuracy_drop:
                continue
            candidates.append({
                "thresholds": [float(t0), float(t1)],
                "selected": selected,
                "accuracy": accuracy,
                "avg_cost": float(selected_cost.mean()),
                "p90_cost": float(np.quantile(selected_cost, 0.9)),
                "p99_cost": float(np.quantile(selected_cost, 0.99)),
            })
    if not candidates:
        raise RuntimeError("No standard BranchyNet policy met the accuracy constraint")
    best = min(candidates, key=lambda row: (row["avg_cost"], row["p99_cost"], -row["accuracy"]))
    selected = best.pop("selected")
    counts = {str(i): int((selected == i).sum()) for i in range(exits)}
    best.update({
        "thresholds": [round(x, 6) for x in best["thresholds"]],
        "accuracy": round(best["accuracy"], 6),
        "avg_cost": round(best["avg_cost"], 6),
        "p90_cost": round(best["p90_cost"], 6),
        "p99_cost": round(best["p99_cost"], 6),
        "exit_counts": counts,
        "exit_rates": {k: round(v / n, 6) for k, v in counts.items()},
    })
    return {"policy": best, "selected": selected}


def forced_decision_risk(correct: np.ndarray, confidence: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    rows = np.arange(correct.shape[0])
    selected_correct = correct[rows, selected]
    selected_conf = confidence[rows, selected]
    final_correct = correct[:, -1]
    final_conf = confidence[:, -1]
    quantiles = [0.05, 0.1, 0.2, 0.3]
    selected_low_conf = {}
    final_low_conf = {}
    for q in quantiles:
        selected_threshold = float(np.quantile(selected_conf, q))
        selected_mask = selected_conf <= selected_threshold
        final_threshold = float(np.quantile(final_conf, q))
        final_mask = final_conf <= final_threshold
        selected_low_conf[f"lowest_{int(q * 100)}pct_selected_conf"] = {
            "threshold": round(selected_threshold, 6),
            "share": round(float(selected_mask.mean()), 6),
            "accuracy": round(float(selected_correct[selected_mask].mean()), 6),
            "false_accept_rate": round(float((~selected_correct[selected_mask]).mean()), 6),
        }
        final_low_conf[f"lowest_{int(q * 100)}pct_final_conf"] = {
            "threshold": round(final_threshold, 6),
            "share": round(float(final_mask.mean()), 6),
            "accuracy": round(float(final_correct[final_mask].mean()), 6),
            "false_accept_rate": round(float((~final_correct[final_mask]).mean()), 6),
        }
    return {
        "standard_policy_forces_all_samples_to_accept": True,
        "overall_selected_accuracy": round(float(selected_correct.mean()), 6),
        "overall_selected_false_accept_rate": round(float((~selected_correct).mean()), 6),
        "overall_final_accuracy": round(float(final_correct.mean()), 6),
        "overall_final_false_accept_rate": round(float((~final_correct).mean()), 6),
        "risk_concentrates_in_low_confidence_selected_decisions": selected_low_conf,
        "risk_concentrates_in_low_confidence_final_decisions": final_low_conf,
    }


def deadline_gap(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray, selected: np.ndarray, deadline_exit: int) -> dict[str, Any]:
    rows = np.arange(correct.shape[0])
    selected_correct = correct[rows, selected]
    selected_cost = exit_costs[selected]
    missed = selected > deadline_exit
    before_deadline = ~missed
    if before_deadline.any():
        on_time_acc = float(selected_correct[before_deadline].mean())
    else:
        on_time_acc = None
    if missed.any():
        late_acc = float(selected_correct[missed].mean())
    else:
        late_acc = None

    # If forced to decide exactly at the deadline without reinspection, what is
    # the quality of those deadline decisions?
    forced_deadline_correct = correct[:, deadline_exit]
    forced_deadline_conf = confidence[:, deadline_exit]
    return {
        "deadline_exit": int(deadline_exit),
        "deadline_cost": round(float(exit_costs[deadline_exit]), 6),
        "standard_policy_deadline_miss_rate": round(float(missed.mean()), 6),
        "standard_policy_on_time_rate": round(float(before_deadline.mean()), 6),
        "standard_policy_on_time_accuracy": None if on_time_acc is None else round(on_time_acc, 6),
        "standard_policy_late_accuracy": None if late_acc is None else round(late_acc, 6),
        "if_forced_to_decide_at_deadline_accuracy": round(float(forced_deadline_correct.mean()), 6),
        "if_forced_to_decide_at_deadline_false_accept_rate": round(float((~forced_deadline_correct).mean()), 6),
        "deadline_confidence_lowest_20pct_false_accept_rate": round(
            float((~forced_deadline_correct[forced_deadline_conf <= np.quantile(forced_deadline_conf, 0.2)]).mean()),
            6,
        ),
    }


def profile_sensitivity(correct: np.ndarray, confidence: np.ndarray, exit_costs: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    n, exits = correct.shape
    hardness = confidence[:, -1]
    order = np.argsort(hardness)
    groups = {
        "hardest_20pct": order[: n // 5],
        "middle_20pct": order[(2 * n) // 5: (3 * n) // 5],
        "easiest_20pct": order[-(n // 5):],
    }
    out = {}
    for name, idx in groups.items():
        selected = np.full(idx.size, exits - 1, dtype=int)
        group_conf = confidence[idx]
        group_correct = correct[idx]
        for exit_idx, threshold in enumerate(thresholds):
            mask = (selected == exits - 1) & (group_conf[:, exit_idx] >= threshold)
            selected[mask] = exit_idx
        rows = np.arange(idx.size)
        selected_correct = group_correct[rows, selected]
        selected_cost = exit_costs[selected]
        counts = {str(i): int((selected == i).sum()) for i in range(exits)}
        out[name] = {
            "accuracy": round(float(selected_correct.mean()), 6),
            "avg_cost": round(float(selected_cost.mean()), 6),
            "final_rate": round(float((selected == exits - 1).mean()), 6),
            "exit_rates": {k: round(v / idx.size, 6) for k, v in counts.items()},
        }
    return out


def analyze_trace(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    exit_costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]

    standard = select_standard_branchynet(correct, confidence, exit_costs, max_accuracy_drop=0.01)
    policy = standard["policy"]
    selected = standard["selected"]
    thresholds = [float(x) for x in policy["thresholds"]]

    return {
        "trace": str(path),
        "n": int(correct.shape[0]),
        "exit_names": exit_names,
        "exit_costs": [round(float(x), 6) for x in exit_costs],
        "standard_branchynet_policy": policy,
        "gap_forced_decision": forced_decision_risk(correct, confidence, selected),
        "gap_deadline_exit0": deadline_gap(correct, confidence, exit_costs, selected, 0),
        "gap_deadline_exit1": deadline_gap(correct, confidence, exit_costs, selected, 1),
        "gap_profile_sensitivity": profile_sensitivity(correct, confidence, exit_costs, thresholds),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        trace = Path(result["trace"]).stem
        forced = result["gap_forced_decision"]
        deadline1 = result["gap_deadline_exit1"]
        profile = result["gap_profile_sensitivity"]
        rows.append({
            "trace": trace,
            "standard_branchynet": {
                "accuracy": result["standard_branchynet_policy"]["accuracy"],
                "avg_cost": result["standard_branchynet_policy"]["avg_cost"],
                "final_rate": result["standard_branchynet_policy"]["exit_rates"][str(len(result["exit_names"]) - 1)],
                "p99_cost": result["standard_branchynet_policy"]["p99_cost"],
            },
            "forced_decision_problem": {
                "overall_false_accept_rate": forced["overall_selected_false_accept_rate"],
                "lowest_20pct_selected_conf_false_accept_rate": forced["risk_concentrates_in_low_confidence_selected_decisions"]["lowest_20pct_selected_conf"]["false_accept_rate"],
                "lowest_20pct_final_conf_false_accept_rate": forced["risk_concentrates_in_low_confidence_final_decisions"]["lowest_20pct_final_conf"]["false_accept_rate"],
            },
            "deadline_exit1_problem": {
                "miss_rate_if_standard_branchynet": deadline1["standard_policy_deadline_miss_rate"],
                "forced_deadline_false_accept_rate": deadline1["if_forced_to_decide_at_deadline_false_accept_rate"],
                "low_conf_deadline_false_accept_rate": deadline1["deadline_confidence_lowest_20pct_false_accept_rate"],
            },
            "profile_problem": {
                "hardest_20pct_final_rate": profile["hardest_20pct"]["final_rate"],
                "middle_20pct_final_rate": profile["middle_20pct"]["final_rate"],
                "easiest_20pct_final_rate": profile["easiest_20pct"]["final_rate"],
                "hardest_20pct_avg_cost": profile["hardest_20pct"]["avg_cost"],
                "easiest_20pct_avg_cost": profile["easiest_20pct"]["avg_cost"],
            },
        })
    return {
        "purpose": "Show why vanilla BranchyNet is not directly sufficient as an inspection system.",
        "vanilla_branchynet_assumption": "Every sample is forced into a class label; confidence controls early exit only for average-cost reduction.",
        "inspection_requirements_not_directly_optimized": [
            "false accept / erroneous pass rate",
            "deadline compliance",
            "reinspection / hold-out option",
            "profile-dependent FPGA tail load",
        ],
        "comparison_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="*", default=DEFAULT_TRACES)
    parser.add_argument("--output", default="results/branchynet_inspection_gap_audit_001_summary.json")
    args = parser.parse_args()

    paths = [Path(p) for p in args.traces]
    missing = [str(path) for path in paths if not path.exists()]
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
