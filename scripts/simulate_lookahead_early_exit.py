import argparse
import json
from pathlib import Path

import numpy as np

from src.experiment_paths import RESULTS_DIR, ensure_dirs


def expected_policy_metrics(levels, costs, keep_beneficial, reject_no_gain):
    exit_count = len(costs)
    never_level = exit_count
    accuracy = 0.0
    energy = 0.0
    final_cost = float(costs[-1])

    for level in levels:
        reach_prob = 1.0
        if level == never_level:
            for checkpoint in range(exit_count):
                stop_prob = reach_prob * reject_no_gain
                energy += stop_prob * float(costs[checkpoint])
                reach_prob *= (1.0 - reject_no_gain)
            energy += reach_prob * final_cost
            continue

        for checkpoint in range(level):
            false_reject_prob = reach_prob * (1.0 - keep_beneficial)
            energy += false_reject_prob * float(costs[checkpoint])
            reach_prob *= keep_beneficial
        energy += reach_prob * float(costs[level])
        accuracy += reach_prob

    sample_count = len(levels)
    return {
        "accuracy": float(accuracy / sample_count),
        "avg_energy": float(energy / sample_count),
        "relative_energy": float((energy / sample_count) / final_cost),
    }


def oracle_early_exit_metrics(levels, costs):
    exit_count = len(costs)
    never_level = exit_count
    energy = []
    correct = []
    for level in levels:
        if level == never_level:
            energy.append(float(costs[-1]))
            correct.append(False)
        else:
            energy.append(float(costs[level]))
            correct.append(True)
    return {
        "accuracy": float(np.mean(correct)),
        "avg_energy": float(np.mean(energy)),
        "relative_energy": float(np.mean(energy) / float(costs[-1])),
    }


def summarize_level_counts(levels, checkpoint_names):
    counts = {}
    for idx, name in enumerate(checkpoint_names):
        counts[name] = int(np.sum(levels == idx))
    counts["never_correct"] = int(np.sum(levels == len(checkpoint_names)))
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="artifacts/early_exit_checkpoint_trace_001.npz")
    parser.add_argument("--output-name", default="lookahead_early_exit_simulation_001.json")
    parser.add_argument("--keep-grid", default="0.80,0.85,0.90,0.95,0.98,0.99,1.00")
    parser.add_argument("--reject-grid", default="0.00,0.25,0.50,0.75,0.90,0.95,0.98,1.00")
    args = parser.parse_args()

    ensure_dirs()
    trace = np.load(args.trace, allow_pickle=True)
    correct = trace["correct"].astype(bool)
    costs = trace["checkpoint_costs"].astype(np.float32)
    checkpoint_names = [str(item) for item in trace["checkpoint_names"].tolist()]

    levels = np.full(correct.shape[0], correct.shape[1], dtype=np.int16)
    for idx, row in enumerate(correct):
        hits = np.flatnonzero(row)
        if len(hits):
            levels[idx] = int(hits[0])

    keep_values = [float(item) for item in args.keep_grid.split(",")]
    reject_values = [float(item) for item in args.reject_grid.split(",")]

    always_final = {
        "accuracy": float(correct[:, -1].mean()),
        "avg_energy": float(costs[-1]),
        "relative_energy": 1.0,
    }
    oracle_early = oracle_early_exit_metrics(levels, costs)

    rows = []
    for keep in keep_values:
        for reject in reject_values:
            metrics = expected_policy_metrics(levels, costs, keep, reject)
            rows.append({
                "keep_beneficial_continue": keep,
                "reject_no_gain": reject,
                **metrics,
                "accuracy_delta_vs_final": float(metrics["accuracy"] - always_final["accuracy"]),
                "energy_saved_vs_final": float(1.0 - metrics["relative_energy"]),
            })

    feasible = [
        row for row in rows
        if row["accuracy"] >= always_final["accuracy"] - 0.01
    ]
    best_energy_under_one_point_accuracy_loss = min(feasible, key=lambda row: row["relative_energy"]) if feasible else None

    summary = {
        "status": "ok",
        "purpose": "Theoretical performance map for predictive early exit that rejects samples unlikely to improve in later checkpoints.",
        "trace": args.trace,
        "samples": int(correct.shape[0]),
        "checkpoint_names": checkpoint_names,
        "checkpoint_costs": costs.astype(float).tolist(),
        "level_definition": "level is the earliest checkpoint that predicts the correct label; never_correct means no checkpoint is correct.",
        "level_counts": summarize_level_counts(levels, checkpoint_names),
        "always_final": always_final,
        "oracle_early_exit_without_no_gain_rejection": oracle_early,
        "prediction_parameters": {
            "keep_beneficial_continue": "probability of correctly continuing a sample that will become correct later",
            "reject_no_gain": "probability of correctly stopping a sample that will not become correct later",
        },
        "best_energy_under_one_point_accuracy_loss": best_energy_under_one_point_accuracy_loss,
        "grid_results": rows,
    }

    output_path = RESULTS_DIR / args.output_name
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
