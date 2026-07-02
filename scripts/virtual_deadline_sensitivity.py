import argparse
import json
from collections import defaultdict

import numpy as np

from scripts.virtual_deadline_simulation import (
    DEFAULT_STAGE_LATENCIES,
    best_by_policy,
    category_counts,
    load_records,
    make_deadlines,
    make_mid_proxy,
    scenarios,
    simulate_cascade,
    simulate_deadline_cascade,
    simulate_lightweight,
    simulate_parallel,
    simulate_proposed,
)
from src.experiment_paths import RESULTS_DIR, ensure_dirs


WEIGHT_PROFILES = {
    "balanced": {"unsafe": 10.0, "safe": 1.0, "miss": 5.0, "energy": 0.02},
    "safety_first": {"unsafe": 20.0, "safe": 1.0, "miss": 10.0, "energy": 0.02},
    "energy_sensitive": {"unsafe": 10.0, "safe": 1.0, "miss": 5.0, "energy": 0.08},
}


def extended_scenarios():
    return scenarios() + [
        {"name": "fixed_D6", "type": "fixed", "value": 6},
        {"name": "fixed_D10", "type": "fixed", "value": 10},
        {"name": "uniform_D6_14", "type": "uniform", "min": 6, "max": 14},
        {
            "name": "mostly_tight_with_slack_tail",
            "type": "mixture",
            "parts": [
                {"prob": 0.55, "min": 3, "max": 7},
                {"prob": 0.30, "min": 7, "max": 12},
                {"prob": 0.15, "min": 12, "max": 20},
            ],
        },
    ]


def stage_capacity(deadlines, latencies):
    deadlines = np.asarray(deadlines)
    return {
        "low_fits": float(np.mean(deadlines >= latencies["low"])),
        "low_mid_fits": float(np.mean(deadlines >= latencies["low"] + latencies["mid"])),
        "low_mid_high_fits": float(np.mean(deadlines >= latencies["low"] + latencies["mid"] + latencies["high"])),
        "high_parallel_fits": float(np.mean(deadlines >= latencies["high"])),
        "cascade_high_fits": float(np.mean(deadlines >= latencies["low"] + latencies["high"])),
        "deadline_mean": float(np.mean(deadlines)),
        "deadline_p10": float(np.quantile(deadlines, 0.10)),
        "deadline_p50": float(np.quantile(deadlines, 0.50)),
        "deadline_p90": float(np.quantile(deadlines, 0.90)),
    }


def theoretical_category_bounds(records):
    counts = defaultdict(int)
    high_correct = defaultdict(int)
    low_correct = defaultdict(int)
    for record in records:
        category = record.get("category")
        if category is None:
            from scripts.difficulty_mechanism_decomposition import category_of

            category = category_of(record)
        counts[category] += 1
        high_correct[category] += int(bool(record["high_correct"]))
        low_correct[category] += int(bool(record["real_low_correct"]))

    total = len(records)
    rows = {}
    for category in sorted(counts):
        rows[category] = {
            "count": counts[category],
            "rate": counts[category] / total,
            "low_correct_rate": low_correct[category] / counts[category],
            "high_correct_rate": high_correct[category] / counts[category],
        }
    rows["all"] = {
        "count": total,
        "rate": 1.0,
        "low_correct_rate": sum(low_correct.values()) / total,
        "high_correct_rate": sum(high_correct.values()) / total,
    }
    return rows


def best_existing_for_context(records, deadlines, low_thresholds, latencies, weights, scenario_name):
    rows = []
    for low_threshold in low_thresholds:
        rows.append(simulate_lightweight(records, deadlines, low_threshold, latencies, weights, scenario_name))
        rows.append(simulate_cascade(records, deadlines, low_threshold, latencies, weights, scenario_name))
        rows.append(simulate_deadline_cascade(records, deadlines, low_threshold, latencies, weights, scenario_name))
        rows.append(simulate_parallel(records, deadlines, low_threshold, latencies, weights, scenario_name, low_fallback=False))
        rows.append(simulate_parallel(records, deadlines, low_threshold, latencies, weights, scenario_name, low_fallback=True))
    per_policy = best_by_policy(rows)
    best = max(per_policy, key=lambda row: row["score"])
    return best, per_policy


def evaluate_proposed_grid(
    records,
    deadlines,
    low_thresholds,
    mid_thresholds,
    mid_envelopes,
    latencies,
    weights,
    scenario_name,
    seed,
):
    rows = []
    for envelope_idx, envelope in enumerate(mid_envelopes):
        mid_correct, mid_conf = make_mid_proxy(records, seed=seed + 1000 * envelope_idx, **envelope)
        for low_threshold in low_thresholds:
            for mid_threshold in mid_thresholds:
                rows.append(
                    simulate_proposed(
                        records,
                        deadlines,
                        low_threshold,
                        mid_threshold,
                        mid_correct,
                        mid_conf,
                        latencies,
                        weights,
                        scenario_name,
                        {"mid_envelope": envelope},
                    )
                )
    best = max(rows, key=lambda row: row["score"])
    return best, rows


def envelope_frontier(rows, baseline_score):
    by_recover = {}
    for row in rows:
        envelope = row["extra"]["mid_envelope"]
        recover = envelope["recover_hard"]
        if row["score"] <= baseline_score:
            continue
        current = by_recover.get(recover)
        if current is None or row["score"] > current["score"]:
            by_recover[recover] = row
    frontier = []
    for recover in sorted(by_recover):
        row = by_recover[recover]
        envelope = row["extra"]["mid_envelope"]
        frontier.append(
            {
                "recover_hard": recover,
                "retain_low_correct": envelope["retain_low_correct"],
                "recover_impossible": envelope["recover_impossible"],
                "score_margin": row["score"] - baseline_score,
                "useful_correct_rate": row["useful_correct_rate"],
                "unsafe_wrong_rate": row["unsafe_wrong_rate"],
                "safe_eject_or_resort_rate": row["safe_eject_or_resort_rate"],
                "deadline_miss_rate": row["deadline_miss_rate"],
                "average_energy_proxy": row["average_energy_proxy"],
            }
        )
    return frontier


def compact_row(row):
    keep = [
        "policy",
        "scenario",
        "low_threshold",
        "mid_threshold",
        "extra",
        "score",
        "useful_correct_rate",
        "unsafe_wrong_rate",
        "safe_eject_or_resort_rate",
        "deadline_miss_rate",
        "average_energy_proxy",
        "accuracy_on_decided",
        "low_decisions_rate",
        "mid_decisions_rate",
        "high_decisions_rate",
        "low_activations_rate",
        "mid_activations_rate",
        "high_activations_rate",
    ]
    return {key: row[key] for key in keep if key in row}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-name", default="virtual_deadline_sensitivity.json")
    parser.add_argument("--seed", type=int, default=314)
    args = parser.parse_args()

    ensure_dirs()
    records = load_records(args.max_samples, args.batch_size)
    low_thresholds = [0.60, 0.75, 0.90, 0.95]
    mid_thresholds = [0.55, 0.65, 0.75]
    mid_latencies = [2.0, 4.0, 6.0]
    mid_envelopes = [
        {"recover_hard": recover, "retain_low_correct": retain, "recover_impossible": recover_impossible}
        for recover in [0.40, 0.55, 0.70, 0.85]
        for retain in [0.94, 0.97, 0.99]
        for recover_impossible in [0.05, 0.12]
    ]

    contexts = []
    wins = 0
    total_contexts = 0
    frontiers = []

    for scenario in extended_scenarios():
        scenario_name, deadlines = make_deadlines(len(records), scenario, args.seed)
        for weight_name, weights in WEIGHT_PROFILES.items():
            for mid_latency in mid_latencies:
                latencies = dict(DEFAULT_STAGE_LATENCIES)
                latencies["mid"] = mid_latency

                baseline_best, baseline_by_policy = best_existing_for_context(
                    records,
                    deadlines,
                    low_thresholds,
                    latencies,
                    weights,
                    scenario_name,
                )
                proposed_best, proposed_rows = evaluate_proposed_grid(
                    records,
                    deadlines,
                    low_thresholds,
                    mid_thresholds,
                    mid_envelopes,
                    latencies,
                    weights,
                    scenario_name,
                    args.seed,
                )

                total_contexts += 1
                proposed_wins = proposed_best["score"] > baseline_best["score"]
                wins += int(proposed_wins)
                frontier = envelope_frontier(proposed_rows, baseline_best["score"])
                if frontier:
                    frontiers.append(
                        {
                            "scenario": scenario_name,
                            "weight_profile": weight_name,
                            "mid_latency": mid_latency,
                            "minimal_recover_hard": min(row["recover_hard"] for row in frontier),
                            "frontier": frontier,
                        }
                    )

                contexts.append(
                    {
                        "scenario": scenario_name,
                        "weight_profile": weight_name,
                        "mid_latency": mid_latency,
                        "stage_capacity": stage_capacity(deadlines, latencies),
                        "baseline_best": compact_row(baseline_best),
                        "baseline_by_policy": [compact_row(row) for row in sorted(baseline_by_policy, key=lambda x: x["policy"])],
                        "proposed_best": compact_row(proposed_best),
                        "proposed_score_margin": proposed_best["score"] - baseline_best["score"],
                        "proposed_wins": proposed_wins,
                        "frontier_count": len(frontier),
                    }
                )

    strong_win_contexts = [
        row for row in contexts
        if row["proposed_wins"]
        and row["proposed_score_margin"] >= 0.05
        and row["proposed_best"]["deadline_miss_rate"] == 0.0
    ]
    robust_contexts = [
        row for row in strong_win_contexts
        if row["proposed_best"]["extra"]["mid_envelope"]["recover_hard"] <= 0.70
        and row["proposed_best"]["extra"]["mid_envelope"]["retain_low_correct"] <= 0.97
    ]

    summary = {
        "status": "ok",
        "purpose": "Sensitivity analysis for virtual-deadline FPGA adaptive inference.",
        "samples": len(records),
        "category_counts": category_counts(records),
        "category_theoretical_bounds": theoretical_category_bounds(records),
        "weight_profiles": WEIGHT_PROFILES,
        "low_thresholds": low_thresholds,
        "mid_thresholds": mid_thresholds,
        "mid_latencies": mid_latencies,
        "mid_envelopes_count": len(mid_envelopes),
        "contexts_count": total_contexts,
        "proposed_win_contexts": wins,
        "proposed_win_rate": wins / total_contexts if total_contexts else None,
        "strong_win_contexts_count": len(strong_win_contexts),
        "robust_contexts_count": len(robust_contexts),
        "strong_win_contexts": strong_win_contexts[:30],
        "robust_contexts": robust_contexts[:30],
        "frontiers": frontiers[:60],
        "contexts": contexts,
        "notes": [
            "This is a parametric MID-stage sensitivity analysis, not a trained-MID result.",
            "A robust context means the proposed policy wins without requiring the strongest MID envelope.",
            "The decisive next step is to train FPGA-feasible MID candidates and compare them to these envelope requirements.",
        ],
    }

    output_path = RESULTS_DIR / args.output_name
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
