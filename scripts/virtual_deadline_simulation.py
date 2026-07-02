import argparse
import json
from collections import defaultdict

import numpy as np

from scripts.evaluate_architectures import FLOPS_HIGH, FLOPS_LOW, add_low_confidence
from scripts.difficulty_mechanism_decomposition import category_of
from src.experiment_paths import ARTIFACT_DIR, DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs


DEFAULT_STAGE_LATENCIES = {
    "low": 1.0,
    "mid": 4.0,
    "high": 12.0,
}


def load_records(max_samples, batch_size):
    records = json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))
    if max_samples:
        records = records[:max_samples]
    return add_low_confidence(records, ARTIFACT_DIR / "cifar100_low_confidence_x1_0.json", batch_size)


def make_deadlines(n, scenario, seed):
    rng = np.random.default_rng(seed)
    name = scenario["name"]
    if scenario["type"] == "fixed":
        values = np.full(n, float(scenario["value"]), dtype=np.float32)
    elif scenario["type"] == "uniform":
        values = rng.uniform(float(scenario["min"]), float(scenario["max"]), size=n).astype(np.float32)
    elif scenario["type"] == "mixture":
        probs = np.asarray([part["prob"] for part in scenario["parts"]], dtype=np.float64)
        probs = probs / probs.sum()
        choices = rng.choice(len(scenario["parts"]), size=n, p=probs)
        values = np.zeros(n, dtype=np.float32)
        for idx, part in enumerate(scenario["parts"]):
            mask = choices == idx
            values[mask] = rng.uniform(float(part["min"]), float(part["max"]), size=int(mask.sum()))
    else:
        raise ValueError(scenario["type"])
    return name, values


def scenarios():
    rows = [
        {"name": "fixed_D2", "type": "fixed", "value": 2},
        {"name": "fixed_D4", "type": "fixed", "value": 4},
        {"name": "fixed_D8", "type": "fixed", "value": 8},
        {"name": "fixed_D12", "type": "fixed", "value": 12},
        {"name": "fixed_D16", "type": "fixed", "value": 16},
        {"name": "uniform_D4_16", "type": "uniform", "min": 4, "max": 16},
        {"name": "uniform_D8_20", "type": "uniform", "min": 8, "max": 20},
        {
            "name": "mixture_tight",
            "type": "mixture",
            "parts": [
                {"prob": 0.4, "min": 2, "max": 5},
                {"prob": 0.4, "min": 5, "max": 12},
                {"prob": 0.2, "min": 12, "max": 20},
            ],
        },
    ]
    return rows


def category_counts(records):
    counts = defaultdict(int)
    for record in records:
        counts[category_of(record)] += 1
    return dict(sorted(counts.items()))


def make_mid_proxy(records, recover_hard, retain_low_correct, recover_impossible, seed):
    rng = np.random.default_rng(seed)
    mid_correct = np.zeros(len(records), dtype=bool)
    mid_conf = np.zeros(len(records), dtype=np.float32)
    for idx, record in enumerate(records):
        category = category_of(record)
        if category == "Easy":
            prob = retain_low_correct
        elif category == "Inverse":
            prob = retain_low_correct
        elif category == "Hard":
            prob = recover_hard
        elif category == "Impossible":
            prob = recover_impossible
        else:
            prob = 0.0
        correct = rng.random() < prob
        mid_correct[idx] = correct
        if correct:
            mid_conf[idx] = float(np.clip(rng.normal(0.78, 0.12), 0.0, 1.0))
        else:
            mid_conf[idx] = float(np.clip(rng.normal(0.42, 0.18), 0.0, 1.0))
    return mid_correct, mid_conf


def empty_stats(policy, scenario, low_threshold, mid_threshold=None, extra=None):
    return {
        "policy": policy,
        "scenario": scenario,
        "low_threshold": float(low_threshold),
        "mid_threshold": None if mid_threshold is None else float(mid_threshold),
        "extra": extra or {},
        "useful_correct": 0,
        "unsafe_wrong": 0,
        "safe_eject_or_resort": 0,
        "deadline_miss": 0,
        "low_decisions": 0,
        "mid_decisions": 0,
        "high_decisions": 0,
        "low_activations": 0,
        "mid_activations": 0,
        "high_activations": 0,
        "energy_proxy": 0.0,
    }


def record_decision(stats, correct, decision_type):
    stats[f"{decision_type}_decisions"] += 1
    if correct:
        stats["useful_correct"] += 1
    else:
        stats["unsafe_wrong"] += 1


def finalize(stats, n, weights):
    stats = dict(stats)
    for key in [
        "useful_correct",
        "unsafe_wrong",
        "safe_eject_or_resort",
        "deadline_miss",
        "low_decisions",
        "mid_decisions",
        "high_decisions",
        "low_activations",
        "mid_activations",
        "high_activations",
    ]:
        stats[f"{key}_rate"] = stats[key] / n
    stats["average_energy_proxy"] = stats["energy_proxy"] / n
    decided = stats["useful_correct"] + stats["unsafe_wrong"]
    stats["accuracy_on_decided"] = stats["useful_correct"] / decided if decided else None
    stats["score"] = (
        stats["useful_correct_rate"]
        - weights["unsafe"] * stats["unsafe_wrong_rate"]
        - weights["safe"] * stats["safe_eject_or_resort_rate"]
        - weights["miss"] * stats["deadline_miss_rate"]
        - weights["energy"] * stats["average_energy_proxy"]
    )
    return stats


def simulate_lightweight(records, deadlines, low_threshold, latencies, weights, scenario_name):
    stats = empty_stats("lightweight_safe_rule", scenario_name, low_threshold)
    for record, deadline in zip(records, deadlines):
        stats["low_activations"] += 1
        stats["energy_proxy"] += latencies["low"]
        if latencies["low"] > deadline:
            stats["deadline_miss"] += 1
            continue
        if record["real_low_conf"] >= low_threshold:
            record_decision(stats, bool(record["real_low_correct"]), "low")
        else:
            stats["safe_eject_or_resort"] += 1
    return finalize(stats, len(records), weights)


def simulate_cascade(records, deadlines, low_threshold, latencies, weights, scenario_name):
    stats = empty_stats("ordinary_cascade", scenario_name, low_threshold)
    for record, deadline in zip(records, deadlines):
        stats["low_activations"] += 1
        stats["energy_proxy"] += latencies["low"]
        if latencies["low"] > deadline:
            stats["deadline_miss"] += 1
            continue
        if record["real_low_conf"] >= low_threshold:
            record_decision(stats, bool(record["real_low_correct"]), "low")
            continue
        stats["high_activations"] += 1
        stats["energy_proxy"] += latencies["high"]
        if latencies["low"] + latencies["high"] <= deadline:
            record_decision(stats, bool(record["high_correct"]), "high")
        else:
            stats["deadline_miss"] += 1
    return finalize(stats, len(records), weights)


def simulate_deadline_cascade(records, deadlines, low_threshold, latencies, weights, scenario_name):
    stats = empty_stats("deadline_aware_low_high", scenario_name, low_threshold)
    for record, deadline in zip(records, deadlines):
        stats["low_activations"] += 1
        stats["energy_proxy"] += latencies["low"]
        if latencies["low"] > deadline:
            stats["safe_eject_or_resort"] += 1
            continue
        if record["real_low_conf"] >= low_threshold:
            record_decision(stats, bool(record["real_low_correct"]), "low")
            continue
        if latencies["low"] + latencies["high"] <= deadline:
            stats["high_activations"] += 1
            stats["energy_proxy"] += latencies["high"]
            record_decision(stats, bool(record["high_correct"]), "high")
        else:
            stats["safe_eject_or_resort"] += 1
    return finalize(stats, len(records), weights)


def simulate_parallel(records, deadlines, low_threshold, latencies, weights, scenario_name, low_fallback):
    policy = "parallel_with_low_fallback" if low_fallback else "parallel_high_required"
    stats = empty_stats(policy, scenario_name, low_threshold)
    for record, deadline in zip(records, deadlines):
        stats["low_activations"] += 1
        stats["high_activations"] += 1
        stats["energy_proxy"] += latencies["low"] + latencies["high"]
        if latencies["high"] <= deadline:
            record_decision(stats, bool(record["high_correct"]), "high")
        elif low_fallback and latencies["low"] <= deadline and record["real_low_conf"] >= low_threshold:
            record_decision(stats, bool(record["real_low_correct"]), "low")
        else:
            stats["deadline_miss"] += 1
    return finalize(stats, len(records), weights)


def simulate_proposed(records, deadlines, low_threshold, mid_threshold, mid_correct, mid_conf, latencies, weights, scenario_name, extra):
    stats = empty_stats("proposed_deadline_aware_low_mid_high", scenario_name, low_threshold, mid_threshold, extra)
    for idx, (record, deadline) in enumerate(zip(records, deadlines)):
        elapsed = 0.0
        stats["low_activations"] += 1
        stats["energy_proxy"] += latencies["low"]
        elapsed += latencies["low"]
        if elapsed > deadline:
            stats["safe_eject_or_resort"] += 1
            continue
        if record["real_low_conf"] >= low_threshold:
            record_decision(stats, bool(record["real_low_correct"]), "low")
            continue

        if elapsed + latencies["mid"] > deadline:
            stats["safe_eject_or_resort"] += 1
            continue
        stats["mid_activations"] += 1
        stats["energy_proxy"] += latencies["mid"]
        elapsed += latencies["mid"]
        if mid_conf[idx] >= mid_threshold:
            record_decision(stats, bool(mid_correct[idx]), "mid")
            continue

        if elapsed + latencies["high"] > deadline:
            stats["safe_eject_or_resort"] += 1
            continue
        stats["high_activations"] += 1
        stats["energy_proxy"] += latencies["high"]
        record_decision(stats, bool(record["high_correct"]), "high")
    return finalize(stats, len(records), weights)


def best_by_policy(rows):
    result = {}
    for row in rows:
        key = (row["scenario"], row["policy"])
        if key not in result or row["score"] > result[key]["score"]:
            result[key] = row
    return list(result.values())


def best_overall_by_scenario(rows):
    result = {}
    for row in rows:
        key = row["scenario"]
        if key not in result or row["score"] > result[key]["score"]:
            result[key] = row
    return list(result.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-name", default="virtual_deadline_simulation.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    records = load_records(args.max_samples, args.batch_size)
    n = len(records)
    latencies = dict(DEFAULT_STAGE_LATENCIES)
    weights = {
        "unsafe": 10.0,
        "safe": 1.0,
        "miss": 5.0,
        "energy": 0.02,
    }

    low_thresholds = np.linspace(0.5, 0.95, 10)
    mid_thresholds = [0.55, 0.65, 0.75]
    mid_envelopes = [
        {"recover_hard": 0.25, "retain_low_correct": 0.90, "recover_impossible": 0.05},
        {"recover_hard": 0.50, "retain_low_correct": 0.94, "recover_impossible": 0.08},
        {"recover_hard": 0.75, "retain_low_correct": 0.97, "recover_impossible": 0.12},
    ]

    all_rows = []
    for scenario in scenarios():
        scenario_name, deadlines = make_deadlines(n, scenario, args.seed)
        for low_threshold in low_thresholds:
            all_rows.append(simulate_lightweight(records, deadlines, low_threshold, latencies, weights, scenario_name))
            all_rows.append(simulate_cascade(records, deadlines, low_threshold, latencies, weights, scenario_name))
            all_rows.append(simulate_deadline_cascade(records, deadlines, low_threshold, latencies, weights, scenario_name))
            all_rows.append(simulate_parallel(records, deadlines, low_threshold, latencies, weights, scenario_name, low_fallback=False))
            all_rows.append(simulate_parallel(records, deadlines, low_threshold, latencies, weights, scenario_name, low_fallback=True))

            for envelope_idx, envelope in enumerate(mid_envelopes):
                mid_correct, mid_conf = make_mid_proxy(
                    records,
                    seed=args.seed + 1000 * envelope_idx,
                    **envelope,
                )
                for mid_threshold in mid_thresholds:
                    all_rows.append(simulate_proposed(
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
                    ))

    best_policy_rows = sorted(best_by_policy(all_rows), key=lambda row: (row["scenario"], row["policy"]))
    best_scenario_rows = sorted(best_overall_by_scenario(all_rows), key=lambda row: row["scenario"])

    summary = {
        "status": "ok",
        "purpose": "Virtual deadline simulation for deadline-aware FPGA adaptive optical sorting policy.",
        "samples": n,
        "category_counts": category_counts(records),
        "latencies": latencies,
        "weights": weights,
        "deadline_scenarios": scenarios(),
        "policies": sorted({row["policy"] for row in all_rows}),
        "best_by_policy": best_policy_rows,
        "best_by_scenario": best_scenario_rows,
        "all_results_count": len(all_rows),
        "notes": [
            "This is a CIFAR-100 proxy experiment, not proof for optical sorting.",
            "Safe eject/re-sort is treated as safe but not useful automatic classification.",
            "The MID stage is a parametric envelope; replace with a real FPGA-feasible model before making strong claims.",
        ],
    }
    output_path = RESULTS_DIR / args.output_name
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
