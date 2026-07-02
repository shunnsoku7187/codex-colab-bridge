import math


def _pct(count, total):
    return 100.0 * count / total if total else 0.0


def category_counts(records):
    counts = {"Easy": 0, "Hard": 0, "Impossible": 0, "Inverse": 0}
    for item in records:
        low_correct = bool(item["low_correct"])
        high_correct = bool(item["high_correct"])
        if low_correct and high_correct:
            counts["Easy"] += 1
        elif not low_correct and high_correct:
            counts["Hard"] += 1
        elif not low_correct and not high_correct:
            counts["Impossible"] += 1
        else:
            counts["Inverse"] += 1
    return counts


def oracle_at_target(records, flops_low, flops_high, flops_router, target_margin_pct=1.0):
    total = len(records)
    counts = category_counts(records)
    high_correct = counts["Easy"] + counts["Hard"]
    low_correct = counts["Easy"] + counts["Inverse"]
    high_accuracy = _pct(high_correct, total)
    low_accuracy = _pct(low_correct, total)
    target_accuracy = high_accuracy - target_margin_pct

    target_correct = int(math.ceil(target_accuracy * total / 100.0))
    correct_without_hard_high = counts["Easy"] + counts["Inverse"]
    needed_hard_high = max(0, target_correct - correct_without_hard_high)
    hard_low_allowed = max(0, counts["Hard"] - needed_hard_high)
    oracle_low = counts["Easy"] + counts["Inverse"] + counts["Impossible"] + hard_low_allowed
    oracle_high = total - oracle_low
    oracle_cost = (total * flops_router + oracle_low * flops_low + oracle_high * flops_high) / total

    return {
        "counts": counts,
        "high_accuracy": high_accuracy,
        "low_accuracy": low_accuracy,
        "target_accuracy": target_accuracy,
        "all_low_meets_target": low_accuracy >= target_accuracy,
        "oracle_low": oracle_low,
        "oracle_high": oracle_high,
        "oracle_low_rate": oracle_low / total if total else 0.0,
        "oracle_high_rate": oracle_high / total if total else 0.0,
        "oracle_cost": float(oracle_cost),
        "hard_low_allowed_at_target": hard_low_allowed,
    }


def guardrail_report(
    records,
    best,
    flops_low,
    flops_high,
    flops_router,
    target_margin_pct=1.0,
    min_high_rate=0.05,
    min_low_rate=0.05,
):
    total = len(records)
    oracle = oracle_at_target(records, flops_low, flops_high, flops_router, target_margin_pct)
    all_low_cost = flops_router + flops_low
    all_high_cost = flops_router + flops_high
    baselines = {
        "all_low": {
            "accuracy": oracle["low_accuracy"],
            "avg_cost": float(all_low_cost),
            "meets_target": oracle["all_low_meets_target"],
        },
        "all_high": {
            "accuracy": oracle["high_accuracy"],
            "avg_cost": float(all_high_cost),
            "meets_target": True,
        },
        "oracle": oracle,
    }

    flags = []
    if oracle["all_low_meets_target"]:
        flags.append("benchmark_degenerate_all_low_meets_target")
    if oracle["oracle_high"] == 0:
        flags.append("benchmark_degenerate_oracle_routes_no_high")

    if best is None:
        flags.append("no_feasible_threshold")
        return {
            "baselines": baselines,
            "candidate": None,
            "flags": flags,
            "valid_for_claim": False,
            "recommended_action": "discard_or_reformulate_threshold_search",
        }

    to_low = int(best["to_low"])
    to_high = int(best["to_high"])
    low_rate = to_low / total if total else 0.0
    high_rate = to_high / total if total else 0.0

    if to_high == 0:
        flags.append("candidate_all_low_escape")
    elif high_rate < min_high_rate:
        flags.append("candidate_near_all_low_escape")

    if to_low == 0:
        flags.append("candidate_all_high_escape")
    elif low_rate < min_low_rate:
        flags.append("candidate_near_all_high_escape")

    oracle_cost = oracle["oracle_cost"]
    if best["avg_cost"] <= oracle_cost + 0.05:
        flags.append("candidate_near_oracle_cost_requires_heldout_validation")

    valid_for_claim = not any(
        flag.startswith("benchmark_degenerate")
        or flag in {"candidate_all_low_escape", "candidate_all_high_escape"}
        for flag in flags
    )
    if valid_for_claim and any(flag.startswith("candidate_near_all") for flag in flags):
        recommended_action = "treat_as_warning_and_require_secondary_validation"
    elif valid_for_claim:
        recommended_action = "eligible_after_heldout_or_cv_validation"
    else:
        recommended_action = "do_not_claim_as_router_result"

    return {
        "baselines": baselines,
        "candidate": {
            "to_low_rate": low_rate,
            "to_high_rate": high_rate,
            "cost_margin_vs_all_low": float(best["avg_cost"] - all_low_cost),
            "cost_margin_vs_all_high": float(best["avg_cost"] - all_high_cost),
            "cost_margin_vs_oracle": float(best["avg_cost"] - oracle_cost),
        },
        "flags": flags,
        "valid_for_claim": valid_for_claim,
        "recommended_action": recommended_action,
    }
