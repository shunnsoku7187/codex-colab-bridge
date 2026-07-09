"""Analyze whether low self-confidence ResNet decisions are rescued later.

This focuses on the user's current question:

* If an early exit is low-confidence, does a later exit actually overturn it
  into a high-confidence correct decision?
* Or are low-confidence samples mostly still unsafe later, making further
  computation less useful for inspection-style decisions?

MobileNet is intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def low_conf_mask(confidence: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(confidence, quantile))
    return confidence <= threshold, threshold


def high_conf_threshold(confidence: np.ndarray, quantile: float) -> float:
    return float(np.quantile(confidence, quantile))


def summarize_transition(
    source_name: str,
    target_name: str,
    source_conf: np.ndarray,
    source_correct: np.ndarray,
    target_conf: np.ndarray,
    target_correct: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> dict[str, Any]:
    mask, low_threshold = low_conf_mask(source_conf, low_quantile)
    high_threshold = high_conf_threshold(target_conf, high_quantile)
    n = int(mask.sum())
    if n == 0:
        raise RuntimeError(f"No samples selected for {source_name}")

    source_wrong = ~source_correct[mask]
    target_fixed = source_wrong & target_correct[mask]
    target_high = target_conf[mask] >= high_threshold
    rescued_high_correct = source_wrong & target_correct[mask] & target_high
    still_unsafe = (~target_correct[mask]) | (~target_high)

    return {
        "source": source_name,
        "target": target_name,
        "low_conf_quantile": low_quantile,
        "source_low_conf_threshold": round(low_threshold, 6),
        "target_high_conf_quantile": high_quantile,
        "target_high_conf_threshold": round(high_threshold, 6),
        "selected_count": n,
        "selected_rate": round(float(mask.mean()), 6),
        "source_accuracy_in_selected": round(float(source_correct[mask].mean()), 6),
        "target_accuracy_in_selected": round(float(target_correct[mask].mean()), 6),
        "target_high_conf_rate_in_selected": round(float(target_high.mean()), 6),
        "wrong_at_source_rate_in_selected": round(float(source_wrong.mean()), 6),
        "rescued_to_correct_rate_among_selected": round(float(target_fixed.mean()), 6),
        "rescued_to_correct_rate_among_source_wrong": round(float(target_fixed.sum() / max(1, source_wrong.sum())), 6),
        "rescued_to_high_conf_correct_rate_among_selected": round(float(rescued_high_correct.mean()), 6),
        "rescued_to_high_conf_correct_rate_among_source_wrong": round(float(rescued_high_correct.sum() / max(1, source_wrong.sum())), 6),
        "still_unsafe_rate_after_target": round(float(still_unsafe.mean()), 6),
    }


def branchy_resnet_transitions(branchy_npz: Path, low_quantiles: list[float], high_quantile: float) -> list[dict[str, Any]]:
    data = np.load(branchy_npz, allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]

    rows = []
    for q in low_quantiles:
        rows.append(summarize_transition(
            exit_names[0],
            exit_names[1],
            confidence[:, 0],
            correct[:, 0],
            confidence[:, 1],
            correct[:, 1],
            q,
            high_quantile,
        ))
        rows.append(summarize_transition(
            exit_names[0],
            exit_names[2],
            confidence[:, 0],
            correct[:, 0],
            confidence[:, 2],
            correct[:, 2],
            q,
            high_quantile,
        ))
        rows.append(summarize_transition(
            exit_names[1],
            exit_names[2],
            confidence[:, 1],
            correct[:, 1],
            confidence[:, 2],
            correct[:, 2],
            q,
            high_quantile,
        ))
    return rows


def standalone_resnet_transitions(small_npz: Path, low_quantiles: list[float], high_quantile: float) -> list[dict[str, Any]]:
    data = np.load(small_npz, allow_pickle=True)
    rows = []
    pairs = [
        ("resnet20", "resnet32"),
        ("resnet20", "resnet44"),
        ("resnet32", "resnet44"),
    ]
    for q in low_quantiles:
        for source, target in pairs:
            rows.append(summarize_transition(
                source,
                target,
                np.asarray(data[f"{source}_confidence"], dtype=float),
                np.asarray(data[f"{source}_correct"], dtype=bool),
                np.asarray(data[f"{target}_confidence"], dtype=float),
                np.asarray(data[f"{target}_correct"], dtype=bool),
                q,
                high_quantile,
            ))
    return rows


def best_rows(rows: list[dict[str, Any]], low_quantile: float) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["low_conf_quantile"] == low_quantile]
    return sorted(
        filtered,
        key=lambda row: (
            -row["rescued_to_high_conf_correct_rate_among_selected"],
            row["still_unsafe_rate_after_target"],
        ),
    )


def summarize(branchy_rows: list[dict[str, Any]], standalone_rows: list[dict[str, Any]], low_quantiles: list[float]) -> dict[str, Any]:
    summary_rows = []
    for q in low_quantiles:
        branchy_best = best_rows(branchy_rows, q)[0]
        standalone_best = best_rows(standalone_rows, q)[0]
        summary_rows.append({
            "low_conf_quantile": q,
            "best_branchy_transition": {
                "source": branchy_best["source"],
                "target": branchy_best["target"],
                "rescued_high_conf_correct_selected": branchy_best["rescued_to_high_conf_correct_rate_among_selected"],
                "rescued_high_conf_correct_source_wrong": branchy_best["rescued_to_high_conf_correct_rate_among_source_wrong"],
                "still_unsafe_after_target": branchy_best["still_unsafe_rate_after_target"],
            },
            "best_standalone_resnet_transition": {
                "source": standalone_best["source"],
                "target": standalone_best["target"],
                "rescued_high_conf_correct_selected": standalone_best["rescued_to_high_conf_correct_rate_among_selected"],
                "rescued_high_conf_correct_source_wrong": standalone_best["rescued_to_high_conf_correct_rate_among_source_wrong"],
                "still_unsafe_after_target": standalone_best["still_unsafe_rate_after_target"],
            },
            "branchy_minus_standalone_rescue_rate": round(
                branchy_best["rescued_to_high_conf_correct_rate_among_selected"]
                - standalone_best["rescued_to_high_conf_correct_rate_among_selected"],
                6,
            ),
        })
    return {
        "purpose": "Decide whether ResNet BranchyNet has value in sending low self-confidence early-exit samples to later exits.",
        "interpretation": {
            "keep_rescue_route": "Keep only if early low-confidence samples are often converted to high-confidence correct decisions later, preferably more than standalone ResNet scaling does.",
            "cut_or_demote": "If low-confidence samples remain unsafe later, or standalone ResNet scaling rescues them as well or better, then this is not a strong BranchyNet-specific route.",
        },
        "summary_rows": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branchy-npz", default="results/0000b_branchynet_reproduce_resnet56_cifar10.npz")
    parser.add_argument("--small-npz", default="results/deadline_vs_small_model_001_small_outputs.npz")
    parser.add_argument("--output", default="results/resnet_low_conf_rescue_analysis_001_summary.json")
    parser.add_argument("--low-quantiles", nargs="*", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--high-quantile", type=float, default=0.7)
    args = parser.parse_args()

    branchy_path = Path(args.branchy_npz)
    small_path = Path(args.small_npz)
    if not branchy_path.exists():
        raise FileNotFoundError(branchy_path)
    if not small_path.exists():
        raise FileNotFoundError(small_path)

    branchy_rows = branchy_resnet_transitions(branchy_path, args.low_quantiles, args.high_quantile)
    standalone_rows = standalone_resnet_transitions(small_path, args.low_quantiles, args.high_quantile)
    payload = {
        "summary": summarize(branchy_rows, standalone_rows, args.low_quantiles),
        "branchy_resnet_transitions": branchy_rows,
        "standalone_resnet_transitions": standalone_rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
