"""Audit selective classification and dual-sided early exit on BranchyNet traces.

This script separates two claims:

1. Selective classification value:
   If we only emit labels for high-confidence samples, how does risk change
   as coverage changes?

2. Dual-sided early-exit value:
   Can an early reject/accept policy approach the same selective risk while
   lowering normalized compute cost?

The implementation uses existing per-exit predictions/confidences from a saved
BranchyNet trace. It is a reproduction of the selective-classification
risk-coverage evaluation framework, not yet a full SelectiveNet training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def round_float(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def coverage_points() -> list[float]:
    return [round(float(x), 2) for x in np.linspace(1.0, 0.1, 19)]


def selective_curve(correct: np.ndarray, score: np.ndarray, coverages: list[float]) -> list[dict[str, Any]]:
    n = len(correct)
    order = np.argsort(-score)
    rows = []
    for coverage in coverages:
        k = max(1, int(round(n * coverage)))
        chosen = order[:k]
        risk = float((~correct[chosen]).mean())
        rows.append(
            {
                "coverage": round_float(k / n),
                "selective_risk": round_float(risk),
                "selective_accuracy": round_float(1.0 - risk),
                "threshold": round_float(float(score[order[k - 1]])),
                "accepted": int(k),
            }
        )
    return rows


def make_thresholds(values: np.ndarray, quantiles: list[float]) -> list[float]:
    return sorted({float(x) for x in np.quantile(values, quantiles)})


def evaluate_dual_policy(
    correct: np.ndarray,
    confidence: np.ndarray,
    costs: np.ndarray,
    lower0: float,
    upper0: float,
    lower1: float,
    upper1: float,
    final_threshold: float,
) -> dict[str, Any]:
    n = correct.shape[0]
    final_idx = correct.shape[1] - 1
    terminal_exit = np.full(n, final_idx, dtype=np.int16)
    accepted = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)
    accepted_correct = np.zeros(n, dtype=bool)

    accept0 = confidence[:, 0] >= upper0
    reject0 = (confidence[:, 0] <= lower0) & ~accept0
    accepted[accept0] = True
    accepted_correct[accept0] = correct[accept0, 0]
    terminal_exit[accept0 | reject0] = 0
    rejected[reject0] = True

    unresolved = ~(accept0 | reject0)
    accept1 = unresolved & (confidence[:, 1] >= upper1)
    reject1 = unresolved & (confidence[:, 1] <= lower1) & ~accept1
    accepted[accept1] = True
    accepted_correct[accept1] = correct[accept1, 1]
    terminal_exit[accept1 | reject1] = 1
    rejected[reject1] = True

    final_mask = ~(accepted | rejected)
    final_accept = final_mask & (confidence[:, final_idx] >= final_threshold)
    final_reject = final_mask & ~final_accept
    accepted[final_accept] = True
    accepted_correct[final_accept] = correct[final_accept, final_idx]
    rejected[final_reject] = True

    accept_count = int(accepted.sum())
    risk = None if accept_count == 0 else float((~accepted_correct[accepted]).mean())
    early_reject = reject0 | reject1
    return {
        "thresholds": {
            "lower0": round_float(lower0),
            "upper0": round_float(upper0),
            "lower1": round_float(lower1),
            "upper1": round_float(upper1),
            "final": round_float(final_threshold),
        },
        "coverage": round_float(float(accepted.mean())),
        "selective_risk": round_float(risk),
        "selective_accuracy": None if risk is None else round_float(1.0 - risk),
        "reject_rate": round_float(float(rejected.mean())),
        "early_reject_rate": round_float(float(early_reject.mean())),
        "early_accept_rate": round_float(float((accept0 | accept1).mean())),
        "final_execution_rate": round_float(float(final_mask.mean())),
        "avg_cost": round_float(float(costs[terminal_exit].mean())),
    }


def nearest_row(curve: list[dict[str, Any]], target_coverage: float) -> dict[str, Any]:
    return min(curve, key=lambda row: abs(row["coverage"] - target_coverage))


def best_dual_for_coverage(
    correct: np.ndarray,
    confidence: np.ndarray,
    costs: np.ndarray,
    target_coverage: float,
    max_risk: float,
    grid_quantiles: list[float],
) -> dict[str, Any] | None:
    lower0_values = make_thresholds(confidence[:, 0], grid_quantiles)
    upper0_values = make_thresholds(confidence[:, 0], grid_quantiles)
    lower1_values = make_thresholds(confidence[:, 1], grid_quantiles)
    upper1_values = make_thresholds(confidence[:, 1], grid_quantiles)
    final_values = make_thresholds(confidence[:, -1], grid_quantiles)
    rows = []
    for lower0 in lower0_values:
        for upper0 in upper0_values:
            if lower0 >= upper0:
                continue
            for lower1 in lower1_values:
                for upper1 in upper1_values:
                    if lower1 >= upper1:
                        continue
                    for final_threshold in final_values:
                        row = evaluate_dual_policy(correct, confidence, costs, lower0, upper0, lower1, upper1, final_threshold)
                        if row["selective_risk"] is None:
                            continue
                        if row["selective_risk"] <= max_risk and row["coverage"] >= target_coverage:
                            rows.append(row)
    if not rows:
        return None
    return min(rows, key=lambda row: (row["avg_cost"], -row["coverage"], row["selective_risk"]))


def write_svg_fallback(payload: dict[str, Any], svg_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    final_curve = payload["selective_classification"]["softmax_response_final"]
    exit0_curve = payload["selective_classification"]["softmax_response_exit0"]
    exit1_curve = payload["selective_classification"]["softmax_response_exit1"]
    dual_rows = [row for row in payload["dual_exit_against_final_sr"].values() if row is not None]

    width, height = 980, 380
    margin = 52
    panel_w = (width - 3 * margin) / 2
    panel_h = height - 2 * margin

    def x(panel: int, coverage: float) -> float:
        left = margin + panel * (panel_w + margin)
        return left + (1.0 - coverage) / 0.9 * panel_w

    def y_risk(risk: float) -> float:
        return margin + (1.0 - min(max(risk / 0.36, 0.0), 1.0)) * panel_h

    def y_cost(cost: float) -> float:
        return margin + (1.0 - min(max(cost, 0.0), 1.0)) * panel_h

    def polyline(curve: list[dict[str, Any]], color: str, value_key: str, y_fn) -> str:
        points = " ".join(f"{x(0, r['coverage']):.1f},{y_fn(r[value_key]):.1f}" for r in curve)
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#0f172a}.title{font-size:16px;font-weight:700}.axis{stroke:#94a3b8;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}</style>',
        f'<text x="{margin}" y="24" class="title">Selective classification: risk-coverage</text>',
        f'<text x="{2 * margin + panel_w}" y="24" class="title">Dual-side early exit under final-SR risk target</text>',
    ]
    for panel in [0, 1]:
        left = margin + panel * (panel_w + margin)
        right = left + panel_w
        bottom = margin + panel_h
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{margin}" x2="{left}" y2="{bottom}"/>')
        for c in [1.0, 0.8, 0.6, 0.4, 0.2]:
            parts.append(f'<line class="grid" x1="{x(panel, c):.1f}" y1="{margin}" x2="{x(panel, c):.1f}" y2="{bottom}"/>')
            parts.append(f'<text x="{x(panel, c) - 10:.1f}" y="{bottom + 18}">{c:.1f}</text>')
    parts += [
        polyline(exit0_curve, "#2563eb", "selective_risk", y_risk),
        polyline(exit1_curve, "#f97316", "selective_risk", y_risk),
        polyline(final_curve, "#16a34a", "selective_risk", y_risk),
        f'<text x="{margin + 10}" y="{height - 12}">coverage</text>',
        f'<text x="{margin - 44}" y="{margin + 10}">risk</text>',
        f'<text x="{margin + 160}" y="{margin + 20}" fill="#2563eb">exit0</text>',
        f'<text x="{margin + 215}" y="{margin + 20}" fill="#f97316">exit1</text>',
        f'<text x="{margin + 270}" y="{margin + 20}" fill="#16a34a">final</text>',
    ]
    right_left = 2 * margin + panel_w
    right_bottom = margin + panel_h
    parts.append(f'<line x1="{right_left}" y1="{y_cost(1.0):.1f}" x2="{right_left + panel_w}" y2="{y_cost(1.0):.1f}" stroke="#64748b" stroke-width="2.5"/>')
    for row in dual_rows:
        parts.append(f'<circle cx="{x(1, row["coverage"]):.1f}" cy="{y_cost(row["avg_cost"]):.1f}" r="5" fill="#16a34a"/>')
    parts += [
        f'<text x="{right_left + 10}" y="{height - 12}">coverage</text>',
        f'<text x="{right_left - 42}" y="{margin + 10}">cost</text>',
        f'<text x="{right_left + 165}" y="{margin + 20}" fill="#64748b">final-only</text>',
        f'<text x="{right_left + 260}" y="{margin + 20}" fill="#16a34a">dual-side</text>',
        "</svg>",
    ]
    svg_path.write_text("\n".join(parts), encoding="utf-8")


def maybe_plot(payload: dict[str, Any], plot_path: Path) -> None:
    write_svg_fallback(payload, plot_path.with_suffix(".svg"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    final_curve = payload["selective_classification"]["softmax_response_final"]
    exit0_curve = payload["selective_classification"]["softmax_response_exit0"]
    exit1_curve = payload["selective_classification"]["softmax_response_exit1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, curve in [("exit0", exit0_curve), ("exit1", exit1_curve), ("final", final_curve)]:
        axes[0].plot([r["coverage"] for r in curve], [100 * r["selective_risk"] for r in curve], marker="o", label=label)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("coverage")
    axes[0].set_ylabel("selective risk [%]")
    axes[0].set_title("Selective classification: risk-coverage")
    axes[0].legend()

    dual_rows = [row for row in payload["dual_exit_against_final_sr"].values() if row is not None]
    if dual_rows:
        axes[1].scatter([r["coverage"] for r in dual_rows], [r["avg_cost"] for r in dual_rows], color="#16a34a", label="dual-side")
    axes[1].plot([r["coverage"] for r in final_curve], [1.0 for _ in final_curve], color="#475569", label="final-only cost")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("coverage")
    axes[1].set_ylabel("normalized compute cost")
    axes[1].set_title("Dual-side early exit under final-SR risk target")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="results/0000b_branchynet_reproduce_resnet56_cifar10.npz")
    parser.add_argument("--output", default="results/selective_dual_exit_audit_001_summary.json")
    parser.add_argument("--plot", default="results/selective_dual_exit_audit_001_risk_coverage.png")
    parser.add_argument("--grid-quantiles", nargs="*", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99])
    args = parser.parse_args()

    data = np.load(Path(args.trace), allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]
    coverages = coverage_points()

    exit_curves = {
        "softmax_response_exit0": selective_curve(correct[:, 0], confidence[:, 0], coverages),
        "softmax_response_exit1": selective_curve(correct[:, 1], confidence[:, 1], coverages),
        "softmax_response_final": selective_curve(correct[:, -1], confidence[:, -1], coverages),
    }

    dual = {}
    for target in [0.95, 0.90, 0.80, 0.70, 0.60]:
        baseline = nearest_row(exit_curves["softmax_response_final"], target)
        dual[f"coverage_at_least_{target:.2f}_risk_le_final_sr"] = best_dual_for_coverage(
            correct,
            confidence,
            costs,
            target_coverage=target,
            max_risk=baseline["selective_risk"],
            grid_quantiles=args.grid_quantiles,
        )

    payload = {
        "purpose": "Reproduce selective-classification risk-coverage evaluation and test whether dual-sided early exit can reduce compute under comparable final softmax-response risk targets.",
        "trace": args.trace,
        "exit_names": exit_names,
        "exit_costs": [round_float(x) for x in costs],
        "definitions": {
            "coverage": "fraction of samples for which the system emits a class label",
            "selective_risk": "classification error among emitted labels only",
            "softmax_response": "baseline selective classifier that accepts samples by descending max softmax confidence",
            "dual_side_early_exit": "accept high-confidence early exits, reject low-confidence early exits, and send middle-confidence samples deeper",
        },
        "selective_classification": exit_curves,
        "dual_exit_against_final_sr": dual,
        "implementation_notes": {
            "current_status": "metric reproduction and policy audit using an existing BranchyNet trace",
            "not_yet_done": "full SelectiveNet training with a learned selection head",
            "fpga_relevance": "risk-coverage gives the software objective; dual-side policies test whether reject decisions can be moved earlier to reduce hardware work",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    maybe_plot(payload, Path(args.plot))
    print(json.dumps({"wrote": str(out), "plot": args.plot, "dual_keys": list(dual)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
