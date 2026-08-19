"""Summarize KSDD2 final-inspection baseline result files.

This script is intentionally CPU-only.  It compares completed KSDD2 baseline
summaries using inspection-facing metrics: defect false-pass rate, good-pass
rate, and seed stability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_label(payload: dict[str, Any], path: Path) -> str:
    model = payload.get("model", {})
    architecture = model.get("architecture")
    encoder = model.get("encoder")
    if architecture and encoder:
        return f"{architecture}/{encoder}"
    if architecture:
        return str(architecture)
    return path.stem.replace("_summary", "")


def best_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("aggregate_rows", [])
    if not isinstance(rows, list):
        return []
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("worst_false_pass_rate_defect", 1.0)),
            -float(row.get("worst_good_pass_rate_good", 0.0)),
            -float(row.get("test_feasible_seeds", 0)),
        ),
    )


def summarize_file(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    rows = best_rows(payload)
    strict_rows = [
        row
        for row in rows
        if float(row.get("max_false_pass_rate_defect", 1.0)) <= 0.01
        and float(row.get("min_good_pass_rate_good", 0.0)) >= 0.90
    ]
    selected = strict_rows[0] if strict_rows else (rows[0] if rows else {})
    return {
        "path": str(path),
        "dataset": payload.get("dataset", {}).get("name", "unknown"),
        "model": model_label(payload, path),
        "score": selected.get("score_name", "-"),
        "constraint_false_pass": selected.get("max_false_pass_rate_defect"),
        "constraint_good_pass": selected.get("min_good_pass_rate_good"),
        "feasible_seeds": selected.get("test_feasible_seeds"),
        "seeds": selected.get("seeds"),
        "mean_good_pass": selected.get("mean_good_pass_rate_good"),
        "worst_good_pass": selected.get("worst_good_pass_rate_good"),
        "mean_false_pass": selected.get("mean_false_pass_rate_defect"),
        "worst_false_pass": selected.get("worst_false_pass_rate_defect"),
        "curve_png": payload.get("curve_png", ""),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# KSDD2 baseline comparison",
        "",
        "This table compares final-only inspection baselines.  Lower defect false-pass is better; higher good-pass is better.",
        "",
        "| result | model | score | target false-pass | target good-pass | feasible seeds | worst false-pass | worst good-pass | curve |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        feasible = "-"
        if row["feasible_seeds"] is not None and row["seeds"] is not None:
            feasible = f"{row['feasible_seeds']}/{row['seeds']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(row["path"]).name,
                    str(row["model"]),
                    str(row["score"]),
                    pct(row["constraint_false_pass"]),
                    pct(row["constraint_good_pass"]),
                    feasible,
                    pct(row["worst_false_pass"]),
                    pct(row["worst_good_pass"]),
                    str(row["curve_png"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Reading guide:",
            "",
            "- If no row has feasible seeds under a 1% false-pass target, the final detector is still not safe enough for inspection claims.",
            "- Prefer the model with the lowest worst false-pass rate before adding early exits.",
            "- Use good-pass rate as the secondary criterion because rejecting good products is a cost, but passing defects is the safety failure.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", default=[])
    parser.add_argument("--glob", default="results/ksdd2*_baseline*_summary.json")
    parser.add_argument("--output", default="results/ksdd2_baseline_comparison.json")
    parser.add_argument("--markdown", default="docs/ksdd2_baseline_comparison.md")
    args = parser.parse_args()

    paths = [Path(item) for item in args.inputs] if args.inputs else sorted(Path(".").glob(args.glob))
    summaries = [summarize_file(path) for path in paths if path.exists()]
    summaries.sort(key=lambda row: (float(row.get("worst_false_pass") or 1.0), -float(row.get("worst_good_pass") or 0.0)))

    payload = {
        "purpose": "Compare KSDD2 final-only inspection baselines before selecting a base model for early-exit experiments.",
        "inputs": [str(path) for path in paths],
        "rows": summaries,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(markdown(summaries), encoding="utf-8")
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "rows": len(summaries)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
