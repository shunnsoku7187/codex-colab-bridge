"""Evaluate single-seed and ensemble KSDD2 score files.

The GPU jobs can save per-seed validation/test defect scores under
`results/<job>_scores/`.  This CPU-only script checks whether averaging seeds
stabilizes the operating trade-off before the baseline is fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.ksdd2_unet_inspection_baseline import candidates, metric, score_auc


def round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def load_score_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def choose_threshold(labels: np.ndarray, scores: np.ndarray, max_false_pass: float, min_good_pass: float, max_candidates: int) -> dict[str, Any] | None:
    best = None
    for threshold in candidates(scores, max_candidates):
        row = metric(labels, scores, threshold)
        if row["false_pass_rate_defect"] <= max_false_pass + 1e-12 and row["good_pass_rate_good"] >= min_good_pass - 1e-12:
            if best is None or (
                float(row["good_pass_rate_good"]),
                -float(row["false_pass_rate_defect"]),
            ) > (
                float(best["val_metric"]["good_pass_rate_good"]),
                -float(best["val_metric"]["false_pass_rate_defect"]),
            ):
                best = {"threshold": threshold, "val_metric": row}
    return best


def evaluate(name: str, val_labels: np.ndarray, val_scores: np.ndarray, test_labels: np.ndarray, test_scores: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    auc = score_auc(test_labels, test_scores)
    for max_fp in args.max_false_pass_rates:
        for min_gp in args.min_good_pass_rates:
            selected = choose_threshold(val_labels, val_scores, max_fp, min_gp, args.max_threshold_candidates)
            if selected is None:
                rows.append(
                    {
                        "name": name,
                        "max_false_pass_rate_defect": max_fp,
                        "min_good_pass_rate_good": min_gp,
                        "feasible_on_val": False,
                        "test_feasible": False,
                        "threshold": None,
                        **auc,
                    }
                )
                continue
            test_metric = metric(test_labels, test_scores, selected["threshold"])
            rows.append(
                {
                    "name": name,
                    "max_false_pass_rate_defect": max_fp,
                    "min_good_pass_rate_good": min_gp,
                    "feasible_on_val": True,
                    "test_feasible": (
                        test_metric["false_pass_rate_defect"] <= max_fp + 1e-12
                        and test_metric["good_pass_rate_good"] >= min_gp - 1e-12
                    ),
                    "threshold": round_float(selected["threshold"]),
                    "test_metric": test_metric,
                    **auc,
                }
            )
    return rows


def rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# KSDD2 score ensemble check",
        "",
        "Seed別スコアとseed平均アンサンブルを、同じvalidation-selected thresholdで比較する。",
        "",
        "| score source | target false-pass | target good-pass | val feasible | test feasible | test AUROC | test AUPR | test false-pass | test good-pass | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric_row = row.get("test_metric", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    pct(row["max_false_pass_rate_defect"]),
                    pct(row["min_good_pass_rate_good"]),
                    "yes" if row["feasible_on_val"] else "no",
                    "yes" if row["test_feasible"] else "no",
                    "-" if row.get("image_auroc") is None else f"{row['image_auroc']:.4f}",
                    "-" if row.get("image_aupr") is None else f"{row['image_aupr']:.4f}",
                    pct(metric_row.get("false_pass_rate_defect")),
                    pct(metric_row.get("good_pass_rate_good")),
                    "-" if row.get("threshold") is None else f"{row['threshold']:.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--markdown", default="")
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.01, 0.05])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.90, 0.95])
    parser.add_argument("--max-threshold-candidates", type=int, default=201)
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    paths = sorted(scores_dir.glob("seed_*_scores.npz"))
    if not paths:
        raise SystemExit(f"no score files found: {scores_dir}")

    loaded = [(path.stem.replace("_scores", ""), load_score_file(path)) for path in paths]
    all_rows: list[dict[str, Any]] = []
    for seed_name, data in loaded:
        for score_name in ["max_score", "topk_score"]:
            all_rows.extend(
                evaluate(
                    f"{seed_name}/{score_name}",
                    data["val_labels"],
                    data[f"val_{score_name}"],
                    data["test_labels"],
                    data[f"test_{score_name}"],
                    args,
                )
            )

    first = loaded[0][1]
    for _name, data in loaded[1:]:
        if not np.array_equal(first["val_labels"], data["val_labels"]) or not np.array_equal(first["test_labels"], data["test_labels"]):
            raise SystemExit("score files use different validation/test ordering; ensemble cannot be averaged safely")

    for score_name in ["max_score", "topk_score"]:
        val_stack = np.stack([data[f"val_{score_name}"] for _name, data in loaded], axis=0)
        test_stack = np.stack([data[f"test_{score_name}"] for _name, data in loaded], axis=0)
        all_rows.extend(
            evaluate(
                f"ensemble_mean/{score_name}",
                first["val_labels"],
                val_stack.mean(axis=0),
                first["test_labels"],
                test_stack.mean(axis=0),
                args,
            )
        )

    payload = {
        "purpose": "Check whether score averaging stabilizes the KSDD2 final-inspection operating point.",
        "scores_dir": str(scores_dir),
        "inputs": [str(path) for path in paths],
        "rows": all_rows,
    }
    output = Path(args.output or scores_dir.with_name(scores_dir.name + "_ensemble_summary.json"))
    markdown = Path(args.markdown or scores_dir.with_name(scores_dir.name + "_ensemble_summary.md"))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(rows_to_markdown(all_rows), encoding="utf-8")
    print(json.dumps({"wrote": str(output), "markdown": str(markdown), "rows": len(all_rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
