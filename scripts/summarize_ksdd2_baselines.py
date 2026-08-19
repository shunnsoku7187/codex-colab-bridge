"""Summarize KSDD2 baseline result files for value-add evaluation.

This script is intentionally CPU-only.  It separates two questions:

1. Does the off-the-shelf baseline behave like a reasonable defect detector?
2. Where can a proposed early-exit / FPGA implementation add value on top of
   that baseline?

Strict false-pass / good-pass operating points are therefore treated as
application constraints, not as proof that the baseline technology itself
failed to reproduce prior work.
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


def mean_test_auc(payload: dict[str, Any], score_name: str) -> dict[str, float | None]:
    aurocs = []
    auprs = []
    for seed in payload.get("seed_results", []):
        auc = seed.get("test_auc", {}).get(score_name, {})
        if auc.get("image_auroc") is not None:
            aurocs.append(float(auc["image_auroc"]))
        if auc.get("image_aupr") is not None:
            auprs.append(float(auc["image_aupr"]))
    return {
        "mean_test_auroc": round(sum(aurocs) / len(aurocs), 6) if aurocs else None,
        "mean_test_aupr": round(sum(auprs) / len(auprs), 6) if auprs else None,
    }


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
    auc = mean_test_auc(payload, str(selected.get("score_name", "topk_score")))
    return {
        "path": str(path),
        "dataset": payload.get("dataset", {}).get("name", "unknown"),
        "model": model_label(payload, path),
        "score": selected.get("score_name", "-"),
        **auc,
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
        "# KSDD2 baseline value-add evaluation",
        "",
        "目的は、既存の欠陥検出モデルを否定することではない。既存技術を土台にしたとき、提案手法やFPGA化でどこに価値を足せるかを確認する。",
        "",
        "読み分け:",
        "",
        "- `mean test AUROC/AUPR`: 既存技術として欠陥スコアが妥当に出ているかを見る再現・土台確認の指標。",
        "- `worst false-pass/good-pass`: 検品ライン風の運用制約に置いたとき、どこが不足しやすいかを見る指標。",
        "- 低いfalse-pass条件は、既存技術の合否判定ではなく、提案手法/FPGA化が改善すべき運用上の負荷を見つけるための条件。",
        "",
        "| result | model | score | mean test AUROC | mean test AUPR | operating false-pass | operating good-pass | feasible seeds | worst false-pass | worst good-pass | value-add reading |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        feasible = "-"
        if row["feasible_seeds"] is not None and row["seeds"] is not None:
            feasible = f"{row['feasible_seeds']}/{row['seeds']}"
        if row["worst_false_pass"] is None:
            reading = "-"
        elif float(row["worst_false_pass"]) <= 0.01:
            reading = "土台検出器は十分強い。次は計算量・電力削減を評価する。"
        else:
            reading = "欠陥スコアは有効だが、運用閾値には改善余地がある。校正・選択的判定・FPGA化の効果を見る。"
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(row["path"]).name,
                    str(row["model"]),
                    str(row["score"]),
                    "-" if row["mean_test_auroc"] is None else f"{row['mean_test_auroc']:.4f}",
                    "-" if row["mean_test_aupr"] is None else f"{row['mean_test_aupr']:.4f}",
                    pct(row["constraint_false_pass"]),
                    pct(row["constraint_good_pass"]),
                    feasible,
                    pct(row["worst_false_pass"]),
                    pct(row["worst_good_pass"]),
                    reading,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "次に見るべきこと:",
            "",
            "- 既存技術の再現はAUROC/AUPRや先行研究の評価条件で見る。",
            "- 提案手法の価値は、同じ土台モデルに対して計算量・消費電力・レイテンシ・棄却率・閾値安定性がどれだけ改善するかで見る。",
            "- FPGA化の価値は、分岐後に動かさない回路を物理的に止める、複数段をパイプライン化する、固定小数点化で電力と資源を見積もる、という追加指標で測る。",
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
        "purpose": "Compare KSDD2 baselines as foundations, then identify where early-exit or FPGA implementation can add value.",
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
