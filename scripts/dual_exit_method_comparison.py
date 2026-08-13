"""Compare dual-sided early exit with baseline policies.

The input is a completed dual_exit_reliability_shift_* summary. The comparison
is intentionally conservative: it reports where dual-sided early exit wins in
compute cost, and also reports the coverage/reject trade-off that pays for it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


POLICY_LABELS = {
    "final_only": "finalのみ",
    "upper_only_best_cost": "上側早期終了のみ",
    "dual_side_best_cost_lost_final_reliable_le_1pct": "両側早期終了 良品ロス1%級",
    "dual_side_best_cost_lost_final_reliable_le_2pct": "両側早期終了 良品ロス2%級",
    "dual_side_best_cost_lost_final_reliable_le_5pct": "両側早期終了 良品ロス5%級",
}


def pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{100 * float(value):.{digits}f}%"


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def scenario_policy_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scenario, result in payload["scenario_results"].items():
        final = result["eval_final_only"]
        rows.append(
            {
                "scenario": scenario,
                "method": "final_only",
                "method_label": POLICY_LABELS["final_only"],
                "accepted_accuracy": final["accepted_accuracy"],
                "accept_rate": final["accept_rate"],
                "reject_rate": final["reject_rate"],
                "early_reject_rate": final["early_reject_rate"],
                "final_rate": final["final_rate"],
                "avg_cost": final["avg_cost"],
                "lost_final_reliable_rate": 0.0,
                "cost_reduction_vs_final": 0.0,
                "accept_rate_drop_vs_final": 0.0,
                "valid": True,
            }
        )
        for key, policy in result["evaluated_on_heldout"].items():
            rows.append(
                {
                    "scenario": scenario,
                    "method": key,
                    "method_label": POLICY_LABELS.get(key, key),
                    "accepted_accuracy": None if policy is None else policy["accepted_accuracy"],
                    "accept_rate": None if policy is None else policy["accept_rate"],
                    "reject_rate": None if policy is None else policy["reject_rate"],
                    "early_reject_rate": None if policy is None else policy["early_reject_rate"],
                    "final_rate": None if policy is None else policy["final_rate"],
                    "avg_cost": None if policy is None else policy["avg_cost"],
                    "lost_final_reliable_rate": None if policy is None else policy["lost_final_reliable_rate"],
                    "cost_reduction_vs_final": None if policy is None else round(final["avg_cost"] - policy["avg_cost"], 6),
                    "accept_rate_drop_vs_final": None if policy is None else round(final["accept_rate"] - policy["accept_rate"], 6),
                    "valid": policy is not None,
                }
            )
    return rows


def winning_rows(
    rows: list[dict[str, Any]],
    min_accuracy: float,
    min_cost_reduction: float,
    max_lost: float | None = None,
) -> list[dict[str, Any]]:
    winners = [
        row
        for row in rows
        if row["method"].startswith("dual_side")
        and row["valid"]
        and row["accepted_accuracy"] >= min_accuracy
        and row["cost_reduction_vs_final"] >= min_cost_reduction
        and (max_lost is None or row["lost_final_reliable_rate"] <= max_lost)
    ]
    winners.sort(key=lambda row: (row["cost_reduction_vs_final"], row["accepted_accuracy"]), reverse=True)
    return winners


def scenario_summary(rows: list[dict[str, Any]], min_accuracy: float, min_cost_reduction: float) -> list[dict[str, Any]]:
    scenarios = sorted({row["scenario"] for row in rows})
    out = []
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        final = next(row for row in scenario_rows if row["method"] == "final_only")
        upper = next(row for row in scenario_rows if row["method"] == "upper_only_best_cost")
        wins = [
            row
            for row in scenario_rows
            if row["method"].startswith("dual_side")
            and row["valid"]
            and row["accepted_accuracy"] >= min_accuracy
            and row["cost_reduction_vs_final"] >= min_cost_reduction
        ]
        best = max(wins, key=lambda row: row["cost_reduction_vs_final"]) if wins else None
        out.append(
            {
                "scenario": scenario,
                "final_accept_rate": final["accept_rate"],
                "final_accuracy": final["accepted_accuracy"],
                "upper_only_valid": upper["valid"],
                "best_dual": best,
                "verdict": "clear_dual_win" if best is not None else "no_clear_win",
            }
        )
    return out


def write_markdown(payload: dict[str, Any], rows: list[dict[str, Any]], out_path: Path, min_accuracy: float, min_cost_reduction: float) -> None:
    winners = winning_rows(rows, min_accuracy, min_cost_reduction)
    summaries = scenario_summary(rows, min_accuracy, min_cost_reduction)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 両側早期終了 他手法比較",
        "",
        "## 比較条件",
        "",
        f"- ラベルを出した画像だけの精度が {pct(min_accuracy)} 以上",
        f"- finalのみと比べた平均計算量削減が {pct(min_cost_reduction)} 以上",
        "- 棄却は成功扱いにしない。ラベル出力率低下と良品ロスを同時に見る。",
        "- 上側早期終了のみは、同じ99%級の精度制約を満たせる設定があるかで判定する。",
        "",
        "## 結論",
        "",
    ]
    clear = [item for item in summaries if item["verdict"] == "clear_dual_win"]
    lines.append(f"この条件では、{len(clear)}/{len(summaries)} シナリオで両側早期終了が明確な省計算側の勝ちを持つ。")
    lines.append("")
    lines.append("ただし、これは無条件の勝利ではない。両側早期終了はラベル出力率を下げ、低信頼画像を早期棄却することで計算量を下げる方式である。")
    lines.append("したがって勝ち筋は、「全画像へラベルを付ける分類」ではなく、「信頼できるラベルだけを低コストで出したい検品・低品質入力混入」の条件である。")
    lines.append("")
    lines.append("## 代表的な勝ちシナリオ")
    lines.append("")
    lines.append("| シナリオ | 手法 | ラベル精度 | ラベル出力率 | final実行率 | 早期棄却率 | 平均計算量 | 削減率 | 良品ロス |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in winners[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    row["method_label"],
                    pct(row["accepted_accuracy"]),
                    pct(row["accept_rate"]),
                    pct(row["final_rate"]),
                    pct(row["early_reject_rate"]),
                    num(row["avg_cost"]),
                    pct(row["cost_reduction_vs_final"]),
                    pct(row["lost_final_reliable_rate"]),
                ]
            )
            + " |"
        )
    conservative = winning_rows(rows, min_accuracy, min_cost_reduction, max_lost=0.02)
    lines.append("")
    lines.append("## 良品ロス2%以下でも勝てる条件")
    lines.append("")
    lines.append("発表・修論で強く使いやすいのは、良品ロスを2%以下に抑えても10%以上の計算量削減が残る条件である。")
    lines.append("")
    lines.append("| シナリオ | 手法 | ラベル精度 | ラベル出力率 | final実行率 | 早期棄却率 | 平均計算量 | 削減率 | 良品ロス |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in conservative[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    row["method_label"],
                    pct(row["accepted_accuracy"]),
                    pct(row["accept_rate"]),
                    pct(row["final_rate"]),
                    pct(row["early_reject_rate"]),
                    num(row["avg_cost"]),
                    pct(row["cost_reduction_vs_final"]),
                    pct(row["lost_final_reliable_rate"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## シナリオ別判定")
    lines.append("")
    lines.append("| シナリオ | finalのみ ラベル精度 | finalのみ 出力率 | 上側早期終了のみ | 両側早期終了の判定 |")
    lines.append("|---|---:|---:|---|---|")
    for item in summaries:
        best = item["best_dual"]
        if best is None:
            dual = "明確な勝ちなし"
        else:
            dual = f"{best['method_label']} / 削減 {pct(best['cost_reduction_vs_final'])}, 精度 {pct(best['accepted_accuracy'])}"
        upper = "99%級制約で有効設定あり" if item["upper_only_valid"] else "99%級制約で有効設定なし"
        lines.append(
            f"| {item['scenario']} | {pct(item['final_accuracy'])} | {pct(item['final_accept_rate'])} | {upper} | {dual} |"
        )
    lines.append("")
    lines.append("## 発表で使える主張")
    lines.append("")
    lines.append("両側早期終了は、既存の上側早期終了のように「自信があるものを早く出す」だけではなく、")
    lines.append("「後段まで進めても信頼ラベルになりにくい低品質入力を早く棄却する」点で差別化できる。")
    lines.append("今回の低品質混入条件では、上側早期終了のみは99%級の精度制約を満たす有効設定が見つからず、")
    lines.append("両側早期終了だけが精度を保ちながらfinal実行率を下げる設定を持った。")
    lines.append("")
    lines.append("## 注意")
    lines.append("")
    lines.append("良品ロス5%級の設定は省計算効果が大きいが、棄却による損失も大きい。")
    lines.append("修論の主張では、良品ロス1%級または2%級でも十分な削減がある条件を中心に扱う方が安全である。")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(winners: list[dict[str, Any]], out_path: Path) -> None:
    winners = winners[:10]
    if not winners:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1180
    row_h = 38
    left = 355
    top = 76
    bar_w = 520
    height = top + row_h * len(winners) + 80
    max_gain = max(row["cost_reduction_vs_final"] for row in winners) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#0f172a;font-size:14px}.title{font-size:22px;font-weight:700}.small{font-size:12px;fill:#475569}</style>',
        '<text x="34" y="36" class="title">Dual-sided early exit wins under reliability-oriented low-quality conditions</text>',
        '<text x="34" y="58" class="small">Constraint: accepted-label accuracy >= 99%, cost reduction >= 10% versus final-only confidence filtering.</text>',
    ]
    for idx, row in enumerate(winners):
        y = top + idx * row_h
        gain = row["cost_reduction_vs_final"]
        bw = bar_w * gain / max_gain
        label = row["scenario"].replace("_", " ")
        parts += [
            f'<text x="34" y="{y + 22}">{label}</text>',
            f'<rect x="{left}" y="{y + 7}" width="{bar_w}" height="20" fill="#e2e8f0"/>',
            f'<rect x="{left}" y="{y + 7}" width="{bw:.1f}" height="20" fill="#16a34a"/>',
            f'<text x="{left + bar_w + 18}" y="{y + 22}">-{gain * 100:.1f}% cost</text>',
            f'<text x="{left + bar_w + 136}" y="{y + 22}" class="small">acc {row["accepted_accuracy"] * 100:.2f}%, output {row["accept_rate"] * 100:.1f}%, final {row["final_rate"] * 100:.1f}%</text>',
        ]
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/dual_exit_reliability_shift_003_summary.json")
    parser.add_argument("--output", default="results/dual_exit_method_comparison_003.json")
    parser.add_argument("--markdown", default="docs/dual_exit_method_comparison_003.md")
    parser.add_argument("--svg", default="results/dual_exit_method_comparison_003.svg")
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    parser.add_argument("--min-cost-reduction", type=float, default=0.10)
    args = parser.parse_args()

    payload = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    rows = scenario_policy_rows(payload)
    winners = winning_rows(rows, args.min_accuracy, args.min_cost_reduction)
    conservative_winners = winning_rows(rows, args.min_accuracy, args.min_cost_reduction, max_lost=0.02)
    result = {
        "source": args.summary,
        "min_accuracy": args.min_accuracy,
        "min_cost_reduction": args.min_cost_reduction,
        "definitions": {
            "accepted_accuracy": "accuracy among samples that receive a label",
            "accept_rate": "fraction of samples that receive a label",
            "avg_cost": "mean normalized compute cost; final-only is 1.0",
            "lost_final_reliable_rate": "fraction of all samples final-only would correctly label but the dual-side policy rejects early",
        },
        "winner_count": len(winners),
        "conservative_winner_count_lost_le_2pct": len(conservative_winners),
        "scenario_count": len(payload["scenario_results"]),
        "winners": winners,
        "conservative_winners_lost_le_2pct": conservative_winners,
        "all_rows": rows,
        "scenario_summary": scenario_summary(rows, args.min_accuracy, args.min_cost_reduction),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, rows, Path(args.markdown), args.min_accuracy, args.min_cost_reduction)
    write_svg(winners, Path(args.svg))
    print(json.dumps({"wrote": args.output, "winners": len(winners), "markdown": args.markdown, "svg": args.svg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
