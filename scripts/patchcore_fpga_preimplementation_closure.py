"""Close the non-implementation experiment phase for PatchCore FPGA work.

This script does not train a model.  It consolidates the already completed
MVTec AD PatchCore sweeps, holdout validation, FPGA cost model, and mode-switch
analysis into a decision record: what is proven enough to implement, what is
rejected, and what must be measured only after FPGA implementation exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.experiment_paths import ensure_dirs


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def ratio(value: float) -> str:
    return f"{value:.4f}x"


def mib(value: float) -> str:
    return f"{value:.3f} MiB"


def find_aggregate(rows: list[dict[str, Any]], target: float, drop: float) -> dict[str, Any]:
    for row in rows:
        if (
            abs(row["target_false_pass_rate_defect"] - target) < 1e-12
            and abs(row["allowed_good_pass_drop"] - drop) < 1e-12
        ):
            return row
    raise ValueError(f"missing aggregate target={target} drop={drop}")


def uniform_rows_for_target(rows: list[dict[str, Any]], target: float) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if abs(row["target_false_pass_rate_defect"] - target) < 1e-12
        and row.get("categories_done") == 15
    ]


def best_uniform_rows(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    target_rows = uniform_rows_for_target(rows, target)
    if not target_rows:
        raise ValueError(f"missing uniform rows for target={target}")
    full = max(target_rows, key=lambda row: row["mean_approx_nn_ops"])
    best_light = min(
        target_rows,
        key=lambda row: (
            -row["mean_good_pass_rate_good"],
            row["relative_nn_ops"],
        ),
    )
    best_under_20 = max(
        [row for row in target_rows if row["relative_nn_ops"] <= 0.2],
        key=lambda row: row["mean_good_pass_rate_good"],
    )
    best_under_10 = max(
        [row for row in target_rows if row["relative_nn_ops"] <= 0.1],
        key=lambda row: row["mean_good_pass_rate_good"],
    )
    return {
        "full": full,
        "best_good_pass": best_light,
        "best_under_20pct_nn": best_under_20,
        "best_under_10pct_nn": best_under_10,
    }


def pick_first_targets(cost_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    viable = [row for row in cost_rows if row["holdout_selected_false_pass"] <= 0.03]
    ranked = sorted(
        viable,
        key=lambda row: (
            -row["holdout_selected_good_pass"],
            row["holdout_selected_false_pass"],
            row["relative_total_proxy_ops"],
        ),
    )
    return ranked[:5]


def summarize_quality_rows(cost_rows: list[dict[str, Any]]) -> dict[str, Any]:
    false_passes = [row["holdout_selected_false_pass"] for row in cost_rows]
    good_passes = [row["holdout_selected_good_pass"] for row in cost_rows]
    total_ratios = [row["relative_total_proxy_ops"] for row in cost_rows]
    nn_ratios = [row["relative_nn_ops"] for row in cost_rows]
    bank_ratios = [row["relative_bank_bytes_int8"] for row in cost_rows]
    return {
        "categories": len(cost_rows),
        "categories_false_pass_le_1pct": sum(v <= 0.01 for v in false_passes),
        "categories_false_pass_le_3pct": sum(v <= 0.03 for v in false_passes),
        "categories_false_pass_le_5pct": sum(v <= 0.05 for v in false_passes),
        "mean_false_pass": round(mean(false_passes), 6),
        "median_false_pass": round(median(false_passes), 6),
        "mean_good_pass": round(mean(good_passes), 6),
        "median_good_pass": round(median(good_passes), 6),
        "mean_total_proxy_ratio": round(mean(total_ratios), 6),
        "median_total_proxy_ratio": round(median(total_ratios), 6),
        "mean_nn_ratio": round(mean(nn_ratios), 6),
        "median_nn_ratio": round(median(nn_ratios), 6),
        "mean_bank_ratio": round(mean(bank_ratios), 6),
        "median_bank_ratio": round(median(bank_ratios), 6),
    }


def make_markdown(payload: dict[str, Any]) -> str:
    u = payload["uniform_comparison"]
    h = payload["holdout_report_setting"]
    q = payload["quality_summary"]
    c = payload["cost_summary"]
    m = payload["mode_switch_summary"]
    targets = payload["recommended_first_targets"]
    closure = payload["closure_decision"]

    lines: list[str] = []
    lines.append("# PatchCore FPGA 実装前実験の完了判定")
    lines.append("")
    lines.append("Date: 2026-08-23")
    lines.append("")
    lines.append("## 結論")
    lines.append("")
    lines.append(closure["short"])
    lines.append("")
    lines.append("ここから先の主要な未確認事項は、GPU上の追加探索ではなく、FPGA実装で実測するべき項目です。")
    lines.append("")
    lines.append("## 実装前に確認できたこと")
    lines.append("")
    lines.append("| 確認項目 | 判定 | 根拠 |")
    lines.append("|---|---:|---|")
    for row in payload["closed_questions"]:
        lines.append(f"| {row['question']} | {row['status']} | {row['evidence']} |")
    lines.append("")
    lines.append("## 主要な数値")
    lines.append("")
    lines.append("| 項目 | 結果 | 読み取り |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| カテゴリ別プロファイルのholdout評価 | 良品通過 {pct(h['selected_holdout_good_pass_mean'])}, "
        f"欠陥誤通過 {pct(h['selected_holdout_false_pass_mean'])}, KNN {ratio(h['relative_nn_ops_mean'])} | "
        "計算削減は再現するが、全カテゴリで安全とは言えない。 |"
    )
    lines.append(
        f"| FPGA向け総コスト近似 | 平均 {ratio(c['mean_relative_total_proxy_ops'])}, "
        f"中央値 {ratio(c['median_relative_total_proxy_ops'])} | CNNとKNNを合わせても大きく下がる。 |"
    )
    lines.append(
        f"| KNN探索コスト | 平均 {ratio(c['mean_relative_nn_ops'])}, "
        f"中央値 {ratio(c['median_relative_nn_ops'])} | 98%級の削減は演算回数の見積もりとして成立している。 |"
    )
    lines.append(
        f"| 全カテゴリ分のメモリバンク | {mib(m['baseline_all_category_bank_mib'])} -> "
        f"{mib(m['selected_all_category_bank_mib'])} | カテゴリ別バンクにより常駐メモリを {pct(m['all_category_bank_reduction'])} 削減できる。 |"
    )
    lines.append(
        f"| 512並列距離計算器モデル | 平均 {m['profiled_mean_knn_ms_512']:.3f} ms, "
        f"最大 {m['profiled_worst_knn_ms_512']:.3f} ms | 削減後バンクはFPGA上の並列KNNに載せる候補として現実的。 |"
    )
    lines.append("")
    lines.append("## 全カテゴリ共通設定との比較")
    lines.append("")
    lines.append("ここが研究テーマの芯です。単にPatchCoreを小型化するのではなく、検品対象ごとに必要な構成だけを使う。")
    lines.append("")
    lines.append("| 設定 | 良品通過 | KNN演算 | コメント |")
    lines.append("|---|---:|---:|---|")
    for name, row in u.items():
        lines.append(
            f"| {name} (`{row['config']}`) | {pct(row['mean_good_pass_rate_good'])} | "
            f"{ratio(row['relative_nn_ops'])} | {row['comment']} |"
        )
    lines.append(
        f"| カテゴリ別選択設定、holdout平均 | {pct(h['selected_holdout_good_pass_mean'])} | "
        f"{ratio(h['relative_nn_ops_mean'])} | KNNは非常に小さいが、安全性はカテゴリ別に扱う必要がある。 |"
    )
    lines.append("")
    lines.append("## 品質面の限界")
    lines.append("")
    lines.append(
        f"MVTec ADの15カテゴリに対して、選択済みプロファイル設定が欠陥誤通過 <= 1% を満たしたのは "
        f"{q['categories_false_pass_le_1pct']}/{q['categories']} カテゴリ、<= 3% は "
        f"{q['categories_false_pass_le_3pct']}/{q['categories']} カテゴリ、<= 5% は "
        f"{q['categories_false_pass_le_5pct']}/{q['categories']} カテゴリでした。"
    )
    lines.append("")
    lines.append(
        "したがって、現時点で主張できるのは「全カテゴリ共通の完成検品器」ではなく、"
        "「カテゴリを定めた検品対象に対して、必要なPatchCore構成を事前に決めてFPGA資源へ落とす方式」です。"
    )
    lines.append("")
    lines.append("## 初回FPGA実装ターゲット")
    lines.append("")
    lines.append("| 順位 | カテゴリ | 選択設定 | 良品通過 | 欠陥誤通過 | 総コスト近似 | KNN演算 | バンク |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(targets, start=1):
        lines.append(
            f"| {i} | {row['category']} | `{row['selected_config']}` | "
            f"{pct(row['holdout_selected_good_pass'])} | {pct(row['holdout_selected_false_pass'])} | "
            f"{ratio(row['relative_total_proxy_ops'])} | {ratio(row['relative_nn_ops'])} | "
            f"{ratio(row['relative_bank_bytes_int8'])} |"
        )
    lines.append("")
    lines.append("## 実装前に広げない項目")
    lines.append("")
    lines.append("| 項目 | 判定 | 理由 |")
    lines.append("|---|---|---|")
    for row in payload["deferred_or_rejected"]:
        lines.append(f"| {row['item']} | {row['decision']} | {row['reason']} |")
    lines.append("")
    lines.append("## FPGA実装へ入る条件")
    lines.append("")
    for item in closure["implementation_entry_conditions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 使用した結果ファイル")
    lines.append("")
    for path in payload["inputs"]:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    deep = load_json(args.deep_frontier)
    holdout = load_json(args.holdout)
    cost = load_json(args.cost_model)
    mode = load_json(args.mode_switch)

    uniform = best_uniform_rows(deep["aggregate_rows"], args.report_false_pass_target)
    report_holdout = find_aggregate(
        holdout["aggregate_summary"],
        args.report_false_pass_target,
        args.allowed_good_pass_drop,
    )
    quality = summarize_quality_rows(cost["rows"])
    first_targets = pick_first_targets(cost["rows"])
    cost_agg = cost["aggregate"]
    storage = mode["storage_summary"]
    arch_by_name = {row["name"]: row for row in mode["architecture_options"]}
    profiled = arch_by_name["profiled_all_resident"]

    uniform_comparison = {
        "全カテゴリ共通: 基準構成": {
            **uniform["full"],
            "comment": "最も重い参照点。",
        },
        "全カテゴリ共通: 良品通過最大": {
            **uniform["best_good_pass"],
            "comment": "良品通過を最大化すると基準構成から小さくならない。",
        },
        "全カテゴリ共通: KNN 20%以下": {
            **uniform["best_under_20pct_nn"],
            "comment": "共通軽量化で品質を残せる限界の目安。",
        },
        "全カテゴリ共通: KNN 10%以下": {
            **uniform["best_under_10pct_nn"],
            "comment": "さらに軽くすると良品通過が大きく下がる。",
        },
    }

    mode_summary = {
        **storage,
        "profiled_mean_knn_ms_512": profiled["mean_knn_ms"],
        "profiled_worst_knn_ms_512": profiled["worst_knn_ms"],
    }

    return {
        "purpose": "Close all non-FPGA-implementation experiments and decide implementation entry.",
        "inputs": [
            args.deep_frontier,
            args.holdout,
            args.cost_model,
            args.mode_switch,
        ],
        "report_setting": {
            "target_false_pass_rate_defect": args.report_false_pass_target,
            "allowed_good_pass_drop": args.allowed_good_pass_drop,
            "quality_false_pass_limit_for_first_target": 0.03,
        },
        "uniform_comparison": uniform_comparison,
        "holdout_report_setting": report_holdout,
        "quality_summary": quality,
        "cost_summary": cost_agg,
        "mode_switch_summary": mode_summary,
        "recommended_first_targets": first_targets,
        "closed_questions": [
            {
                "question": "小さいPatchCoreバンク/探索で本当に計算を減らせるか",
                "status": "完了",
                "evidence": f"mean NN ops {ratio(cost_agg['mean_relative_nn_ops'])}, median {ratio(cost_agg['median_relative_nn_ops'])}",
            },
            {
                "question": "98%級のKNN削減は単なる測定ノイズではないか",
                "status": "完了",
                "evidence": "演算回数、バンクサイズ、探索時間の監査が同じ方向を示した。",
            },
            {
                "question": "全カテゴリ共通の軽量設定で十分か",
                "status": "否定的",
                "evidence": "共通設定ではカテゴリ依存の良品通過低下が出る。カテゴリ別プロファイルの方が筋が良い。",
            },
            {
                "question": "15カテゴリすべてで厳しい検品品質を満たせるか",
                "status": "否定的",
                "evidence": f"欠陥誤通過 <= 3% を満たすのは {quality['categories_false_pass_le_3pct']}/{quality['categories']} カテゴリのみ。",
            },
            {
                "question": "モード切替構造に技術的な意味はあるか",
                "status": "完了",
                "evidence": f"全カテゴリ分バンクが {mib(storage['baseline_all_category_bank_mib'])} -> {mib(storage['selected_all_category_bank_mib'])}。",
            },
        ],
        "deferred_or_rejected": [
            {
                "item": "広いGPU探索の追加",
                "decision": "いったん停止",
                "reason": "初回FPGAターゲットは大きく変わりにくく、現在のボトルネックは実装して実測差を見ること。",
            },
            {
                "item": "全カテゴリでの安全な完成検品器という主張",
                "decision": "現時点では不可",
                "reason": "holdoutでの欠陥誤通過がカテゴリごとに大きく異なる。",
            },
            {
                "item": "正確な電力・実行時間",
                "decision": "実装評価へ回す",
                "reason": "実際のデータパス、メモリ配置、クロック、合成・配置配線が必要。",
            },
            {
                "item": "固定小数点/int8での品質保証",
                "decision": "実装受け入れ試験へ回す",
                "reason": "バンク削減率はビット幅に依存しないが、スコア変動は採用する数値形式で決まる。",
            },
            {
                "item": "新規データセット追加",
                "decision": "スコープ拡大時のみ",
                "reason": "MVTec ADで15種類の工業検品カテゴリを扱えており、次に欠けている証拠はハードウェア実測。",
            },
        ],
        "closure_decision": {
            "short": (
                "カテゴリ別プロファイル型PatchCore-liteとしてFPGA実装へ進む。"
                "初回ターゲットはhazelnut、wood/cable/zipperは比較・バックアップ候補にする。"
            ),
            "implementation_entry_conditions": [
                "まず1カテゴリの特徴抽出・メモリバンク・距離計算構成を実装する。",
                "欠陥誤通過と良品通過を分けて報告し、AUROCだけで品質低下を隠さない。",
                "総コスト近似と実際のFPGA遅延・資源量・電力の差を測る。",
                "その後、単一カテゴリ実装からモード切替型の複数カテゴリ対応へ拡張する。",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-frontier", default="results/mvtec_patchcore_all15_deep_frontier_001_summary.json")
    parser.add_argument("--holdout", default="results/mvtec_patchcore_profiled_holdout_validation_001_summary.json")
    parser.add_argument("--cost-model", default="results/mvtec_patchcore_fpga_cost_model_001_summary.json")
    parser.add_argument("--mode-switch", default="results/mvtec_patchcore_mode_switch_analysis_003_summary.json")
    parser.add_argument("--output", default="results/patchcore_fpga_preimplementation_closure_001_summary.json")
    parser.add_argument("--markdown", default="docs/patchcore_fpga_preimplementation_closure_001.md")
    parser.add_argument("--report-false-pass-target", type=float, default=0.01)
    parser.add_argument("--allowed-good-pass-drop", type=float, default=0.02)
    args = parser.parse_args()

    ensure_dirs()
    payload = build_payload(args)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(make_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "markdown": args.markdown,
        "decision": payload["closure_decision"]["short"],
        "first_target": payload["recommended_first_targets"][0]["category"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
