"""Summarize KSDD2 final-detector candidates for foundation-model selection.

The goal is not to prove the proposed early-exit method yet.  This summary
answers a simpler question: which existing final detector is the least bad
foundation for the next experiments, and what metrics should be used after an
early-exit mechanism is attached?
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


RESULT_FILES = [
    Path("results/ksdd2_smp_final_inspection_baseline_caviar9_001_summary.json"),
    Path("results/ksdd2_smp_final_inspection_baseline_caviar9_unet_resnet50_001_summary.json"),
    Path("results/ksdd2_smp_final_inspection_baseline_caviar9_fpn_resnet50_001_summary.json"),
    Path("results/ksdd2_unet_inspection_baseline_001_summary.json"),
    Path("results/ksdd2_industrial_anomaly_baselines_caviar9_001_summary.json"),
]


def rf(x: Any, ndigits: int = 6) -> float | None:
    if x is None:
        return None
    return round(float(x), ndigits)


def pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.2f}%"


def model_name(payload: dict[str, Any], result_file: Path, method: str | None = None) -> str:
    if method:
        return method
    model = payload.get("model") or {}
    if model:
        arch = model.get("architecture", "model")
        encoder = model.get("encoder", "")
        return f"{arch}/{encoder}".strip("/")
    if "ksdd2_unet_inspection" in result_file.name:
        cfg = payload.get("training_config", {})
        return f"small_unet/base{cfg.get('base_channels', '')}"
    return result_file.stem.replace("_summary", "")


def iter_candidates(result_file: Path) -> list[dict[str, Any]]:
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    candidates = []
    for seed_result in payload.get("seed_results", []):
        seed = seed_result.get("seed")
        methods = seed_result.get("methods")
        if methods is None:
            methods = [
                {
                    "method": None,
                    "val_auc": seed_result.get("val_auc", {}),
                    "test_auc": seed_result.get("test_auc", {}),
                    "threshold_rows": seed_result.get("threshold_rows", []),
                    "test_curve_rows": seed_result.get("test_curve_rows", {}),
                }
            ]
        for method in methods:
            candidates.append(
                {
                    "result_file": result_file.name,
                    "model": model_name(payload, result_file, method.get("method")),
                    "seed": seed,
                    "test_auc": method.get("test_auc", {}),
                    "threshold_rows": method.get("threshold_rows", []),
                    "test_curve_rows": method.get("test_curve_rows", {}),
                }
            )
    return candidates


def best_safe_from_curve(rows: list[dict[str, Any]], max_false_pass: float) -> dict[str, Any] | None:
    safe = [row for row in rows if float(row["false_pass_rate_defect"]) <= max_false_pass + 1e-12]
    if not safe:
        return None
    return max(safe, key=lambda row: (float(row["good_pass_rate_good"]), -float(row["false_pass_rate_defect"])))


def near_good_pass(rows: list[dict[str, Any]], target_good_pass: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["good_pass_rate_good"]) - target_good_pass))


def selected_threshold_rows(candidate: dict[str, Any], score_name: str, max_false_pass: float, min_good_pass: float) -> list[dict[str, Any]]:
    out = []
    for row in candidate["threshold_rows"]:
        constraint = row.get("constraint", {})
        if row.get("score_name") != score_name:
            continue
        if abs(float(constraint.get("max_false_pass_rate_defect", -1)) - max_false_pass) > 1e-12:
            continue
        if abs(float(constraint.get("min_good_pass_rate_good", -1)) - min_good_pass) > 1e-12:
            continue
        out.append(row)
    return out


def summarize_group(group: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    aucs, auprs = [], []
    best_safe_rows, gp90_rows = [], []
    selected_5_90_rows = []
    selected_0_90_rows = []
    for candidate in group:
        auc = candidate["test_auc"].get(score_name, {})
        if auc.get("image_auroc") is not None:
            aucs.append(float(auc["image_auroc"]))
        if auc.get("image_aupr") is not None:
            auprs.append(float(auc["image_aupr"]))
        curve = candidate["test_curve_rows"].get(score_name, [])
        best_safe = best_safe_from_curve(curve, 0.05)
        gp90 = near_good_pass(curve, 0.90)
        if best_safe is not None:
            best_safe_rows.append(best_safe)
        if gp90 is not None:
            gp90_rows.append(gp90)
        selected_5_90_rows.extend(selected_threshold_rows(candidate, score_name, 0.05, 0.90))
        selected_0_90_rows.extend(selected_threshold_rows(candidate, score_name, 0.0, 0.90))

    def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
        return rf(mean(float(row[field]) for row in rows)) if rows else None

    return {
        "score_name": score_name,
        "seed_count": len(group),
        "mean_test_auroc": rf(mean(aucs)) if aucs else None,
        "mean_test_aupr": rf(mean(auprs)) if auprs else None,
        "best_good_pass_when_false_pass_le_5_mean": mean_field(best_safe_rows, "good_pass_rate_good"),
        "best_good_pass_when_false_pass_le_5_worst": rf(min(float(row["good_pass_rate_good"]) for row in best_safe_rows)) if best_safe_rows else None,
        "false_pass_near_good_pass_90_mean": mean_field(gp90_rows, "false_pass_rate_defect"),
        "false_pass_near_good_pass_90_worst": rf(max(float(row["false_pass_rate_defect"]) for row in gp90_rows)) if gp90_rows else None,
        "selected_5fp_90gp_feasible": sum(bool(row.get("test_feasible")) for row in selected_5_90_rows),
        "selected_5fp_90gp_seeds": len(selected_5_90_rows),
        "selected_0fp_90gp_feasible": sum(bool(row.get("test_feasible")) for row in selected_0_90_rows),
        "selected_0fp_90gp_seeds": len(selected_0_90_rows),
    }


def score_for_selection(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    """Prefer safe operating ability, then discrimination quality."""

    safe_gp = float(summary.get("best_good_pass_when_false_pass_le_5_mean") or 0.0)
    gp90_fp = 1.0 - float(summary.get("false_pass_near_good_pass_90_mean") or 1.0)
    auc = float(summary.get("mean_test_auroc") or 0.0)
    aupr = float(summary.get("mean_test_aupr") or 0.0)
    return safe_gp, gp90_fp, auc, aupr


def main() -> None:
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result_file in RESULT_FILES:
        if not result_file.exists():
            continue
        for candidate in iter_candidates(result_file):
            by_model.setdefault((candidate["result_file"], candidate["model"]), []).append(candidate)

    rows = []
    for (result_file, model), group in sorted(by_model.items()):
        for score_name in ["max_score", "topk_score"]:
            if not any(score_name in candidate["test_auc"] for candidate in group):
                continue
            summary = summarize_group(group, score_name)
            summary.update({"result_file": result_file, "model": model})
            rows.append(summary)

    rows.sort(key=score_for_selection, reverse=True)
    selected = rows[0] if rows else None
    payload = {
        "purpose": "Select the current KSDD2 foundation detector and define evaluation axes for later early-exit/FPGA work.",
        "selection_rule": [
            "First require useful final-detector operation, not just high AUROC.",
            "Main check: how much good data can pass when defect false-pass is constrained to 5% or lower.",
            "Secondary check: defect false-pass near 90% good-pass.",
            "Then use AUROC/AUPR to judge score quality.",
        ],
        "selected_foundation": selected,
        "candidate_rows": rows,
        "evaluation_axes": {
            "foundation_model": [
                "image-level AUROC/AUPR",
                "good pass rate under defect false-pass <= 5%",
                "defect false-pass near good-pass ~= 90%",
                "threshold stability across seeds/splits",
            ],
            "proposed_method_on_fixed_foundation": [
                "defect false-pass must not worsen beyond the chosen safety budget",
                "good pass / good loss trade-off",
                "average executed stages or equivalent compute",
                "worst-case latency and throughput",
                "FPGA resource, power, and memory-access estimate",
            ],
        },
    }

    out_json = Path("results/ksdd2_foundation_model_selection.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_md = Path("docs/ksdd2_foundation_model_selection.md")
    lines = [
        "# KSDD2土台モデルの性能再確認と評価軸",
        "",
        "## 結論",
        "",
    ]
    if selected:
        lines += [
            f"現時点で一番マシな土台モデルは `{selected['model']}` です。",
            f"根拠ファイルは `{selected['result_file']}`、採用スコアは `{selected['score_name']}` です。",
            "",
            f"欠陥誤通過を5%以下に抑える条件では、良品通過率は平均 {pct(selected['best_good_pass_when_false_pass_le_5_mean'])}、最悪seedでも {pct(selected['best_good_pass_when_false_pass_le_5_worst'])} でした。",
            f"また、良品通過率90%付近では欠陥誤通過が平均 {pct(selected['false_pass_near_good_pass_90_mean'])} です。",
            "",
            "ただし完全な土台モデルではありません。検証データで選んだ「欠陥誤通過5%以下・良品通過90%以上」の閾値がtestでも成功したのは 1/2 seed です。つまり、スコア分離能力は最も高いが、閾値安定性にはまだ改善または再確認が必要です。",
            "",
        ]
    lines += [
        "## なぜAUROCだけで決めないか",
        "",
        "検品タスクで一番避けたい失敗は、欠陥品を良品として通してしまうことです。AUROCが高くても、欠陥誤通過を低く抑えようとした瞬間に良品まで大量に捨てるモデルでは、実運用の土台として弱いです。",
        "",
        "そのため、土台モデル選定では「欠陥スコアの順位付け性能」だけでなく、「実際に閾値を置いたときの良品通過率と欠陥誤通過率」を重視します。",
        "",
        "## 候補モデルの再確認",
        "",
        "| 順位 | モデル | スコア | 元結果 | 平均AUROC | 平均AUPR | 欠陥誤通過5%以下での良品通過 | 良品通過90%付近の欠陥誤通過 | 検証閾値の成功数 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        feasible = f"{row['selected_5fp_90gp_feasible']}/{row['selected_5fp_90gp_seeds']}"
        lines.append(
            f"| {idx} | {row['model']} | {row['score_name']} | {row['result_file']} | "
            f"{row['mean_test_auroc']} | {row['mean_test_aupr']} | "
            f"{pct(row['best_good_pass_when_false_pass_le_5_mean'])} | "
            f"{pct(row['false_pass_near_good_pass_90_mean'])} | {feasible} |"
        )
    lines += [
        "",
        "## 今後の評価軸",
        "",
        "### 1. 土台モデル単体の評価",
        "",
        "- 欠陥誤通過率: 欠陥品を良品として通した割合。安全性の最重要制約。",
        "- 良品通過率: 良品を良品として通した割合。歩留まり・生産性の指標。",
        "- 良品棄却率: 良品を止めた割合。欠陥誤通過よりは許容されるが、コストとして扱う。",
        "- AUROC/AUPR: 欠陥スコアそのものの順位付け性能。補助指標。",
        "- 閾値安定性: validationで選んだ閾値がtestでも崩れないか。",
        "",
        "### 2. 提案手法を載せた後の評価",
        "",
        "- 欠陥誤通過率は土台モデルと同じ安全予算内に維持する。",
        "- その条件で良品通過率・良品棄却率がどう変わるかを見る。",
        "- 精度だけでなく、平均実行段数・平均計算量・平均消費電力を比較する。",
        "- 平均レイテンシと最悪レイテンシを分けて報告する。",
        "- FPGA化では、使用LUT/DSP/BRAM、メモリアクセス、推定電力、パイプライン化、分岐先回路の停止可能性を評価する。",
        "",
        "## 現時点の読み",
        "",
        "現時点では、PatchCore-liteやPaDiM-diagonalよりも、U-Net/ResNet50系のセグメンテーションモデルを土台にするのが妥当です。",
        "",
        "PatchCoreは保険テーマとしては残せますが、今回のKSDD2実験では土台モデルとして最有力とは言えません。もしPatchCore FPGA化を主題にするなら、まずPatchCore本体をより忠実に再現し、性能面でU-Net/ResNet50に近づくか上回ることを確認する必要があります。",
        "",
        "次にやるべきことは、`unet/resnet50` を固定土台として、閾値校正とseed/split安定性を再確認することです。その後、この固定土台に対して両側早期終了またはFPGA化による計算量削減を評価します。",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(out_json), "markdown": str(out_md), "selected": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
