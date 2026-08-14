"""Speed-oriented comparison for dual-sided early exit.

This experiment reframes the proposal as a high-speed safe-pass system:

* false pass is controlled by accepted-label accuracy
* false reject / over-reject is exposed through accept rate and lost reliable
* speed is reported through final execution rate and normalized average cost

It compares representative dynamic switching baselines:

* HIGH only
* BN upper-only early exit
* LOW/HIGH cascade
* LOW/HIGH parallel
* proposed dual-sided early exit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.compare_final_threshold_vs_dual_exit import choose_best
from scripts.dual_exit_reliability_shift_experiment import (
    collect_scenario,
    concat_data,
    load_branchynet,
    make_loader,
    sweep_policies,
)
from scripts.branchynet_cifar_sweep import (
    DEFAULT_EXITS,
    estimate_costs,
    make_datasets,
    parse_csv_ints,
    transform_eval,
)
from src.experiment_paths import ensure_dirs


def mixture_specs() -> dict[str, list[tuple[str, float]]]:
    return {
        "clean_only": [("clean", 1.0)],
        "clean90_mild_quality10": [("clean", 0.90), ("blur_r1", 0.04), ("noise_10", 0.03), ("jpeg_q20", 0.03)],
        "clean80_mild_quality20": [("clean", 0.80), ("blur_r1", 0.07), ("noise_10", 0.06), ("jpeg_q20", 0.04), ("low_contrast_50", 0.03)],
        "clean75_occlude16_25": [("clean", 0.75), ("occlude_16", 0.25)],
        "clean75_jpeg10_25": [("clean", 0.75), ("jpeg_q10", 0.25)],
        "clean40_mixed_low_quality60": [("clean", 0.40), ("blur_r2", 0.25), ("occlude_16", 0.25), ("low_contrast_50", 0.10)],
    }


def required_scenarios(specs: dict[str, list[tuple[str, float]]]) -> list[str]:
    scenarios = sorted({scenario for spec in specs.values() for scenario, _ in spec})
    return scenarios


def row_from_policy(
    scenario: str,
    target_accuracy: float,
    method: str,
    method_label: str,
    policy: dict[str, Any] | None,
    final_only: dict[str, Any],
    low_cost: float,
    high_cost: float,
    cost_override: float | None = None,
    final_rate_override: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    if policy is None:
        return {
            "scenario": scenario,
            "target_accuracy": target_accuracy,
            "max_false_pass_rate": round(1.0 - target_accuracy, 6),
            "method": method,
            "method_label": method_label,
            "valid": False,
            "note": note,
        }
    avg_cost = policy["avg_cost"] if cost_override is None else cost_override
    final_rate = policy["final_rate"] if final_rate_override is None else final_rate_override
    accepted_accuracy = policy["accepted_accuracy"]
    return {
        "scenario": scenario,
        "target_accuracy": target_accuracy,
        "max_false_pass_rate": round(1.0 - target_accuracy, 6),
        "method": method,
        "method_label": method_label,
        "valid": True,
        "accepted_accuracy": accepted_accuracy,
        "false_pass_rate": None if accepted_accuracy is None else round(1.0 - accepted_accuracy, 6),
        "pass_rate": policy["accept_rate"],
        "reject_rate": policy["reject_rate"],
        "early_reject_rate": policy["early_reject_rate"],
        "final_execution_rate": final_rate,
        "fast_decision_rate": round(1.0 - final_rate, 6),
        "avg_cost": round(float(avg_cost), 6),
        "speedup_vs_high_only": round(float(final_only["avg_cost"] / avg_cost), 6) if avg_cost else None,
        "cost_reduction_vs_high_only": round(float(final_only["avg_cost"] - avg_cost), 6),
        "lost_final_reliable_rate": policy.get("lost_final_reliable_rate", 0.0),
        "worst_case_compute": round(max(high_cost, avg_cost), 6),
        "note": note,
    }


def build_comparison_rows(scenario: str, target_accuracy: float, result: dict[str, Any], low_cost: float, high_cost: float) -> list[dict[str, Any]]:
    rows = []
    final_only = result["eval_final_only"]
    evaluated = result["evaluated_on_heldout"]
    upper = evaluated.get("upper_only_best_cost")

    rows.append(
        row_from_policy(
            scenario,
            target_accuracy,
            "high_only",
            "HIGHのみ",
            final_only,
            final_only,
            low_cost,
            high_cost,
            note="HIGHを最後まで実行し、信頼できるものだけ通過。",
        )
    )
    rows.append(
        row_from_policy(
            scenario,
            target_accuracy,
            "bn_upper_only",
            "BN",
            upper,
            final_only,
            low_cost,
            high_cost,
            note="高確信なら早期通過。低確信の早期棄却はしない。",
        )
    )
    cascade_cost = None if upper is None else low_cost + upper["final_rate"] * high_cost
    rows.append(
        row_from_policy(
            scenario,
            target_accuracy,
            "cascade_low_high",
            "カスケード",
            upper,
            final_only,
            low_cost,
            high_cost,
            cost_override=cascade_cost,
            note="LOWで高確信なら通過。そうでなければHIGHを追加実行。",
        )
    )
    rows.append(
        row_from_policy(
            scenario,
            target_accuracy,
            "parallel_low_high",
            "パラレル",
            final_only,
            final_only,
            low_cost,
            high_cost,
            cost_override=low_cost + high_cost,
            final_rate_override=1.0,
            note="LOW/HIGHを同時実行。信頼判定はHIGH側で行う。",
        )
    )
    for key, label in [
        ("dual_side_best_cost_lost_final_reliable_le_1pct", "提案 良品ロス1%級"),
        ("dual_side_best_cost_lost_final_reliable_le_2pct", "提案 良品ロス2%級"),
        ("dual_side_best_cost_lost_final_reliable_le_5pct", "提案 良品ロス5%級"),
    ]:
        rows.append(
            row_from_policy(
                scenario,
                target_accuracy,
                key,
                label,
                evaluated.get(key),
                final_only,
                low_cost,
                high_cost,
                note="高確信なら早期通過、低確信なら早期棄却、中間だけ後段へ送る。",
            )
        )
    return rows


def invalid_rows(scenario: str, target_accuracy: float, reason: str) -> list[dict[str, Any]]:
    methods = [
        ("high_only", "HIGHのみ"),
        ("bn_upper_only", "BN"),
        ("cascade_low_high", "カスケード"),
        ("parallel_low_high", "パラレル"),
        ("dual_side_best_cost_lost_final_reliable_le_1pct", "提案 良品ロス1%級"),
        ("dual_side_best_cost_lost_final_reliable_le_2pct", "提案 良品ロス2%級"),
        ("dual_side_best_cost_lost_final_reliable_le_5pct", "提案 良品ロス5%級"),
    ]
    return [
        {
            "scenario": scenario,
            "target_accuracy": target_accuracy,
            "max_false_pass_rate": round(1.0 - target_accuracy, 6),
            "method": method,
            "method_label": label,
            "valid": False,
            "note": reason,
        }
        for method, label in methods
    ]


def best_rows(rows: list[dict[str, Any]], target_accuracy: float, min_cost_reduction: float) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("valid")
        and row["method"].startswith("dual_side")
        and row["target_accuracy"] == target_accuracy
        and row["accepted_accuracy"] >= target_accuracy
        and row["cost_reduction_vs_high_only"] >= min_cost_reduction
    ]
    candidates.sort(key=lambda row: (row["cost_reduction_vs_high_only"], row["pass_rate"]), reverse=True)
    return candidates


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["rows"]
    lines = [
        "# 速さ軸でのトレードオフ比較",
        "",
        "## 評価の見方",
        "",
        "- 誤通過率: ラベルを出した画像のうち、分類が間違っていた割合。",
        "- 通過率: システムが信頼ラベルを出した割合。",
        "- 早期棄却率: 後段まで進めず、途中で棄却した割合。",
        "- final実行率: 最後の重い段まで到達した割合。低いほど速さに効く。",
        "- 平均計算量: HIGHのみを1.0とした正規化コスト。",
        "",
        "## 結論",
        "",
        "速さ特化で見ると、提案手法の価値は「誤通過を抑えたまま、低信頼入力を早く棄却してfinal実行率を下げる」点にある。",
        "BN/カスケードは低信頼入力を早く捨てられないため、信頼ラベル制約を厳しくすると有効設定が出にくい。",
        "パラレルは速度の最悪遅延では有利だが、LOWとHIGHを常に動かすため平均計算量・電力では不利である。",
        "",
    ]
    for target in payload["target_accuracies"]:
        target_rows = best_rows(rows, target, payload["min_cost_reduction"])
        lines.append(f"## 誤通過率 {100 * (1 - target):.1f}% 以下")
        lines.append("")
        lines.append("| シナリオ | 手法 | 誤通過率 | 通過率 | 早期棄却率 | final実行率 | 平均計算量 | HIGH比削減 | HIGH比速度 | 良品ロス |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in target_rows[:12]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["scenario"],
                        row["method_label"],
                        f"{100 * row['false_pass_rate']:.2f}%",
                        f"{100 * row['pass_rate']:.2f}%",
                        f"{100 * row['early_reject_rate']:.2f}%",
                        f"{100 * row['final_execution_rate']:.2f}%",
                        f"{row['avg_cost']:.4f}",
                        f"{100 * row['cost_reduction_vs_high_only']:.2f}%",
                        f"{row['speedup_vs_high_only']:.2f}x",
                        f"{100 * row['lost_final_reliable_rate']:.2f}%",
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.append("## 主張の形")
    lines.append("")
    lines.append("提案手法は、全画像を分類する方式ではなく、高速ライン上で「安全に通せるものだけを即時通過させる」方式である。")
    lines.append("誤棄却は避けられないが、その代わり誤通過を抑えたままfinal実行率を下げられる。")
    lines.append("FPGA化では、早期棄却後に後段回路を動かさない設計が可能になり、平均遅延・動的電力の削減に接続できる。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(payload: dict[str, Any], path: Path) -> None:
    rows = best_rows(payload["rows"], 0.99, payload["min_cost_reduction"])[:10]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1180
    row_h = 38
    top = 76
    left = 360
    bar_w = 500
    height = top + row_h * len(rows) + 74
    max_gain = max(row["cost_reduction_vs_high_only"] for row in rows) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#0f172a;font-size:14px}.title{font-size:22px;font-weight:700}.small{font-size:12px;fill:#475569}</style>',
        '<text x="34" y="36" class="title">Speed-oriented safe-pass trade-off</text>',
        '<text x="34" y="58" class="small">False pass <= 1%. Bars show normalized compute reduction versus HIGH-only.</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        gain = row["cost_reduction_vs_high_only"]
        bw = bar_w * gain / max_gain
        parts += [
            f'<text x="34" y="{y + 22}">{row["scenario"].replace("_", " ")}</text>',
            f'<rect x="{left}" y="{y + 7}" width="{bar_w}" height="20" fill="#e2e8f0"/>',
            f'<rect x="{left}" y="{y + 7}" width="{bw:.1f}" height="20" fill="#16a34a"/>',
            f'<text x="{left + bar_w + 18}" y="{y + 22}">-{100 * gain:.1f}% cost</text>',
            f'<text x="{left + bar_w + 135}" y="{y + 22}" class="small">pass {100 * row["pass_rate"]:.1f}%, final {100 * row["final_execution_rate"]:.1f}%, false pass {100 * row["false_pass_rate"]:.2f}%</text>',
        ]
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/dual_exit_speed_tradeoff_001_summary.json")
    parser.add_argument("--markdown", default="docs/dual_exit_speed_tradeoff_001.md")
    parser.add_argument("--svg", default="results/dual_exit_speed_tradeoff_001.svg")
    parser.add_argument("--model-output-name", default="0000b_branchynet_reproduce_resnet56_cifar10")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--arch", default="resnet56", choices=["resnet56"])
    parser.add_argument("--exit-modules", default="")
    parser.add_argument("--branch-depths", default="3,2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-val-size", type=int, default=5000)
    parser.add_argument("--target-accuracies", nargs="*", type=float, default=[0.98, 0.99, 0.995])
    parser.add_argument("--min-cost-reduction", type=float, default=0.10)
    parser.add_argument("--grid-quantiles", nargs="*", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99])
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    specs = mixture_specs()
    scenarios = required_scenarios(specs)
    exit_modules = [item.strip() for item in (args.exit_modules or DEFAULT_EXITS[args.arch]).split(",") if item.strip()]
    branch_depths = parse_csv_ints(args.branch_depths)
    costs = np.asarray(estimate_costs(args.arch, exit_modules), dtype=np.float32)
    low_cost = float(costs[1])
    high_cost = float(costs[-1])
    model, checkpoint_payload = load_branchynet(args.model_output_name, args.arch, args.dataset, exit_modules, branch_depths, device)
    _, val_set, eval_set = make_datasets(args.dataset, args.threshold_val_size, seed=123)
    val_loader = make_loader(val_set, args.batch_size)
    eval_loader = make_loader(eval_set, args.batch_size)
    base_transform = transform_eval(args.dataset)

    val_by_scenario = {}
    eval_by_scenario = {}
    for scenario in scenarios:
        val_by_scenario[scenario] = collect_scenario(model, val_loader, base_transform, scenario, device, model.exit_names, costs)
        eval_by_scenario[scenario] = collect_scenario(model, eval_loader, base_transform, scenario, device, model.exit_names, costs)

    rows = []
    detailed_results = {}
    for name, spec in specs.items():
        val_mix = concat_data([val_by_scenario[key] for key, _ in spec], [weight for _, weight in spec])
        eval_mix = concat_data([eval_by_scenario[key] for key, _ in spec], [weight for _, weight in spec])
        detailed_results[name] = {}
        for target_accuracy in args.target_accuracies:
            print(f"Sweeping speed trade-off scenario={name} target_accuracy={target_accuracy}", flush=True)
            try:
                result = sweep_policies(val_mix, eval_mix, target_accuracy, args.grid_quantiles)
            except RuntimeError as exc:
                detailed_results[name][str(target_accuracy)] = {"status": "invalid", "reason": str(exc)}
                rows.extend(invalid_rows(name, target_accuracy, str(exc)))
                continue
            detailed_results[name][str(target_accuracy)] = result
            rows.extend(build_comparison_rows(name, target_accuracy, result, low_cost, high_cost))

    payload = {
        "purpose": "Speed-oriented trade-off comparison for safe-pass dual-sided early exit.",
        "target_accuracies": args.target_accuracies,
        "min_cost_reduction": args.min_cost_reduction,
        "model": {
            "checkpoint_output_name": args.model_output_name,
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "dataset": args.dataset,
            "arch": args.arch,
            "exit_modules": exit_modules,
            "branch_depths": branch_depths,
            "exit_names": model.exit_names,
            "exit_costs": [float(x) for x in costs],
        },
        "definitions": {
            "false_pass_rate": "error rate among samples that receive a label",
            "pass_rate": "fraction of samples that receive a reliable label",
            "early_reject_rate": "fraction of samples rejected before the final heavy stage",
            "final_execution_rate": "fraction of samples reaching the final heavy stage",
            "avg_cost": "normalized mean compute cost; HIGH-only is 1.0",
        },
        "rows": rows,
        "detailed_results": detailed_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    write_svg(payload, Path(args.svg))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "svg": args.svg, "rows": len(rows)}, ensure_ascii=False, indent=2), flush=True)
    model.close()


if __name__ == "__main__":
    main()
