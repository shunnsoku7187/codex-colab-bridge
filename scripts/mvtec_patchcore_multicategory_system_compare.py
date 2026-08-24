"""Compare multi-category PatchCore inspection system options.

The research claim is not that a single-category PatchCore can be tuned.  The
claim is that a shared inspection accelerator can switch among category-specific
tuned profiles and therefore behave close to multiple dedicated systems while
avoiding a worst-case common design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SOURCE = Path("results/mvtec_patchcore_backbone_floor_probe_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_multicategory_system_compare_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_multicategory_system_compare_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_multicategory_system_compare_001.png")


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def best_for_target(row: dict, target: float) -> dict | None:
    for best in row["best_rows"]:
        if abs(float(best["target"]) - target) < 1e-12:
            return best
    return None


def row_good(row: dict, target: float) -> float | None:
    best = best_for_target(row, target)
    return None if best is None else best["good_pass_rate_good"]


def rows_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["category"], []).append(row)
    return out


def selected_minimal(summary: dict, target: float, tolerance: float) -> dict[str, dict]:
    out = {}
    for row in summary["minimal_table"]:
        if abs(row["max_false_pass_rate_defect"] - target) < 1e-12 and abs(row["allowed_good_pass_drop"] - tolerance) < 1e-12:
            out[row["category"]] = row
    return out


def candidate_summary(name: str, description: str, category_rows: dict[str, dict], baseline_by_category: dict[str, dict], target: float) -> dict:
    rows = []
    for category, row in sorted(category_rows.items()):
        baseline = baseline_by_category[category]
        good = row_good(row, target) if "best_rows" in row else row["selected_good_pass"]
        baseline_good = row_good(baseline, target)
        rows.append(
            {
                "category": category,
                "config": row["config"] if "config" in row else row["selected_config"],
                "good_pass": round_float(good),
                "baseline_good_pass": round_float(baseline_good),
                "good_pass_drop_from_baseline": round_float((baseline_good or 0.0) - (good or 0.0)),
                "relative_nn_ops": row["relative_nn_ops"],
                "relative_bank": row.get("relative_bank_int8", row.get("relative_bank", None)),
                "patch_grid": row["patch_grid"],
                "backbone": row["backbone"],
                "out_indices": row["out_indices"],
                "bank_patches": row.get("actual_bank_patches", row.get("bank_patches")),
                "feature_dim": row["footprint"]["feature_dim"] if "footprint" in row else row["feature_dim"],
            }
        )
    good_values = [row["good_pass"] for row in rows if row["good_pass"] is not None]
    rel_ops = [row["relative_nn_ops"] for row in rows if row["relative_nn_ops"] is not None]
    rel_bank = [row["relative_bank"] for row in rows if row["relative_bank"] is not None]
    return {
        "name": name,
        "description": description,
        "category_rows": rows,
        "mean_good_pass": round_float(mean(good_values)) if good_values else None,
        "min_good_pass": round_float(min(good_values)) if good_values else None,
        "mean_relative_nn_ops": round_float(mean(rel_ops)) if rel_ops else None,
        "median_relative_nn_ops": round_float(median(rel_ops)) if rel_ops else None,
        "max_relative_nn_ops": round_float(max(rel_ops)) if rel_ops else None,
        "mean_relative_bank": round_float(mean(rel_bank)) if rel_bank else None,
        "max_relative_bank": round_float(max(rel_bank)) if rel_bank else None,
    }


def choose_best_common(rows: list[dict], baseline_by_category: dict[str, dict], target: float, tolerance: float) -> tuple[str, dict[str, dict]] | None:
    configs = sorted({row["config"] for row in rows})
    best = None
    for config in configs:
        by_category = {row["category"]: row for row in rows if row["config"] == config}
        if len(by_category) != len(baseline_by_category):
            continue
        feasible = True
        for category, baseline in baseline_by_category.items():
            good = row_good(by_category[category], target)
            baseline_good = row_good(baseline, target)
            if good is None or baseline_good is None or good < baseline_good - tolerance:
                feasible = False
                break
        if not feasible:
            continue
        summary = candidate_summary("candidate", "", by_category, baseline_by_category, target)
        key = (summary["mean_relative_nn_ops"], summary["mean_relative_bank"], -summary["mean_good_pass"])
        if best is None or key < best[0]:
            best = (key, config, by_category)
    return None if best is None else (best[1], best[2])


def choose_best_common_under_budget(rows: list[dict], target: float, max_mean_ops: float) -> tuple[str, dict[str, dict]] | None:
    configs = sorted({row["config"] for row in rows})
    best = None
    for config in configs:
        by_category = {row["category"]: row for row in rows if row["config"] == config}
        if len(by_category) < 15:
            continue
        rel = [row["relative_nn_ops"] for row in by_category.values()]
        if mean(rel) > max_mean_ops:
            continue
        goods = [row_good(row, target) for row in by_category.values()]
        if any(value is None for value in goods):
            continue
        key = (mean(goods), -mean(rel))
        if best is None or key > best[0]:
            best = (key, config, by_category)
    return None if best is None else (best[1], best[2])


def choose_bank_only_switch(rows_by_cat: dict[str, list[dict]], baseline_by_category: dict[str, dict], target: float, tolerance: float) -> dict[str, dict]:
    selected = {}
    for category, rows in rows_by_cat.items():
        baseline_good = row_good(baseline_by_category[category], target)
        candidates = []
        for row in rows:
            if row["backbone"] != "wide_resnet50_2" or row["out_indices"] != [1, 2] or int(row["patch_grid"]) != 14:
                continue
            good = row_good(row, target)
            if good is not None and baseline_good is not None and good >= baseline_good - tolerance:
                candidates.append(row)
        selected[category] = min(
            candidates,
            key=lambda row: (row["relative_nn_ops"], row["relative_bank_int8"], -(row_good(row, target) or 0.0)),
        )
    return selected


def build(summary: dict, target: float, tolerance: float, common_budget: float) -> dict:
    rows = summary["variant_rows"]
    by_cat = rows_by_category(rows)
    baseline_config = summary["baseline_config"]
    baseline_by_category = {
        category: next(row for row in category_rows if row["config"] == baseline_config)
        for category, category_rows in by_cat.items()
    }

    full_common = candidate_summary(
        "common_full_patchcore",
        "All categories use the standard full PatchCore profile.",
        baseline_by_category,
        baseline_by_category,
        target,
    )

    common_quality = choose_best_common(rows, baseline_by_category, target, tolerance)
    if common_quality is None:
        common_quality_summary = None
    else:
        config, category_rows = common_quality
        common_quality_summary = candidate_summary(
            "common_light_quality_constrained",
            f"All categories use one shared profile, selected to stay within {pct(tolerance)} of each baseline good-pass.",
            category_rows,
            baseline_by_category,
            target,
        )
        common_quality_summary["shared_config"] = config

    common_budget_choice = choose_best_common_under_budget(rows, target, common_budget)
    common_budget_summary = None
    if common_budget_choice is not None:
        config, category_rows = common_budget_choice
        common_budget_summary = candidate_summary(
            "common_light_budget_constrained",
            f"All categories use one shared lightweight profile under mean NN ops <= {common_budget:.2f}x.",
            category_rows,
            baseline_by_category,
            target,
        )
        common_budget_summary["shared_config"] = config

    bank_only_rows = choose_bank_only_switch(by_cat, baseline_by_category, target, tolerance)
    bank_only = candidate_summary(
        "bank_only_mode_switch",
        "The product mode switches only bank/top-k/threshold while keeping the full backbone/layers/grid fixed.",
        bank_only_rows,
        baseline_by_category,
        target,
    )

    profiled_rows = selected_minimal(summary, target, tolerance)
    profiled = candidate_summary(
        "proposed_full_profile_mode_switch",
        "The product mode switches backbone/layers/grid/bank/top-k/threshold.",
        profiled_rows,
        baseline_by_category,
        target,
    )

    systems = [full_common, bank_only, profiled]
    if common_quality_summary is not None:
        systems.insert(1, common_quality_summary)
    if common_budget_summary is not None:
        systems.insert(2, common_budget_summary)

    profile_rows = profiled["category_rows"]
    all_profile_bank = sum(row["relative_bank"] for row in profile_rows if row["relative_bank"] is not None)
    dedicated_resource_proxy = sum(row["max_relative_nn_ops"] for row in [profiled])
    return {
        "systems": systems,
        "theory": {
            "common_runtime_cost": "C_common",
            "switching_expected_cost": "E[C_switch] = sum_i p_i C_i",
            "uniform_category_usage": "if p_i = 1/K, E[C_switch] = mean_i C_i",
            "nn_cost_formula": "C_NN(i) = P_i * B_i * D_i",
            "relative_formula": "C'_NN / C_NN = (P'/P) * (B'/B) * (D'/D)",
            "quality_constraint": "GoodPass_i(profile) >= GoodPass_i(baseline) - epsilon and FalsePass_i(profile) <= alpha",
            "mode_switch_advantage_condition": "sum_i p_i C_i < C_common",
            "all_resident_bank_proxy_sum_relative_to_one_full_bank": round_float(all_profile_bank),
            "dedicated_multi_accelerator_compute_proxy": "roughly sum or duplicate of category-specific accelerators; proposed shares one configurable fabric",
            "unused_placeholder": dedicated_resource_proxy,
        },
    }


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 多カテゴリ兼用PatchCore検品システムの比較",
        "",
        "## 位置づけ",
        "",
        "提案は，1カテゴリ専用PatchCoreを軽量化すること自体ではない。",
        "提案は，A/B/...複数カテゴリに対応する兼用検品システムにおいて，カテゴリごとの専用チューニング状態を切り替えることである。",
        "",
        "## 理論式",
        "",
        "- 共通兼用機: 全カテゴリで同じ構成を使うため，毎回 `C_common` を支払う。",
        "- 提案切替機: `E[C_switch] = sum_i p_i C_i`。カテゴリ使用頻度が一様なら平均カテゴリコストになる。",
        "- PatchCoreの最近傍探索コスト: `C_NN(i) = P_i * B_i * D_i`。",
        "- 相対削減率: `C'_NN / C_NN = (P'/P) * (B'/B) * (D'/D)`。",
        "- 品質制約: 欠陥誤通過率を `alpha` 以下にし，良品通過率を基準構成から `epsilon` 以内に保つ。",
        "",
        "## システム比較",
        "",
        "| システム | 平均良品通過 | 最低良品通過 | 平均NN計算量 | 中央NN計算量 | 最大NN計算量 | 平均bank | 備考 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for system in payload["systems"]:
        lines.append(
            f"| {system['name']} | {pct(system['mean_good_pass'])} | {pct(system['min_good_pass'])} | "
            f"{system['mean_relative_nn_ops']:.6f}x | {system['median_relative_nn_ops']:.6f}x | "
            f"{system['max_relative_nn_ops']:.6f}x | {system['mean_relative_bank']:.6f}x | "
            f"{system.get('shared_config', system['description'])} |"
        )
    by_name = {system["name"]: system for system in payload["systems"]}
    proposed = by_name["proposed_full_profile_mode_switch"]
    bank_only = by_name["bank_only_mode_switch"]
    full = by_name["common_full_patchcore"]
    budget = by_name.get("common_light_budget_constrained")
    lines += [
        "",
        "## 性能差の読み取り",
        "",
        f"- 標準共通PatchCoreに対し，提案切替は平均NN計算量を `{proposed['mean_relative_nn_ops']:.6f}x` にする。これは約 `{100.0*(1.0-proposed['mean_relative_nn_ops']):.2f}%` 削減である。",
        f"- bankだけ切替に対し，提案切替は平均NN計算量を `{proposed['mean_relative_nn_ops']/bank_only['mean_relative_nn_ops']:.4f}x` にする。つまり約 `{100.0*(1.0-proposed['mean_relative_nn_ops']/bank_only['mean_relative_nn_ops']):.2f}%` さらに削れる。",
        f"- bankだけ切替と提案切替の平均良品通過は `{pct(bank_only['mean_good_pass'])}` と `{pct(proposed['mean_good_pass'])}` でほぼ同等である。",
        f"- 提案切替の最大NN計算量は `{proposed['max_relative_nn_ops']:.6f}x` なので，最悪カテゴリでも標準共通PatchCoreから約 `{100.0*(1.0-proposed['max_relative_nn_ops']):.2f}%` 削減できる。",
    ]
    if budget is not None:
        lines.append(
            f"- 共通軽量構成を平均NN計算量10%以下に制限すると，平均良品通過は `{pct(budget['mean_good_pass'])}`，最低良品通過は `{pct(budget['min_good_pass'])}` まで落ちる。提案切替は同程度以上に軽くしつつ，最低良品通過を `{pct(proposed['min_good_pass'])}` まで保つ。"
        )
    lines += [
        "",
        "## 解釈",
        "",
        "- 標準共通PatchCoreは品質基準だが，すべてのカテゴリで常に最大構成を動かすため重い。",
        "- bankだけ切替は「メモリを減らしただけではないか」という反論に対する比較対象である。",
        "- 提案切替はbankだけでなく，patch数と特徴次元も同時に削るため，bankだけ切替より大きく計算量を下げられる。",
        "- 共通軽量構成は一部カテゴリで品質が崩れる。提案切替はカテゴリごとに強い構成を残せるため，兼用機としての品質低下を抑えやすい。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_font(size: int):
    for font in ["C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/NotoSansJP-VF.ttf", "C:/Windows/Fonts/YuGothM.ttc"]:
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            pass
    return ImageFont.load_default()


def write_figure(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    systems = payload["systems"]
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    body_font = load_font(22)
    small_font = load_font(18)
    draw.text((40, 30), "AB兼用PatchCore検品システムの比較", fill=(20, 30, 45), font=title_font)
    draw.text((42, 78), "同じ検品対象群を扱う兼用機として、共通構成とカテゴリ別切替構成を比較", fill=(70, 80, 95), font=small_font)

    display_names = {
        "common_full_patchcore": "標準共通",
        "common_light_quality_constrained": "品質維持の共通軽量",
        "common_light_budget_constrained": "10%制約の共通軽量",
        "bank_only_mode_switch": "bankだけ切替",
        "proposed_full_profile_mode_switch": "提案: full profile切替",
    }
    x0, y0 = 360, 150
    bar_w = 360
    row_h = 90
    max_cost = max(system["mean_relative_nn_ops"] for system in systems if system["mean_relative_nn_ops"] is not None)
    for idx, system in enumerate(systems):
        y = y0 + idx * row_h
        name = display_names.get(system["name"], system["name"])
        draw.text((40, y + 6), name, fill=(35, 35, 35), font=small_font)
        cost = system["mean_relative_nn_ops"]
        good = system["mean_good_pass"]
        cost_w = int(bar_w * cost / max_cost)
        draw.rectangle((x0, y, x0 + bar_w, y + 24), fill=(230, 235, 240))
        draw.rectangle((x0, y, x0 + max(2, cost_w), y + 24), fill=(242, 142, 43))
        draw.text((x0 + bar_w + 20, y - 1), f"NN {cost:.4f}x", fill=(35, 35, 35), font=small_font)
        draw.rectangle((x0, y + 38, x0 + bar_w, y + 62), fill=(230, 235, 240))
        draw.rectangle((x0, y + 38, x0 + int(bar_w * good), y + 62), fill=(70, 155, 85))
        draw.text((x0 + bar_w + 20, y + 37), f"good {100*good:.1f}%", fill=(35, 35, 35), font=small_font)
    draw.text((40, 660), "提案の優位条件:  Σ p_i C_i  <  C_common。カテゴリ使用頻度が偏っていても、軽いカテゴリが多いほど有利。", fill=(30, 45, 65), font=body_font)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--false-pass-target", type=float, default=0.03)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--common-budget", type=float, default=0.10)
    args = parser.parse_args()

    summary = json.loads(args.source.read_text(encoding="utf-8"))
    payload = {
        "purpose": "compare compatible multi-category PatchCore inspection systems",
        "config": {
            "source": str(args.source),
            "false_pass_target": args.false_pass_target,
            "tolerance": args.tolerance,
            "common_budget": args.common_budget,
        },
        **build(summary, args.false_pass_target, args.tolerance, args.common_budget),
        "outputs": {"markdown": str(args.markdown), "figure": str(args.figure)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "systems": [s["name"] for s in payload["systems"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
