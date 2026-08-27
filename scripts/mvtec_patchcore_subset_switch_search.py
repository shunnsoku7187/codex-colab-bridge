"""Search claim-friendly AB/ABC category sets for profiled PatchCore switching.

This script reuses the existing per-category PatchCore sweep results.  It does
not recompute mixed memory-bank anomaly scores, so the output is a candidate
screening table: useful for choosing AB/ABC sets that should be re-evaluated on
GPU with true mixed banks.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SOURCE = Path("results/mvtec_patchcore_backbone_floor_probe_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_subset_switch_search_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_subset_switch_search_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_subset_switch_search_001.png")


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
    if "best_rows" in row:
        best = best_for_target(row, target)
        return None if best is None else float(best["good_pass_rate_good"])
    return float(row["selected_good_pass"])


def row_threshold(row: dict, target: float) -> float | None:
    if "best_rows" in row:
        best = best_for_target(row, target)
        return None if best is None else float(best["threshold"])
    return row.get("threshold")


def rows_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["category"], []).append(row)
    return out


def profile_key(row: dict) -> tuple[str, tuple[int, ...], int]:
    return (row["backbone"], tuple(row["out_indices"]), int(row["patch_grid"]))


def config_parts(row: dict) -> dict:
    footprint = row.get("footprint", {})
    return {
        "config": row.get("config", row.get("selected_config")),
        "backbone": row["backbone"],
        "out_indices": list(row["out_indices"]),
        "patch_grid": int(row["patch_grid"]),
        "patch_count": int(row.get("patch_count", footprint.get("patch_count"))),
        "bank_patches": int(row.get("actual_bank_patches", row.get("bank_patches", footprint.get("bank_patches")))),
        "feature_dim": int(row.get("feature_dim", footprint.get("feature_dim"))),
        "topk_fraction": row.get("topk_fraction"),
    }


def selected_minimal(summary: dict, target: float, tolerance: float) -> dict[str, dict]:
    selected = {}
    for row in summary["minimal_table"]:
        if abs(float(row["max_false_pass_rate_defect"]) - target) < 1e-12 and abs(float(row["allowed_good_pass_drop"]) - tolerance) < 1e-12:
            selected[row["category"]] = row
    return selected


def choose_rows_for_profile(
    category_rows: list[dict],
    target: float,
    wanted_profile: tuple[str, tuple[int, ...], int],
) -> dict | None:
    candidates = [row for row in category_rows if profile_key(row) == wanted_profile and row_good(row, target) is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row_good(row, target) or 0.0,
            -row["footprint"]["patch_count"] * row["footprint"]["bank_patches"] * row["footprint"]["feature_dim"],
        ),
    )


def summarize_rows(
    name: str,
    rows: dict[str, dict],
    subset: tuple[str, ...],
    baseline_bank_total: int,
    baseline_patch_count: int,
    baseline_dim: int,
    target: float,
) -> dict:
    category_rows = []
    costs = []
    goods = []
    banks = []
    denom = baseline_patch_count * baseline_dim * baseline_bank_total
    for category in subset:
        row = rows[category]
        parts = config_parts(row)
        cost = parts["patch_count"] * parts["bank_patches"] * parts["feature_dim"] / denom
        good = row_good(row, target)
        costs.append(cost)
        if good is not None:
            goods.append(good)
        banks.append(parts["bank_patches"] / baseline_bank_total)
        category_rows.append(
            {
                "category": category,
                **parts,
                "good_pass": round_float(good),
                "threshold": round_float(row_threshold(row, target)),
                "relative_nn_ops_to_subset_standard": round_float(cost, 9),
                "relative_bank_to_subset_standard": round_float(parts["bank_patches"] / baseline_bank_total, 9),
            }
        )
    return {
        "name": name,
        "category_rows": category_rows,
        "mean_good_pass": round_float(mean(goods)) if goods else None,
        "min_good_pass": round_float(min(goods)) if goods else None,
        "mean_relative_nn_ops_to_subset_standard": round_float(mean(costs), 9),
        "max_relative_nn_ops_to_subset_standard": round_float(max(costs), 9),
        "mean_relative_bank_to_subset_standard": round_float(mean(banks), 9),
    }


def full_standard_system(
    subset: tuple[str, ...],
    baseline_by_category: dict[str, dict],
    baseline_bank_total: int,
    target: float,
) -> dict:
    rows = {}
    for category in subset:
        baseline = baseline_by_category[category]
        parts = config_parts(baseline)
        row = dict(baseline)
        row["actual_bank_patches"] = baseline_bank_total
        row["footprint"] = dict(baseline["footprint"])
        row["footprint"]["bank_patches"] = baseline_bank_total
        row["footprint"]["approx_nn_ops"] = parts["patch_count"] * baseline_bank_total * parts["feature_dim"]
        rows[category] = row
    parts = config_parts(next(iter(baseline_by_category.values())))
    return summarize_rows("subset_standard_merged_bank", rows, subset, baseline_bank_total, parts["patch_count"], parts["feature_dim"], target)


def bank_only_system(
    subset: tuple[str, ...],
    baseline_by_category: dict[str, dict],
    baseline_bank_total: int,
    target: float,
) -> dict:
    parts = config_parts(next(iter(baseline_by_category.values())))
    rows = {category: baseline_by_category[category] for category in subset}
    return summarize_rows("bank_only_switch", rows, subset, baseline_bank_total, parts["patch_count"], parts["feature_dim"], target)


def fixed_profile_system(
    fixed_category: str,
    subset: tuple[str, ...],
    by_category: dict[str, list[dict]],
    selected_by_category: dict[str, dict],
    baseline_bank_total: int,
    target: float,
    baseline_patch_count: int,
    baseline_dim: int,
) -> dict | None:
    fixed = selected_by_category[fixed_category]
    wanted = profile_key(fixed)
    rows = {}
    for category in subset:
        row = choose_rows_for_profile(by_category[category], target, wanted)
        if row is None:
            return None
        rows[category] = row
    system = summarize_rows(
        f"profile_fixed_to_{fixed_category}_bank_switch",
        rows,
        subset,
        baseline_bank_total,
        baseline_patch_count,
        baseline_dim,
        target,
    )
    system["fixed_profile_category"] = fixed_category
    return system


def proposed_system(
    subset: tuple[str, ...],
    selected_by_category: dict[str, dict],
    baseline_bank_total: int,
    target: float,
    baseline_patch_count: int,
    baseline_dim: int,
) -> dict:
    rows = {category: selected_by_category[category] for category in subset}
    return summarize_rows("proposed_profile_and_bank_switch", rows, subset, baseline_bank_total, baseline_patch_count, baseline_dim, target)


def evaluate_subset(
    subset: tuple[str, ...],
    by_category: dict[str, list[dict]],
    baseline_by_category: dict[str, dict],
    selected_by_category: dict[str, dict],
    target: float,
) -> dict | None:
    if any(category not in selected_by_category for category in subset):
        return None
    baseline_parts = config_parts(baseline_by_category[subset[0]])
    baseline_bank_total = sum(config_parts(baseline_by_category[category])["bank_patches"] for category in subset)
    standard = full_standard_system(subset, baseline_by_category, baseline_bank_total, target)
    bank_only = bank_only_system(subset, baseline_by_category, baseline_bank_total, target)
    fixed = [
        fixed_profile_system(category, subset, by_category, selected_by_category, baseline_bank_total, target, baseline_parts["patch_count"], baseline_parts["feature_dim"])
        for category in subset
    ]
    fixed = [system for system in fixed if system is not None]
    proposed = proposed_system(subset, selected_by_category, baseline_bank_total, target, baseline_parts["patch_count"], baseline_parts["feature_dim"])
    best_fixed_good = max((system["mean_good_pass"] or 0.0 for system in fixed), default=0.0)
    best_fixed_min_good = max((system["min_good_pass"] or 0.0 for system in fixed), default=0.0)
    score = (
        100.0 * (1.0 - proposed["mean_relative_nn_ops_to_subset_standard"])
        + 30.0 * (1.0 - proposed["mean_relative_nn_ops_to_subset_standard"] / max(bank_only["mean_relative_nn_ops_to_subset_standard"], 1e-12))
        + 50.0 * max(0.0, (proposed["min_good_pass"] or 0.0) - best_fixed_min_good)
    )
    return {
        "subset": list(subset),
        "subset_size": len(subset),
        "baseline_bank_total": baseline_bank_total,
        "systems": [standard, bank_only, *fixed, proposed],
        "claim_metrics": {
            "proposed_vs_subset_standard_nn_reduction": round_float(1.0 - proposed["mean_relative_nn_ops_to_subset_standard"], 6),
            "proposed_vs_bank_only_nn_reduction": round_float(1.0 - proposed["mean_relative_nn_ops_to_subset_standard"] / max(bank_only["mean_relative_nn_ops_to_subset_standard"], 1e-12), 6),
            "proposed_mean_good_pass": proposed["mean_good_pass"],
            "proposed_min_good_pass": proposed["min_good_pass"],
            "best_fixed_profile_mean_good_pass": round_float(best_fixed_good),
            "best_fixed_profile_min_good_pass": round_float(best_fixed_min_good),
            "proposed_min_good_advantage_over_best_fixed": round_float((proposed["min_good_pass"] or 0.0) - best_fixed_min_good),
            "screening_score": round_float(score),
        },
        "caveat": "subset_standard_merged_bank accuracy is not measured here; mixed-bank accuracy requires a GPU re-evaluation.",
    }


def load_font(size: int, bold: bool = False):
    fonts = ["C:/Windows/Fonts/YuGothB.ttc" if bold else "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"]
    for font in fonts:
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            pass
    return ImageFont.load_default()


def write_markdown(payload: dict, path: Path) -> None:
    def fmt(x):
        return "-" if x is None else f"{x:.6f}"

    lines = [
        "# A/B・A/B/C兼用PatchCore切替候補の総当たり探索",
        "",
        "## 目的",
        "",
        "提案手法の比較対象を「同じ対象カテゴリ集合に対応する兼用機」に直す。",
        "たとえばA/B検品なら，A-Z全カテゴリを持つ標準構成ではなく，A/Bだけを持つ標準構成と比較する。",
        "",
        "## 比較した構成",
        "",
        "1. `subset_standard_merged_bank`: 標準profileで，対象カテゴリすべてのbankを結合して持つ。",
        "2. `bank_only_switch`: 標準profileのまま，カテゴリごとにbankのみ切り替える。",
        "3. `profile_fixed_to_X_bank_switch`: X用profileを固定し，bankだけカテゴリごとに切り替える。",
        "4. `proposed_profile_and_bank_switch`: profileとbankの両方をカテゴリごとに切り替える。",
        "",
        "## 注意",
        "",
        "この集計は既存の単独カテゴリ評価から作った候補探索である。`subset_standard_merged_bank` の混合bank精度は未実測なので，最終主張には上位候補のGPU再評価が必要である。",
        "",
    ]
    for size_key, title in [("top_pairs", "AB候補"), ("top_triples", "ABC候補")]:
        lines += [f"## {title}", ""]
        lines.append("| rank | subset | proposed NN | vs standard削減 | vs bank-only追加削減 | proposed平均良品通過 | proposed最低良品通過 | best固定profile最低 | 最低良品通過差 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for rank, item in enumerate(payload[size_key], start=1):
            m = item["claim_metrics"]
            proposed = next(s for s in item["systems"] if s["name"] == "proposed_profile_and_bank_switch")
            lines.append(
                f"| {rank} | {' + '.join(item['subset'])} | {fmt(proposed['mean_relative_nn_ops_to_subset_standard'])}x | "
                f"{pct(m['proposed_vs_subset_standard_nn_reduction'])} | {pct(m['proposed_vs_bank_only_nn_reduction'])} | "
                f"{pct(m['proposed_mean_good_pass'])} | {pct(m['proposed_min_good_pass'])} | "
                f"{pct(m['best_fixed_profile_min_good_pass'])} | {pct(m['proposed_min_good_advantage_over_best_fixed'])} |"
            )
        lines.append("")
    lines += [
        "## 解釈",
        "",
        "- 対象カテゴリ数が2から3へ増えると，標準merged bankは対象カテゴリ数ぶんbankを持つため，bank-onlyやprofile切替の相対コストは原理的に下がりやすい。",
        "- 提案が本当に強い組み合わせは，`vs bank-only追加削減` が大きく，かつ `固定profile最低` より `proposed最低良品通過` が高い組み合わせである。",
        "- 次は上位AB/ABCについて，混合bankを実際に作るGPUジョブで `subset_standard_merged_bank` の検知精度まで測る。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    title = load_font(34, True)
    body = load_font(22)
    small = load_font(17)
    draw.text((42, 30), "AB/ABC兼用PatchCore: 切替候補の総当たり探索", fill=(20, 32, 48), font=title)
    draw.text((44, 78), "同じ対象カテゴリ集合に対応する標準構成を基準に再比較", fill=(70, 82, 96), font=body)

    def panel(items, x, y, label):
        draw.text((x, y), label, fill=(20, 32, 48), font=body)
        y += 42
        max_bar = 1.0
        for i, item in enumerate(items[:6]):
            m = item["claim_metrics"]
            proposed = next(s for s in item["systems"] if s["name"] == "proposed_profile_and_bank_switch")
            row_y = y + i * 105
            name = "+".join(item["subset"])
            cost = proposed["mean_relative_nn_ops_to_subset_standard"]
            good = m["proposed_min_good_pass"] or 0.0
            draw.text((x, row_y), f"{i+1}. {name}", fill=(35, 35, 35), font=small)
            draw.rectangle((x + 260, row_y + 4, x + 580, row_y + 27), fill=(231, 236, 242))
            draw.rectangle((x + 260, row_y + 4, x + 260 + max(2, int(320 * cost / max_bar)), row_y + 27), fill=(242, 142, 43))
            draw.text((x + 595, row_y + 1), f"NN {cost:.4f}x", fill=(35, 35, 35), font=small)
            draw.rectangle((x + 260, row_y + 41, x + 580, row_y + 64), fill=(231, 236, 242))
            draw.rectangle((x + 260, row_y + 41, x + 260 + int(320 * good), row_y + 64), fill=(70, 155, 85))
            draw.text((x + 595, row_y + 38), f"min good {100*good:.1f}%", fill=(35, 35, 35), font=small)

    panel(payload["top_pairs"], 42, 140, "AB候補")
    panel(payload["top_triples"], 780, 140, "ABC候補")
    draw.text((44, 835), "注: これは単独カテゴリ評価からの候補探索。混合bankの実精度は上位候補で再評価する。", fill=(80, 80, 80), font=small)
    image.save(path)


def build(args: argparse.Namespace) -> dict:
    summary = json.loads(args.source.read_text(encoding="utf-8"))
    rows = summary["variant_rows"]
    by_category = rows_by_category(rows)
    baseline_config = summary["baseline_config"]
    baseline_by_category = {
        category: next(row for row in category_rows if row["config"] == baseline_config)
        for category, category_rows in by_category.items()
    }
    selected_by_category = selected_minimal(summary, args.false_pass_target, args.tolerance)
    categories = sorted(baseline_by_category)
    all_results = []
    for size in args.subset_sizes:
        for subset in itertools.combinations(categories, size):
            result = evaluate_subset(subset, by_category, baseline_by_category, selected_by_category, args.false_pass_target)
            if result is not None:
                all_results.append(result)
    all_results.sort(key=lambda item: item["claim_metrics"]["screening_score"], reverse=True)
    pairs = [item for item in all_results if item["subset_size"] == 2]
    triples = [item for item in all_results if item["subset_size"] == 3]
    return {
        "purpose": "screen AB/ABC category subsets for category-wise profile and bank switching claims",
        "config": {
            "source": str(args.source),
            "false_pass_target": args.false_pass_target,
            "tolerance": args.tolerance,
            "subset_sizes": args.subset_sizes,
            "top_k": args.top_k,
        },
        "counts": {"pairs": len(pairs), "triples": len(triples)},
        "top_pairs": pairs[: args.top_k],
        "top_triples": triples[: args.top_k],
        "all_results": all_results,
        "next_required_experiment": "GPU re-evaluation with true merged memory banks for the top AB/ABC candidates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--false-pass-target", type=float, default=0.03)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(
        json.dumps(
            {
                "wrote": str(args.output),
                "pairs": payload["counts"]["pairs"],
                "triples": payload["counts"]["triples"],
                "top_pair": payload["top_pairs"][0]["subset"] if payload["top_pairs"] else None,
                "top_triple": payload["top_triples"][0]["subset"] if payload["top_triples"] else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
