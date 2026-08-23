"""Audit PatchCore-lite reductions that are not just memory-bank shrinkage.

The previous minimal search already swept many PatchCore variants.  This script
re-reads those variants and asks category-wise questions that are closer to the
thesis objection:

* Does resnet18 remain competitive with wide_resnet50_2 when the bank search is
  already controlled by the same candidate pool?
* Can the patch grid be reduced?
* Can the feature layer set be simplified?

The output is intentionally explanation-facing: a JSON/CSV record, a Japanese
markdown interpretation, and a compact PNG heatmap for slides.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = Path("results/mvtec_patchcore_profiled_minimal_search_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_nonbank_factor_audit_001_summary.json")
DEFAULT_CSV = Path("results/mvtec_patchcore_nonbank_factor_audit_001.csv")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_nonbank_factor_audit_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_nonbank_factor_audit_001.png")

TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.1f}%"


def best_for_target(row: dict, target: float) -> dict | None:
    for best in row.get("best_rows", []):
        if abs(float(best["target"]) - target) < 1e-12:
            return best
    return None


def layer_kind(row: dict) -> str:
    indices = tuple(row.get("out_indices", []))
    if len(indices) == 1:
        return f"l{indices[0] + 1}"
    return "multi"


def row_score(row: dict, target: float) -> float | None:
    best = best_for_target(row, target)
    if best is None:
        return None
    return best.get("good_pass_rate_good")


def choose_best(rows: list[dict], target: float, predicate: Callable[[dict], bool] | None = None) -> tuple[dict, dict] | None:
    candidates: list[tuple[dict, dict]] = []
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        best = best_for_target(row, target)
        if best is None or best.get("good_pass_rate_good") is None:
            continue
        candidates.append((row, best))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda pair: (
            pair[1]["good_pass_rate_good"],
            -(pair[0].get("relative_nn_ops") or float("inf")),
            -(pair[0].get("relative_bank_int8") or float("inf")),
        ),
    )


def summarize_choice(choice: tuple[dict, dict] | None, best_all_good: float | None) -> dict:
    if choice is None:
        return {
            "config": None,
            "good_pass": None,
            "delta_from_best_all": None,
            "relative_nn_ops": None,
            "relative_bank_int8": None,
            "backbone": None,
            "patch_grid": None,
            "out_indices": None,
            "bank_patches": None,
            "feature_dim": None,
            "patch_count": None,
        }
    row, best = choice
    good = best.get("good_pass_rate_good")
    footprint = row["footprint"]
    return {
        "config": row["config"],
        "good_pass": round_float(good),
        "delta_from_best_all": round_float(best_all_good - good) if best_all_good is not None and good is not None else None,
        "relative_nn_ops": row.get("relative_nn_ops"),
        "relative_bank_int8": row.get("relative_bank_int8"),
        "backbone": row.get("backbone"),
        "patch_grid": row.get("patch_grid"),
        "out_indices": row.get("out_indices"),
        "layer_kind": layer_kind(row),
        "bank_patches": row.get("actual_bank_patches"),
        "feature_dim": footprint.get("feature_dim"),
        "patch_count": footprint.get("patch_count"),
    }


def status_from_delta(delta: float | None, best_good: float | None, tolerance: float, min_interpretable_good: float) -> str:
    if delta is None or best_good is None:
        return "no_data"
    if best_good < min_interpretable_good:
        return "baseline_weak"
    if delta <= tolerance:
        return "ok"
    if delta <= 2.5 * tolerance:
        return "borderline"
    return "ng"


def status_text(status: str) -> str:
    return {
        "ok": "削れる",
        "borderline": "要注意",
        "ng": "落とせない",
        "baseline_weak": "土台弱い",
        "no_data": "データなし",
    }.get(status, status)


def build_category_rows(payload: dict, target: float, tolerance: float, min_interpretable_good: float) -> list[dict]:
    rows = payload["variant_rows"]
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    output: list[dict] = []
    for category, category_rows in sorted(by_category.items()):
        baseline = next((row for row in category_rows if row["config"] == payload["baseline_config"]), None)
        baseline_good = row_score(baseline, target) if baseline is not None else None

        best_all = choose_best(category_rows, target)
        best_all_good = best_all[1]["good_pass_rate_good"] if best_all else None
        best_wrn = choose_best(category_rows, target, lambda row: row["backbone"] == "wide_resnet50_2")
        best_res18 = choose_best(category_rows, target, lambda row: row["backbone"] == "resnet18")
        best_grid7 = choose_best(category_rows, target, lambda row: int(row["patch_grid"]) <= 7)
        best_grid5 = choose_best(category_rows, target, lambda row: int(row["patch_grid"]) <= 5)
        best_single = choose_best(category_rows, target, lambda row: len(row.get("out_indices", [])) == 1)
        best_multi = choose_best(category_rows, target, lambda row: len(row.get("out_indices", [])) > 1)

        res18_summary = summarize_choice(best_res18, best_all_good)
        grid7_summary = summarize_choice(best_grid7, best_all_good)
        grid5_summary = summarize_choice(best_grid5, best_all_good)
        single_summary = summarize_choice(best_single, best_all_good)
        best_summary = summarize_choice(best_all, best_all_good)
        wrn_summary = summarize_choice(best_wrn, best_all_good)
        multi_summary = summarize_choice(best_multi, best_all_good)

        statuses = {
            "backbone_res18": status_from_delta(
                res18_summary["delta_from_best_all"], best_all_good, tolerance, min_interpretable_good
            ),
            "grid_le7": status_from_delta(
                grid7_summary["delta_from_best_all"], best_all_good, tolerance, min_interpretable_good
            ),
            "grid_le5": status_from_delta(
                grid5_summary["delta_from_best_all"], best_all_good, tolerance, min_interpretable_good
            ),
            "single_layer": status_from_delta(
                single_summary["delta_from_best_all"], best_all_good, tolerance, min_interpretable_good
            ),
        }
        ok_count = sum(1 for status in statuses.values() if status == "ok")
        if best_all_good is not None and best_all_good < min_interpretable_good:
            category_class = "土台精度が弱く、削減可否の主張には使いにくい"
        elif statuses["backbone_res18"] == "ok" and statuses["grid_le7"] == "ok" and statuses["single_layer"] == "ok":
            category_class = "大きく削れる"
        elif ok_count >= 2:
            category_class = "一部を削れる"
        elif ok_count == 1:
            category_class = "一点だけ削れる"
        else:
            category_class = "強い構成が必要"

        output.append(
            {
                "category": category,
                "category_type": "texture" if category in TEXTURE_CATEGORIES else "object",
                "target_false_pass_rate_defect": target,
                "baseline_good_pass": round_float(baseline_good),
                "best_all": best_summary,
                "best_wrn": wrn_summary,
                "best_res18": res18_summary,
                "best_grid_le7": grid7_summary,
                "best_grid_le5": grid5_summary,
                "best_single_layer": single_summary,
                "best_multi_layer": multi_summary,
                "statuses": statuses,
                "category_class": category_class,
            }
        )
    return output


def aggregate(category_rows: list[dict]) -> dict:
    status_counts: dict[str, dict[str, int]] = {}
    for factor in ["backbone_res18", "grid_le7", "grid_le5", "single_layer"]:
        counts: dict[str, int] = {}
        for row in category_rows:
            status = row["statuses"][factor]
            counts[status] = counts.get(status, 0) + 1
        status_counts[factor] = counts

    class_counts: dict[str, int] = {}
    type_rows: dict[str, list[dict]] = {}
    for row in category_rows:
        class_counts[row["category_class"]] = class_counts.get(row["category_class"], 0) + 1
        type_rows.setdefault(row["category_type"], []).append(row)

    def mean_good(rows: list[dict], key: str) -> float | None:
        values = [row[key]["good_pass"] for row in rows if row[key]["good_pass"] is not None]
        return round_float(mean(values)) if values else None

    return {
        "status_counts": status_counts,
        "category_class_counts": class_counts,
        "mean_best_all_good_pass": mean_good(category_rows, "best_all"),
        "mean_best_res18_good_pass": mean_good(category_rows, "best_res18"),
        "mean_best_grid_le7_good_pass": mean_good(category_rows, "best_grid_le7"),
        "mean_best_single_layer_good_pass": mean_good(category_rows, "best_single_layer"),
        "by_category_type": {
            category_type: {
                "categories": len(rows),
                "mean_best_all_good_pass": mean_good(rows, "best_all"),
                "mean_best_res18_good_pass": mean_good(rows, "best_res18"),
                "mean_best_grid_le7_good_pass": mean_good(rows, "best_grid_le7"),
                "mean_best_single_layer_good_pass": mean_good(rows, "best_single_layer"),
            }
            for category_type, rows in sorted(type_rows.items())
        },
    }


def write_csv(category_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "category_type",
        "category_class",
        "baseline_good_pass",
        "best_all_good",
        "best_all_config",
        "best_all_rel_nn_ops",
        "res18_good",
        "res18_delta",
        "res18_status",
        "grid_le7_good",
        "grid_le7_delta",
        "grid_le7_status",
        "grid_le5_good",
        "grid_le5_delta",
        "grid_le5_status",
        "single_layer_good",
        "single_layer_delta",
        "single_layer_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in category_rows:
            writer.writerow(
                {
                    "category": row["category"],
                    "category_type": row["category_type"],
                    "category_class": row["category_class"],
                    "baseline_good_pass": row["baseline_good_pass"],
                    "best_all_good": row["best_all"]["good_pass"],
                    "best_all_config": row["best_all"]["config"],
                    "best_all_rel_nn_ops": row["best_all"]["relative_nn_ops"],
                    "res18_good": row["best_res18"]["good_pass"],
                    "res18_delta": row["best_res18"]["delta_from_best_all"],
                    "res18_status": row["statuses"]["backbone_res18"],
                    "grid_le7_good": row["best_grid_le7"]["good_pass"],
                    "grid_le7_delta": row["best_grid_le7"]["delta_from_best_all"],
                    "grid_le7_status": row["statuses"]["grid_le7"],
                    "grid_le5_good": row["best_grid_le5"]["good_pass"],
                    "grid_le5_delta": row["best_grid_le5"]["delta_from_best_all"],
                    "grid_le5_status": row["statuses"]["grid_le5"],
                    "single_layer_good": row["best_single_layer"]["good_pass"],
                    "single_layer_delta": row["best_single_layer"]["delta_from_best_all"],
                    "single_layer_status": row["statuses"]["single_layer"],
                }
            )


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["category_rows"]
    aggregate_payload = payload["aggregate"]
    lines = [
        "# PatchCore-lite 非メモリバンク要素の監査",
        "",
        "## 目的",
        "",
        "カテゴリを減らすとメモリバンクが減るのは当然なので、この実験ではそれ以外の要素を分けて見る。",
        "具体的には、既存の全探索結果を使い、各カテゴリで「バックボーンをresnet18まで落とせるか」「patch gridを7以下まで落とせるか」「特徴層を単層化できるか」を調べた。",
        "",
        "## 判定基準",
        "",
        f"- 欠陥誤通過率の上限: {pct(payload['config']['target_false_pass_rate_defect'])}",
        f"- 各カテゴリで最良構成からの良品通過率低下が {pct(payload['config']['factor_tolerance'])} 以下なら「削れる」とした。",
        f"- ただし最良構成そのものの良品通過率が {pct(payload['config']['min_interpretable_good_pass'])} 未満のカテゴリは、土台が弱いため削減可否の根拠としては扱いにくい。",
        "",
        "## 全体結果",
        "",
        "| 要素 | 削れる | 要注意 | 落とせない | 土台弱い |",
        "|---|---:|---:|---:|---:|",
    ]
    for factor, label in [
        ("backbone_res18", "バックボーン: resnet18"),
        ("grid_le7", "patch grid: 7以下"),
        ("grid_le5", "patch grid: 5以下"),
        ("single_layer", "特徴層: 単層"),
    ]:
        counts = aggregate_payload["status_counts"][factor]
        lines.append(
            f"| {label} | {counts.get('ok', 0)} | {counts.get('borderline', 0)} | "
            f"{counts.get('ng', 0)} | {counts.get('baseline_weak', 0)} |"
        )
    lines.extend(
        [
            "",
            "## カテゴリ別の読み取り",
            "",
            "| category | 種別 | 判定 | best good | best config | res18 | grid<=7 | grid<=5 | 単層 |",
            "|---|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['category_type']} | {row['category_class']} | "
            f"{pct(row['best_all']['good_pass'])} | `{row['best_all']['config']}` | "
            f"{pct(row['best_res18']['good_pass'])} ({status_text(row['statuses']['backbone_res18'])}) | "
            f"{pct(row['best_grid_le7']['good_pass'])} ({status_text(row['statuses']['grid_le7'])}) | "
            f"{pct(row['best_grid_le5']['good_pass'])} ({status_text(row['statuses']['grid_le5'])}) | "
            f"{pct(row['best_single_layer']['good_pass'])} ({status_text(row['statuses']['single_layer'])}) |"
        )
    lines.extend(
        [
            "",
            "## 現時点の解釈",
            "",
            "1. すべてのカテゴリで同じ削り方が通るわけではない。これは「メモリバンクだけを小さくした」という単純な話ではなく、カテゴリごとに必要な特徴抽出の強さと空間解像度が違うことを示している。",
            "2. resnet18でも最良構成に近いカテゴリがある一方、wide_resnet50_2側が必要なカテゴリも残る。つまりカテゴリ別プロファイルにより、検品直前に使うバックボーン/特徴層/gridを切り替える設計に意味がある。",
            "3. gridを5まで落としても崩れないカテゴリと、grid 7以下でも崩れるカテゴリが分かれる。細い傷や局所欠陥を拾うカテゴリでは、単純なgrid削減が危険になりやすい可能性がある。",
            "4. 次に必要なのは、resnet18よりさらに軽いバックボーンを追加したGPU sweepである。今回の既存探索にはMobileNet/EfficientNet系が入っていないため、「もっと落としても成立するか」は追加実験で詰める。",
            "",
            "## 次のGPU実験案",
            "",
            "- `mobilenetv3_small_100`, `mobilenetv3_large_100`, `efficientnet_b0` を候補に入れる。",
            "- bank上限を固定した条件を入れ、メモリバンク削減とバックボーン削減を分離する。",
            "- 今回「削れたカテゴリ」「落とせなかったカテゴリ」をそれぞれ代表カテゴリとして選び、軽量バックボーン追加時の傾向が保存されるか確認する。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in [
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/NotoSansJP-VF.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_figure(category_rows: list[dict], aggregate_payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1780, 920
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    header_font = load_font(22)
    body_font = load_font(20)
    small_font = load_font(17)
    draw.text((40, 28), "PatchCore-lite: bank以外の削減可否", fill=(25, 35, 50), font=title_font)
    draw.text((42, 78), "判定: 最良構成から良品通過率の低下が2pt以内なら削れる", fill=(70, 80, 92), font=small_font)

    factor_labels = [
        ("backbone_res18", "resnet18"),
        ("grid_le7", "grid<=7"),
        ("grid_le5", "grid<=5"),
        ("single_layer", "単層"),
    ]
    colors = {
        "ok": (56, 142, 60),
        "borderline": (245, 166, 35),
        "ng": (198, 57, 57),
        "baseline_weak": (130, 130, 130),
        "no_data": (210, 210, 210),
    }
    x0, y0 = 40, 135
    row_h = 37
    col_w = 135
    draw.text((x0, y0), "category", fill=(20, 20, 20), font=header_font)
    for i, (_factor, label) in enumerate(factor_labels):
        draw.text((x0 + 235 + i * col_w, y0), label, fill=(20, 20, 20), font=header_font)
    config_x = x0 + 235 + len(factor_labels) * col_w + 25
    draw.text((config_x, y0), "best config", fill=(20, 20, 20), font=header_font)

    for idx, row in enumerate(category_rows):
        y = y0 + 45 + idx * row_h
        draw.text((x0, y + 5), row["category"], fill=(35, 35, 35), font=body_font)
        for i, (factor, _label) in enumerate(factor_labels):
            status = row["statuses"][factor]
            x = x0 + 240 + i * col_w
            draw.rounded_rectangle((x, y, x + 100, y + 28), radius=5, fill=colors[status])
            draw.text((x + 14, y + 3), status_text(status), fill="white", font=small_font)
        config = row["best_all"]["config"] or "-"
        if len(config) > 46:
            config = config[:43] + "..."
        draw.text((config_x, y + 5), config, fill=(55, 55, 55), font=small_font)

    bx, by = 1230, 190
    draw.text((bx, by - 38), "集計", fill=(20, 20, 20), font=header_font)
    for i, (factor, label) in enumerate(factor_labels):
        counts = aggregate_payload["status_counts"][factor]
        y = by + i * 58
        draw.text((bx, y + 4), label, fill=(35, 35, 35), font=body_font)
        start = bx + 150
        total = max(1, sum(counts.values()))
        cursor = start
        for status in ["ok", "borderline", "ng", "baseline_weak"]:
            w = int(270 * counts.get(status, 0) / total)
            if w > 0:
                draw.rectangle((cursor, y + 5, cursor + w, y + 28), fill=colors[status])
            cursor += w
        draw.text((start + 285, y + 3), f"削れる {counts.get('ok', 0)}/15", fill=(35, 35, 35), font=small_font)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--target-false-pass", type=float, default=0.03)
    parser.add_argument("--factor-tolerance", type=float, default=0.02)
    parser.add_argument("--min-interpretable-good-pass", type=float, default=0.50)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    category_rows = build_category_rows(
        payload,
        target=args.target_false_pass,
        tolerance=args.factor_tolerance,
        min_interpretable_good=args.min_interpretable_good_pass,
    )
    aggregate_payload = aggregate(category_rows)
    output_payload = {
        "purpose": "Audit PatchCore-lite reductions that are not explained only by memory-bank shrinkage.",
        "config": {
            "source": str(args.input),
            "target_false_pass_rate_defect": args.target_false_pass,
            "factor_tolerance": args.factor_tolerance,
            "min_interpretable_good_pass": args.min_interpretable_good_pass,
            "baseline_config": payload.get("baseline_config"),
            "available_backbones": sorted({row["backbone"] for row in payload["variant_rows"]}),
            "available_patch_grids": sorted({row["patch_grid"] for row in payload["variant_rows"]}),
            "available_bank_patches": sorted({row["requested_bank_patches"] for row in payload["variant_rows"]}),
        },
        "aggregate": aggregate_payload,
        "category_rows": category_rows,
        "outputs": {
            "csv": str(args.csv),
            "markdown": str(args.markdown),
            "figure": str(args.figure),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(category_rows, args.csv)
    write_markdown(output_payload, args.markdown)
    write_figure(category_rows, aggregate_payload, args.figure)
    print(json.dumps(output_payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
