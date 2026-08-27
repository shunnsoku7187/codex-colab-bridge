"""Create presentation-friendly tables for AB/ABC PatchCore switching comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "mvtec_patchcore_subset_mixed_bank_verify_001_summary.json"
OUT_MD = ROOT / "docs" / "mvtec_patchcore_subset_switch_comparison_tables.md"
OUT_CSV = ROOT / "results" / "mvtec_patchcore_subset_switch_comparison_tables.csv"


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def x(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}x"


def find_system(item: dict, name: str) -> dict:
    if name == "true_mixed_standard":
        return item["true_mixed_standard"]
    return item[name]


def fixed_systems(item: dict) -> list[dict]:
    return item["fixed_profile_systems"]


def split_cost_factors(system: dict) -> tuple[str, str, str]:
    rows = system.get("category_rows", [])
    if not rows:
        return "-", "-", "-"
    bank = []
    pd = []
    for row in rows:
        rel_nn = row.get("relative_nn_ops_to_subset_standard")
        rel_bank = row.get("relative_bank_to_subset_standard")
        if rel_nn is None or rel_bank is None:
            continue
        rel_bank = float(rel_bank)
        bank.append(rel_bank)
        pd.append(float(rel_nn) / rel_bank if rel_bank > 0 else None)
    bank_values = [value for value in bank if value is not None]
    pd_values = [value for value in pd if value is not None]
    if not bank_values:
        return "-", "-", "-"
    mean_bank = sum(bank_values) / len(bank_values)
    mean_pd = sum(pd_values) / len(pd_values)
    return x(mean_bank), x(mean_pd), x(mean_bank * mean_pd)


def row_for_system(subset: str, label: str, system: dict, note: str) -> dict:
    bank_factor, profile_factor, recomposed = split_cost_factors(system)
    return {
        "subset": subset,
        "system": label,
        "mean_good_pass": pct(system.get("mean_good_pass")),
        "min_good_pass": pct(system.get("min_good_pass")),
        "mean_nn": x(system.get("mean_relative_nn_ops_to_subset_standard", 1.0)),
        "max_nn": x(system.get("max_relative_nn_ops_to_subset_standard", 1.0)),
        "mean_bank": x(system.get("mean_relative_bank_to_subset_standard", 1.0)),
        "bank_factor": bank_factor,
        "profile_factor": profile_factor,
        "recomposed_nn": recomposed,
        "note": note,
    }


def build_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for item in payload["verified_subsets"]:
        subset = " + ".join(item["subset"])
        rows.append(
            row_for_system(
                subset,
                "① 標準profile + 混合bank",
                item["true_mixed_standard"],
                "対象カテゴリだけの正常bankを結合した標準構成",
            )
        )
        rows.append(
            row_for_system(
                subset,
                "② 標準profile + bank切替",
                item["bank_only_switch"],
                "CNN/層/gridは標準のままbankのみカテゴリ別",
            )
        )
        for fixed in fixed_systems(item):
            category = fixed["name"].replace("profile_fixed_to_", "").replace("_bank_switch", "")
            rows.append(
                row_for_system(
                    subset,
                    f"固定profile({category}) + bank切替",
                    fixed,
                    f"{category}用profileを全カテゴリに固定",
                )
            )
        rows.append(
            row_for_system(
                subset,
                "★ profile + bank両切替",
                item["proposed_profile_and_bank_switch"],
                "カテゴリごとにprofileとbankを両方切替",
            )
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "system",
                "mean_good_pass",
                "min_good_pass",
                "mean_nn",
                "max_nn",
                "mean_bank",
                "bank_factor",
                "profile_factor",
                "recomposed_nn",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(payload: dict, rows: list[dict]) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AB/ABC兼用PatchCore切替方式の比較表",
        "",
        "同じ対象カテゴリ集合に対して，①標準混合bank，②bankのみ切替，③以降の固定profile，★提案方式を同じ表で比較する。",
        "NN計算量とbank量は，各subsetの標準profile + 混合bankを1.0xとした相対値である。",
        "",
        "注意: `bank量` は単なるA+BをA/Bに分けた効果だけではない。profile切替後の小型bank数まで含む。",
        "そのため，②bankのみ切替はABで約0.5x，ABCで約0.333xになる。一方，★ではbank数そのものも125や750まで削るため，さらに小さくなる。",
        "",
    ]
    for item in payload["verified_subsets"]:
        subset = " + ".join(item["subset"])
        subset_rows = [row for row in rows if row["subset"] == subset]
        lines += [
            f"## {subset}",
            "",
            "| 方式 | 平均良品通過 | 最低良品通過 | 平均NN計算量 | bank要因 | profile要因(P×D) | 平均bank量 | 読み取り |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in subset_rows:
            lines.append(
                f"| {row['system']} | {row['mean_good_pass']} | {row['min_good_pass']} | "
                f"{row['mean_nn']} | {row['bank_factor']} | {row['profile_factor']} | {row['mean_bank']} | {row['note']} |"
            )
        lines.append("")
    lines += [
        "## 発表で使いやすい抜粋",
        "",
        "| subset | 標準最低良品通過 | 提案最低良品通過 | 提案平均NN | profile固定で一番良い最低良品通過 | 読み取り |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in payload["verified_subsets"]:
        subset = " + ".join(item["subset"])
        fixed_min = max(float(s["min_good_pass"]) for s in item["fixed_profile_systems"])
        mixed = item["true_mixed_standard"]
        proposed = item["proposed_profile_and_bank_switch"]
        if subset in {"hazelnut + tile", "toothbrush + wood", "bottle + toothbrush + wood"}:
            if subset == "hazelnut + tile":
                note = "品質が最もきれいで，提案方式の安全な代表例"
            elif subset == "toothbrush + wood":
                note = "固定profileとの差が大きく，両切替の必要性を示しやすいAB例"
            else:
                note = "ABCに増やしても同じ傾向が残る代表例"
            lines.append(
                f"| {subset} | {pct(mixed['min_good_pass'])} | {pct(proposed['min_good_pass'])} | "
                f"{x(proposed['mean_relative_nn_ops_to_subset_standard'])} | {pct(fixed_min)} | {note} |"
            )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = build_rows(payload)
    write_csv(rows)
    write_markdown(payload, rows)
    print(json.dumps({"markdown": str(OUT_MD), "csv": str(OUT_CSV), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
