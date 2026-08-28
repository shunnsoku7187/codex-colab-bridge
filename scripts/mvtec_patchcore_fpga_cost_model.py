"""FPGA-facing cost model for fixed-bank PatchCore profile switching.

The previous experiment showed algorithmic performance.  This script translates
the same systems into implementation-facing quantities: stored feature memory,
nearest-neighbor search work, expected cycles for a simple parallel distance
engine, and mode-switch metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

DEFAULT_SOURCE = Path("results/mvtec_patchcore_fixed_bank_profile_switch_002_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_fpga_cost_model_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_fpga_cost_model_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_fpga_cost_model_001.png")


SYSTEM_LABELS = [
    "default profile + common bank",
    "default profile + category bank switch",
    "best fixed profile + category bank switch",
    "proposed profile + category bank switch",
]

STANDARD_FEATURE_DIM = 768


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def mib(bytes_value: float) -> float:
    return bytes_value / (1024.0 * 1024.0)


def bram36(bits: int) -> int:
    return math.ceil(bits / 36_864)


def uram288(bits: int) -> int:
    return math.ceil(bits / 294_912)


def category_feature_values(system: dict, common_bank: bool) -> int:
    rows = system["category_rows"]
    if common_bank:
        first = rows[0]
        return int(first["bank_patches_searched"] * first["feature_dim"])
    return int(sum(row["bank_patches_searched"] * row["feature_dim"] for row in rows))


def searched_bank_patches(system: dict) -> float:
    return float(mean(row["bank_patches_searched"] for row in system["category_rows"]))


def category_max_dim(system: dict) -> int:
    return int(max(row["feature_dim"] for row in system["category_rows"]))


def category_mean_dim(system: dict) -> float:
    return float(mean(row["feature_dim"] for row in system["category_rows"]))


def category_mean_patch_count(system: dict) -> float:
    return float(mean(row["patch_count"] for row in system["category_rows"]))


def nn_cycles(row: dict, lanes: int) -> int:
    return int(row["patch_count"] * row["bank_patches_searched"] * math.ceil(row["feature_dim"] / lanes))


def system_cycles(system: dict, lanes: int) -> dict:
    cycles = [nn_cycles(row, lanes) for row in system["category_rows"]]
    return {
        "mean_cycles": round_float(mean(cycles)),
        "median_cycles": round_float(median(cycles)),
        "max_cycles": int(max(cycles)),
    }


def best_fixed_system(item: dict) -> dict:
    return max(item["systems"][2:-1], key=lambda system: system["min_good_pass"] if system["min_good_pass"] is not None else -1.0)


def normalized_systems(item: dict) -> list[dict]:
    return [
        item["systems"][0],
        item["systems"][1],
        best_fixed_system(item),
        item["systems"][-1],
    ]


def summarize_bank(items: list[dict], args: argparse.Namespace) -> list[dict]:
    rows = []
    for system_index, label in enumerate(SYSTEM_LABELS):
        systems = [normalized_systems(item)[system_index] for item in items]
        common_bank = system_index == 0
        memory_values = [category_feature_values(system, common_bank) for system in systems]
        bits = [value * args.feature_bits for value in memory_values]
        bank_patch_values = [system["total_stored_bank_patches"] for system in systems]
        fixed_width_bits = [value * STANDARD_FEATURE_DIM * args.feature_bits for value in bank_patch_values]
        nn_ops = [system["mean_nn_ops_per_image"] for system in systems]
        good_min = [system["min_good_pass"] for system in systems if system["min_good_pass"] is not None]
        good_mean = [system["mean_good_pass"] for system in systems if system["mean_good_pass"] is not None]
        cycle_rows = [system_cycles(system, args.distance_lanes)["mean_cycles"] for system in systems]
        rows.append(
            {
                "system": label,
                "mean_min_good_pass": round_float(mean(good_min)) if good_min else None,
                "mean_good_pass": round_float(mean(good_mean)) if good_mean else None,
                "mean_nn_ops": round_float(mean(nn_ops)),
                "relative_nn_ops": None,
                "mean_total_bank_patches": round_float(mean(bank_patch_values)),
                "mean_searched_bank_patches_per_image": round_float(mean(searched_bank_patches(system) for system in systems)),
                "mean_stored_feature_values": round_float(mean(memory_values)),
                "mean_compact_feature_memory_mib": round_float(mib(mean(bits) / 8.0)),
                "mean_fixed_width_bank_memory_mib": round_float(mib(mean(fixed_width_bits) / 8.0)),
                "mean_compact_bram36": round_float(mean(bram36(int(bit)) for bit in bits)),
                "mean_compact_uram288": round_float(mean(uram288(int(bit)) for bit in bits)),
                "mean_fixed_width_bram36": round_float(mean(bram36(int(bit)) for bit in fixed_width_bits)),
                "mean_fixed_width_uram288": round_float(mean(uram288(int(bit)) for bit in fixed_width_bits)),
                "max_feature_dim": int(max(category_max_dim(system) for system in systems)),
                "mean_feature_dim_per_mode": round_float(mean(category_mean_dim(system) for system in systems)),
                "mean_patch_count_per_mode": round_float(mean(category_mean_patch_count(system) for system in systems)),
                "mean_cycles_at_lanes": round_float(mean(cycle_rows)),
                "relative_cycles_at_lanes": None,
            }
        )
    base_ops = rows[0]["mean_nn_ops"]
    base_cycles = rows[0]["mean_cycles_at_lanes"]
    for row in rows:
        row["relative_nn_ops"] = round_float(row["mean_nn_ops"] / base_ops)
        row["relative_cycles_at_lanes"] = round_float(row["mean_cycles_at_lanes"] / base_cycles)
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    cfg = payload["config"]
    lines = [
        "# FPGA実装に向けた理論コスト比較",
        "",
        "## 目的",
        "",
        "性能評価で有利でも，FPGA実装時に大きな回路・メモリコストが必要なら利点は弱くなる。ここでは実測ではなく，固定bank数実験の結果を用いて，NN探索量・メモリ量・切替オーバーヘッドを理論値で比較する。",
        "",
        "## 前提",
        "",
        "- CNN本体は `wide_resnet50_2` で固定し，複数CNNをFPGAに載せる構成にはしない。",
        "- 切り替えるのは，使用する中間特徴，特徴次元，grid，top-k，bankのベースアドレス，探索bank長，閾値である。",
        "- bank値は量子化後に保持する想定で，ここでは1特徴値を "
        f"{cfg['feature_bits']} bit として見積もる。",
        "- BRAM36は36 Kbit，URAM288は288 Kbitとして概算する。",
        f"- NN探索器は1サイクルに {cfg['distance_lanes']} 個の特徴差分を処理する単純モデルとする。",
        "- 本実験では総bank点数を固定する。したがってbank点数の削減は主張しない。",
        "- メモリ量は，768次元固定幅でbank RAMを組む場合と，profileごとの有効特徴次元だけを保持する場合を分けて示す。",
        "- mode切替に必要な設定値は，bank本体に比べて十分小さいため，主コストはbank特徴量メモリとNN探索で評価する。",
        "",
        "## 比較対象",
        "",
        "| 記号 | 方式 | 実装上の意味 |",
        "|---|---|---|",
        "| ① | デフォルトprofile + 共通bank | 対象カテゴリ集合のbankを毎回すべて探索する。実装は単純だが探索量が最大。 |",
        "| ② | デフォルトprofile + category bank切替 | 総bank数は同じだが，検品対象カテゴリのbankだけを探索する。 |",
        "| ③ | どれか1つの専用profile固定 + category bank切替 | 軽いprofileを1つだけ採用する。回路は単純だが，カテゴリ相性を外すと性能が下振れる。 |",
        "| ④ | 提案: category別profile + category bank切替 | CNNは共通のまま，カテゴリごとに特徴抽出profileとbankを切り替える。 |",
        "",
    ]
    for bank_size, rows in payload["summary_by_bank_per_category"].items():
        lines += [
            f"## bank/category = {bank_size}",
            "",
            "| 方式 | 最低良品通過率 | 平均良品通過率 | 平均総bank点数 | 平均探索bank点数/画像 | NN演算量 | 固定幅bankメモリ | 有効特徴量メモリ | 最大特徴次元 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['system']} | {pct(row['mean_min_good_pass'])} | {pct(row['mean_good_pass'])} | "
                f"{row['mean_total_bank_patches']:.0f} | {row['mean_searched_bank_patches_per_image']:.0f} | "
                f"{row['relative_nn_ops']:.3f}x | {row['mean_fixed_width_bank_memory_mib']:.3f} MiB | "
                f"{row['mean_compact_feature_memory_mib']:.3f} MiB | {row['max_feature_dim']} |"
            )
        prop = rows[-1]
        bank_only = rows[1]
        fixed = rows[2]
        lines += [
            "",
            "読み取り:",
            "",
            f"- ②に対して④は，NN演算量を {pct(1.0 - prop['relative_nn_ops'] / bank_only['relative_nn_ops'])} 追加削減する。",
            f"- ③に対して④は，平均コストは近いが，カテゴリ別profileを使うためprofile選択ミスによる性能下振れを避ける設計である。",
            f"- 総bank点数は同じなので，bank点数を削った効果ではない。固定幅RAMなら保存メモリは同じで，有効特徴次元だけを詰めて持つ設計なら特徴量メモリも減る。",
            "",
        ]
    lines += [
        "## FPGA実装上の主張",
        "",
        "1. **複数CNNを載せない**: backboneを固定するため，profile切替のためにCNN回路を複製しない。",
        "2. **切替コストが小さい**: category IDでbank開始アドレス，探索長，特徴マスク，top-k数，閾値を切り替えるだけなので，追加制御は小さい。",
        "3. **NN探索の削減が直接効く**: PatchCoreの重い部分はpatch特徴とbankの距離計算であり，探索bank長と特徴次元の削減はサイクル数・メモリアクセス量・消費電力に直接効く。",
        "4. **bank点数削減とは分ける**: 本比較では総bank点数は固定であり，提案方式の利点は探索対象bankの切替と有効特徴次元の削減である。",
        "5. **未使用次元を止められる**: 提案方式では512次元profileのカテゴリでは768次元全体を使わず，距離演算器の一部をクロックゲートまたは無効化できる。",
        "6. **再構成不要**: カテゴリ変更はbitstream再構成ではなくmode registerの更新として扱えるため，検品ラインの段取り替えに合わせやすい。",
        "",
        "## 注意点",
        "",
        "- CNN本体の畳み込み計算は固定backboneなので，本見積もりの主対象はPatchCore後段の特徴保持・NN探索である。",
        "- 実際の速度・消費電力はメモリ帯域，量子化方式，距離演算器の並列度，bank配置に依存するため，最終的にはRTLまたはHLSで実測する必要がある。",
        "- 固定幅768次元のRAM構成では，総bank点数固定のため保存メモリ量は方式間でほぼ同じになる。この場合の利点は，探索bank点数と有効特徴次元を減らせることによるサイクル数・読み出し量・演算器稼働率の低下である。",
        "- 有効特徴次元だけを詰めて格納する可変幅または分割bank構成なら，保存メモリ量も減る。ただしこれは実装方式に依存するため，固定幅メモリ削減としては主張しない。",
        "- ただし理論値上，提案方式は大きな追加メモリや複数CNNを要求しないため，性能面の利点をFPGA実装コストが帳消しにする構造ではない。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        return False

    bank_size = "3000" if "3000" in payload["summary_by_bank_per_category"] else sorted(payload["summary_by_bank_per_category"])[-1]
    rows = payload["summary_by_bank_per_category"][bank_size]
    labels = ["①", "②", "③", "④"]
    ops = [row["relative_nn_ops"] for row in rows]
    mem = [row["mean_compact_feature_memory_mib"] for row in rows]
    good = [row["mean_min_good_pass"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]
    axes[0].bar(labels, ops, color=colors)
    axes[0].set_title("NN search work")
    axes[0].set_ylabel("relative to ①")
    axes[0].set_ylim(0, 1.08)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, mem, color=colors)
    axes[1].set_title("Compact feature storage")
    axes[1].set_ylabel("MiB")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[2].bar(labels, good, color=colors)
    axes[2].set_title("Minimum good-pass")
    axes[2].set_ylabel("rate")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"FPGA-facing cost model, bank/category={bank_size}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--feature-bits", type=int, default=8)
    parser.add_argument("--distance-lanes", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    summary_by_bank = {
        bank_size: summarize_bank(items, args)
        for bank_size, items in source["results_by_bank_per_category"].items()
    }
    payload = {
        "purpose": "FPGA-facing theoretical cost model for category-wise PatchCore profile and bank switching.",
        "config": {
            "source": str(args.source),
            "feature_bits": args.feature_bits,
            "distance_lanes": args.distance_lanes,
        },
        "summary_by_bank_per_category": summary_by_bank,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.markdown)
    wrote_figure = write_figure(payload, args.figure)
    print(
        json.dumps(
            {"wrote": str(args.output), "markdown": str(args.markdown), "figure": str(args.figure) if wrote_figure else None},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
