"""FPGA-facing cost model for profiled minimal PatchCore configurations.

This script converts the current experimental evidence into hardware-facing
numbers: feature-extractor MACs, nearest-neighbor operations, memory-bank size,
bank traffic, and estimated KNN latency for several parallel distance-engine
widths.  It is intentionally an estimate, not a substitute for RTL/HLS
implementation.  Its purpose is to decide whether the thesis can move into
"implement and measure on FPGA" mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.mvtec_patchcore_cost_credibility_audit import parse_profile_from_config
from scripts.train_kolektor_strong_final import round_float
from src.experiment_paths import ensure_dirs


def estimate_profile_macs(profile: dict, image_size: tuple[int, int]) -> int:
    """Return approximate multiply-adds for the timm feature extractor."""
    import torch
    import timm
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        profile["backbone"],
        pretrained=False,
        features_only=True,
        out_indices=tuple(profile["out_indices"]),
    ).eval().to(device)
    info = summary(
        model,
        input_size=(1, 3, image_size[0], image_size[1]),
        verbose=0,
        col_names=("mult_adds",),
        device=str(device),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return int(info.total_mult_adds or 0)


@lru_cache(maxsize=None)
def profile_macs_from_config(config_name: str, image_height: int, image_width: int) -> int:
    profile, _bank, _topk = parse_profile_from_config(config_name)
    return estimate_profile_macs(profile, (image_height, image_width))


def cycles_to_ms(cycles: int, clock_mhz: float) -> float:
    return 1000.0 * cycles / (clock_mhz * 1_000_000.0)


def build_rows(args: argparse.Namespace, cost_payload: dict, holdout_payload: dict) -> list[dict]:
    holdout_by_category = {
        row["category"]: row
        for row in holdout_payload["category_summary"]
        if row["runs"] > 0
    }
    rows = []
    for row in cost_payload["rows"]:
        category = row["category"]
        base = cost_payload["baseline_details"][category]
        selected = cost_payload["selected_details"][category]
        holdout = holdout_by_category.get(category, {})
        base_fp = base["footprint"]
        sel_fp = selected["footprint"]

        base_cnn_macs = profile_macs_from_config(
            row["baseline_config"], args.image_height, args.image_width
        )
        sel_cnn_macs = profile_macs_from_config(
            row["selected_config"], args.image_height, args.image_width
        )
        base_nn_ops = int(base_fp["per_image_nn_ops"])
        sel_nn_ops = int(sel_fp["per_image_nn_ops"])
        base_total_ops = base_cnn_macs + args.distance_op_weight * base_nn_ops
        sel_total_ops = sel_cnn_macs + args.distance_op_weight * sel_nn_ops

        base_bank_bytes_int8 = int(base_fp["bank_bytes_int8"])
        sel_bank_bytes_int8 = int(sel_fp["bank_bytes_int8"])
        base_stream_bytes_int8 = int(base_fp["patch_count"] * base_bank_bytes_int8)
        sel_stream_bytes_int8 = int(sel_fp["patch_count"] * sel_bank_bytes_int8)
        base_cached_bytes_int8 = int(base_bank_bytes_int8 + base_fp["per_image_feature_values"])
        sel_cached_bytes_int8 = int(sel_bank_bytes_int8 + sel_fp["per_image_feature_values"])

        lane_estimates = {}
        for lanes in args.knn_lanes:
            base_cycles = math.ceil(base_nn_ops / lanes)
            sel_cycles = math.ceil(sel_nn_ops / lanes)
            lane_estimates[str(lanes)] = {
                "baseline_knn_cycles": base_cycles,
                "selected_knn_cycles": sel_cycles,
                "baseline_knn_ms": round_float(cycles_to_ms(base_cycles, args.clock_mhz)),
                "selected_knn_ms": round_float(cycles_to_ms(sel_cycles, args.clock_mhz)),
            }

        rows.append(
            {
                "category": category,
                "selected_config": row["selected_config"],
                "holdout_baseline_good_pass": holdout.get("baseline_holdout_good_pass_mean"),
                "holdout_selected_good_pass": holdout.get("selected_holdout_good_pass_mean"),
                "holdout_selected_false_pass": holdout.get("selected_holdout_false_pass_mean"),
                "baseline_cnn_macs": base_cnn_macs,
                "selected_cnn_macs": sel_cnn_macs,
                "relative_cnn_macs": round_float(sel_cnn_macs / base_cnn_macs) if base_cnn_macs else None,
                "baseline_nn_ops_per_image": base_nn_ops,
                "selected_nn_ops_per_image": sel_nn_ops,
                "relative_nn_ops": round_float(sel_nn_ops / base_nn_ops) if base_nn_ops else None,
                "baseline_total_proxy_ops": int(base_total_ops),
                "selected_total_proxy_ops": int(sel_total_ops),
                "relative_total_proxy_ops": round_float(sel_total_ops / base_total_ops) if base_total_ops else None,
                "baseline_bank_bytes_int8": base_bank_bytes_int8,
                "selected_bank_bytes_int8": sel_bank_bytes_int8,
                "relative_bank_bytes_int8": round_float(sel_bank_bytes_int8 / base_bank_bytes_int8)
                if base_bank_bytes_int8
                else None,
                "baseline_stream_bytes_int8_per_image": base_stream_bytes_int8,
                "selected_stream_bytes_int8_per_image": sel_stream_bytes_int8,
                "relative_stream_bytes_int8": round_float(sel_stream_bytes_int8 / base_stream_bytes_int8)
                if base_stream_bytes_int8
                else None,
                "baseline_cached_bytes_int8_per_image": base_cached_bytes_int8,
                "selected_cached_bytes_int8_per_image": sel_cached_bytes_int8,
                "relative_cached_bytes_int8": round_float(sel_cached_bytes_int8 / base_cached_bytes_int8)
                if base_cached_bytes_int8
                else None,
                "lane_estimates": lane_estimates,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    keys = [
        "relative_cnn_macs",
        "relative_nn_ops",
        "relative_total_proxy_ops",
        "relative_bank_bytes_int8",
        "relative_stream_bytes_int8",
        "relative_cached_bytes_int8",
    ]
    out = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        out[f"mean_{key}"] = round_float(float(np.mean(values))) if values else None
        out[f"median_{key}"] = round_float(float(np.median(values))) if values else None
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    columns = [
        "category",
        "selected_config",
        "holdout_baseline_good_pass",
        "holdout_selected_good_pass",
        "holdout_selected_false_pass",
        "relative_cnn_macs",
        "relative_nn_ops",
        "relative_total_proxy_ops",
        "relative_bank_bytes_int8",
        "relative_stream_bytes_int8",
        "relative_cached_bytes_int8",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def plot_summary(payload: dict, path: Path) -> None:
    rows = payload["rows"]
    labels = [row["category"] for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    rel_cnn = [100.0 * row["relative_cnn_macs"] for row in rows]
    rel_nn = [100.0 * row["relative_nn_ops"] for row in rows]
    rel_total = [100.0 * row["relative_total_proxy_ops"] for row in rows]
    rel_bank = [100.0 * row["relative_bank_bytes_int8"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.4), sharex=True)
    axes[0].bar(x - width, rel_cnn, width=width, label="CNN MAC", color="#4c78a8")
    axes[0].bar(x, rel_nn, width=width, label="NN ops", color="#f58518")
    axes[0].bar(x + width, rel_total, width=width, label="total proxy", color="#54a24b")
    axes[0].set_ylabel("selected / baseline [%]")
    axes[0].set_title("FPGA-facing compute proxy")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - 0.16, rel_bank, width=0.32, label="bank storage", color="#b279a2")
    axes[1].bar(
        x + 0.16,
        [100.0 * row["relative_stream_bytes_int8"] for row in rows],
        width=0.32,
        label="streamed bank traffic",
        color="#e45756",
    )
    axes[1].set_ylabel("selected / baseline [%]")
    axes[1].set_title("FPGA-facing memory proxy")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path) -> None:
    agg = payload["aggregate"]
    lines = [
        "# FPGA cost model for profiled PatchCore",
        "",
        "Purpose: translate the current PatchCore reduction evidence into FPGA-facing resource and latency proxies.",
        "",
        "## Scope",
        "",
        "This is a pre-RTL estimate.  It does not claim final FPGA power or timing.",
        "It separates the parts that must be implemented and measured next:",
        "",
        "- CNN feature extraction MACs",
        "- PatchCore nearest-neighbor distance operations",
        "- Memory-bank storage",
        "- Memory-bank read traffic",
        "- KNN latency under several parallel distance-lane counts",
        "",
        "## Aggregate ratios",
        "",
        f"- mean CNN MAC ratio: `{agg['mean_relative_cnn_macs']:.4f}x`",
        f"- mean NN operation ratio: `{agg['mean_relative_nn_ops']:.4f}x`",
        f"- mean total proxy ratio: `{agg['mean_relative_total_proxy_ops']:.4f}x`",
        f"- mean memory-bank ratio: `{agg['mean_relative_bank_bytes_int8']:.4f}x`",
        f"- mean streamed-bank traffic ratio: `{agg['mean_relative_stream_bytes_int8']:.4f}x`",
        "",
        "## Category table",
        "",
        "| category | selected config | good-pass | false-pass | CNN MAC | NN ops | total proxy | bank | streamed traffic |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['category']} | `{row['selected_config']}` | "
            f"{pct(row['holdout_selected_good_pass'])} | {pct(row['holdout_selected_false_pass'])} | "
            f"{row['relative_cnn_macs']:.4f}x | {row['relative_nn_ops']:.4f}x | "
            f"{row['relative_total_proxy_ops']:.4f}x | {row['relative_bank_bytes_int8']:.4f}x | "
            f"{row['relative_stream_bytes_int8']:.4f}x |"
        )
    lines += [
        "",
        "## Interpretation for thesis lock",
        "",
        "- The nearest-neighbor search reduction is mathematically explained by patch count, bank size, and feature dimension.",
        "- After the NN search is reduced, CNN feature extraction becomes the dominant remaining compute block.",
        "- Therefore the FPGA thesis should not claim only `PatchCore is 98% lighter`.",
        "- The defensible claim is: category profiling can shrink the memory-bank search engine dramatically, and the final FPGA implementation must measure how much of that reduction survives after CNN and memory-system costs are included.",
        "",
        f"CSV: `{payload['csv']}`",
        f"Figure: `{payload['figure']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-summary", default="results/mvtec_patchcore_cost_credibility_audit_002_summary.json")
    parser.add_argument("--holdout-summary", default="results/mvtec_patchcore_profiled_holdout_validation_001_summary.json")
    parser.add_argument("--output", default="results/mvtec_patchcore_fpga_cost_model_001_summary.json")
    parser.add_argument("--csv", default="results/mvtec_patchcore_fpga_cost_model_001.csv")
    parser.add_argument("--markdown", default="docs/mvtec_patchcore_fpga_cost_model_001.md")
    parser.add_argument("--figure", default="results/mvtec_patchcore_fpga_cost_model_001.png")
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--clock-mhz", type=float, default=200.0)
    parser.add_argument("--distance-op-weight", type=float, default=3.0)
    parser.add_argument("--knn-lanes", nargs="*", type=int, default=[64, 128, 256, 512, 1024, 2048])
    args = parser.parse_args()

    ensure_dirs()
    cost_payload = json.loads(Path(args.cost_summary).read_text(encoding="utf-8"))
    holdout_payload = json.loads(Path(args.holdout_summary).read_text(encoding="utf-8"))
    rows = build_rows(args, cost_payload, holdout_payload)
    payload = {
        "purpose": "FPGA-facing cost model for category-profiled minimal PatchCore.",
        "config": vars(args),
        "aggregate": summarize(rows),
        "rows": rows,
        "csv": args.csv,
        "figure": args.figure,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, Path(args.csv))
    plot_summary(payload, Path(args.figure))
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "csv": args.csv, "markdown": args.markdown, "figure": args.figure}, indent=2))


if __name__ == "__main__":
    main()
