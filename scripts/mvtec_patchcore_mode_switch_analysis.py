"""Mode-switching analysis for category-profiled PatchCore FPGA designs.

The thesis-facing question is whether the current category-profiled PatchCore
results support a mode-switching FPGA architecture:

  setup product/category mode before inspection starts
  -> load only the needed memory bank/config/threshold
  -> run a fixed low-latency inspection pipeline during production

This is deliberately different from per-image dynamic routing.  The switching
cost is paid between production modes, not on every image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.train_kolektor_strong_final import round_float
from src.experiment_paths import ensure_dirs


def bytes_to_kib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / 1024.0


def bytes_to_mib(value: int | float | None) -> float | None:
    return None if value is None else float(value) / (1024.0 * 1024.0)


def switch_time_ms(bytes_value: int, bandwidth_mib_s: float) -> float:
    bytes_per_s = bandwidth_mib_s * 1024.0 * 1024.0
    return 1000.0 * bytes_value / bytes_per_s


def summarize_mode_storage(rows: list[dict], bandwidths: list[float], quality_false_pass_limit: float) -> dict:
    baseline_bank_sum = sum(row["baseline_bank_bytes_int8"] for row in rows)
    selected_bank_sum = sum(row["selected_bank_bytes_int8"] for row in rows)
    baseline_bank_max = max(row["baseline_bank_bytes_int8"] for row in rows)
    selected_bank_max = max(row["selected_bank_bytes_int8"] for row in rows)
    selected_bank_median = float(np.median([row["selected_bank_bytes_int8"] for row in rows]))
    quality_rows = [row for row in rows if row["holdout_selected_false_pass"] <= quality_false_pass_limit]
    return {
        "categories": len(rows),
        "quality_false_pass_limit": quality_false_pass_limit,
        "quality_rows": len(quality_rows),
        "baseline_all_category_bank_mib": round_float(bytes_to_mib(baseline_bank_sum)),
        "selected_all_category_bank_mib": round_float(bytes_to_mib(selected_bank_sum)),
        "all_category_bank_ratio": round_float(selected_bank_sum / baseline_bank_sum) if baseline_bank_sum else None,
        "all_category_bank_reduction": round_float(1.0 - selected_bank_sum / baseline_bank_sum) if baseline_bank_sum else None,
        "baseline_active_bank_mib": round_float(bytes_to_mib(baseline_bank_max)),
        "max_selected_active_bank_mib": round_float(bytes_to_mib(selected_bank_max)),
        "median_selected_active_bank_mib": round_float(bytes_to_mib(selected_bank_median)),
        "max_active_bank_ratio": round_float(selected_bank_max / baseline_bank_max) if baseline_bank_max else None,
        "median_active_bank_ratio": round_float(selected_bank_median / baseline_bank_max) if baseline_bank_max else None,
        "selected_all_bank_switch_time_ms": {
            f"{bandwidth:g}_MiB_s": round_float(switch_time_ms(selected_bank_sum, bandwidth))
            for bandwidth in bandwidths
        },
        "max_selected_bank_switch_time_ms": {
            f"{bandwidth:g}_MiB_s": round_float(switch_time_ms(selected_bank_max, bandwidth))
            for bandwidth in bandwidths
        },
    }


def build_mode_rows(rows: list[dict], bandwidths: list[float], lane_choices: list[int]) -> list[dict]:
    out = []
    for row in rows:
        selected_bank = int(row["selected_bank_bytes_int8"])
        baseline_bank = int(row["baseline_bank_bytes_int8"])
        lane_estimates = {}
        for lanes in lane_choices:
            lane = row["lane_estimates"].get(str(lanes), {})
            lane_estimates[str(lanes)] = {
                "baseline_knn_ms": lane.get("baseline_knn_ms"),
                "selected_knn_ms": lane.get("selected_knn_ms"),
                "speedup": round_float(lane["baseline_knn_ms"] / lane["selected_knn_ms"])
                if lane.get("selected_knn_ms")
                else None,
            }
        out.append(
            {
                "category": row["category"],
                "selected_config": row["selected_config"],
                "good_pass": row["holdout_selected_good_pass"],
                "false_pass": row["holdout_selected_false_pass"],
                "baseline_bank_bytes_int8": baseline_bank,
                "selected_bank_bytes_int8": selected_bank,
                "baseline_bank_mib": round_float(bytes_to_mib(baseline_bank)),
                "selected_bank_kib": round_float(bytes_to_kib(selected_bank)),
                "selected_bank_mib": round_float(bytes_to_mib(selected_bank)),
                "bank_ratio": row["relative_bank_bytes_int8"],
                "stream_ratio": row["relative_stream_bytes_int8"],
                "nn_ratio": row["relative_nn_ops"],
                "total_proxy_ratio": row["relative_total_proxy_ops"],
                "switch_time_ms": {
                    f"{bandwidth:g}_MiB_s": round_float(switch_time_ms(selected_bank, bandwidth))
                    for bandwidth in bandwidths
                },
                "lane_estimates": lane_estimates,
            }
        )
    return out


def build_architecture_options(summary: dict, rows: list[dict], lane: int) -> list[dict]:
    selected_max_latency = max(row["lane_estimates"][str(lane)]["selected_knn_ms"] for row in rows)
    baseline_max_latency = max(row["lane_estimates"][str(lane)]["baseline_knn_ms"] for row in rows)
    selected_mean_latency = float(np.mean([row["lane_estimates"][str(lane)]["selected_knn_ms"] for row in rows]))
    baseline_mean_latency = float(np.mean([row["lane_estimates"][str(lane)]["baseline_knn_ms"] for row in rows]))
    return [
        {
            "name": "full_baseline_mode",
            "description": "Use the full baseline PatchCore configuration for every product category.",
            "resident_bank_mib": summary["baseline_all_category_bank_mib"],
            "active_bank_mib": summary["baseline_active_bank_mib"],
            "mean_knn_ms": round_float(baseline_mean_latency),
            "worst_knn_ms": round_float(baseline_max_latency),
            "strength": "highest reference capacity",
            "weakness": "large bank storage and KNN latency",
        },
        {
            "name": "profiled_active_load",
            "description": "Load only the selected category bank before inspection starts.",
            "resident_bank_mib": summary["max_selected_active_bank_mib"],
            "active_bank_mib": summary["max_selected_active_bank_mib"],
            "mean_knn_ms": round_float(selected_mean_latency),
            "worst_knn_ms": round_float(selected_max_latency),
            "strength": "minimum on-chip active memory and deterministic during inspection",
            "weakness": "mode switch must load the next category bank",
        },
        {
            "name": "profiled_all_resident",
            "description": "Keep all selected category banks resident and switch pointers/configs.",
            "resident_bank_mib": summary["selected_all_category_bank_mib"],
            "active_bank_mib": summary["max_selected_active_bank_mib"],
            "mean_knn_ms": round_float(selected_mean_latency),
            "worst_knn_ms": round_float(selected_max_latency),
            "strength": "near-zero mode switch latency after initialization",
            "weakness": "requires enough on-chip/off-chip storage for every selected bank",
        },
    ]


def plot(payload: dict, path: Path, lane: int) -> None:
    rows = payload["mode_rows"]
    labels = [row["category"] for row in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(12.2, 9.0), sharex=True)
    axes[0].bar(x, [100.0 * row["bank_ratio"] for row in rows], color="#4c78a8")
    axes[0].set_ylabel("bank [%]")
    axes[0].set_title("Mode-specific memory bank size vs full baseline")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, [100.0 * row["nn_ratio"] for row in rows], color="#f58518")
    axes[1].set_ylabel("NN ops [%]")
    axes[1].set_title("Mode-specific KNN search cost vs full baseline")
    axes[1].grid(True, axis="y", alpha=0.3)

    false_pass = [100.0 * row["false_pass"] for row in rows]
    good_pass = [100.0 * row["good_pass"] for row in rows]
    axes[2].bar(x - 0.18, good_pass, width=0.36, label="good-pass", color="#54a24b")
    axes[2].bar(x + 0.18, false_pass, width=0.36, label="false-pass", color="#e45756")
    axes[2].set_ylabel("rate [%]")
    axes[2].set_title(f"Inspection quality of selected mode; KNN latency estimated separately at {lane} lanes")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend()
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.2f}%"


def write_markdown(payload: dict, path: Path, lane: int) -> None:
    summary = payload["storage_summary"]
    lines = [
        "# Mode-switching PatchCore FPGA analysis",
        "",
        "Purpose: evaluate a product/category mode-switching architecture instead of per-image dynamic routing.",
        "",
        "## Architecture idea",
        "",
        "Before inspection starts, the FPGA enters the mode for the current product category.",
        "During inspection, the configuration is fixed, so per-image routing mistakes and variable control latency are avoided.",
        "",
        "Switchable mode contents:",
        "",
        "- memory bank",
        "- anomaly threshold",
        "- top-k score setting",
        "- patch grid / feature profile setting",
        "- optional feature-layer selection",
        "",
        "## Storage summary",
        "",
        f"- categories: `{summary['categories']}`",
        f"- full baseline banks for all categories: `{summary['baseline_all_category_bank_mib']:.3f} MiB`",
        f"- selected banks for all categories: `{summary['selected_all_category_bank_mib']:.3f} MiB`",
        f"- all-category bank ratio: `{summary['all_category_bank_ratio']:.4f}x`",
        f"- all-category bank reduction: `{pct(summary['all_category_bank_reduction'])}`",
        f"- largest selected active bank: `{summary['max_selected_active_bank_mib']:.3f} MiB`",
        f"- median selected active bank: `{summary['median_selected_active_bank_mib']:.3f} MiB`",
        "",
        "## Architecture options",
        "",
        f"KNN latency below is estimated with `{lane}` parallel distance lanes.",
        "",
        "| option | resident bank | active bank | mean KNN | worst KNN | strength | weakness |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for option in payload["architecture_options"]:
        lines.append(
            f"| {option['name']} | {option['resident_bank_mib']:.3f} MiB | "
            f"{option['active_bank_mib']:.3f} MiB | {option['mean_knn_ms']:.4f} ms | "
            f"{option['worst_knn_ms']:.4f} ms | {option['strength']} | {option['weakness']} |"
        )
    lines += [
        "",
        "## Per-category mode table",
        "",
        "| category | selected config | good-pass | false-pass | bank | NN ops | total proxy | KNN ms | load time @ 100 MiB/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["mode_rows"]:
        lines.append(
            f"| {row['category']} | `{row['selected_config']}` | {pct(row['good_pass'])} | {pct(row['false_pass'])} | "
            f"{row['bank_ratio']:.4f}x | {row['nn_ratio']:.4f}x | {row['total_proxy_ratio']:.4f}x | "
            f"{row['lane_estimates'][str(lane)]['selected_knn_ms']:.4f} ms | "
            f"{row['switch_time_ms']['100_MiB_s']:.4f} ms |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This supports a mode-switching design, not a per-image dynamic routing design.",
        "- The large reduction comes from loading only the category-specific normal bank and feature profile needed for the current product.",
        "- The strongest hardware story is `profiled_all_resident` if all selected banks fit, because mode switching becomes a pointer/config change.",
        "- The safer first implementation is `profiled_active_load`, because it only needs one selected bank on the FPGA at a time.",
        "- Quality is still the limiting issue.  The best first FPGA target should balance false-pass, good-pass, bank size, and KNN latency.",
        "",
        f"Figure: `{payload['figure']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpga-cost-summary", default="results/mvtec_patchcore_fpga_cost_model_001_summary.json")
    parser.add_argument("--output", default="results/mvtec_patchcore_mode_switch_analysis_001_summary.json")
    parser.add_argument("--markdown", default="docs/mvtec_patchcore_mode_switch_analysis_001.md")
    parser.add_argument("--figure", default="results/mvtec_patchcore_mode_switch_analysis_001.png")
    parser.add_argument("--bandwidths-mib-s", nargs="*", type=float, default=[100.0, 500.0, 1000.0, 5000.0])
    parser.add_argument("--lane-choices", nargs="*", type=int, default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--report-lane", type=int, default=512)
    parser.add_argument("--quality-false-pass-limit", type=float, default=0.03)
    args = parser.parse_args()

    ensure_dirs()
    source = json.loads(Path(args.fpga_cost_summary).read_text(encoding="utf-8"))
    source_rows = source["rows"]
    mode_rows = build_mode_rows(source_rows, args.bandwidths_mib_s, args.lane_choices)
    storage_summary = summarize_mode_storage(mode_rows, args.bandwidths_mib_s, args.quality_false_pass_limit)
    payload = {
        "purpose": "Mode-switching architecture analysis for category-profiled PatchCore FPGA design.",
        "config": vars(args),
        "storage_summary": storage_summary,
        "architecture_options": build_architecture_options(storage_summary, mode_rows, args.report_lane),
        "mode_rows": mode_rows,
        "figure": args.figure,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(payload, Path(args.figure), args.report_lane)
    write_markdown(payload, Path(args.markdown), args.report_lane)
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "figure": args.figure}, indent=2))


if __name__ == "__main__":
    main()
