"""Sweep PatchCore-lite settings on representative MVTec AD categories.

This is the next step after confirming that PatchCore-lite is a strong
inspection baseline on MVTec AD.  The experiment asks a thesis-facing question:
how much of PatchCore's inspection strength remains when we reduce the pieces
that are expensive for an FPGA implementation, especially patch count, feature
dimension, and memory-bank size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.mvtec_ad_parquet_anomaly_probe import (
    MVTecSample,
    collect_features,
    curve_rows,
    find_materialized_samples,
    image_scores_from_patch_scores,
    make_backbone,
    patchcore_scores,
    sample_normal_patch_bank,
    score_auc,
)
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


DEFAULT_CATEGORIES = ["bottle", "hazelnut", "tile", "cable", "pill", "screw"]


def parse_config(text: str) -> dict:
    values: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    name = values["name"]
    return {
        "name": name,
        "backbone": values.get("backbone", "wide_resnet50_2"),
        "out_indices": [int(v) for v in values.get("out", "1:2").split(":") if v != ""],
        "patch_grid": int(values.get("grid", "14")),
        "bank_patches": int(values.get("bank", "12000")),
        "topk_fraction": float(values.get("topk", "0.01")),
    }


def normal_train_and_test(samples: list[MVTecSample]) -> tuple[list[MVTecSample], list[MVTecSample]]:
    train = [sample for sample in samples if sample.split == "train" and sample.label == 0]
    test = [sample for sample in samples if sample.split == "test"]
    return train, test


def best_under_false_pass(rows: list[dict], target: float) -> dict:
    feasible = [row for row in rows if row["false_pass_rate_defect"] is not None and row["false_pass_rate_defect"] <= target]
    if not feasible:
        return {"target": target, "good_pass_rate_good": None, "good_loss_rate_good": None, "threshold": None}
    best = min(feasible, key=lambda row: row["good_loss_rate_good"])
    return {
        "target": target,
        "good_pass_rate_good": best["good_pass_rate_good"],
        "good_loss_rate_good": best["good_loss_rate_good"],
        "threshold": best["threshold"],
    }


def approx_nn_ops(test_samples: int, patch_count: int, bank_patches: int, feature_dim: int) -> int:
    return int(test_samples * patch_count * bank_patches * feature_dim)


def run_config(args: argparse.Namespace, config: dict, categories: list[str], materialized_root: Path, device: torch.device) -> dict:
    model = make_backbone(config["backbone"], tuple(config["out_indices"]), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (config["patch_grid"], config["patch_grid"])
    category_results = []
    for category in categories:
        samples = find_materialized_samples(materialized_root, category)
        train, test = normal_train_and_test(samples)
        if not train or not test or len({sample.label for sample in test}) < 2:
            category_results.append({"category": category, "status": "skipped", "reason": "insufficient train/test labels"})
            continue
        train_features, _ = collect_features(
            model, train, image_size, args.batch_size, patch_grid, device, f"{config['name']} {category} train"
        )
        test_features, test_labels = collect_features(
            model, test, image_size, args.batch_size, patch_grid, device, f"{config['name']} {category} test"
        )
        train_labels = np.zeros(len(train_features), dtype=np.int64)
        bank = sample_normal_patch_bank(train_features, train_labels, config["bank_patches"], args.seed)
        patch_scores = patchcore_scores(test_features, bank, args.nn_chunk_size)
        image_scores = image_scores_from_patch_scores(patch_scores, config["topk_fraction"])
        score_name = args.score_name
        rows = curve_rows(test_labels, image_scores[score_name], args.curve_points)
        feature_dim = int(train_features.shape[-1])
        patch_count = int(train_features.shape[1])
        category_results.append(
            {
                "category": category,
                "status": "done",
                "sample_counts": {
                    "train_normal": len(train),
                    "test": len(test),
                    "test_good": int((test_labels == 0).sum()),
                    "test_defect": int((test_labels == 1).sum()),
                },
                "auc": {name: score_auc(test_labels, scores) for name, scores in image_scores.items()},
                "selected_score": score_name,
                "best_rows": [best_under_false_pass(rows, target) for target in args.false_pass_targets],
                "footprint": {
                    "patch_count": patch_count,
                    "feature_dim": feature_dim,
                    "bank_patches": int(len(bank)),
                    "approx_nn_ops": approx_nn_ops(len(test), patch_count, len(bank), feature_dim),
                },
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"config": config, "category_results": category_results}


def aggregate_rows(results: list[dict], target: float) -> list[dict]:
    rows = []
    for result in results:
        config = result["config"]
        good_pass_values = []
        ops_values = []
        auc_values = []
        done = 0
        for category in result["category_results"]:
            if category.get("status") != "done":
                continue
            done += 1
            best = next(row for row in category["best_rows"] if row["target"] == target)
            if best["good_pass_rate_good"] is not None:
                good_pass_values.append(best["good_pass_rate_good"])
            ops_values.append(category["footprint"]["approx_nn_ops"])
            auc = category["auc"][category["selected_score"]]["image_auroc"]
            if auc is not None:
                auc_values.append(auc)
        rows.append(
            {
                "config": config["name"],
                "target_false_pass_rate_defect": target,
                "categories_done": done,
                "mean_good_pass_rate_good": round_float(float(np.mean(good_pass_values))) if good_pass_values else None,
                "min_good_pass_rate_good": round_float(float(np.min(good_pass_values))) if good_pass_values else None,
                "mean_auc": round_float(float(np.mean(auc_values))) if auc_values else None,
                "mean_approx_nn_ops": int(np.mean(ops_values)) if ops_values else None,
            }
        )
    if rows and rows[0]["mean_approx_nn_ops"]:
        base_ops = rows[0]["mean_approx_nn_ops"]
        for row in rows:
            row["relative_nn_ops"] = round_float(row["mean_approx_nn_ops"] / base_ops) if row["mean_approx_nn_ops"] else None
    return rows


def plot_summary(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["aggregate_rows"]
    target = payload["config"]["false_pass_targets"][1] if len(payload["config"]["false_pass_targets"]) > 1 else payload["config"]["false_pass_targets"][0]
    rows = [row for row in rows if row["target_false_pass_rate_defect"] == target]
    labels = [row["config"] for row in rows]
    good_pass = [100.0 * row["mean_good_pass_rate_good"] if row["mean_good_pass_rate_good"] is not None else 0.0 for row in rows]
    rel_ops = [row["relative_nn_ops"] if row["relative_nn_ops"] is not None else 0.0 for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(labels))
    ax1.bar(x - 0.18, good_pass, width=0.36, label="mean good pass @ constraint", color="#2878b5")
    ax1.set_ylabel("mean good pass [%]")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, rel_ops, width=0.36, label="relative NN ops", color="#d95f02")
    ax2.set_ylabel("relative NN search ops")
    ax2.set_ylim(0, max(1.05, max(rel_ops) * 1.15 if rel_ops else 1.0))
    ax1.set_title(f"PatchCore-lite lightweight sweep at defect false-pass <= {100*target:.1f}%")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MVTec PatchCore-lite lightweight sweep",
        "",
        "Purpose: find a lighter inspection baseline that still keeps the defect false-pass constraint useful.",
        "",
        "## Configurations",
        "",
        "| config | backbone | out indices | grid | bank patches | top-k fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in payload["sweep_configs"]:
        lines.append(
            f"| {config['name']} | {config['backbone']} | {config['out_indices']} | {config['patch_grid']} | "
            f"{config['bank_patches']} | {config['topk_fraction']} |"
        )
    lines += [
        "",
        "## Aggregate result",
        "",
        "| config | max defect false-pass | mean good pass | min good pass | mean AUROC | relative NN ops |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_rows"]:
        mean_good = "" if row["mean_good_pass_rate_good"] is None else f"{100*row['mean_good_pass_rate_good']:.2f}%"
        min_good = "" if row["min_good_pass_rate_good"] is None else f"{100*row['min_good_pass_rate_good']:.2f}%"
        rel_ops = "" if row["relative_nn_ops"] is None else f"{row['relative_nn_ops']:.3f}x"
        lines.append(
            f"| {row['config']} | {100*row['target_false_pass_rate_defect']:.1f}% | {mean_good} | "
            f"{min_good} | {row['mean_auc']} | {rel_ops} |"
        )
    lines += [
        "",
        "## Per-category result at each false-pass target",
        "",
        "| config | category | max defect false-pass | good pass | AUROC | relative NN ops vs first config |",
        "|---|---|---:|---:|---:|---:|",
    ]
    base_ops = None
    for result in payload["results"]:
        for category in result["category_results"]:
            if category.get("status") == "done":
                base_ops = category["footprint"]["approx_nn_ops"]
                break
        if base_ops:
            break
    for result in payload["results"]:
        for category in result["category_results"]:
            if category.get("status") != "done":
                continue
            auc = category["auc"][category["selected_score"]]["image_auroc"]
            rel = category["footprint"]["approx_nn_ops"] / base_ops if base_ops else None
            for best in category["best_rows"]:
                good = "" if best["good_pass_rate_good"] is None else f"{100*best['good_pass_rate_good']:.2f}%"
                lines.append(
                    f"| {result['config']['name']} | {category['category']} | {100*best['target']:.1f}% | "
                    f"{good} | {auc} | {rel:.3f}x |"
                )
    lines += [
        "",
        "## Interpretation guide",
        "",
        "- High good pass with low relative NN ops is the best FPGA-oriented region.",
        "- If a small bank keeps most of the good pass, memory-bank reduction is promising.",
        "- If a small grid keeps most of the good pass, patch-count reduction is promising.",
        "- If a smaller backbone collapses, the feature extractor is still the core bottleneck.",
        "",
        f"Summary figure: `{payload['figure']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", default="/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2")
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    parser.add_argument("--sweep-config", action="append", default=[])
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--false-pass-targets", nargs="*", type=float, default=[0.0, 0.01, 0.05])
    parser.add_argument("--score-name", default="topk_score", choices=["max_score", "topk_score"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="results/mvtec_patchcore_lightweight_sweep_001_summary.json")
    parser.add_argument("--markdown", default="docs/mvtec_patchcore_lightweight_sweep_001.md")
    parser.add_argument("--figure", default="results/mvtec_patchcore_lightweight_sweep_001.png")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    configs = [parse_config(text) for text in args.sweep_config] or [
        parse_config("name=base_wrn14_12k,backbone=wide_resnet50_2,out=1:2,grid=14,bank=12000,topk=0.01"),
        parse_config("name=bank3k_wrn14,backbone=wide_resnet50_2,out=1:2,grid=14,bank=3000,topk=0.01"),
        parse_config("name=grid7_bank3k_wrn,backbone=wide_resnet50_2,out=1:2,grid=7,bank=3000,topk=0.01"),
        parse_config("name=resnet18_grid14_3k,backbone=resnet18,out=1:2,grid=14,bank=3000,topk=0.01"),
    ]

    materialized_root = Path(args.materialized_root)
    results = [run_config(args, config, args.categories, materialized_root, device) for config in configs]
    aggregate = []
    for target in args.false_pass_targets:
        aggregate.extend(aggregate_rows(results, target))
    payload = {
        "purpose": "Sweep PatchCore-lite lightweight settings on representative MVTec AD categories.",
        "config": vars(args),
        "sweep_configs": configs,
        "results": results,
        "aggregate_rows": aggregate,
        "figure": args.figure,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    plot_summary(payload, Path(args.figure))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "figure": args.figure}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
