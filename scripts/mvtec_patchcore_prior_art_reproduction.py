"""Reproduce standard PatchCore reduction techniques on MVTec AD.

This job is intentionally not the proposed method.  It checks the technologies
that prior PatchCore/FPGA work can reasonably use as baselines:

* k-center coreset memory-bank pruning
* random pruning as a weak control
* INT8/INT4 feature quantization as a hardware-oriented proxy

The goal is to separate "known effective bank compression" from the thesis
claim about category-wise profile switching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.mvtec_ad_parquet_anomaly_probe import (
    collect_features,
    curve_rows,
    find_materialized_samples,
    image_scores_from_patch_scores,
    make_backbone,
    sample_normal_patch_bank,
    score_auc,
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass, normal_train_and_test
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


ALL_MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def flatten_features(features: np.ndarray) -> np.ndarray:
    return features.reshape(-1, features.shape[-1]).astype(np.float32, copy=False)


@torch.no_grad()
def nearest_patch_scores(features: np.ndarray, bank: np.ndarray, chunk_size: int, device: torch.device) -> np.ndarray:
    shape = features.shape[:2]
    flat = flatten_features(features)
    bank_t = torch.from_numpy(bank.astype(np.float32, copy=False)).to(device)
    out = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        chunk = torch.from_numpy(flat[start : start + chunk_size]).to(device)
        distances = torch.cdist(chunk, bank_t)
        values = distances.min(dim=1).values
        out[start : start + len(chunk)] = values.detach().cpu().numpy()
    return out.reshape(shape)


@torch.no_grad()
def kcenter_prefix_indices(points: np.ndarray, max_size: int, seed: int, device: torch.device, batch_size: int) -> np.ndarray:
    if len(points) <= max_size:
        return np.arange(len(points), dtype=np.int64)
    points_t = torch.from_numpy(points.astype(np.float32, copy=False)).to(device)
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(points)))
    selected: list[int] = []
    min_dist = torch.full((len(points_t),), float("inf"), device=device)
    for _ in range(max_size):
        selected.append(idx)
        chosen = points_t[idx]
        for start in range(0, len(points_t), batch_size):
            chunk = points_t[start : start + batch_size]
            dist = torch.sum((chunk - chosen[None, :]) ** 2, dim=1)
            min_dist[start : start + len(chunk)] = torch.minimum(min_dist[start : start + len(chunk)], dist)
        min_dist[idx] = 0.0
        idx = int(torch.argmax(min_dist).item())
    return np.asarray(selected, dtype=np.int64)


def random_bank(full_bank: np.ndarray, size: int, seed: int) -> np.ndarray:
    if len(full_bank) <= size:
        return full_bank.copy()
    rng = np.random.default_rng(seed)
    return full_bank[rng.choice(len(full_bank), size=size, replace=False)].astype(np.float32, copy=False)


def quantize_dequant_pair(
    test_features: np.ndarray,
    bank: np.ndarray,
    bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    if bits >= 32:
        return test_features.astype(np.float32, copy=False), bank.astype(np.float32, copy=False)
    levels = float((1 << bits) - 1)
    flat_bank = flatten_features(bank.reshape(1, len(bank), bank.shape[-1]))
    lo = flat_bank.min(axis=0)
    hi = flat_bank.max(axis=0)
    scale = (hi - lo) / levels
    scale[scale == 0] = 1.0

    def qdq(x: np.ndarray) -> np.ndarray:
        q = np.rint((x - lo) / scale)
        q = np.clip(q, 0, levels)
        return (q * scale + lo).astype(np.float32)

    return qdq(test_features), qdq(bank)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def correlation(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float | None:
    if len(a) < 2:
        return None
    x = rankdata(a) if rank else a.astype(np.float64)
    y = rankdata(b) if rank else b.astype(np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return round_float(float(np.corrcoef(x, y)[0, 1]))


def evaluate_variant(
    name: str,
    method: str,
    bank: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    full_scores: np.ndarray,
    full_bank_size: int,
    topk_fraction: float,
    bits: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    q_test, q_bank = quantize_dequant_pair(test_features, bank, bits)
    patch_scores = nearest_patch_scores(q_test, q_bank, args.nn_chunk_size, device)
    scores = image_scores_from_patch_scores(patch_scores, topk_fraction)[args.score_name]
    rows = curve_rows(test_labels, scores, args.curve_points)
    best_rows = [best_under_false_pass(rows, target) for target in args.false_pass_targets]
    feature_dim = int(bank.shape[-1])
    memory_bytes = int(len(bank) * feature_dim * (bits if bits < 32 else 32) / 8)
    return {
        "name": name,
        "method": method,
        "quant_bits": int(bits),
        "bank_patches": int(len(bank)),
        "relative_bank_points": round_float(float(len(bank) / full_bank_size)),
        "relative_nn_ops": round_float(float(len(bank) / full_bank_size)),
        "feature_dim": feature_dim,
        "memory_bytes": memory_bytes,
        "relative_memory_bytes_fp32_full": round_float(memory_bytes / (full_bank_size * feature_dim * 4.0)),
        "auc": score_auc(test_labels, scores),
        "best_rows": best_rows,
        "score_pearson_to_fp32_full": correlation(full_scores, scores),
        "score_spearman_to_fp32_full": correlation(full_scores, scores, rank=True),
    }


def run_category(category: str, args: argparse.Namespace, device: torch.device) -> dict:
    samples = find_materialized_samples(args.materialized_root, category)
    train, test = normal_train_and_test(samples)
    if not train or not test or len({sample.label for sample in test}) < 2:
        return {"category": category, "status": "skipped", "reason": "insufficient train/test labels"}

    model = make_backbone(args.backbone, tuple(args.out_indices), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (args.patch_grid, args.patch_grid)
    train_features, _ = collect_features(model, train, image_size, args.batch_size, patch_grid, device, f"{category} train")
    test_features, test_labels = collect_features(model, test, image_size, args.batch_size, patch_grid, device, f"{category} test")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_labels = np.zeros(len(train_features), dtype=np.int64)
    full_size = min(args.full_bank_patches, len(flatten_features(train_features)))
    full_bank = sample_normal_patch_bank(train_features, train_labels, full_size, args.seed)
    full_patch_scores = nearest_patch_scores(test_features, full_bank, args.nn_chunk_size, device)
    full_scores = image_scores_from_patch_scores(full_patch_scores, args.topk_fraction)[args.score_name]

    target_sizes = sorted({min(size, len(full_bank)) for size in args.bank_sizes if size > 0}, reverse=True)
    kcenter_sizes = [size for size in target_sizes if size < len(full_bank)]
    kcenter_order = np.array([], dtype=np.int64)
    if kcenter_sizes:
        kcenter_order = kcenter_prefix_indices(
            full_bank,
            max(kcenter_sizes),
            args.seed + 1009,
            device,
            args.distance_batch_size,
        )

    variants = [
        evaluate_variant(
            "fp32_full_bank",
            "full",
            full_bank,
            test_features,
            test_labels,
            full_scores,
            len(full_bank),
            args.topk_fraction,
            32,
            args,
            device,
        )
    ]

    for bits in args.quant_bits:
        if bits != 32:
            variants.append(
                evaluate_variant(
                    f"int{bits}_full_bank",
                    "full_quantized",
                    full_bank,
                    test_features,
                    test_labels,
                    full_scores,
                    len(full_bank),
                    args.topk_fraction,
                    bits,
                    args,
                    device,
                )
            )

    for size in target_sizes:
        if size >= len(full_bank):
            continue
        for seed_offset in range(args.random_repeats):
            bank = random_bank(full_bank, size, args.seed + 17 + seed_offset)
            variants.append(
                evaluate_variant(
                    f"random_b{size}_r{seed_offset}",
                    "random",
                    bank,
                    test_features,
                    test_labels,
                    full_scores,
                    len(full_bank),
                    args.topk_fraction,
                    32,
                    args,
                    device,
                )
            )
        k_bank = full_bank[kcenter_order[:size]]
        variants.append(
            evaluate_variant(
                f"kcenter_b{size}",
                "kcenter_coreset",
                k_bank,
                test_features,
                test_labels,
                full_scores,
                len(full_bank),
                args.topk_fraction,
                32,
                args,
                device,
            )
        )
        for bits in args.quant_bits:
            if bits != 32:
                variants.append(
                    evaluate_variant(
                        f"kcenter_b{size}_int{bits}",
                        "kcenter_coreset_quantized",
                        k_bank,
                        test_features,
                        test_labels,
                        full_scores,
                        len(full_bank),
                        args.topk_fraction,
                        bits,
                        args,
                        device,
                    )
                )

    return {
        "category": category,
        "status": "done",
        "sample_counts": {
            "train_normal": int(len(train)),
            "test": int(len(test)),
            "test_good": int((test_labels == 0).sum()),
            "test_defect": int((test_labels == 1).sum()),
        },
        "profile": {
            "backbone": args.backbone,
            "out_indices": args.out_indices,
            "patch_grid": args.patch_grid,
            "topk_fraction": args.topk_fraction,
            "score_name": args.score_name,
        },
        "full_bank_patches": int(len(full_bank)),
        "patch_count": int(train_features.shape[1]),
        "feature_dim": int(train_features.shape[-1]),
        "variants": variants,
    }


def aggregate(payload: dict) -> list[dict]:
    rows = []
    by_name: dict[str, list[dict]] = {}
    for category in payload["category_results"]:
        if category.get("status") != "done":
            continue
        for variant in category["variants"]:
            by_name.setdefault(variant["name"], []).append(variant)

    for name, items in sorted(by_name.items()):
        for target in payload["config"]["false_pass_targets"]:
            good_values = []
            auc_values = []
            spearman_values = []
            rel_ops = []
            rel_mem = []
            for item in items:
                best = next(row for row in item["best_rows"] if row["target"] == target)
                if best["good_pass_rate_good"] is not None:
                    good_values.append(best["good_pass_rate_good"])
                auc = item["auc"]["image_auroc"]
                if auc is not None:
                    auc_values.append(auc)
                if item["score_spearman_to_fp32_full"] is not None:
                    spearman_values.append(item["score_spearman_to_fp32_full"])
                rel_ops.append(item["relative_nn_ops"])
                rel_mem.append(item["relative_memory_bytes_fp32_full"])
            rows.append(
                {
                    "variant": name,
                    "method": items[0]["method"],
                    "target_false_pass_rate_defect": target,
                    "categories": len(items),
                    "mean_good_pass_rate_good": round_float(mean(good_values)) if good_values else None,
                    "min_good_pass_rate_good": round_float(min(good_values)) if good_values else None,
                    "mean_image_auroc": round_float(mean(auc_values)) if auc_values else None,
                    "mean_score_spearman_to_fp32_full": round_float(mean(spearman_values)) if spearman_values else None,
                    "mean_relative_nn_ops": round_float(mean(rel_ops)),
                    "mean_relative_memory_bytes_fp32_full": round_float(mean(rel_mem)),
                }
            )
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# PatchCore既存削減技術の再現実験",
        "",
        "## 目的",
        "",
        "PatchCoreをFPGAへ載せる先行研究で前提になりうる既存技術を，こちらのMVTec AD条件で再現する。",
        "ここでは提案手法の優位性ではなく，標準PatchCoreに対して次の削減がどれだけ成立するかを確認する。",
        "",
        "- k-center coreset: 正常特徴分布を覆う代表点を意図的に選んでbankを削る既存手法。",
        "- random: 同じbank数でも，選び方がランダムだとどれだけ不安定かを見る対照群。",
        "- INT8/INT4: FPGA実装を想定した特徴量・bank量子化の近似評価。",
        "",
        "## 評価条件",
        "",
        f"- backbone: `{payload['config']['backbone']}`",
        f"- 中間特徴層: `{payload['config']['out_indices']}`",
        f"- patch grid: `{payload['config']['patch_grid']} x {payload['config']['patch_grid']}`",
        f"- full bank上限: `{payload['config']['full_bank_patches']}`",
        f"- score: `{payload['config']['score_name']}`, top-k fraction `{payload['config']['topk_fraction']}`",
        "",
        "## 集計結果",
        "",
        "| variant | method | 欠陥誤通過上限 | 平均良品通過率 | 最低良品通過率 | 平均AUROC | fullとの順位相関 | NN演算量 | bankメモリ量 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    target = payload["config"]["report_target_false_pass"]
    rows = [row for row in payload["aggregate_rows"] if row["target_false_pass_rate_defect"] == target]
    rows.sort(key=lambda row: (row["method"], row["mean_relative_nn_ops"], row["variant"]))
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['method']} | {pct(row['target_false_pass_rate_defect'])} | "
            f"{pct(row['mean_good_pass_rate_good'])} | {pct(row['min_good_pass_rate_good'])} | "
            f"{row['mean_image_auroc']} | {row['mean_score_spearman_to_fp32_full']} | "
            f"{row['mean_relative_nn_ops']:.4f}x | {row['mean_relative_memory_bytes_fp32_full']:.4f}x |"
        )
    lines += [
        "",
        "## 読み取り方",
        "",
        "- `良品通過率` は，欠陥誤通過率を指定上限以下に抑えたうえで，正常品を正常として通せた割合である。",
        "- `fullとの順位相関` は，full bank fp32 PatchCoreの異常スコア順位をどれだけ保てたかを示す。1に近いほど削減後も同じ判断順序を保っている。",
        "- k-centerがrandomより安定すれば，bank削減は「たまたま当たった」ではなく，正常特徴空間の代表点選択として扱える。",
        "- INT4まで落として順位相関や良品通過率が保てる場合，MAD-Flow型の低bit KNN実装を比較対象として採用しやすい。",
        "",
        f"図: `{payload['figure']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    target = payload["config"]["report_target_false_pass"]
    rows = [row for row in payload["aggregate_rows"] if row["target_false_pass_rate_defect"] == target]
    keep = [
        "fp32_full_bank",
        "int8_full_bank",
        "int4_full_bank",
        "kcenter_b6000",
        "kcenter_b3000",
        "kcenter_b1500",
        "kcenter_b750",
        "kcenter_b3000_int8",
        "kcenter_b3000_int4",
    ]
    row_map = {row["variant"]: row for row in rows}
    rows = [row_map[name] for name in keep if name in row_map]
    labels = [row["variant"].replace("_bank", "").replace("kcenter_", "kc_") for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    panels = [
        ("mean_good_pass_rate_good", "平均良品通過率 [%]", 100.0, (0, 105)),
        ("mean_image_auroc", "平均AUROC [%]", 100.0, (70, 101)),
        ("mean_relative_nn_ops", "NN演算量 [full=1]", 1.0, (0, 1.05)),
        ("mean_relative_memory_bytes_fp32_full", "bankメモリ量 [fp32 full=1]", 1.0, (0, 1.05)),
    ]
    colors = ["#2878b5", "#2878b5", "#d95f02", "#d95f02"]
    for ax, (key, ylabel, scale, ylim), color in zip(axes.ravel(), panels, colors):
        values = [(row[key] or 0.0) * scale for row in rows]
        ax.bar(x, values, color=color)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"PatchCore prior-art reduction reproduction at defect false-pass <= {100*target:.1f}%")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, default=Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2"))
    parser.add_argument("--categories", nargs="*", default=ALL_MVTEC_CATEGORIES)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--out-indices", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--patch-grid", type=int, default=14)
    parser.add_argument("--full-bank-patches", type=int, default=12000)
    parser.add_argument("--bank-sizes", type=parse_int_list, default=parse_int_list("6000,3000,1500,750"))
    parser.add_argument("--quant-bits", type=parse_int_list, default=parse_int_list("32,8,4"))
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--score-name", default="topk_score", choices=["topk_score", "max_score"])
    parser.add_argument("--false-pass-targets", type=float, nargs="*", default=[0.01, 0.03, 0.05])
    parser.add_argument("--report-target-false-pass", type=float, default=0.03)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nn-chunk-size", type=int, default=8192)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("results/mvtec_patchcore_prior_art_reproduction_001_summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/mvtec_patchcore_prior_art_reproduction_001.md"))
    parser.add_argument("--figure", type=Path, default=Path("results/mvtec_patchcore_prior_art_reproduction_001.png"))
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required.")
    results = [run_category(category, args, device) for category in args.categories]
    payload = {
        "purpose": "Reproduce standard PatchCore bank pruning and quantization techniques before evaluating the proposed category-profile switch.",
        "config": {
            **vars(args),
            "materialized_root": str(args.materialized_root),
            "output": str(args.output),
            "markdown": str(args.markdown),
            "figure": str(args.figure),
            "device": str(device),
        },
        "category_results": results,
        "figure": str(args.figure),
    }
    payload["aggregate_rows"] = aggregate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "categories": len(results), "device": str(device)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
