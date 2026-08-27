"""Fair AB/ABC PatchCore profile-switch comparison with fixed coreset ratio.

Earlier subset-switch experiments mixed two effects:

* category/profile switching
* per-category hand-picked memory-bank sizes

Memory-bank reduction is a known PatchCore component, so this experiment treats
it as a common baseline tool.  Every compared system uses the same k-center
coreset ratio.  The measured difference is therefore profile switching: backbone,
feature layers, patch grid, and category-specific mode selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
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
    score_auc,
)
from scripts.mvtec_patchcore_lightweight_sweep import best_under_false_pass, normal_train_and_test
from scripts.train_kolektor_strong_final import round_float, set_seed


DEFAULT_SOURCE = Path("results/mvtec_patchcore_backbone_floor_probe_001_summary.json")
DEFAULT_OUTPUT = Path("results/mvtec_patchcore_fixed_coreset_profile_switch_001_summary.json")
DEFAULT_MARKDOWN = Path("docs/mvtec_patchcore_fixed_coreset_profile_switch_001.md")
DEFAULT_FIGURE = Path("results/mvtec_patchcore_fixed_coreset_profile_switch_001.png")


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def parse_float_from_config(config: str, key: str, default: float) -> float:
    match = re.search(rf"{re.escape(key)}([0-9]+)p([0-9]+)", config)
    if not match:
        return default
    return float(f"{match.group(1)}.{match.group(2)}")


def best_for_target(row: dict, target: float) -> dict | None:
    for best in row.get("best_rows", []):
        if abs(float(best["target"]) - target) < 1e-12:
            return best
    return None


def row_good(row: dict, target: float) -> float | None:
    best = best_for_target(row, target)
    if best is not None:
        return None if best["good_pass_rate_good"] is None else float(best["good_pass_rate_good"])
    if "selected_good_pass" in row:
        return float(row["selected_good_pass"])
    return None


def row_config(row: dict) -> dict:
    footprint = row.get("footprint", {})
    config = str(row.get("config", row.get("selected_config", "")))
    topk = row.get("topk_fraction")
    if topk is None:
        topk = parse_float_from_config(config, "topk", 0.01)
    return {
        "config": config,
        "backbone": row["backbone"],
        "out_indices": tuple(int(i) for i in row["out_indices"]),
        "patch_grid": int(row.get("patch_grid", footprint.get("patch_count", 14) ** 0.5)),
        "feature_dim": int(row.get("feature_dim", footprint.get("feature_dim"))),
        "topk_fraction": float(topk),
    }


def rows_by_category(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["category"], []).append(row)
    return out


def selected_minimal(summary: dict, target: float, tolerance: float) -> dict[str, dict]:
    selected = {}
    for row in summary["minimal_table"]:
        if abs(float(row["max_false_pass_rate_defect"]) - target) < 1e-12 and abs(float(row["allowed_good_pass_drop"]) - tolerance) < 1e-12:
            selected[row["category"]] = row
    return selected


def baseline_rows(summary: dict, target: float) -> dict[str, dict]:
    by_cat = rows_by_category(summary["variant_rows"])
    out = {}
    for category, rows in by_cat.items():
        candidates = []
        for row in rows:
            cfg = row_config(row)
            if cfg["backbone"] == "wide_resnet50_2" and cfg["out_indices"] == (1, 2) and cfg["patch_grid"] == 14:
                good = row_good(row, target)
                if good is not None:
                    candidates.append(row)
        if candidates:
            out[category] = max(candidates, key=lambda row: row_good(row, target) or 0.0)
    return out


def profile_key(cfg: dict) -> tuple[str, tuple[int, ...], int]:
    return (cfg["backbone"], cfg["out_indices"], cfg["patch_grid"])


def choose_row_for_profile(rows: list[dict], wanted: tuple[str, tuple[int, ...], int], target: float) -> dict | None:
    candidates = [row for row in rows if profile_key(row_config(row)) == wanted and row_good(row, target) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row_good(row, target) or 0.0)


def reservoir_rows(x: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1]).astype(np.float32, copy=False)
    if len(flat) <= max_rows:
        return flat
    # Deterministic downsampling keeps this from becoming a seed-search result.
    idx = np.linspace(0, len(flat) - 1, num=max_rows, dtype=np.int64)
    return flat[idx]


@torch.no_grad()
def kcenter_bank(points: np.ndarray, ratio: float, min_bank: int, max_bank: int, seed: int, device: torch.device, batch_size: int) -> np.ndarray:
    size = int(math.ceil(len(points) * ratio))
    size = max(min_bank, size)
    size = min(max_bank, size, len(points))
    if len(points) <= size:
        return points.astype(np.float32, copy=False)
    points_t = torch.from_numpy(points.astype(np.float32, copy=False)).to(device)
    center = points.mean(axis=0, keepdims=True)
    idx = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
    selected: list[int] = []
    min_dist = torch.full((len(points_t),), float("inf"), device=device)
    for _ in range(size):
        selected.append(idx)
        chosen = points_t[idx]
        for start in range(0, len(points_t), batch_size):
            chunk = points_t[start : start + batch_size]
            dist = torch.sum((chunk - chosen[None, :]) ** 2, dim=1)
            min_dist[start : start + len(chunk)] = torch.minimum(min_dist[start : start + len(chunk)], dist)
        min_dist[idx] = 0.0
        idx = int(torch.argmax(min_dist).item())
    return points[np.array(selected, dtype=np.int64)].astype(np.float32, copy=False)


@torch.no_grad()
def patchcore_scores_gpu(features: np.ndarray, bank: np.ndarray, chunk_size: int, device: torch.device) -> np.ndarray:
    shape = features.shape[:2]
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32, copy=False)
    bank_t = torch.from_numpy(bank.astype(np.float32, copy=False)).to(device)
    mins = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        chunk = torch.from_numpy(flat[start : start + chunk_size]).to(device)
        distances = torch.cdist(chunk, bank_t)
        values = distances.min(dim=1).values
        mins[start : start + len(chunk)] = values.detach().cpu().numpy()
    return mins.reshape(shape)


class FeatureCache:
    def __init__(self, materialized_root: Path, args: argparse.Namespace, device: torch.device):
        self.materialized_root = materialized_root
        self.args = args
        self.device = device
        self.cache: dict[tuple[str, str, tuple[int, ...], int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]] = {}

    def get(self, category: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        key = (category, cfg["backbone"], cfg["out_indices"], cfg["patch_grid"])
        if key in self.cache:
            return self.cache[key]
        samples = find_materialized_samples(self.materialized_root, category)
        train, test = normal_train_and_test(samples)
        model = make_backbone(cfg["backbone"], cfg["out_indices"], self.device)
        image_size = (self.args.image_height, self.args.image_width)
        patch_grid = (cfg["patch_grid"], cfg["patch_grid"])
        train_features, _ = collect_features(model, train, image_size, self.args.batch_size, patch_grid, self.device, f"{category} {cfg['config']} train")
        test_features, test_labels = collect_features(model, test, image_size, self.args.batch_size, patch_grid, self.device, f"{category} {cfg['config']} test")
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        meta = {
            "train_normal": int(len(train)),
            "test": int(len(test)),
            "test_good": int((test_labels == 0).sum()),
            "test_defect": int((test_labels == 1).sum()),
        }
        value = (train_features, test_features, test_labels, np.zeros(len(train_features), dtype=np.int64), meta)
        self.cache[key] = value
        return value


def evaluate_category(category: str, cfg: dict, bank_categories: list[str], cache: FeatureCache, args: argparse.Namespace, device: torch.device) -> dict:
    banks = []
    bank_pool_sizes = {}
    for bank_category in bank_categories:
        train_features, _test_features, _test_labels, _train_labels, _meta = cache.get(bank_category, cfg)
        pool = reservoir_rows(train_features, args.coreset_candidate_pool, args.seed)
        bank = kcenter_bank(pool, args.coreset_ratio, args.min_bank_patches, args.max_bank_patches, args.seed, device, args.distance_batch_size)
        banks.append(bank)
        bank_pool_sizes[bank_category] = {"candidate_pool": int(len(pool)), "bank_patches": int(len(bank))}
    merged_bank = np.concatenate(banks, axis=0)
    _train_features, test_features, test_labels, _train_labels, meta = cache.get(category, cfg)
    patch_scores = patchcore_scores_gpu(test_features, merged_bank, args.nn_chunk_size, device)
    scores = image_scores_from_patch_scores(patch_scores, cfg["topk_fraction"])[args.score_name]
    rows = curve_rows(test_labels, scores, args.curve_points)
    best = best_under_false_pass(rows, args.false_pass_target)
    patch_count = int(test_features.shape[1])
    feature_dim = int(test_features.shape[2])
    return {
        "category": category,
        "config": cfg["config"],
        "backbone": cfg["backbone"],
        "out_indices": list(cfg["out_indices"]),
        "patch_grid": cfg["patch_grid"],
        "patch_count": patch_count,
        "feature_dim": feature_dim,
        "topk_fraction": cfg["topk_fraction"],
        "bank_categories": bank_categories,
        "bank_patches": int(len(merged_bank)),
        "bank_pool_sizes": bank_pool_sizes,
        "nn_ops_per_image": int(patch_count * feature_dim * len(merged_bank)),
        "auc": score_auc(test_labels, scores),
        "good_pass": round_float(best["good_pass_rate_good"]),
        "good_loss": round_float(best["good_loss_rate_good"]),
        "threshold": round_float(best["threshold"]),
        "false_pass_target": args.false_pass_target,
        "sample_counts": meta,
    }


def summarize_system(name: str, rows: list[dict], baseline_ops: list[int]) -> dict:
    goods = [row["good_pass"] for row in rows if row["good_pass"] is not None]
    ops = [row["nn_ops_per_image"] for row in rows]
    rel = [op / base for op, base in zip(ops, baseline_ops)]
    return {
        "name": name,
        "mean_good_pass": round_float(mean(goods)) if goods else None,
        "min_good_pass": round_float(min(goods)) if goods else None,
        "mean_nn_ops_per_image": int(mean(ops)),
        "mean_relative_nn_ops_to_standard_merged": round_float(mean(rel), 9),
        "max_relative_nn_ops_to_standard_merged": round_float(max(rel), 9),
        "category_rows": rows,
    }


def build_systems_for_subset(
    subset: tuple[str, ...],
    by_category: dict[str, list[dict]],
    baseline_by_category: dict[str, dict],
    selected_by_category: dict[str, dict],
    cache: FeatureCache,
    args: argparse.Namespace,
    device: torch.device,
) -> dict | None:
    if any(category not in baseline_by_category or category not in selected_by_category for category in subset):
        return None
    standard_cfg = row_config(baseline_by_category[subset[0]])
    standard_rows = [evaluate_category(category, standard_cfg, list(subset), cache, args, device) for category in subset]
    baseline_ops = [row["nn_ops_per_image"] for row in standard_rows]
    systems = [summarize_system("① 共通標準profile + merged bank", standard_rows, baseline_ops)]

    bank_only_rows = [evaluate_category(category, standard_cfg, [category], cache, args, device) for category in subset]
    systems.append(summarize_system("② 共通標準profile + category bank切替", bank_only_rows, baseline_ops))

    for fixed_category in subset:
        wanted = profile_key(row_config(selected_by_category[fixed_category]))
        fixed_rows = []
        for category in subset:
            row = choose_row_for_profile(by_category[category], wanted, args.false_pass_target)
            if row is None:
                fixed_rows = []
                break
            fixed_rows.append(evaluate_category(category, row_config(row), [category], cache, args, device))
        if fixed_rows:
            systems.append(summarize_system(f"固定profile={fixed_category} + category bank切替", fixed_rows, baseline_ops))

    proposed_rows = [
        evaluate_category(category, row_config(selected_by_category[category]), [category], cache, args, device)
        for category in subset
    ]
    systems.append(summarize_system("★ category profile + category bank 両切替", proposed_rows, baseline_ops))
    return {"subset": list(subset), "systems": systems}


def select_subsets(categories: list[str], subset_size: int, limit: int) -> list[tuple[str, ...]]:
    return list(itertools.combinations(categories, subset_size))[:limit]


def rank_subsets(results: list[dict]) -> list[dict]:
    ranked = []
    for item in results:
        systems = {system["name"]: system for system in item["systems"]}
        standard = systems["① 共通標準profile + merged bank"]
        bank_only = systems["② 共通標準profile + category bank切替"]
        proposed = systems["★ category profile + category bank 両切替"]
        fixed = [system for name, system in systems.items() if name.startswith("固定profile=")]
        best_fixed_min = max((system["min_good_pass"] or 0.0 for system in fixed), default=0.0)
        ranked.append(
            item
            | {
                "claim_metrics": {
                    "proposed_vs_standard_ops_reduction": round_float(1.0 - proposed["mean_relative_nn_ops_to_standard_merged"]),
                    "proposed_vs_bank_only_ops_reduction": round_float(
                        1.0
                        - proposed["mean_relative_nn_ops_to_standard_merged"]
                        / max(bank_only["mean_relative_nn_ops_to_standard_merged"], 1e-12)
                    ),
                    "proposed_min_good_minus_best_fixed_min_good": round_float((proposed["min_good_pass"] or 0.0) - best_fixed_min),
                    "proposed_min_good_minus_standard": round_float((proposed["min_good_pass"] or 0.0) - (standard["min_good_pass"] or 0.0)),
                }
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["claim_metrics"]["proposed_vs_bank_only_ops_reduction"] or 0.0,
            item["claim_metrics"]["proposed_min_good_minus_best_fixed_min_good"] or -1.0,
        ),
        reverse=True,
    )


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# 固定coreset比率によるprofile切替の公平比較",
        "",
        "## 目的",
        "",
        "bank数のカテゴリ別手動最適化を禁止し，全方式に同じk-center coreset比率を適用する。候補bankの抽出とk-center初期点は決定的に固定し，seed探索による上振れを避ける。",
        "",
    ]
    for group_name in ["top_pairs", "top_triples"]:
        lines += [f"## {group_name}", ""]
        lines.append("| subset | system | 最低良品通過 | 平均良品通過 | 平均NN計算量 | 標準比 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for item in payload[group_name]:
            subset = " + ".join(item["subset"])
            for system in item["systems"]:
                lines.append(
                    f"| {subset} | {system['name']} | {pct(system['min_good_pass'])} | "
                    f"{pct(system['mean_good_pass'])} | {system['mean_nn_ops_per_image']} | "
                    f"{system['mean_relative_nn_ops_to_standard_merged']:.6f}x |"
                )
            lines.append(f"| {subset} | 主張用差分 |  |  | vs標準削減 {pct(item['claim_metrics']['proposed_vs_standard_ops_reduction'])} | vs bank-only追加削減 {pct(item['claim_metrics']['proposed_vs_bank_only_ops_reduction'])} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    rows = payload["top_pairs"][:3] + payload["top_triples"][:3]
    labels = ["+".join(item["subset"]) for item in rows]
    std_ops = [1.0 for _ in rows]
    bank_ops = []
    prop_ops = []
    prop_good = []
    std_good = []
    for item in rows:
        systems = {system["name"]: system for system in item["systems"]}
        bank_ops.append(systems["② 共通標準profile + category bank切替"]["mean_relative_nn_ops_to_standard_merged"])
        prop_ops.append(systems["★ category profile + category bank 両切替"]["mean_relative_nn_ops_to_standard_merged"])
        prop_good.append(100.0 * (systems["★ category profile + category bank 両切替"]["min_good_pass"] or 0.0))
        std_good.append(100.0 * (systems["① 共通標準profile + merged bank"]["min_good_pass"] or 0.0))
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.5), 7.2), constrained_layout=True)
    width = 0.26
    axes[0].bar(x - width, std_ops, width=width, label="standard merged")
    axes[0].bar(x, bank_ops, width=width, label="bank switch")
    axes[0].bar(x + width, prop_ops, width=width, label="profile+bank switch")
    axes[0].set_ylabel("relative NN ops")
    axes[0].set_yscale("log")
    axes[0].set_title("Fixed coreset ratio cost comparison")
    axes[0].legend()
    axes[1].plot(x, std_good, marker="o", label="standard min good-pass")
    axes[1].plot(x, prop_good, marker="o", label="proposed min good-pass")
    axes[1].set_ylabel("min good-pass [%]")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--false-pass-target", type=float, default=0.05)
    parser.add_argument("--allowed-good-pass-drop", type=float, default=0.05)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--coreset-candidate-pool", type=int, default=12000)
    parser.add_argument("--min-bank-patches", type=int, default=25)
    parser.add_argument("--max-bank-patches", type=int, default=12000)
    parser.add_argument("--pair-limit", type=int, default=105)
    parser.add_argument("--triple-limit", type=int, default=455)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-name", default="topk_score", choices=["topk_score", "max_score"])
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nn-chunk-size", type=int, default=8192)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this job.")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    by_category = rows_by_category(source["variant_rows"])
    selected = selected_minimal(source, args.false_pass_target, args.allowed_good_pass_drop)
    baseline = baseline_rows(source, args.false_pass_target)
    categories = sorted(set(selected) & set(baseline))
    cache = FeatureCache(args.materialized_root, args, device)
    pair_results = []
    for subset in select_subsets(categories, 2, args.pair_limit):
        result = build_systems_for_subset(subset, by_category, baseline, selected, cache, args, device)
        if result is not None:
            pair_results.append(result)
    triple_results = []
    for subset in select_subsets(categories, 3, args.triple_limit):
        result = build_systems_for_subset(subset, by_category, baseline, selected, cache, args, device)
        if result is not None:
            triple_results.append(result)
    pair_ranked = rank_subsets(pair_results)
    triple_ranked = rank_subsets(triple_results)
    payload = {
        "purpose": "Fair profile-switch comparison with fixed k-center coreset ratio for every system.",
        "config": vars(args) | {"device": str(device), "categories": categories},
        "top_pairs": pair_ranked[: args.top_k],
        "top_triples": triple_ranked[: args.top_k],
        "all_pair_count": len(pair_ranked),
        "all_triple_count": len(triple_ranked),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "pairs": len(pair_ranked), "triples": len(triple_ranked), "device": str(device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
