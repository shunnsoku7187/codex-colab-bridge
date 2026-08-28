"""Diagnose and reproduce a more faithful PatchCore baseline on MVTec AD.

Earlier experiments used a PatchCore-lite profile that may be too far from the
usual PatchCore setting: WideResNet50, ImageNet preprocessing, and layer2/layer3
features.  This script compares those choices directly so that later FPGA
experiments have a defensible baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from scripts.mvtec_ad_parquet_anomaly_probe import (
    MVTecSample,
    curve_rows,
    find_materialized_samples,
    image_scores_from_patch_scores,
    make_backbone,
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


@dataclass(frozen=True)
class PatchCoreProfile:
    name: str
    backbone: str
    out_indices: tuple[int, ...]
    resize: int
    crop: int
    grid: int
    topk_fraction: float
    bank_policy: str
    bank_value: float


class MVTecPatchCoreDataset(Dataset):
    def __init__(self, samples: list[MVTecSample], resize: int, crop: int):
        self.samples = samples
        self.resize = resize
        self.crop = crop
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, [self.resize, self.resize], interpolation=TF.InterpolationMode.BILINEAR)
            if self.crop > 0 and self.crop < self.resize:
                image = TF.center_crop(image, [self.crop, self.crop])
            x = TF.to_tensor(image)
            x = (x - self.mean) / self.std
        return x, torch.tensor(sample.label, dtype=torch.long)


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def parse_out_indices(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(":") if part.strip())


def parse_profile(text: str) -> PatchCoreProfile:
    values: dict[str, str] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    bank = values.get("bank", "ratio:0.1")
    if ":" in bank:
        policy, raw = bank.split(":", 1)
        bank_value = float(raw)
    else:
        policy, bank_value = "fixed", float(bank)
    return PatchCoreProfile(
        name=values["name"],
        backbone=values.get("backbone", "wide_resnet50_2"),
        out_indices=parse_out_indices(values.get("out", "2:3")),
        resize=int(values.get("resize", "256")),
        crop=int(values.get("crop", "224")),
        grid=int(values.get("grid", "28")),
        topk_fraction=float(values.get("topk", "0.01")),
        bank_policy=policy,
        bank_value=bank_value,
    )


def flatten_features(features: np.ndarray) -> np.ndarray:
    return features.reshape(-1, features.shape[-1]).astype(np.float32, copy=False)


def make_loader(samples: list[MVTecSample], profile: PatchCoreProfile, batch_size: int) -> DataLoader:
    dataset = MVTecPatchCoreDataset(samples, resize=profile.resize, crop=profile.crop)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


@torch.no_grad()
def batch_patch_features(model, x: torch.Tensor, grid: int) -> torch.Tensor:
    feats = model(x)
    resized = [F.interpolate(feat, size=(grid, grid), mode="bilinear", align_corners=False) for feat in feats]
    emb = torch.cat(resized, dim=1)
    emb = F.normalize(emb, dim=1)
    return emb.permute(0, 2, 3, 1).reshape(x.shape[0], grid * grid, -1)


@torch.no_grad()
def collect_features(model, samples: list[MVTecSample], profile: PatchCoreProfile, batch_size: int, device: torch.device, desc: str) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for x, label in tqdm(make_loader(samples, profile, batch_size), desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        emb = batch_patch_features(model, x, profile.grid).detach().cpu().numpy().astype(np.float32)
        features.append(emb)
        labels.append(label.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels).astype(np.int64)


def deterministic_pool(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(np.float32, copy=False)
    idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[idx].astype(np.float32, copy=False)


@torch.no_grad()
def kcenter_bank(points: np.ndarray, bank_size: int, device: torch.device, batch_size: int) -> np.ndarray:
    if len(points) <= bank_size:
        return points.astype(np.float32, copy=False)
    points_t = torch.from_numpy(points.astype(np.float32, copy=False)).to(device)
    center = points.mean(axis=0, keepdims=True)
    idx = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
    selected: list[int] = []
    min_dist = torch.full((len(points_t),), float("inf"), device=device)
    for _ in range(bank_size):
        selected.append(idx)
        chosen = points_t[idx]
        for start in range(0, len(points_t), batch_size):
            chunk = points_t[start : start + batch_size]
            dist = torch.sum((chunk - chosen[None, :]) ** 2, dim=1)
            min_dist[start : start + len(chunk)] = torch.minimum(min_dist[start : start + len(chunk)], dist)
        min_dist[idx] = 0.0
        idx = int(torch.argmax(min_dist).item())
    return points[np.asarray(selected, dtype=np.int64)].astype(np.float32, copy=False)


@torch.no_grad()
def patchcore_scores_gpu(features: np.ndarray, bank: np.ndarray, chunk_size: int, device: torch.device) -> np.ndarray:
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


def choose_bank_size(profile: PatchCoreProfile, normal_patches: int, args: argparse.Namespace) -> int:
    if profile.bank_policy == "ratio":
        size = int(math.ceil(normal_patches * profile.bank_value))
    elif profile.bank_policy == "fixed":
        size = int(profile.bank_value)
    else:
        raise ValueError(f"unknown bank policy: {profile.bank_policy}")
    return max(1, min(size, args.max_bank_patches, normal_patches))


def evaluate_category(category: str, profile: PatchCoreProfile, args: argparse.Namespace, device: torch.device) -> dict:
    samples = find_materialized_samples(args.materialized_root, category)
    train, test = normal_train_and_test(samples)
    if not train or not test or len({sample.label for sample in test}) < 2:
        return {"category": category, "profile": profile.name, "status": "skipped", "reason": "insufficient samples"}

    model = make_backbone(profile.backbone, profile.out_indices, device)
    train_features, _ = collect_features(model, train, profile, args.batch_size, device, f"{profile.name} {category} train")
    test_features, test_labels = collect_features(model, test, profile, args.batch_size, device, f"{profile.name} {category} test")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    normal_patches = flatten_features(train_features)
    bank_size = choose_bank_size(profile, len(normal_patches), args)
    pool = deterministic_pool(normal_patches, args.coreset_candidate_pool)
    bank = kcenter_bank(pool, min(bank_size, len(pool)), device, args.distance_batch_size)
    patch_scores = patchcore_scores_gpu(test_features, bank, args.nn_chunk_size, device)
    image_scores = image_scores_from_patch_scores(patch_scores, profile.topk_fraction)
    selected_scores = image_scores[args.score_name]
    rows = curve_rows(test_labels, selected_scores, args.curve_points)
    best_rows = [best_under_false_pass(rows, target) for target in args.false_pass_targets]
    return {
        "category": category,
        "profile": profile.name,
        "status": "done",
        "sample_counts": {
            "train_normal": len(train),
            "test": len(test),
            "test_good": int((test_labels == 0).sum()),
            "test_defect": int((test_labels == 1).sum()),
        },
        "footprint": {
            "normal_patch_candidates": int(len(normal_patches)),
            "coreset_candidate_pool": int(len(pool)),
            "bank_patches": int(len(bank)),
            "bank_fraction_of_normal_patches": round_float(float(len(bank) / len(normal_patches))),
            "patch_count": int(test_features.shape[1]),
            "feature_dim": int(test_features.shape[2]),
            "nn_ops_per_image": int(test_features.shape[1] * test_features.shape[2] * len(bank)),
        },
        "auc": {name: score_auc(test_labels, scores) for name, scores in image_scores.items()},
        "selected_score": args.score_name,
        "best_rows": best_rows,
    }


def aggregate_results(category_results: list[dict], profiles: list[PatchCoreProfile], target: float, score_name: str) -> list[dict]:
    out = []
    for profile in profiles:
        rows = [row for row in category_results if row.get("status") == "done" and row["profile"] == profile.name]
        goods = []
        mins = []
        aucs = []
        ops = []
        banks = []
        for row in rows:
            best = next(item for item in row["best_rows"] if abs(item["target"] - target) < 1e-12)
            if best["good_pass_rate_good"] is not None:
                goods.append(best["good_pass_rate_good"])
                mins.append(best["good_pass_rate_good"])
            auc = row["auc"][score_name]["image_auroc"]
            if auc is not None:
                aucs.append(auc)
            ops.append(row["footprint"]["nn_ops_per_image"])
            banks.append(row["footprint"]["bank_patches"])
        out.append(
            {
                "profile": profile.name,
                "categories": len(rows),
                "mean_good_pass_rate_good": round_float(mean(goods)) if goods else None,
                "min_good_pass_rate_good": round_float(min(mins)) if mins else None,
                "mean_image_auroc": round_float(mean(aucs)) if aucs else None,
                "mean_nn_ops_per_image": round_float(mean(ops)) if ops else None,
                "mean_bank_patches": round_float(mean(banks)) if banks else None,
            }
        )
    if out and out[0]["mean_nn_ops_per_image"]:
        base = out[0]["mean_nn_ops_per_image"]
        for row in out:
            row["relative_nn_ops_to_first_profile"] = round_float(row["mean_nn_ops_per_image"] / base) if row["mean_nn_ops_per_image"] else None
    return out


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# PatchCore標準設定の再現診断",
        "",
        "## 目的",
        "",
        "これまでのPatchCore-lite結果が先行研究水準より低く見える原因を切り分ける。",
        "特に，特徴層が浅すぎた可能性，入力前処理の違い，bank数の扱いを確認する。",
        "",
        "## 比較profile",
        "",
        "| profile | backbone | out indices | resize/crop | grid | bank policy | top-k |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for profile in payload["profiles"]:
        lines.append(
            f"| {profile['name']} | {profile['backbone']} | {profile['out_indices']} | "
            f"{profile['resize']}/{profile['crop']} | {profile['grid']} | "
            f"{profile['bank_policy']}:{profile['bank_value']} | {profile['topk_fraction']} |"
        )
    lines += [
        "",
        f"## 集計結果: 欠陥誤通過率 <= {pct(payload['config']['report_false_pass_target'])}",
        "",
        "| profile | 平均良品通過率 | 最低良品通過率 | 平均AUROC | 平均bank数 | 相対NN演算量 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_rows"]:
        lines.append(
            f"| {row['profile']} | {pct(row['mean_good_pass_rate_good'])} | {pct(row['min_good_pass_rate_good'])} | "
            f"{row['mean_image_auroc']} | {row['mean_bank_patches']} | {row['relative_nn_ops_to_first_profile']:.4f}x |"
        )
    lines += [
        "",
        "## カテゴリ別AUROC",
        "",
        "| category | profile | AUROC | 良品通過率 | bank数 | patch数 | 特徴次元 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    target = payload["config"]["report_false_pass_target"]
    for row in payload["category_results"]:
        if row.get("status") != "done":
            continue
        best = next(item for item in row["best_rows"] if abs(item["target"] - target) < 1e-12)
        foot = row["footprint"]
        lines.append(
            f"| {row['category']} | {row['profile']} | {row['auc'][payload['config']['score_name']]['image_auroc']} | "
            f"{pct(best['good_pass_rate_good'])} | {foot['bank_patches']} | {foot['patch_count']} | {foot['feature_dim']} |"
        )
    lines += [
        "",
        "## 判断",
        "",
        "- 原論文寄りprofileでAUROCが大きく上がるなら，これまでの低スコアは提案手法の問題ではなくbaseline設定の問題である。",
        "- 原論文寄りprofileでもAUROCが低いなら，MVTec mirrorの読み込み・前処理・score計算・PatchCore reweighting欠落をさらに疑う。",
        "- AUROCは高いが良品通過率が低い場合，先行研究指標と検品動作点指標の違いが原因である。",
        "",
        f"図: `{payload['figure']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(payload: dict, path: Path) -> None:
    rows = payload["aggregate_rows"]
    labels = [row["profile"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(x, [100 * (row["mean_image_auroc"] or 0) for row in rows], color="#2878b5")
    axes[0].set_ylabel("Mean image AUROC [%]")
    axes[0].set_ylim(80, 101)
    axes[1].bar(x, [100 * (row["mean_good_pass_rate_good"] or 0) for row in rows], color="#59a14f")
    axes[1].set_ylabel("Mean good-pass at constraint [%]")
    axes[1].set_ylim(0, 105)
    axes[2].bar(x, [row["relative_nn_ops_to_first_profile"] or 0 for row in rows], color="#f28e2b")
    axes[2].set_ylabel("Relative NN ops")
    axes[2].set_ylim(0, max(1.05, max(row["relative_nn_ops_to_first_profile"] or 0 for row in rows) * 1.1))
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("PatchCore baseline diagnosis")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-root", type=Path, default=Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2"))
    parser.add_argument("--categories", nargs="*", default=ALL_MVTEC_CATEGORIES)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--max-bank-patches", type=int, default=24000)
    parser.add_argument("--coreset-candidate-pool", type=int, default=60000)
    parser.add_argument("--false-pass-targets", type=float, nargs="*", default=[0.01, 0.03, 0.05])
    parser.add_argument("--report-false-pass-target", type=float, default=0.03)
    parser.add_argument("--score-name", default="topk_score", choices=["topk_score", "max_score"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nn-chunk-size", type=int, default=8192)
    parser.add_argument("--distance-batch-size", type=int, default=8192)
    parser.add_argument("--curve-points", type=int, default=180)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("results/mvtec_patchcore_faithful_reproduction_001_summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/mvtec_patchcore_faithful_reproduction_001.md"))
    parser.add_argument("--figure", type=Path, default=Path("results/mvtec_patchcore_faithful_reproduction_001.png"))
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
    profile_specs = args.profile or [
        "name=current_lite_out12_g14_b6000,backbone=wide_resnet50_2,out=1:2,resize=224,crop=224,grid=14,bank=fixed:6000,topk=0.01",
        "name=patchcore_l23_g14_b6000,backbone=wide_resnet50_2,out=2:3,resize=256,crop=224,grid=14,bank=fixed:6000,topk=0.01",
        "name=patchcore_l23_g28_b6000,backbone=wide_resnet50_2,out=2:3,resize=256,crop=224,grid=28,bank=fixed:6000,topk=0.01",
        "name=patchcore_l23_g28_core10,backbone=wide_resnet50_2,out=2:3,resize=256,crop=224,grid=28,bank=ratio:0.10,topk=0.01",
    ]
    profiles = [parse_profile(spec) for spec in profile_specs]
    category_results = []
    for profile in profiles:
        for category in args.categories:
            category_results.append(evaluate_category(category, profile, args, device))
    payload = {
        "purpose": "Diagnose whether previous MVTec PatchCore-lite scores were low because the baseline was not faithful to standard PatchCore settings.",
        "config": {
            **vars(args),
            "materialized_root": str(args.materialized_root),
            "output": str(args.output),
            "markdown": str(args.markdown),
            "figure": str(args.figure),
            "device": str(device),
        },
        "profiles": [
            {
                "name": p.name,
                "backbone": p.backbone,
                "out_indices": list(p.out_indices),
                "resize": p.resize,
                "crop": p.crop,
                "grid": p.grid,
                "topk_fraction": p.topk_fraction,
                "bank_policy": p.bank_policy,
                "bank_value": p.bank_value,
            }
            for p in profiles
        ],
        "category_results": category_results,
        "figure": str(args.figure),
    }
    payload["aggregate_rows"] = aggregate_results(category_results, profiles, args.report_false_pass_target, args.score_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(payload, args.markdown)
    write_figure(payload, args.figure)
    print(json.dumps({"wrote": str(args.output), "profiles": len(profiles), "categories": len(args.categories), "device": str(device)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
