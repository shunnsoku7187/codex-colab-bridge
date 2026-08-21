"""KSDD2 baselines with industrial anomaly-detection style methods.

This script is intentionally separate from the U-Net baselines.  It checks
whether established inspection-style approaches can provide a credible final
detector before we attach early exits or FPGA-oriented changes.

Implemented baselines:

* PatchCore-lite: nearest-neighbour distance from test patch features to a
  coreset of normal training patch features.
* PaDiM-diagonal: per-patch diagonal Gaussian distance fitted on normal
  training patch features.

Both use a frozen pretrained CNN backbone and the same final inspection metrics
as the previous KSDD2 experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from scripts.ksdd2_conservative_safe_exit_training import Sample, download_and_extract, find_samples, make_split, split_counts
from scripts.ksdd2_unet_inspection_baseline import aggregate, evaluate_thresholds, make_curve_rows, maybe_pixel_metrics
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


class KSDD2ImageDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: tuple[int, int]):
        self.samples = samples
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
            x = TF.to_tensor(image)
            x = (x - self.mean) / self.std
        return x, torch.tensor(sample.label, dtype=torch.long), torch.tensor(idx, dtype=torch.long)


def make_loader(samples: list[Sample], image_size: tuple[int, int], batch_size: int) -> DataLoader:
    dataset = KSDD2ImageDataset(samples, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def make_backbone(name: str, out_indices: tuple[int, ...], device: torch.device):
    import timm

    model = timm.create_model(name, pretrained=True, features_only=True, out_indices=out_indices)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def batch_patch_features(model, x: torch.Tensor, patch_grid: tuple[int, int]) -> torch.Tensor:
    feats = model(x)
    resized = [F.interpolate(feat, size=patch_grid, mode="bilinear", align_corners=False) for feat in feats]
    emb = torch.cat(resized, dim=1)
    emb = F.normalize(emb, dim=1)
    return emb.permute(0, 2, 3, 1).reshape(x.shape[0], patch_grid[0] * patch_grid[1], -1)


@torch.no_grad()
def collect_patch_features(
    model,
    samples: list[Sample],
    image_size: tuple[int, int],
    batch_size: int,
    patch_grid: tuple[int, int],
    device: torch.device,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(samples, image_size, batch_size)
    labels, features = [], []
    for x, label, _idx in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        emb = batch_patch_features(model, x, patch_grid).detach().cpu().numpy().astype(np.float32)
        features.append(emb)
        labels.append(label.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels).astype(np.int64)


def image_scores_from_patch_scores(patch_scores: np.ndarray, topk_fraction: float) -> dict[str, np.ndarray]:
    k = max(1, int(round(patch_scores.shape[1] * topk_fraction)))
    topk = np.partition(patch_scores, kth=patch_scores.shape[1] - k, axis=1)[:, -k:]
    return {
        "max_score": patch_scores.max(axis=1).astype(np.float32),
        "topk_score": topk.mean(axis=1).astype(np.float32),
    }


def score_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"image_auroc": None, "image_aupr": None}
    return {
        "image_auroc": round_float(roc_auc_score(labels, scores)),
        "image_aupr": round_float(average_precision_score(labels, scores)),
    }


def sample_normal_patch_bank(features: np.ndarray, labels: np.ndarray, max_patches: int, seed: int) -> np.ndarray:
    normal = features[labels == 0].reshape(-1, features.shape[-1])
    if len(normal) == 0:
        raise RuntimeError("No normal training features were available for anomaly baseline fitting.")
    rng = np.random.default_rng(seed)
    if len(normal) > max_patches:
        idx = rng.choice(len(normal), size=max_patches, replace=False)
        normal = normal[idx]
    return normal.astype(np.float32)


def patchcore_scores(features: np.ndarray, bank: np.ndarray, chunk_size: int) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    mins = np.empty(len(flat), dtype=np.float32)
    bank_t = torch.from_numpy(bank)
    for start in tqdm(range(0, len(flat), chunk_size), desc="patchcore nn", leave=False):
        chunk = torch.from_numpy(flat[start : start + chunk_size])
        distances = torch.cdist(chunk, bank_t)
        mins[start : start + len(chunk)] = distances.min(dim=1).values.numpy()
    return mins.reshape(features.shape[0], features.shape[1])


def padim_diag_fit(features: np.ndarray, labels: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    normal = features[labels == 0]
    mean = normal.mean(axis=0)
    std = normal.std(axis=0) + eps
    return mean.astype(np.float32), std.astype(np.float32)


def padim_diag_scores(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (features - mean[None, :, :]) / std[None, :, :]
    return np.mean(z * z, axis=2).astype(np.float32)


def plot_tradeoff(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.4, 5.0))
    for result in payload["seed_results"]:
        if result["seed"] != payload["training_config"]["seeds"][0]:
            continue
        for method in result["methods"]:
            rows = method["test_curve_rows"]["topk_score"]
            x = [100.0 * float(r["good_loss_rate_good"]) for r in rows]
            y = [100.0 * float(r["false_pass_rate_defect"]) for r in rows]
            plt.plot(x, y, marker="o", markersize=2.2, linewidth=1.3, label=method["method"])
    plt.xlabel("good loss rate [%]")
    plt.ylabel("defect false pass rate [%]")
    plt.title("KSDD2 industrial anomaly baselines")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def run_method(args: argparse.Namespace, method: str, train_features, train_labels, val_features, val_labels, test_features, test_labels, seed: int) -> dict:
    if method == "patchcore_lite":
        bank = sample_normal_patch_bank(train_features, train_labels, args.patchcore_bank_patches, seed)
        val_patch_scores = patchcore_scores(val_features, bank, args.nn_chunk_size)
        test_patch_scores = patchcore_scores(test_features, bank, args.nn_chunk_size)
        method_info = {"normal_patch_bank": int(len(bank))}
    elif method == "padim_diag":
        mean, std = padim_diag_fit(train_features, train_labels, args.padim_eps)
        val_patch_scores = padim_diag_scores(val_features, mean, std)
        test_patch_scores = padim_diag_scores(test_features, mean, std)
        method_info = {"normal_patch_positions": int(mean.shape[0]), "feature_dim": int(mean.shape[1])}
    else:
        raise ValueError(f"Unsupported method: {method}")

    val_image = {"labels": val_labels, **image_scores_from_patch_scores(val_patch_scores, args.topk_fraction)}
    test_image = {"labels": test_labels, **image_scores_from_patch_scores(test_patch_scores, args.topk_fraction)}
    rows = evaluate_thresholds(val_image, test_image, args)
    return {
        "method": method,
        "method_info": method_info,
        "val_auc": {name: score_auc(val_labels, val_image[name]) for name in ["max_score", "topk_score"]},
        "test_auc": {name: score_auc(test_labels, test_image[name]) for name in ["max_score", "topk_score"]},
        "threshold_rows": rows,
        "test_curve_rows": {name: make_curve_rows(test_image, name, args.curve_points) for name in ["max_score", "topk_score"]},
    }


def aggregate_methods(seed_results: list[dict]) -> list[dict]:
    out = []
    methods = sorted({m["method"] for r in seed_results for m in r["methods"]})
    for method in methods:
        pseudo = []
        for result in seed_results:
            row = next(m for m in result["methods"] if m["method"] == method)
            pseudo.append({"threshold_rows": row["threshold_rows"]})
        for row in aggregate(pseudo):
            row["method"] = method
            out.append(row)
    return out


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KSDD2 industrial anomaly baselines",
        "",
        "Purpose: test existing inspection-style methods before building more custom early-exit logic.",
        "",
        "## Dataset",
        "",
        f"- Samples: {payload['dataset']['sample_count']}",
        f"- Good: {payload['dataset']['good_count']}",
        f"- Defects: {payload['dataset']['defect_count']}",
        "",
        "## Aggregate threshold result",
        "",
        "| method | score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_rows"]:
        lines.append(
            f"| {row['method']} | {row['score_name']} | {100*row['max_false_pass_rate_defect']:.1f}% | "
            f"{100*row['min_good_pass_rate_good']:.1f}% | {row['test_feasible_seeds']}/{row['seeds']} | "
            f"{100*row['mean_good_pass_rate_good']:.2f}% | {100*row['mean_false_pass_rate_defect']:.2f}% | "
            f"{100*row['worst_false_pass_rate_defect']:.2f}% |"
        )
    lines += [
        "",
        "## Per-seed AUC",
        "",
        "| seed | method | score | val AUROC | val AUPR | test AUROC | test AUPR |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for result in payload["seed_results"]:
        for method in result["methods"]:
            for score_name in ["max_score", "topk_score"]:
                va = method["val_auc"][score_name]
                ta = method["test_auc"][score_name]
                lines.append(
                    f"| {result['seed']} | {method['method']} | {score_name} | {va['image_auroc']} | "
                    f"{va['image_aupr']} | {ta['image_auroc']} | {ta['image_aupr']} |"
                )
    lines += [
        "",
        "## Interpretation guide",
        "",
        "- If PatchCore-lite is clearly stronger, use it as the performance upper-bound and compare FPGA cost.",
        "- If PaDiM-diagonal is close enough, it is more FPGA-friendly because it avoids nearest-neighbour search.",
        "- If both fail under false-pass constraints, continue with the best previous segmentation baseline.",
        "",
        f"Curve image: `{payload['curve_png']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_methods(values: Iterable[str]) -> list[str]:
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ksdd2_industrial_anomaly_baselines_001_summary.json")
    parser.add_argument("--markdown", default="docs/ksdd2_industrial_anomaly_baselines_001.md")
    parser.add_argument("--curve-png", default="results/ksdd2_industrial_anomaly_baselines_001_tradeoff.png")
    parser.add_argument("--scores-dir", default="results/ksdd2_industrial_anomaly_baselines_001_scores")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--out-indices", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--methods", nargs="*", default=["patchcore_lite", "padim_diag"])
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--patch-grid-height", type=int, default=14)
    parser.add_argument("--patch-grid-width", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--patchcore-bank-patches", type=int, default=12000)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--padim-eps", type=float, default=1e-6)
    parser.add_argument("--seeds", nargs="*", type=int, default=[123, 456])
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.01, 0.05])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.90, 0.95])
    parser.add_argument("--max-threshold-candidates", type=int, default=101)
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    methods = parse_methods(args.methods)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    cache_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    samples = find_samples(download_and_extract(cache_root / "kolektor_sdd2"))
    model = make_backbone(args.backbone, tuple(args.out_indices), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (args.patch_grid_height, args.patch_grid_width)
    seed_results = []
    scores_dir = Path(args.scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        set_seed(seed)
        split = make_split(samples, seed)
        train_features, train_labels = collect_patch_features(
            model, split["train"], image_size, args.batch_size, patch_grid, device, f"features train seed={seed}"
        )
        val_features, val_labels = collect_patch_features(
            model, split["val"], image_size, args.batch_size, patch_grid, device, f"features val seed={seed}"
        )
        test_features, test_labels = collect_patch_features(
            model, split["eval"], image_size, args.batch_size, patch_grid, device, f"features test seed={seed}"
        )
        method_results = [
            run_method(args, method, train_features, train_labels, val_features, val_labels, test_features, test_labels, seed)
            for method in methods
        ]
        np.savez_compressed(
            scores_dir / f"seed_{seed}_labels.npz",
            val_labels=val_labels,
            test_labels=test_labels,
        )
        seed_results.append(
            {
                "seed": seed,
                "split_counts": split_counts(split),
                "methods": method_results,
            }
        )

    payload = {
        "purpose": "Compare existing industrial anomaly-detection baselines on KSDD2 before custom early-exit work.",
        "dataset": {
            "name": "KolektorSDD2",
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
        },
        "training_config": vars(args),
        "seed_results": seed_results,
        "aggregate_rows": aggregate_methods(seed_results),
        "curve_png": args.curve_png,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    plot_tradeoff(payload, Path(args.curve_png))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "curve_png": args.curve_png}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
