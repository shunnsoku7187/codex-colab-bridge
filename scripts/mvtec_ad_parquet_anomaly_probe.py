"""Probe MVTec AD anomaly baselines from a Hugging Face Parquet mirror.

The downloaded mirror does not use the usual MVTec directory layout.  This
script reads the Parquet shards directly, materializes selected categories into
a persistent image cache, then runs lightweight PatchCore and PaDiM-style
baselines.  The goal is not to publish a final benchmark yet; it is to check
whether MVTec AD gives a stable inspection baseline that is worth using for
the early-exit/FPGA work.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass
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

from scripts.ksdd2_industrial_anomaly_baselines import (
    batch_patch_features,
    image_scores_from_patch_scores,
    padim_diag_fit,
    padim_diag_scores,
    patchcore_scores,
    sample_normal_patch_bank,
)
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


DEFAULT_CATEGORIES = ["bottle", "cable", "hazelnut", "metal_nut", "screw"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MVTecSample:
    image_path: Path
    mask_path: Path | None
    category: str
    split: str
    defect: str
    label: int


class MVTecImageDataset(Dataset):
    def __init__(self, samples: list[MVTecSample], image_size: tuple[int, int]):
        self.samples = samples
        self.image_size = image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
            x = TF.to_tensor(image)
            x = (x - self.mean) / self.std
        return x, torch.tensor(sample.label, dtype=torch.long)


def decode_image_cell(value) -> Image.Image | None:
    """Decode common Hugging Face Image feature encodings from Parquet."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(bytes(raw)))
        path = value.get("path")
        if path:
            path_obj = Path(path)
            if path_obj.exists():
                return Image.open(path_obj)
        return None
    path_obj = Path(str(value))
    if path_obj.exists():
        return Image.open(path_obj)
    return None


def safe_stem(value: str, fallback: str, max_len: int = 80) -> str:
    stem = Path(str(value)).stem if value else fallback
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    cleaned = cleaned.strip("._")
    return (cleaned or fallback)[:max_len]


def stable_row_stem(parquet_path: Path, row_idx: int) -> str:
    shard = safe_stem(parquet_path.stem, "shard", max_len=42)
    key = f"{parquet_path.name}:{row_idx}".encode("utf-8", errors="replace")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return f"{shard}_{row_idx:06d}_{digest}"


def find_parquet_files(dataset_root: Path) -> list[Path]:
    direct = sorted((dataset_root / "data").glob("*.parquet"))
    if direct:
        return direct
    return sorted(dataset_root.rglob("*.parquet"))


def materialize_from_parquet(dataset_root: Path, output_root: Path, categories: list[str], limit_per_category: int) -> dict:
    import pyarrow.parquet as pq

    output_root.mkdir(parents=True, exist_ok=True)
    parquet_files = find_parquet_files(dataset_root)
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {dataset_root}")

    counts: dict[str, dict[str, int]] = {
        category: {"train": 0, "test": 0, "good": 0, "defect": 0, "masks": 0} for category in categories
    }
    written = 0
    skipped_existing = 0
    inspected_rows = 0

    for parquet_path in tqdm(parquet_files, desc="read parquet shards"):
        table = pq.read_table(parquet_path)
        columns = set(table.column_names)
        rows = table.to_pylist()
        for row_idx, row in enumerate(rows):
            inspected_rows += 1
            category = str(row.get("object") or row.get("category") or "")
            if category not in categories:
                continue
            if limit_per_category > 0 and counts[category]["train"] + counts[category]["test"] >= limit_per_category:
                continue
            split = str(row.get("split") or "test")
            label = int(row.get("label") or 0)
            defect = str(row.get("defect") or ("good" if label == 0 else "defect"))
            image_source = row.get("image") if "image" in columns else row.get("image_path")
            mask_source = row.get("mask") if "mask" in columns else row.get("mask_path")

            image = decode_image_cell(image_source)
            if image is None:
                continue

            # Some HF Image rows expose bytes-like data through image_path.  Never
            # trust dataset metadata as a filesystem name.
            base_name = stable_row_stem(parquet_path, row_idx)
            category_dir = output_root / category
            image_dir = category_dir / split / defect
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{base_name}.png"
            if image_path.exists():
                skipped_existing += 1
            else:
                image.convert("RGB").save(image_path)
                written += 1

            if label != 0:
                counts[category]["defect"] += 1
            else:
                counts[category]["good"] += 1
            if split in ("train", "test"):
                counts[category][split] += 1

            mask = decode_image_cell(mask_source)
            if mask is not None:
                mask_dir = category_dir / "ground_truth" / defect
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{base_name}_mask.png"
                if not mask_path.exists():
                    mask.convert("L").save(mask_path)
                counts[category]["masks"] += 1

    return {
        "dataset_root": str(dataset_root),
        "materialized_root": str(output_root),
        "parquet_files": len(parquet_files),
        "inspected_rows": inspected_rows,
        "written_images": written,
        "skipped_existing_images": skipped_existing,
        "categories": counts,
    }


def find_materialized_samples(root: Path, category: str) -> list[MVTecSample]:
    samples: list[MVTecSample] = []
    category_root = root / category
    for split in ["train", "test"]:
        split_root = category_root / split
        if not split_root.exists():
            continue
        for image_path in sorted(path for path in split_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            defect = image_path.parent.name
            label = 0 if defect == "good" else 1
            mask_path = category_root / "ground_truth" / defect / f"{image_path.stem}_mask.png"
            samples.append(
                MVTecSample(
                    image_path=image_path,
                    mask_path=mask_path if mask_path.exists() else None,
                    category=category,
                    split=split,
                    defect=defect,
                    label=label,
                )
            )
    return samples


def make_backbone(name: str, out_indices: tuple[int, ...], device: torch.device):
    import timm

    model = timm.create_model(name, pretrained=True, features_only=True, out_indices=out_indices)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def make_loader(samples: list[MVTecSample], image_size: tuple[int, int], batch_size: int) -> DataLoader:
    return DataLoader(MVTecImageDataset(samples, image_size), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


@torch.no_grad()
def collect_features(model, samples: list[MVTecSample], image_size: tuple[int, int], batch_size: int, patch_grid: tuple[int, int], device, desc: str):
    features, labels = [], []
    for x, label in tqdm(make_loader(samples, image_size, batch_size), desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        emb = batch_patch_features(model, x, patch_grid).detach().cpu().numpy().astype(np.float32)
        features.append(emb)
        labels.append(label.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels).astype(np.int64)


def score_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"image_auroc": None, "image_aupr": None}
    return {
        "image_auroc": round_float(roc_auc_score(labels, scores)),
        "image_aupr": round_float(average_precision_score(labels, scores)),
    }


def curve_rows(labels: np.ndarray, scores: np.ndarray, points: int) -> list[dict]:
    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, points))
    rows = []
    good = labels == 0
    defect = labels == 1
    for threshold in thresholds:
        predicted_defect = scores >= threshold
        false_pass = float((~predicted_defect[defect]).mean()) if defect.any() else None
        good_loss = float(predicted_defect[good].mean()) if good.any() else None
        rows.append(
            {
                "threshold": round_float(float(threshold)),
                "good_loss_rate_good": round_float(good_loss) if good_loss is not None else None,
                "false_pass_rate_defect": round_float(false_pass) if false_pass is not None else None,
                "good_pass_rate_good": round_float(1.0 - good_loss) if good_loss is not None else None,
                "defect_reject_rate_defect": round_float(1.0 - false_pass) if false_pass is not None else None,
            }
        )
    return rows


def evaluate_method(args, method: str, train_features, test_features, test_labels, seed: int) -> dict:
    if method == "patchcore_lite":
        train_labels = np.zeros(len(train_features), dtype=np.int64)
        bank = sample_normal_patch_bank(train_features, train_labels, args.patchcore_bank_patches, seed)
        patch_scores = patchcore_scores(test_features, bank, args.nn_chunk_size)
        info = {"normal_patch_bank": int(len(bank))}
    elif method == "padim_diag":
        train_labels = np.zeros(len(train_features), dtype=np.int64)
        mean, std = padim_diag_fit(train_features, train_labels, args.padim_eps)
        patch_scores = padim_diag_scores(test_features, mean, std)
        info = {"normal_patch_positions": int(mean.shape[0]), "feature_dim": int(mean.shape[1])}
    else:
        raise ValueError(method)

    image_scores = image_scores_from_patch_scores(patch_scores, args.topk_fraction)
    return {
        "method": method,
        "method_info": info,
        "auc": {name: score_auc(test_labels, scores) for name, scores in image_scores.items()},
        "curve_rows": {name: curve_rows(test_labels, scores, args.curve_points) for name, scores in image_scores.items()},
    }


def plot_category_curves(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 2
    rows = int(np.ceil(len(payload["category_results"]) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.0, max(3.0, 3.0 * rows)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, category_result in zip(axes.ravel(), payload["category_results"]):
        ax.axis("on")
        for method in category_result["methods"]:
            rows_data = method["curve_rows"]["topk_score"]
            x = [100.0 * row["good_loss_rate_good"] for row in rows_data if row["good_loss_rate_good"] is not None]
            y = [100.0 * row["false_pass_rate_defect"] for row in rows_data if row["false_pass_rate_defect"] is not None]
            ax.plot(x, y, linewidth=1.4, label=method["method"])
        ax.set_title(category_result["category"])
        ax.set_xlabel("good loss [%]")
        ax.set_ylabel("defect false pass [%]")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize_best_rows(category_results: list[dict], false_pass_targets: Iterable[float]) -> list[dict]:
    rows = []
    for result in category_results:
        for method in result["methods"]:
            for score_name, curve in method["curve_rows"].items():
                for target in false_pass_targets:
                    feasible = [row for row in curve if row["false_pass_rate_defect"] is not None and row["false_pass_rate_defect"] <= target]
                    if feasible:
                        best = min(feasible, key=lambda row: row["good_loss_rate_good"])
                        rows.append(
                            {
                                "category": result["category"],
                                "method": method["method"],
                                "score": score_name,
                                "max_false_pass_rate_defect": target,
                                "best_good_loss_rate_good": best["good_loss_rate_good"],
                                "best_good_pass_rate_good": best["good_pass_rate_good"],
                                "threshold": best["threshold"],
                            }
                        )
                    else:
                        rows.append(
                            {
                                "category": result["category"],
                                "method": method["method"],
                                "score": score_name,
                                "max_false_pass_rate_defect": target,
                                "best_good_loss_rate_good": None,
                                "best_good_pass_rate_good": None,
                                "threshold": None,
                            }
                        )
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MVTec AD Parquet anomaly baseline probe",
        "",
        "Purpose: check whether MVTec AD can provide a stronger inspection baseline than the small KSDD splits.",
        "",
        "## Materialization",
        "",
        f"- Dataset root: `{payload['materialization']['dataset_root']}`",
        f"- Materialized root: `{payload['materialization']['materialized_root']}`",
        f"- Parquet shards: `{payload['materialization']['parquet_files']}`",
        f"- Written images this run: `{payload['materialization']['written_images']}`",
        "",
        "## Category sample counts",
        "",
        "| category | train | test | good rows | defect rows | masks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, row in payload["materialization"]["categories"].items():
        lines.append(
            f"| {category} | {row['train']} | {row['test']} | {row['good']} | {row['defect']} | {row['masks']} |"
        )
    lines += [
        "",
        "## Image-level anomaly separation",
        "",
        "| category | method | score | AUROC | AUPR |",
        "|---|---|---|---:|---:|",
    ]
    for category_result in payload["category_results"]:
        for method in category_result["methods"]:
            for score_name, auc in method["auc"].items():
                lines.append(
                    f"| {category_result['category']} | {method['method']} | {score_name} | "
                    f"{auc['image_auroc']} | {auc['image_aupr']} |"
                )
    lines += [
        "",
        "## Best good retention under false-pass constraints",
        "",
        "| category | method | score | max defect false-pass | best good pass | good loss |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["best_constraint_rows"]:
        good_pass = "" if row["best_good_pass_rate_good"] is None else f"{100*row['best_good_pass_rate_good']:.2f}%"
        good_loss = "" if row["best_good_loss_rate_good"] is None else f"{100*row['best_good_loss_rate_good']:.2f}%"
        lines.append(
            f"| {row['category']} | {row['method']} | {row['score']} | "
            f"{100*row['max_false_pass_rate_defect']:.1f}% | {good_pass} | {good_loss} |"
        )
    lines += [
        "",
        "## Reading the result",
        "",
        "- AUROC/AUPR checks whether normal and defect images separate at all.",
        "- The tradeoff curve checks the inspection question: how much good product is lost when defect false-pass is restricted.",
        "- If PatchCore is strong and PaDiM is close, PaDiM is the more FPGA-friendly baseline candidate.",
        "- If both are strong, the next step is attaching early-exit style acceleration to this inspection baseline.",
        "",
        f"Curve image: `{payload['curve_png']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--materialized-root", default="")
    parser.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=["patchcore_lite", "padim_diag"])
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--out-indices", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--patch-grid-height", type=int, default=14)
    parser.add_argument("--patch-grid-width", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--patchcore-bank-patches", type=int, default=12000)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--padim-eps", type=float, default=1e-6)
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--false-pass-targets", nargs="*", type=float, default=[0.0, 0.01, 0.05])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="results/mvtec_ad_parquet_probe_001_summary.json")
    parser.add_argument("--markdown", default="docs/mvtec_ad_parquet_probe_001.md")
    parser.add_argument("--curve-png", default="results/mvtec_ad_parquet_probe_001_tradeoff.png")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    data_root = Path(os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    dataset_root = Path(args.dataset_root or data_root / "mvtec_ad")
    materialized_root = Path(args.materialized_root or data_root / "mvtec_ad_materialized")
    materialization = materialize_from_parquet(dataset_root, materialized_root, args.categories, args.limit_per_category)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    model = make_backbone(args.backbone, tuple(args.out_indices), device)
    image_size = (args.image_height, args.image_width)
    patch_grid = (args.patch_grid_height, args.patch_grid_width)
    category_results = []
    for category in args.categories:
        samples = find_materialized_samples(materialized_root, category)
        train = [sample for sample in samples if sample.split == "train" and sample.label == 0]
        test = [sample for sample in samples if sample.split == "test"]
        if not train or not test or len({sample.label for sample in test}) < 2:
            category_results.append(
                {
                    "category": category,
                    "status": "skipped",
                    "reason": "category lacks normal train samples or mixed-label test samples",
                    "sample_counts": {"train_normal": len(train), "test": len(test)},
                    "methods": [],
                }
            )
            continue
        train_features, _train_labels = collect_features(
            model, train, image_size, args.batch_size, patch_grid, device, f"{category} train features"
        )
        test_features, test_labels = collect_features(
            model, test, image_size, args.batch_size, patch_grid, device, f"{category} test features"
        )
        methods = [evaluate_method(args, method, train_features, test_features, test_labels, args.seed) for method in args.methods]
        category_results.append(
            {
                "category": category,
                "status": "done",
                "sample_counts": {
                    "train_normal": len(train),
                    "test": len(test),
                    "test_good": int(sum(sample.label == 0 for sample in test)),
                    "test_defect": int(sum(sample.label == 1 for sample in test)),
                },
                "methods": methods,
            }
        )

    payload = {
        "purpose": "Probe PatchCore-lite and PaDiM-diagonal on selected MVTec AD categories from the Hugging Face Parquet mirror.",
        "config": vars(args),
        "materialization": materialization,
        "category_results": category_results,
        "best_constraint_rows": summarize_best_rows(category_results, args.false_pass_targets),
        "curve_png": args.curve_png,
        "tools": {"python": shutil.which("python"), "git": shutil.which("git")},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    plot_category_curves(payload, Path(args.curve_png))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "curve_png": args.curve_png}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
