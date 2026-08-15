"""Train stronger final-only inspection models on KolektorSDD.

This is a prerequisite for evaluating dual-sided early exit on non-CIFAR data.
The previous tiny model was too weak as a final inspection model, so this script
trains ImageNet-initialized final classifiers and reports inspection-centric
metrics: false pass, good loss, defect recall, and selective pass curves.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

from src.experiment_paths import ensure_dirs


DATASET_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip"
SPLITS_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None
    item: str
    label: int  # 0 good, 1 defect


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def round_float(value: float | np.floating | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def download_and_extract(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "KolektorSDD"
    if dataset_dir.exists() and any(dataset_dir.rglob("*.jpg")):
        return dataset_dir

    archive = root / "KolektorSDD.zip"
    if not archive.exists():
        print(f"Downloading KolektorSDD from {DATASET_URL}", flush=True)
        urlretrieve(DATASET_URL, archive)

    print(f"Extracting {archive}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)

    if dataset_dir.exists():
        return dataset_dir
    candidates = [path for path in root.iterdir() if path.is_dir() and "Kolektor" in path.name]
    if candidates:
        return candidates[0]
    return root


def mask_has_defect(mask_path: Path | None) -> bool:
    if mask_path is None or not mask_path.exists():
        return False
    with Image.open(mask_path) as mask:
        arr = np.asarray(mask.convert("L"))
    return bool((arr > 0).any())


def find_samples(dataset_dir: Path) -> list[Sample]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples: list[Sample] = []
    for image_path in sorted(dataset_dir.rglob("*")):
        if image_path.suffix.lower() not in image_exts:
            continue
        stem = image_path.stem.lower()
        if stem.endswith("_label") or stem.endswith("_gt") or "mask" in stem:
            continue
        mask_candidates = [
            image_path.with_name(image_path.stem + "_label.bmp"),
            image_path.with_name(image_path.stem + "_label.png"),
            image_path.with_name(image_path.stem + "_GT.png"),
            image_path.with_name(image_path.stem + "_mask.png"),
        ]
        mask_path = next((path for path in mask_candidates if path.exists()), None)
        label = 1 if mask_has_defect(mask_path) else 0
        samples.append(Sample(image_path=image_path, mask_path=mask_path, item=image_path.parent.name, label=label))
    if not samples:
        raise RuntimeError(f"No image samples found under {dataset_dir}")
    return samples


def split_by_item(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    rng = np.random.default_rng(seed)
    items = sorted({sample.item for sample in samples})
    best = None
    for attempt in range(300):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_items = set(shuffled[: int(n * 0.6)])
        val_items = set(shuffled[int(n * 0.6) : int(n * 0.8)])
        eval_items = set(shuffled[int(n * 0.8) :])
        split = {
            "train": [s for s in samples if s.item in train_items],
            "val": [s for s in samples if s.item in val_items],
            "eval": [s for s in samples if s.item in eval_items],
        }
        best = split
        if all(sum(s.label for s in group) > 0 for group in split.values()):
            return split
    if best is None:
        raise RuntimeError("Could not create item split")
    return best


class KolektorDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: tuple[int, int], train: bool):
        self.samples = samples
        aug = []
        if train:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.18)], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            ]
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                *aug,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as image:
            x = self.transform(image.convert("RGB"))
        return x, torch.tensor(sample.label, dtype=torch.long)


def make_loader(samples: list[Sample], image_size: tuple[int, int], batch_size: int, train: bool) -> DataLoader:
    dataset = KolektorDataset(samples, image_size=image_size, train=train)
    sampler = None
    shuffle = train
    if train:
        labels = np.asarray([s.label for s in samples], dtype=np.int64)
        counts = np.bincount(labels, minlength=2).astype(np.float32)
        weights = 1.0 / np.maximum(counts, 1.0)
        sampler = WeightedRandomSampler(weights[labels].tolist(), num_samples=len(samples), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=2, pin_memory=True)


def make_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    raise ValueError(f"Unknown arch: {arch}")


def class_weight(samples: list[Sample], device: torch.device) -> torch.Tensor:
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    labels_all = []
    probs_all = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        labels_all.append(y.numpy())
        probs_all.append(probs.detach().cpu().numpy())
    labels = np.concatenate(labels_all).astype(np.int64)
    probs = np.concatenate(probs_all).astype(np.float32)
    pred = np.argmax(probs, axis=1)
    return {"labels": labels, "probs": probs, "pred": pred}


def metrics_from_outputs(outputs: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, float | None]:
    labels = outputs["labels"]
    p_defect = outputs["probs"][:, 1]
    pred = (p_defect >= threshold).astype(np.int64)
    good = labels == 0
    defect = labels == 1
    false_pass = defect & (pred == 0)
    false_reject = good & (pred == 1)
    out: dict[str, float | None] = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, pred),
        "good_recall": float((pred[good] == 0).mean()) if good.any() else None,
        "defect_recall": float((pred[defect] == 1).mean()) if defect.any() else None,
        "false_pass_rate_all": float(false_pass.mean()),
        "false_pass_rate_defect": float(false_pass.sum() / max(defect.sum(), 1)),
        "good_loss_rate_all": float(false_reject.mean()),
        "good_loss_rate_good": float(false_reject.sum() / max(good.sum(), 1)),
    }
    if len(np.unique(labels)) > 1:
        out["auroc_defect"] = roc_auc_score(labels, p_defect)
        out["average_precision_defect"] = average_precision_score(labels, p_defect)
    else:
        out["auroc_defect"] = None
        out["average_precision_defect"] = None
    return out


def threshold_sweep(outputs: dict[str, np.ndarray], max_false_pass_rates: list[float]) -> list[dict[str, float | None]]:
    labels = outputs["labels"]
    p_defect = outputs["probs"][:, 1]
    good = labels == 0
    defect = labels == 1
    rows = []
    thresholds = np.unique(np.quantile(p_defect, np.linspace(0.0, 1.0, 501)))
    for max_fp in max_false_pass_rates:
        best = None
        for threshold in thresholds:
            pred_defect = p_defect >= threshold
            false_pass_rate = float(((defect) & (~pred_defect)).sum() / max(defect.sum(), 1))
            if false_pass_rate > max_fp:
                continue
            good_loss = float((good & pred_defect).sum() / max(good.sum(), 1))
            row = {
                "max_false_pass_rate_defect": max_fp,
                "threshold": float(threshold),
                "good_pass_rate_good": float((good & ~pred_defect).sum() / max(good.sum(), 1)),
                "good_loss_rate_good": good_loss,
                "false_pass_rate_defect": false_pass_rate,
                "defect_recall": float((defect & pred_defect).sum() / max(defect.sum(), 1)),
                "pass_rate_all": float((~pred_defect).mean()),
            }
            if best is None or (row["good_pass_rate_good"], -row["false_pass_rate_defect"]) > (
                best["good_pass_rate_good"],
                -best["false_pass_rate_defect"],
            ):
                best = row
        if best is not None:
            rows.append(best)
    return rows


def train_one(
    arch: str,
    split: dict[str, list[Sample]],
    image_size: tuple[int, int],
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
    pretrained: bool,
    checkpoint_path: Path,
) -> dict[str, object]:
    set_seed(seed)
    model = make_model(arch, pretrained=pretrained).to(device)
    weights = class_weight(split["train"], device)
    train_loader = make_loader(split["train"], image_size, batch_size, train=True)
    val_loader = make_loader(split["val"], image_size, batch_size, train=False)
    eval_loader = make_loader(split["eval"], image_size, batch_size, train=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_state = None
    best_score = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"{arch} epoch {epoch + 1}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, label_smoothing=0.02)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_outputs = evaluate(model, val_loader, device)
        val_metrics = metrics_from_outputs(val_outputs)
        score = float(val_metrics["defect_recall"] or 0.0) + float(val_metrics["good_recall"] or 0.0) + float(val_metrics["auroc_defect"] or 0.0)
        row = {
            "epoch": epoch + 1,
            "loss": round_float(np.mean(losses)),
            **{k: round_float(v) for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps({"arch": arch, **row}, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "pretrained": pretrained,
            "image_size": list(image_size),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    eval_outputs = evaluate(model, eval_loader, device)
    eval_metrics = metrics_from_outputs(eval_outputs)
    return {
        "arch": arch,
        "pretrained": pretrained,
        "checkpoint": str(checkpoint_path),
        "image_size": list(image_size),
        "epochs": epochs,
        "history": history,
        "best_val_score": round_float(best_score),
        "eval_metrics_0_5": {k: round_float(v) for k, v in eval_metrics.items()},
        "eval_threshold_sweep": [{k: round_float(v) for k, v in row.items()} for row in threshold_sweep(eval_outputs, [0.0, 0.05, 0.1, 0.2])],
    }


def write_markdown(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KolektorSDD strong final model",
        "",
        "## Purpose",
        "",
        "Before evaluating dual-sided early exit, this job checks whether a strong final-only inspection model can be trained on KolektorSDD.",
        "",
        "## Dataset split",
        "",
        "| split | samples | good | defect |",
        "|---|---:|---:|---:|",
    ]
    split_counts = payload["dataset"]["split_counts"]  # type: ignore[index]
    for name in ["train", "val", "eval"]:
        row = split_counts[name]
        lines.append(f"| {name} | {row['samples']} | {row['good']} | {row['defects']} |")
    lines += [
        "",
        "## Final-only model results",
        "",
        "| arch | acc | good recall | defect recall | AUROC | AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:  # type: ignore[index]
        m = result["eval_metrics_0_5"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["arch"],
                    f"{100 * m['accuracy']:.2f}%",
                    f"{100 * m['good_recall']:.2f}%",
                    f"{100 * m['defect_recall']:.2f}%",
                    "n/a" if m["auroc_defect"] is None else f"{m['auroc_defect']:.3f}",
                    "n/a" if m["average_precision_defect"] is None else f"{m['average_precision_defect']:.3f}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Safety-threshold view",
        "",
        "Rows show the best good-pass rate when the allowed false-pass rate among defects is constrained.",
        "",
        "| arch | max false pass | good pass | good loss | defect recall | threshold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:  # type: ignore[index]
        for row in result["eval_threshold_sweep"]:
            lines.append(
                f"| {result['arch']} | {100 * row['max_false_pass_rate_defect']:.1f}% | "
                f"{100 * row['good_pass_rate_good']:.2f}% | {100 * row['good_loss_rate_good']:.2f}% | "
                f"{100 * row['defect_recall']:.2f}% | {row['threshold']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results = payload["results"]  # type: ignore[index]
    width, height = 1080, 420
    margin = 74
    bar_area = 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827;font-size:15px}.title{font-size:24px;font-weight:700}.small{fill:#4b5563;font-size:13px}</style>',
        '<text x="36" y="42" class="title">KolektorSDD final-only inspection baseline</text>',
        '<text x="36" y="66" class="small">A strong final model is required before evaluating early rejection.</text>',
    ]
    metrics = [("good_recall", "Good recall", "#2563eb"), ("defect_recall", "Defect recall", "#dc2626")]
    y = 112
    for result in results:
        parts.append(f'<text x="36" y="{y + 6}" font-weight="700">{result["arch"]}</text>')
        for i, (key, label, color) in enumerate(metrics):
            m = result["eval_metrics_0_5"]
            value = float(m[key])
            yy = y + 30 + i * 38
            parts.append(f'<text x="70" y="{yy + 16}" class="small">{label}</text>')
            parts.append(f'<rect x="200" y="{yy}" width="{bar_area}" height="22" fill="#e5e7eb"/>')
            parts.append(f'<rect x="200" y="{yy}" width="{bar_area * value:.1f}" height="22" fill="{color}"/>')
            parts.append(f'<text x="940" y="{yy + 16}">{100 * value:.1f}%</text>')
        y += 110
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/kolektor_strong_final_001_summary.json")
    parser.add_argument("--markdown", default="docs/kolektor_strong_final_001.md")
    parser.add_argument("--svg", default="results/kolektor_strong_final_001.svg")
    parser.add_argument("--checkpoint-dir", default="artifacts/kolektor_strong_final_001")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--archs", nargs="*", default=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    cache_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    dataset_dir = download_and_extract(cache_root / "kolektor_sdd")
    samples = find_samples(dataset_dir)
    split = split_by_item(samples, args.seed)
    image_size = (args.image_height, args.image_width)

    results = []
    for arch in args.archs:
        results.append(
            train_one(
                arch=arch,
                split=split,
                image_size=image_size,
                batch_size=args.batch_size,
                epochs=args.epochs,
                seed=args.seed,
                device=device,
                pretrained=not args.no_pretrained,
                checkpoint_path=Path(args.checkpoint_dir) / f"{arch}.pt",
            )
        )

    payload = {
        "purpose": "Train strong final-only inspection baselines on KolektorSDD before dual-sided early-exit evaluation.",
        "dataset": {
            "name": "KolektorSDD",
            "url": DATASET_URL,
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
            "split_counts": {
                key: {"samples": len(value), "defects": int(sum(s.label for s in value)), "good": int(sum(1 - s.label for s in value))}
                for key, value in split.items()
            },
        },
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    write_svg(payload, Path(args.svg))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "svg": args.svg}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
