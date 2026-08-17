"""Train a lightweight U-Net inspection baseline on KolektorSDD2.

The earlier ResNet classifier was not safe enough as a final inspection model.
This script switches the final detector to a segmentation-style model, which is
more natural for surface inspection and can later receive early exits at
multiple decoder depths.

Image-level decision:

* U-Net predicts a defect heatmap.
* A scalar defect score is derived from that heatmap.
* Validation data selects a threshold.
* Test data reports defect false pass and good pass under fixed thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF
from tqdm import tqdm

from scripts.ksdd2_conservative_safe_exit_training import Sample, download_and_extract, find_samples, make_split, split_counts
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


class KSDD2MaskDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: tuple[int, int], train: bool):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, sample: Sample) -> Image.Image:
        if sample.mask_path is None or not sample.mask_path.exists():
            return Image.new("L", self.image_size[::-1], 0)
        with Image.open(sample.mask_path) as mask:
            return mask.convert("L")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            mask = self._load_mask(sample)

            image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

            if self.train and random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if self.train and random.random() < 0.5:
                image = TF.adjust_brightness(image, 1.0 + random.uniform(-0.10, 0.10))
                image = TF.adjust_contrast(image, 1.0 + random.uniform(-0.14, 0.14))

            x = TF.to_tensor(image)
            x = (x - self.mean) / self.std
            y = (TF.to_tensor(mask) > 0.5).float()
        return x, y, torch.tensor(sample.label, dtype=torch.long)


def make_loader(samples: list[Sample], image_size: tuple[int, int], batch_size: int, train: bool) -> DataLoader:
    dataset = KSDD2MaskDataset(samples, image_size=image_size, train=train)
    sampler = None
    shuffle = train
    if train:
        labels = np.asarray([s.label for s in samples], dtype=np.int64)
        counts = np.bincount(labels, minlength=2).astype(np.float32)
        weights = 1.0 / np.maximum(counts, 1.0)
        sampler = WeightedRandomSampler(weights[labels].tolist(), num_samples=len(samples), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=2, pin_memory=True)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class SmallUNet(nn.Module):
    def __init__(self, base_channels: int = 24):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(3, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 8)
        self.up1 = Up(c * 8, c * 8, c * 4)
        self.up2 = Up(c * 4, c * 4, c * 2)
        self.up3 = Up(c * 2, c * 2, c)
        self.up4 = Up(c, c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def train_one_epoch(model: nn.Module, loader: DataLoader, opt, device: torch.device, pos_weight: float) -> float:
    model.train()
    losses = []
    weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    for x, mask, _label in tqdm(loader, desc="train unet", leave=False):
        x = x.to(device)
        mask = mask.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=weight) + dice_loss(logits, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def collect_scores(model: nn.Module, loader: DataLoader, device: torch.device, topk_fraction: float) -> dict[str, np.ndarray]:
    model.eval()
    labels, max_scores, topk_scores, pixel_scores, pixel_labels = [], [], [], [], []
    for x, mask, label in tqdm(loader, desc="collect unet", leave=False):
        x = x.to(device)
        probs = torch.sigmoid(model(x)).detach().cpu()
        flat = probs.flatten(1)
        k = max(1, int(round(flat.shape[1] * topk_fraction)))
        labels.append(label.numpy())
        max_scores.append(flat.max(dim=1).values.numpy())
        topk_scores.append(flat.topk(k, dim=1).values.mean(dim=1).numpy())
        # Pixel metrics are sampled to avoid oversized result files.
        if len(pixel_scores) < 64:
            pixel_scores.append(probs.flatten().numpy())
            pixel_labels.append(mask.flatten().numpy())
    out = {
        "labels": np.concatenate(labels).astype(np.int64),
        "max_score": np.concatenate(max_scores).astype(np.float32),
        "topk_score": np.concatenate(topk_scores).astype(np.float32),
    }
    if pixel_scores:
        ps = np.concatenate(pixel_scores).astype(np.float32)
        pl = np.concatenate(pixel_labels).astype(np.uint8)
        if len(ps) > 2_000_000:
            rng = np.random.default_rng(123)
            idx = rng.choice(len(ps), size=2_000_000, replace=False)
            ps = ps[idx]
            pl = pl[idx]
        out["pixel_scores"] = ps
        out["pixel_labels"] = pl
    return out


def metric(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int | float | None]:
    good = labels == 0
    defect = labels == 1
    reject = scores >= threshold
    pass_good = ~reject
    false_pass = defect & pass_good
    good_loss = good & reject
    return {
        "samples": int(len(labels)),
        "good_count": int(good.sum()),
        "defect_count": int(defect.sum()),
        "threshold": round_float(threshold),
        "good_pass_rate_good": round_float(float((good & pass_good).sum() / max(good.sum(), 1))),
        "good_loss_rate_good": round_float(float(good_loss.sum() / max(good.sum(), 1))),
        "false_pass_rate_defect": round_float(float(false_pass.sum() / max(defect.sum(), 1))),
        "defect_recall": round_float(float((defect & reject).sum() / max(defect.sum(), 1))),
        "false_pass_count": int(false_pass.sum()),
        "good_loss_count": int(good_loss.sum()),
    }


def score_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"image_auroc": None, "image_aupr": None}
    return {
        "image_auroc": round_float(roc_auc_score(labels, scores)),
        "image_aupr": round_float(average_precision_score(labels, scores)),
    }


def candidates(scores: np.ndarray, max_count: int) -> list[float]:
    unique = np.unique(np.asarray(scores, dtype=np.float32))
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) >= 2 else np.asarray([], dtype=np.float32)
    pooled = np.unique(np.concatenate([unique, mids, np.asarray([0.0, 1.0], dtype=np.float32)]))
    if len(pooled) > max_count:
        pooled = np.unique(np.quantile(pooled, np.linspace(0, 1, max_count)).astype(np.float32))
    return sorted(float(x) for x in pooled)


def choose_threshold(val: dict[str, np.ndarray], score_name: str, max_false_pass: float, min_good_pass: float, max_candidates: int) -> dict | None:
    best = None
    scores = val[score_name]
    for threshold in candidates(scores, max_candidates):
        row = metric(val["labels"], scores, threshold)
        if row["false_pass_rate_defect"] <= max_false_pass + 1e-12 and row["good_pass_rate_good"] >= min_good_pass - 1e-12:
            cand = {"score_name": score_name, "threshold": threshold, "val_metric": row}
            if best is None or (
                float(cand["val_metric"]["good_pass_rate_good"]),
                -float(cand["val_metric"]["false_pass_rate_defect"]),
            ) > (
                float(best["val_metric"]["good_pass_rate_good"]),
                -float(best["val_metric"]["false_pass_rate_defect"]),
            ):
                best = cand
    return best


def evaluate_thresholds(val: dict[str, np.ndarray], test: dict[str, np.ndarray], args: argparse.Namespace) -> list[dict]:
    rows = []
    for score_name in ["max_score", "topk_score"]:
        for max_fp in args.max_false_pass_rates:
            for min_gp in args.min_good_pass_rates:
                selected = choose_threshold(val, score_name, max_fp, min_gp, args.max_threshold_candidates)
                if selected is None:
                    continue
                test_metric = metric(test["labels"], test[score_name], selected["threshold"])
                rows.append(
                    {
                        "score_name": score_name,
                        "constraint": {"max_false_pass_rate_defect": max_fp, "min_good_pass_rate_good": min_gp},
                        "threshold": round_float(selected["threshold"]),
                        "val_metric": selected["val_metric"],
                        "test_metric": test_metric,
                        "test_feasible": (
                            test_metric["false_pass_rate_defect"] <= max_fp + 1e-12
                            and test_metric["good_pass_rate_good"] >= min_gp - 1e-12
                        ),
                    }
                )
    return rows


def make_curve_rows(data: dict[str, np.ndarray], score_name: str, max_points: int) -> list[dict]:
    rows = []
    for threshold in candidates(data[score_name], max_points):
        row = metric(data["labels"], data[score_name], threshold)
        rows.append(row)
    return rows


def plot_curve(curves: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.8))
    for name, rows in curves.items():
        x = [100.0 * float(r["good_loss_rate_good"]) for r in rows]
        y = [100.0 * float(r["false_pass_rate_defect"]) for r in rows]
        plt.plot(x, y, marker="o", markersize=2.8, linewidth=1.4, label=name)
    plt.xlabel("good loss rate [%]")
    plt.ylabel("defect false pass rate [%]")
    plt.title("KSDD2 U-Net inspection trade-off")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def maybe_pixel_metrics(data: dict[str, np.ndarray]) -> dict[str, float | None]:
    if "pixel_scores" not in data or len(np.unique(data["pixel_labels"])) < 2:
        return {"pixel_auroc_sampled": None, "pixel_aupr_sampled": None}
    return {
        "pixel_auroc_sampled": round_float(roc_auc_score(data["pixel_labels"], data["pixel_scores"])),
        "pixel_aupr_sampled": round_float(average_precision_score(data["pixel_labels"], data["pixel_scores"])),
    }


def run_one_seed(args: argparse.Namespace, samples: list[Sample], seed: int, device: torch.device) -> dict:
    set_seed(seed)
    split = make_split(samples, seed)
    image_size = (args.image_height, args.image_width)
    train_loader = make_loader(split["train"], image_size, args.batch_size, train=True)
    val_loader = make_loader(split["val"], image_size, args.batch_size, train=False)
    test_loader = make_loader(split["eval"], image_size, args.batch_size, train=False)

    model = SmallUNet(base_channels=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_state = None
    best_score = -math.inf
    history = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, opt, device, args.positive_pixel_weight)
        scheduler.step()
        val_scores = collect_scores(model, val_loader, device, args.topk_fraction)
        auc = score_auc(val_scores["labels"], val_scores["topk_score"])
        row = {"epoch": epoch + 1, "loss": round_float(loss), **auc}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        history.append(row)
        score = float(auc["image_auroc"] or 0.0) + float(auc["image_aupr"] or 0.0)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = Path(args.checkpoint_dir) / f"seed_{seed}" / "ksdd2_small_unet.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "image_size": list(image_size), "base_channels": args.base_channels}, checkpoint)

    val_scores = collect_scores(model, val_loader, device, args.topk_fraction)
    test_scores = collect_scores(model, test_loader, device, args.topk_fraction)
    curve_rows = {name: make_curve_rows(test_scores, name, args.curve_points) for name in ["max_score", "topk_score"]}
    if seed == args.seeds[0]:
        plot_curve(curve_rows, Path(args.curve_png))
    return {
        "seed": seed,
        "split_counts": split_counts(split),
        "checkpoint": str(checkpoint),
        "history": history,
        "val_auc": {name: score_auc(val_scores["labels"], val_scores[name]) for name in ["max_score", "topk_score"]},
        "test_auc": {name: score_auc(test_scores["labels"], test_scores[name]) for name in ["max_score", "topk_score"]},
        "pixel_metrics_sampled": maybe_pixel_metrics(test_scores),
        "threshold_rows": evaluate_thresholds(val_scores, test_scores, args),
        "test_curve_rows": curve_rows,
    }


def aggregate(seed_results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for result in seed_results:
        for row in result["threshold_rows"]:
            c = row["constraint"]
            key = (row["score_name"], float(c["max_false_pass_rate_defect"]), float(c["min_good_pass_rate_good"]))
            groups.setdefault(key, []).append(row)
    out = []
    for (score_name, max_fp, min_gp), rows in sorted(groups.items()):
        fp = [float(r["test_metric"]["false_pass_rate_defect"]) for r in rows]
        gp = [float(r["test_metric"]["good_pass_rate_good"]) for r in rows]
        out.append(
            {
                "score_name": score_name,
                "max_false_pass_rate_defect": max_fp,
                "min_good_pass_rate_good": min_gp,
                "seeds": len(rows),
                "test_feasible_seeds": int(sum(bool(r["test_feasible"]) for r in rows)),
                "mean_good_pass_rate_good": round_float(np.mean(gp)),
                "mean_false_pass_rate_defect": round_float(np.mean(fp)),
                "worst_false_pass_rate_defect": round_float(np.max(fp)),
                "worst_good_pass_rate_good": round_float(np.min(gp)),
            }
        )
    return out


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KSDD2 U-Net inspection baseline",
        "",
        "Purpose: build a stronger final inspection model before adding early exits.",
        "",
        "## Dataset",
        "",
        f"- Samples: {payload['dataset']['sample_count']}",
        f"- Good: {payload['dataset']['good_count']}",
        f"- Defects: {payload['dataset']['defect_count']}",
        "",
        "## Aggregate image-level thresholds",
        "",
        "| score | max false pass | min good pass | feasible seeds | mean good pass | mean false pass | worst false pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_rows"]:
        lines.append(
            f"| {row['score_name']} | {100*row['max_false_pass_rate_defect']:.1f}% | {100*row['min_good_pass_rate_good']:.1f}% | "
            f"{row['test_feasible_seeds']}/{row['seeds']} | {100*row['mean_good_pass_rate_good']:.2f}% | "
            f"{100*row['mean_false_pass_rate_defect']:.2f}% | {100*row['worst_false_pass_rate_defect']:.2f}% |"
        )
    lines += [
        "",
        "## Per-seed AUC",
        "",
        "| seed | score | val AUROC | val AUPR | test AUROC | test AUPR | sampled pixel AUROC | sampled pixel AUPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["seed_results"]:
        for score_name in ["max_score", "topk_score"]:
            va = result["val_auc"][score_name]
            ta = result["test_auc"][score_name]
            pm = result["pixel_metrics_sampled"]
            lines.append(
                f"| {result['seed']} | {score_name} | {va['image_auroc']} | {va['image_aupr']} | "
                f"{ta['image_auroc']} | {ta['image_aupr']} | {pm['pixel_auroc_sampled']} | {pm['pixel_aupr_sampled']} |"
            )
    lines += [
        "",
        "## Per-seed threshold rows",
        "",
        "| seed | score | max false pass | min good pass | feasible | good pass | false pass | threshold |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["seed_results"]:
        for row in result["threshold_rows"]:
            c = row["constraint"]
            tm = row["test_metric"]
            lines.append(
                f"| {result['seed']} | {row['score_name']} | {100*c['max_false_pass_rate_defect']:.1f}% | "
                f"{100*c['min_good_pass_rate_good']:.1f}% | {'yes' if row['test_feasible'] else 'no'} | "
                f"{100*tm['good_pass_rate_good']:.2f}% | {100*tm['false_pass_rate_defect']:.2f}% | {row['threshold']} |"
            )
    lines.append("")
    lines.append(f"Curve image: `{payload['curve_png']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ksdd2_unet_inspection_baseline_001_summary.json")
    parser.add_argument("--markdown", default="docs/ksdd2_unet_inspection_baseline_001.md")
    parser.add_argument("--checkpoint-dir", default="artifacts/ksdd2_unet_inspection_baseline_001")
    parser.add_argument("--curve-png", default="results/ksdd2_unet_inspection_baseline_001_tradeoff.png")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-pixel-weight", type=float, default=40.0)
    parser.add_argument("--topk-fraction", type=float, default=0.001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[123, 456])
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.01, 0.05])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.90, 0.95])
    parser.add_argument("--max-threshold-candidates", type=int, default=51)
    parser.add_argument("--curve-points", type=int, default=80)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    cache_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    samples = find_samples(download_and_extract(cache_root / "kolektor_sdd2"))
    seed_results = [run_one_seed(args, samples, seed, device) for seed in args.seeds]
    payload = {
        "purpose": "Train a segmentation-based final inspection baseline before adding early exits.",
        "dataset": {
            "name": "KolektorSDD2",
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
        },
        "training_config": vars(args),
        "seed_results": seed_results,
        "aggregate_rows": aggregate(seed_results),
        "curve_png": args.curve_png,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "curve_png": args.curve_png}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
