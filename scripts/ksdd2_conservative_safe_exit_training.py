"""Conservative safe-exit training on KolektorSDD2.

KolektorSDD is tiny: evaluation may change dramatically when only a few defect
images move across splits.  KolektorSDD2 keeps the same industrial surface
inspection flavor but is much larger and has an official train/test split.

This experiment repeats the conservative safe-exit test under the same
inspection constraints:

* false pass among defects must stay below the target;
* good pass among normal samples must stay above the target;
* thresholds are selected on validation data and then fixed on test data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from scripts.kolektor_conservative_safe_exit_training import (
    ResNet18ConservativeSafeBranchy,
    aggregate_rows,
    apply_policy,
    choose_safe_exit_policy,
    collect_outputs,
    train_model,
)
from scripts.kolektor_late_recovery_predictor import choose_bn, choose_final_threshold
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


KSDD2_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None
    label: int  # 0 good, 1 defect
    official_split: str | None


def download_and_extract(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "KolektorSDD2"
    if dataset_dir.exists() and any(dataset_dir.rglob("*")):
        return dataset_dir

    archive = root / "KolektorSDD2.zip"
    if not archive.exists():
        print(f"Downloading KolektorSDD2 from {KSDD2_URL}", flush=True)
        urlretrieve(KSDD2_URL, archive)

    print(f"Extracting {archive}", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)

    candidates = [p for p in root.iterdir() if p.is_dir() and "KolektorSDD2" in p.name]
    if candidates:
        return candidates[0]
    return root


def mask_has_defect(mask_path: Path | None) -> bool:
    if mask_path is None or not mask_path.exists():
        return False
    with Image.open(mask_path) as mask:
        arr = np.asarray(mask.convert("L"))
    return bool((arr > 0).any())


def looks_like_mask(path: Path) -> bool:
    name = path.stem.lower()
    parts = {p.lower() for p in path.parts}
    return (
        name.endswith("_label")
        or name.endswith("_gt")
        or "mask" in name
        or "label" in name
        or "ground_truth" in parts
        or "masks" in parts
    )


def find_mask(image_path: Path) -> Path | None:
    stems = [
        image_path.stem + "_label",
        image_path.stem + "_GT",
        image_path.stem + "_gt",
        image_path.stem + "_mask",
        image_path.stem + "_defect",
    ]
    suffixes = [".png", ".bmp", ".tif", ".tiff", ".jpg"]
    candidates = []
    for stem in stems:
        candidates += [image_path.with_name(stem + suffix) for suffix in suffixes]
    for parent in [image_path.parent, *image_path.parents]:
        for dirname in ["masks", "mask", "ground_truth", "labels", "label"]:
            mask_dir = parent / dirname
            if mask_dir.exists():
                candidates += [mask_dir / (image_path.stem + suffix) for suffix in suffixes]
                candidates += [mask_dir / (image_path.stem + "_label" + suffix) for suffix in suffixes]
    return next((p for p in candidates if p.exists()), None)


def split_name(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    if "train" in parts or "training" in parts:
        return "train"
    if "test" in parts or "testing" in parts:
        return "test"
    return None


def label_from_path(path: Path, mask_path: Path | None) -> int:
    if mask_path is not None:
        return 1 if mask_has_defect(mask_path) else 0
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    defect_words = {"defect", "defective", "bad", "positive", "anomaly", "anomalous"}
    good_words = {"good", "ok", "normal", "negative"}
    if any(word in parts or word in name for word in defect_words):
        return 1
    if any(word in parts or word in name for word in good_words):
        return 0
    return 0


def find_samples(dataset_dir: Path) -> list[Sample]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples = []
    for image_path in sorted(dataset_dir.rglob("*")):
        if image_path.suffix.lower() not in image_exts or looks_like_mask(image_path):
            continue
        mask_path = find_mask(image_path)
        label = label_from_path(image_path, mask_path)
        samples.append(Sample(image_path=image_path, mask_path=mask_path, label=label, official_split=split_name(image_path)))
    if not samples:
        raise RuntimeError(f"No image samples found under {dataset_dir}")
    if sum(s.label for s in samples) == 0:
        raise RuntimeError("No defect samples were detected. Update the KSDD2 parser before trusting this experiment.")
    return samples


def stratified_split(samples: list[Sample], ratios: tuple[float, float, float], seed: int) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    by_label = {0: [s for s in samples if s.label == 0], 1: [s for s in samples if s.label == 1]}
    for group in by_label.values():
        rng.shuffle(group)
    out = {"train": [], "val": [], "eval": []}
    for group in by_label.values():
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        out["train"].extend(group[:n_train])
        out["val"].extend(group[n_train : n_train + n_val])
        out["eval"].extend(group[n_train + n_val :])
    for group in out.values():
        rng.shuffle(group)
    return out


def make_split(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    train_pool = [s for s in samples if s.official_split == "train"]
    test_pool = [s for s in samples if s.official_split == "test"]
    if train_pool and test_pool and sum(s.label for s in train_pool) > 0 and sum(s.label for s in test_pool) > 0:
        train_val = stratified_split(train_pool, (0.82, 0.18, 0.0), seed)
        split = {"train": train_val["train"], "val": train_val["val"], "eval": test_pool}
    else:
        split = stratified_split(samples, (0.60, 0.20, 0.20), seed)
    for name, group in split.items():
        if not group or sum(s.label for s in group) == 0 or sum(1 - s.label for s in group) == 0:
            raise RuntimeError(f"Bad split for {name}: samples={len(group)}, defects={sum(s.label for s in group)}")
    return split


class KSDD2Dataset(Dataset):
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
    dataset = KSDD2Dataset(samples, image_size=image_size, train=train)
    sampler = None
    shuffle = train
    if train:
        labels = np.asarray([s.label for s in samples], dtype=np.int64)
        counts = np.bincount(labels, minlength=2).astype(np.float32)
        weights = 1.0 / np.maximum(counts, 1.0)
        sampler = WeightedRandomSampler(weights[labels].tolist(), num_samples=len(samples), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=2, pin_memory=True)


def class_weight(samples: list[Sample], device: torch.device) -> torch.Tensor:
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def split_counts(split: dict[str, list[Sample]]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "samples": len(group),
            "defects": int(sum(s.label for s in group)),
            "good": int(sum(1 - s.label for s in group)),
        }
        for name, group in split.items()
    }


def run_one_seed(args: argparse.Namespace, samples: list[Sample], seed: int, device: torch.device) -> dict:
    set_seed(seed)
    split = make_split(samples, seed)
    image_size = (args.image_height, args.image_width)
    train_loader = make_loader(split["train"], image_size, args.batch_size, train=True)
    val_loader = make_loader(split["val"], image_size, args.batch_size, train=False)
    eval_loader = make_loader(split["eval"], image_size, args.batch_size, train=False)

    model = ResNet18ConservativeSafeBranchy(pretrained=True).to(device)
    training = train_model(
        model,
        train_loader,
        val_loader,
        class_weight(split["train"], device),
        args.epochs,
        device,
        args.safe_loss_weight,
        args.unsafe_loss_weight,
        args.final_good_threshold,
        args.safe_warmup_epochs,
    )

    checkpoint = Path(args.checkpoint_dir) / f"seed_{seed}" / "ksdd2_resnet18_conservative_safe_branchy.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "image_size": list(image_size), "exit_costs": model.exit_costs.tolist()}, checkpoint)

    val_data = collect_outputs(model, val_loader, device)
    eval_data = collect_outputs(model, eval_loader, device)
    policy_rows = []
    for max_fp in args.max_false_pass_rates:
        for min_gp in args.min_good_pass_rates:
            constraint = {"max_false_pass_rate_defect": max_fp, "min_good_pass_rate_good": min_gp}
            selected = [
                choose_final_threshold(val_data, max_fp, min_gp, args.max_threshold_candidates),
                choose_bn(val_data, max_fp, min_gp, args.max_threshold_candidates),
                choose_safe_exit_policy(val_data, max_fp, min_gp, args.max_threshold_candidates),
            ]
            for row in selected:
                if row is None:
                    continue
                row_out = {"seed": seed, "constraint": constraint, **row, "eval_metric": apply_policy(eval_data, row)}
                row_out["eval_feasible"] = (
                    row_out["eval_metric"]["false_pass_rate_defect"] <= max_fp + 1e-12
                    and row_out["eval_metric"]["good_pass_rate_good"] >= min_gp - 1e-12
                )
                policy_rows.append(row_out)
    return {
        "seed": seed,
        "split_counts": split_counts(split),
        "checkpoint": str(checkpoint),
        "training": training,
        "policy_rows": policy_rows,
    }


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KolektorSDD2 conservative safe-exit training",
        "",
        "Purpose: repeat the KolektorSDD conservative safe-exit test on the larger KolektorSDD2 dataset.",
        "",
        "Decision rule:",
        "",
        "- If final_selective is unstable, the final model/dataset setup is not ready for safe inspection claims.",
        "- If final_selective is feasible but conservative_safe_dual_exit is not, the current early safe-exit mechanism is still weak.",
        "- If conservative_safe_dual_exit is feasible and faster than final_selective / upper-only BranchyNet, the KolektorSDD failure was likely small-data or split related.",
        "",
        "## Dataset",
        "",
        f"- Dataset: {payload['dataset']['name']}",
        f"- Samples found: {payload['dataset']['sample_count']}",
        f"- Defects: {payload['dataset']['defect_count']}",
        f"- Good: {payload['dataset']['good_count']}",
        "",
        "## Aggregate",
        "",
        "| max false pass | min good pass | policy | feasible seeds | mean good pass | mean false pass | worst false pass | mean speedup |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_rows"]:
        lines.append(
            f"| {100 * row['max_false_pass_rate_defect']:.1f}% | {100 * row['min_good_pass_rate_good']:.1f}% | "
            f"{row['policy']} | {row['eval_feasible_seeds']}/{row['seeds']} | "
            f"{100 * row['mean_good_pass_rate_good']:.2f}% | {100 * row['mean_false_pass_rate_defect']:.2f}% | "
            f"{100 * row['worst_false_pass_rate_defect']:.2f}% | {row['mean_speedup']:.2f}x |"
        )
    lines += [
        "",
        "## Per-seed rows",
        "",
        "| seed | max false pass | min good pass | policy | feasible | good pass | false pass | avg cost | speedup |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["seed_results"]:
        for row in result["policy_rows"]:
            e = row["eval_metric"]
            lines.append(
                f"| {row['seed']} | {100 * row['constraint']['max_false_pass_rate_defect']:.1f}% | "
                f"{100 * row['constraint']['min_good_pass_rate_good']:.1f}% | {row['policy']} | "
                f"{'yes' if row.get('eval_feasible') else 'no'} | {100 * e['good_pass_rate_good']:.2f}% | "
                f"{100 * e['false_pass_rate_defect']:.2f}% | {e['avg_cost']:.4f} | {e['speedup_vs_final_only']:.2f}x |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ksdd2_conservative_safe_exit_training_001_summary.json")
    parser.add_argument("--markdown", default="docs/ksdd2_conservative_safe_exit_training_001.md")
    parser.add_argument("--checkpoint-dir", default="artifacts/ksdd2_conservative_safe_exit_training_001")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=704)
    parser.add_argument("--seeds", nargs="*", type=int, default=[123, 456])
    parser.add_argument("--safe-loss-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-loss-weight", type=float, default=4.0)
    parser.add_argument("--final-good-threshold", type=float, default=0.9)
    parser.add_argument("--safe-warmup-epochs", type=int, default=4)
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.05, 0.1])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.90, 0.95])
    parser.add_argument("--max-threshold-candidates", type=int, default=15)
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
        "purpose": "Check whether KolektorSDD failures were caused by small data / split dependence.",
        "dataset": {
            "name": "KolektorSDD2",
            "source": KSDD2_URL,
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
        },
        "training_config": {
            "epochs": args.epochs,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "safe_loss_weight": args.safe_loss_weight,
            "unsafe_loss_weight": args.unsafe_loss_weight,
            "final_good_threshold": args.final_good_threshold,
            "safe_warmup_epochs": args.safe_warmup_epochs,
            "seeds": args.seeds,
        },
        "seed_results": seed_results,
        "aggregate_rows": aggregate_rows(seed_results),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "seeds": args.seeds}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
