"""Non-CIFAR inspection experiment for the dual-sided early-exit idea.

Dataset: Kolektor Surface-Defect Dataset (KolektorSDD).

Task framing:
* class 0 = good surface, class 1 = defective surface
* a "pass" means accepting a good-surface label
* a false pass means a defective surface is accepted as good
* a lower exit rejector predicts whether the final stage will fail to provide a
  reliable good pass, so that the final stage can be skipped.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.experiment_paths import ensure_dirs


DATASET_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip"


def round_float(value: float | np.floating | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None
    item: str
    label: int  # 0 good, 1 defect


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
        item = image_path.parent.name
        label = 1 if mask_has_defect(mask_path) else 0
        samples.append(Sample(image_path=image_path, mask_path=mask_path, item=item, label=label))
    if not samples:
        raise RuntimeError(f"No image samples found under {dataset_dir}")
    return samples


def split_by_item(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    rng = np.random.default_rng(seed)
    items = sorted({sample.item for sample in samples})
    rng.shuffle(items)

    # Re-sample until every split contains at least one defect if possible.
    best = None
    for attempt in range(200):
        if attempt:
            rng.shuffle(items)
        n = len(items)
        train_items = set(items[: int(n * 0.6)])
        val_items = set(items[int(n * 0.6) : int(n * 0.8)])
        eval_items = set(items[int(n * 0.8) :])
        split = {
            "train": [s for s in samples if s.item in train_items],
            "val": [s for s in samples if s.item in val_items],
            "eval": [s for s in samples if s.item in eval_items],
        }
        defect_counts = {key: sum(s.label for s in value) for key, value in split.items()}
        best = split
        if all(count > 0 for count in defect_counts.values()):
            return split
    if best is None:
        raise RuntimeError("Could not create split")
    return best


class KolektorDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: tuple[int, int]):
        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as image:
            x = self.transform(image.convert("RGB"))
        y = torch.tensor(sample.label, dtype=torch.long)
        return x, y


class TinyInspectionBranchyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.exit0 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))
        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))
        self.final = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2))
        self.exit_names = ["exit0", "exit1", "final"]
        self.exit_costs = np.asarray([0.22, 0.55, 1.0], dtype=np.float32)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h0 = self.stage0(x)
        out0 = self.exit0(h0)
        h1 = self.stage1(h0)
        out1 = self.exit1(h1)
        h2 = self.stage2(h1)
        out2 = self.final(h2)
        return [out0, out1, out2]


def make_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)


def make_loader(samples: list[Sample], batch_size: int, image_size: tuple[int, int], train: bool) -> DataLoader:
    dataset = KolektorDataset(samples, image_size=image_size)
    sampler = make_sampler(samples) if train else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False if sampler else train, sampler=sampler, num_workers=2, pin_memory=True)


def class_weight(samples: list[Sample], device: torch.device) -> torch.Tensor:
    labels = np.asarray([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    model: TinyInspectionBranchyNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weights: torch.Tensor,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_state = None
    best_score = -1.0
    exit_loss_weights = [0.4, 0.7, 1.0]
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = sum(w * F.cross_entropy(logits, y, weight=weights) for w, logits in zip(exit_loss_weights, outputs))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = collect_outputs(model, val_loader, device)
        final_acc = float(val["correct"][:, -1].mean())
        defect_recall = defect_recall_score(val["labels"], val["pred"][:, -1])
        score = final_acc + defect_recall
        history.append({"epoch": epoch + 1, "loss": round_float(np.mean(losses)), "val_final_acc": round_float(final_acc), "val_defect_recall": round_float(defect_recall)})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_score": round_float(best_score)}


def defect_recall_score(labels: np.ndarray, pred: np.ndarray) -> float:
    defect = labels == 1
    if not defect.any():
        return 0.0
    return float((pred[defect] == 1).mean())


@torch.no_grad()
def collect_outputs(model: TinyInspectionBranchyNet, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    labels_all, pred_all, correct_all, confidence_all, entropy_all, prob_good_all = [], [], [], [], [], []
    for x, y in tqdm(loader, desc="Collecting outputs"):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        labels_all.append(y.detach().cpu().numpy())
        preds = []
        corrects = []
        confs = []
        ents = []
        pgood = []
        for logits in outputs:
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(2.0)
            preds.append(pred.detach().cpu().numpy())
            corrects.append((pred == y).detach().cpu().numpy())
            confs.append(probs.max(dim=1).values.detach().cpu().numpy())
            ents.append(ent.detach().cpu().numpy())
            pgood.append(probs[:, 0].detach().cpu().numpy())
        pred_all.append(np.stack(preds, axis=1))
        correct_all.append(np.stack(corrects, axis=1))
        confidence_all.append(np.stack(confs, axis=1))
        entropy_all.append(np.stack(ents, axis=1))
        prob_good_all.append(np.stack(pgood, axis=1))
    return {
        "labels": np.concatenate(labels_all).astype(np.int16),
        "pred": np.concatenate(pred_all).astype(np.int16),
        "correct": np.concatenate(correct_all).astype(bool),
        "confidence": np.concatenate(confidence_all).astype(np.float32),
        "entropy": np.concatenate(entropy_all).astype(np.float32),
        "prob_good": np.concatenate(prob_good_all).astype(np.float32),
        "exit_costs": model.exit_costs,
        "exit_names": np.asarray(model.exit_names, dtype=object),
    }


def calibrate_final_good_threshold(data: dict[str, np.ndarray], target_precision: float) -> dict[str, Any] | None:
    labels = data["labels"]
    pgood = data["prob_good"][:, -1]
    pred_good = data["pred"][:, -1] == 0
    good = labels == 0
    rows = []
    for threshold in np.unique(np.quantile(pgood[pred_good], np.linspace(0.0, 1.0, 301))) if pred_good.any() else []:
        accept = pred_good & (pgood >= threshold)
        if not accept.any():
            continue
        precision = float(good[accept].mean())
        rows.append({"threshold": float(threshold), "pass_rate": float(accept.mean()), "pass_precision": precision})
    valid = [row for row in rows if row["pass_precision"] >= target_precision]
    if not valid:
        return None
    return max(valid, key=lambda row: (row["pass_rate"], row["pass_precision"], -row["threshold"]))


def final_reliable_good(data: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    labels = data["labels"]
    return (labels == 0) & (data["pred"][:, -1] == 0) & (data["prob_good"][:, -1] >= threshold)


def make_features(data: dict[str, np.ndarray], exit_idx: int, feature_set: str) -> np.ndarray:
    pgood = data["prob_good"]
    conf = data["confidence"]
    ent = data["entropy"]
    pred = data["pred"]
    cols = [
        pgood[:, exit_idx],
        1.0 - pgood[:, exit_idx],
        conf[:, exit_idx],
        ent[:, exit_idx],
        (pred[:, exit_idx] == 0).astype(np.float32),
    ]
    if feature_set in {"trace", "class_aware"} and exit_idx >= 1:
        cols += [
            pgood[:, 0],
            conf[:, 0],
            ent[:, 0],
            pgood[:, exit_idx] - pgood[:, 0],
            conf[:, exit_idx] - conf[:, 0],
            ent[:, exit_idx] - ent[:, 0],
            (pred[:, exit_idx] == pred[:, 0]).astype(np.float32),
        ]
    if feature_set == "class_aware":
        for cls in [0, 1]:
            cols.append((pred[:, exit_idx] == cls).astype(np.float32))
    return np.stack(cols, axis=1).astype(np.float32)


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n * 0.6))
    return idx[:cut], idx[cut:]


def model_specs(seed: int) -> list[tuple[str, Any, str]]:
    return [
        ("logistic_l2", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)), "fixed-point linear score"),
        ("tree_depth2", DecisionTreeClassifier(max_depth=2, min_samples_leaf=4, class_weight="balanced", random_state=seed), "few comparators / LUT"),
        ("tree_depth3", DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, class_weight="balanced", random_state=seed), "small comparator tree"),
        ("mlp_8", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), alpha=1e-3, max_iter=800, random_state=seed)), "tiny nonlinear head"),
    ]


def non_reliable_score(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    idx = classes.index(False) if False in classes else classes.index(0)
    return np.asarray(proba[:, idx], dtype=np.float32)


def choose_threshold(score: np.ndarray, y_final_reliable: np.ndarray, max_lost: float) -> dict[str, Any] | None:
    best = None
    total = len(score)
    for threshold in np.unique(np.quantile(score, np.linspace(0.0, 1.0, 301))):
        reject = score >= threshold
        lost = float((reject & y_final_reliable).sum() / total)
        if lost > max_lost:
            continue
        row = {
            "threshold": float(threshold),
            "early_reject_rate": float(reject.mean()),
            "lost_final_reliable_rate": lost,
            "reject_precision_non_reliable": None if not reject.any() else float((reject & ~y_final_reliable).sum() / reject.sum()),
        }
        if best is None or (row["early_reject_rate"], -row["lost_final_reliable_rate"]) > (best["early_reject_rate"], -best["lost_final_reliable_rate"]):
            best = row
    return best


def eval_policy(score: np.ndarray, y_final_reliable: np.ndarray, threshold: float, exit_cost: float, final_cost: float) -> dict[str, Any]:
    reject = score >= threshold
    avg_cost = float(reject.mean() * exit_cost + (1.0 - reject.mean()) * final_cost)
    return {
        "early_reject_rate": round_float(float(reject.mean())),
        "final_execution_rate": round_float(float(1.0 - reject.mean())),
        "avg_cost": round_float(avg_cost),
        "speedup_vs_final_only": round_float(final_cost / avg_cost),
        "cost_reduction_vs_final_only": round_float(final_cost - avg_cost),
        "lost_final_reliable_rate": round_float(float((reject & y_final_reliable).sum() / len(reject))),
        "reject_precision_non_reliable": round_float(float((reject & ~y_final_reliable).sum() / reject.sum()) if reject.any() else None),
    }


def score_quality(score: np.ndarray, y_final_reliable: np.ndarray) -> dict[str, Any]:
    y_non_reliable = ~y_final_reliable
    out = {"final_reliable_good_rate": round_float(float(y_final_reliable.mean()))}
    if len(np.unique(y_non_reliable)) > 1:
        out["auroc_non_reliable"] = round_float(roc_auc_score(y_non_reliable, score))
        out["average_precision_non_reliable"] = round_float(average_precision_score(y_non_reliable, score))
    else:
        out["auroc_non_reliable"] = None
        out["average_precision_non_reliable"] = None
    return out


def run_rejector_sweep(
    val_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    target_precision: float,
    max_lost_rates: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    threshold_info = calibrate_final_good_threshold(val_data, target_precision)
    if threshold_info is None:
        return [{"valid": False, "target_pass_precision": target_precision, "reason": "no final threshold satisfies target pass precision"}]
    y_val = final_reliable_good(val_data, threshold_info["threshold"])
    y_eval = final_reliable_good(eval_data, threshold_info["threshold"])
    train_idx, cal_idx = split_indices(len(y_val), seed)
    costs = np.asarray(eval_data["exit_costs"], dtype=np.float32)
    for max_lost in max_lost_rates:
        for exit_idx in [0, 1]:
            raw_val_score = 1.0 - val_data["prob_good"][:, exit_idx]
            raw_eval_score = 1.0 - eval_data["prob_good"][:, exit_idx]
            raw_t = choose_threshold(raw_val_score[cal_idx], y_val[cal_idx], max_lost)
            if raw_t is not None:
                rows.append({
                    "valid": True,
                    "target_pass_precision": target_precision,
                    "max_lost_final_reliable_rate": max_lost,
                    "exit_idx": exit_idx,
                    "feature_set": "prob_good_only",
                    "predictor": "raw_prob_good_threshold",
                    "validation_final_threshold": {key: round_float(value) for key, value in threshold_info.items()},
                    "selected_threshold": {key: round_float(value) if isinstance(value, float) else value for key, value in raw_t.items()},
                    "score_quality_eval": score_quality(raw_eval_score, y_eval),
                    **eval_policy(raw_eval_score, y_eval, raw_t["threshold"], float(costs[exit_idx]), float(costs[-1])),
                    "fpga_note": "single comparator on good probability",
                })
            for feature_set in ["scalar", "trace", "class_aware"]:
                if exit_idx == 0 and feature_set == "trace":
                    continue
                x_val = make_features(val_data, exit_idx, feature_set)
                x_eval = make_features(eval_data, exit_idx, feature_set)
                for name, model, note in model_specs(seed):
                    try:
                        model.fit(x_val[train_idx], y_val[train_idx])
                        cal_score = non_reliable_score(model, x_val[cal_idx])
                        t = choose_threshold(cal_score, y_val[cal_idx], max_lost)
                        if t is None:
                            rows.append({"valid": False, "target_pass_precision": target_precision, "max_lost_final_reliable_rate": max_lost, "exit_idx": exit_idx, "feature_set": feature_set, "predictor": name, "reason": "no threshold satisfies max_lost", "fpga_note": note})
                            continue
                        eval_score = non_reliable_score(model, x_eval)
                        rows.append({
                            "valid": True,
                            "target_pass_precision": target_precision,
                            "max_lost_final_reliable_rate": max_lost,
                            "exit_idx": exit_idx,
                            "feature_set": feature_set,
                            "predictor": name,
                            "validation_final_threshold": {key: round_float(value) for key, value in threshold_info.items()},
                            "selected_threshold": {key: round_float(value) if isinstance(value, float) else value for key, value in t.items()},
                            "score_quality_eval": score_quality(eval_score, y_eval),
                            **eval_policy(eval_score, y_eval, t["threshold"], float(costs[exit_idx]), float(costs[-1])),
                            "fpga_note": note,
                        })
                    except Exception as exc:  # noqa: BLE001
                        rows.append({"valid": False, "target_pass_precision": target_precision, "max_lost_final_reliable_rate": max_lost, "exit_idx": exit_idx, "feature_set": feature_set, "predictor": name, "reason": repr(exc), "fpga_note": note})
    return rows


def best_rows(rows: list[dict[str, Any]], strict_eval: bool = True) -> list[dict[str, Any]]:
    valid = []
    for row in rows:
        if not row.get("valid"):
            continue
        if strict_eval and row["lost_final_reliable_rate"] > row["max_lost_final_reliable_rate"] + 1e-12:
            continue
        valid.append(row)
    grouped: dict[tuple[float, float], dict[str, Any]] = {}
    for row in valid:
        key = (row["target_pass_precision"], row["max_lost_final_reliable_rate"])
        current = grouped.get(key)
        if current is None or (row["avg_cost"], -row["early_reject_rate"], row["lost_final_reliable_rate"]) < (current["avg_cost"], -current["early_reject_rate"], current["lost_final_reliable_rate"]):
            grouped[key] = row
    return sorted(grouped.values(), key=lambda row: (row["target_pass_precision"], row["max_lost_final_reliable_rate"]))


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KolektorSDD non-CIFAR inspection experiment",
        "",
        "## Purpose",
        "",
        "This experiment checks whether the lower-side early reject idea still works on a real industrial surface-defect dataset, not CIFAR with synthetic corruption.",
        "",
        "## Dataset",
        "",
        "- Kolektor Surface-Defect Dataset",
        "- Binary inspection task: good surface vs visible defect",
        "- Pass means accepting a good-surface decision",
        "- False pass means a defective surface is accepted as good",
        "",
        "## Best strict-evaluation rows",
        "",
        "| target pass precision | max good loss | predictor | exit | early reject | final rate | avg cost | speedup | measured good loss |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["best_rows_strict_eval"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{100 * row['target_pass_precision']:.1f}%",
                    f"{100 * row['max_lost_final_reliable_rate']:.1f}%",
                    f"{row['feature_set']} {row['predictor']}",
                    f"exit{row['exit_idx']}",
                    f"{100 * row['early_reject_rate']:.2f}%",
                    f"{100 * row['final_execution_rate']:.2f}%",
                    f"{row['avg_cost']:.4f}",
                    f"{row['speedup_vs_final_only']:.2f}x",
                    f"{100 * row['lost_final_reliable_rate']:.2f}%",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(payload: dict[str, Any], path: Path) -> None:
    rows = payload["best_rows_strict_eval"]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1120
    top = 72
    row_h = 42
    height = top + row_h * len(rows) + 70
    left = 300
    bar_w = 470
    max_gain = max(row["cost_reduction_vs_final_only"] for row in rows) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#0f172a;font-size:14px}.title{font-size:22px;font-weight:700}.small{fill:#475569;font-size:12px}</style>',
        '<text x="34" y="36" class="title">KolektorSDD: lower-side late-recovery predictor</text>',
        '<text x="34" y="58" class="small">Bars show normalized compute reduction versus final-only on a real industrial surface-defect dataset.</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        gain = row["cost_reduction_vs_final_only"]
        bw = bar_w * gain / max_gain
        label = f"pass {100 * row['target_pass_precision']:.1f}%, loss {100 * row['max_lost_final_reliable_rate']:.1f}%"
        parts += [
            f'<text x="34" y="{y + 24}">{label}</text>',
            f'<rect x="{left}" y="{y + 8}" width="{bar_w}" height="22" fill="#e2e8f0"/>',
            f'<rect x="{left}" y="{y + 8}" width="{bw:.1f}" height="22" fill="#2563eb"/>',
            f'<text x="{left + bar_w + 18}" y="{y + 24}">-{100 * gain:.1f}% cost</text>',
            f'<text x="{left + bar_w + 134}" y="{y + 24}" class="small">{row["feature_set"]} {row["predictor"]}, reject {100 * row["early_reject_rate"]:.1f}%</text>',
        ]
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/kolektor_late_recovery_001_summary.json")
    parser.add_argument("--markdown", default="docs/kolektor_late_recovery_001.md")
    parser.add_argument("--svg", default="results/kolektor_late_recovery_001.svg")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--target-pass-precisions", nargs="*", type=float, default=[0.95, 0.98, 0.99])
    parser.add_argument("--max-lost-rates", nargs="*", type=float, default=[0.02, 0.05, 0.10])
    parser.add_argument("--seed", type=int, default=123)
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
    train_loader = make_loader(split["train"], args.batch_size, image_size, train=True)
    val_loader = make_loader(split["val"], args.batch_size, image_size, train=False)
    eval_loader = make_loader(split["eval"], args.batch_size, image_size, train=False)

    model = TinyInspectionBranchyNet().to(device)
    history = train_model(model, train_loader, val_loader, class_weight(split["train"], device), args.epochs, device)
    val_data = collect_outputs(model, val_loader, device)
    eval_data = collect_outputs(model, eval_loader, device)
    rows = []
    for target in args.target_pass_precisions:
        rows.extend(run_rejector_sweep(val_data, eval_data, target, args.max_lost_rates, args.seed))

    payload = {
        "purpose": "Non-CIFAR real industrial inspection validation for lower-side late-recovery predictor.",
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
        "model": {
            "arch": "TinyInspectionBranchyNet",
            "exit_names": model.exit_names,
            "exit_costs": [float(x) for x in model.exit_costs],
            "image_size": [args.image_height, args.image_width],
            "epochs": args.epochs,
            "training": history,
        },
        "eval_final": {
            "accuracy": round_float(float(eval_data["correct"][:, -1].mean())),
            "good_recall": round_float(float(((eval_data["labels"] == 0) & (eval_data["pred"][:, -1] == 0)).sum() / max((eval_data["labels"] == 0).sum(), 1))),
            "defect_recall": round_float(defect_recall_score(eval_data["labels"], eval_data["pred"][:, -1])),
        },
        "definitions": {
            "pass_precision": "precision among samples accepted as good",
            "lost_final_reliable_rate": "fraction of all samples that final-only would reliably pass as good but early rejector rejects",
            "early_reject_rate": "fraction of samples stopped before final execution",
            "avg_cost": "normalized mean compute cost; final-only is 1.0",
        },
        "rows": rows,
        "best_rows_strict_eval": best_rows(rows, strict_eval=True),
        "best_rows_validation_selected": best_rows(rows, strict_eval=False),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    write_svg(payload, Path(args.svg))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "svg": args.svg, "rows": len(rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
