"""Compare BranchyNet deadline exits against standalone small models.

This is a decision experiment for the "deadline-aware" research direction.
If a large BranchyNet truncated at time t is not better than a standalone model
whose worst-case execution fits in t, then deadline-aware BranchyNet is a weak
main theme. Both sides are also evaluated with confidence-based reject, because
reject/reinspection is not unique to BranchyNet.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.experiment_paths import DATA_DIR, RESULTS_DIR, ensure_dirs


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
DEFAULT_BRANCHY_TRACES = {
    "branchy_resnet56": "results/0000b_branchynet_reproduce_resnet56_cifar10.npz",
    "branchy_mobilenet": "results/0001_branchynet_mobilenet_cifar10_retry.npz",
}
SMALL_MODEL_COSTS_VS_RESNET56 = {
    # Rough CIFAR ResNet convolution-depth ratios: (1 + 6n) / 55.
    "resnet20": 19 / 55,
    "resnet32": 31 / 55,
    "resnet44": 43 / 55,
}


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def transform_eval():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def make_eval_loader(batch_size: int, threshold_val_size: int, seed: int) -> DataLoader:
    test_set = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=None)
    val_size = min(threshold_val_size, len(test_set) // 2)
    eval_size = len(test_set) - val_size
    _, eval_set = random_split(test_set, [val_size, eval_size], generator=torch.Generator().manual_seed(seed))
    return DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate)


@torch.no_grad()
def collect_small_model(arch: str, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model_name = f"cifar10_{arch}"
    print(f"Loading pretrained small model: {model_name}", flush=True)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True).to(device)
    model.eval()
    tfm = transform_eval()
    labels_all = []
    pred_all = []
    correct_all = []
    confidence_all = []
    entropy_all = []
    for images, labels in tqdm(loader, desc=f"Collecting {arch} outputs"):
        labels = labels.to(device)
        batch = torch.stack([tfm(image) for image in images]).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(probs.shape[1])
        labels_all.append(labels.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())
        correct_all.append((pred == labels).detach().cpu().numpy())
        confidence_all.append(confidence.detach().cpu().numpy())
        entropy_all.append(entropy.detach().cpu().numpy())
    return {
        "labels": np.concatenate(labels_all).astype(np.int16),
        "pred": np.concatenate(pred_all).astype(np.int16),
        "correct": np.concatenate(correct_all).astype(bool),
        "confidence": np.concatenate(confidence_all).astype(np.float32),
        "entropy": np.concatenate(entropy_all).astype(np.float32),
    }


def reject_curve(correct: np.ndarray, confidence: np.ndarray) -> list[dict[str, Any]]:
    thresholds = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, 21)))
    rows = []
    for threshold in thresholds:
        accept = confidence >= threshold
        if accept.any():
            accepted_accuracy = float(correct[accept].mean())
            false_accept_rate = float((~correct[accept]).mean())
        else:
            accepted_accuracy = None
            false_accept_rate = None
        rows.append({
            "threshold": round(float(threshold), 6),
            "accept_rate": round(float(accept.mean()), 6),
            "reject_rate": round(float((~accept).mean()), 6),
            "accepted_accuracy": None if accepted_accuracy is None else round(accepted_accuracy, 6),
            "false_accept_rate": None if false_accept_rate is None else round(false_accept_rate, 6),
        })
    return rows


def best_under_false_accept(curve: list[dict[str, Any]], targets=(0.01, 0.02, 0.05, 0.10)) -> dict[str, Any]:
    out = {}
    for target in targets:
        valid = [
            row for row in curve
            if row["false_accept_rate"] is not None and row["false_accept_rate"] <= target
        ]
        out[f"false_accept_le_{target:.2f}"] = None if not valid else max(valid, key=lambda row: row["accept_rate"])
    return out


def candidate_summary(name: str, cost: float, correct: np.ndarray, confidence: np.ndarray, kind: str) -> dict[str, Any]:
    curve = reject_curve(correct, confidence)
    return {
        "name": name,
        "kind": kind,
        "estimated_cost": round(float(cost), 6),
        "forced_accuracy": round(float(correct.mean()), 6),
        "forced_false_accept_rate": round(float((~correct).mean()), 6),
        "reject_curve": curve,
        "best_under_false_accept": best_under_false_accept(curve),
    }


def load_branchy_candidates(trace_name: str, trace_path: Path) -> list[dict[str, Any]]:
    data = np.load(trace_path, allow_pickle=True)
    correct = np.asarray(data["correct"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=float)
    costs = np.asarray(data["exit_costs"], dtype=float)
    exit_names = [str(x) for x in data["exit_names"].tolist()]
    candidates = []
    for idx in range(correct.shape[1] - 1):
        candidates.append(candidate_summary(
            name=f"{trace_name}_{exit_names[idx]}",
            cost=float(costs[idx]),
            correct=correct[:, idx],
            confidence=confidence[:, idx],
            kind="branchy_deadline_exit",
        ))
    return candidates


def pairwise_deadline_comparison(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    branchy = [row for row in candidates if row["kind"] == "branchy_deadline_exit"]
    small = [row for row in candidates if row["kind"] == "standalone_small_model"]
    rows = []
    for b in branchy:
        eligible = [s for s in small if s["estimated_cost"] <= b["estimated_cost"] * 1.10]
        if not eligible:
            eligible = sorted(small, key=lambda s: abs(s["estimated_cost"] - b["estimated_cost"]))[:1]
        for s in eligible:
            row = {
                "branchy": b["name"],
                "small": s["name"],
                "branchy_cost": b["estimated_cost"],
                "small_cost": s["estimated_cost"],
                "forced_accuracy_delta_branchy_minus_small": round(b["forced_accuracy"] - s["forced_accuracy"], 6),
                "forced_false_accept_delta_branchy_minus_small": round(b["forced_false_accept_rate"] - s["forced_false_accept_rate"], 6),
                "reject_accept_rate_delta_at_false_accept_le_5pct": None,
                "verdict": "undecided",
            }
            b5 = b["best_under_false_accept"]["false_accept_le_0.05"]
            s5 = s["best_under_false_accept"]["false_accept_le_0.05"]
            if b5 is not None and s5 is not None:
                delta = b5["accept_rate"] - s5["accept_rate"]
                row["reject_accept_rate_delta_at_false_accept_le_5pct"] = round(float(delta), 6)
                if delta > 0.05:
                    row["verdict"] = "branchy_exit_has_meaningful_advantage"
                elif delta < -0.05:
                    row["verdict"] = "small_model_is_better_or_enough"
                else:
                    row["verdict"] = "roughly_tied"
            elif b5 is not None and s5 is None:
                row["verdict"] = "branchy_meets_safety_constraint_small_does_not"
            elif b5 is None and s5 is not None:
                row["verdict"] = "small_meets_safety_constraint_branchy_does_not"
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/deadline_vs_small_model_001_summary.json")
    parser.add_argument("--small-models", nargs="*", default=["resnet20", "resnet32", "resnet44"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold-val-size", type=int, default=5000)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. Select a GPU runtime before running this job.")

    candidates: list[dict[str, Any]] = []
    for trace_name, trace_path_text in DEFAULT_BRANCHY_TRACES.items():
        trace_path = Path(trace_path_text)
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing BranchyNet trace: {trace_path}")
        candidates.extend(load_branchy_candidates(trace_name, trace_path))

    loader = make_eval_loader(args.batch_size, args.threshold_val_size, seed=123)
    small_npz_payload = {}
    for arch in args.small_models:
        outputs = collect_small_model(arch, loader, device)
        cost = SMALL_MODEL_COSTS_VS_RESNET56.get(arch)
        if cost is None:
            raise ValueError(f"No cost estimate registered for {arch}")
        candidates.append(candidate_summary(
            name=f"standalone_{arch}",
            cost=cost,
            correct=outputs["correct"],
            confidence=outputs["confidence"],
            kind="standalone_small_model",
        ))
        for key, value in outputs.items():
            small_npz_payload[f"{arch}_{key}"] = value

    comparison = pairwise_deadline_comparison(candidates)
    payload = {
        "purpose": "Decide whether the deadline route has value beyond a small model that already fits the deadline.",
        "decision_rule": {
            "keep_deadline_route": "Keep only if BranchyNet deadline exits beat same-cost small models under the same confidence-reject false-accept constraint.",
            "cut_deadline_route": "Cut if standalone small models are comparable or better, because reject/reinspection can also be attached to small models.",
        },
        "cost_caveat": "Costs are rough normalized compute estimates. ResNet20/32/44 costs use CIFAR ResNet depth ratios versus ResNet56; cross-architecture comparisons are indicative, not final FPGA measurements.",
        "candidates": candidates,
        "pairwise_deadline_comparison": comparison,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(RESULTS_DIR / "deadline_vs_small_model_001_small_outputs.npz", **small_npz_payload)
    print(json.dumps({
        "purpose": payload["purpose"],
        "decision_rule": payload["decision_rule"],
        "pairwise_deadline_comparison": comparison,
    }, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
