import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from scripts.prepare_difficulty_labels import load_low_model
from src.experiment_paths import DATA_DIR, RESULTS_DIR, ensure_dirs


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def entropy_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(probs.shape[1])


class ConvExitBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        hidden_channels = min(max(in_channels, 32), 256)
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x):
        return self.branch(x)


class LinearExitBranch(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.branch(x)


class BranchyMobileNetV2(nn.Module):
    def __init__(self, base_model, exit_indices, num_classes=100):
        super().__init__()
        if not hasattr(base_model, "features"):
            raise RuntimeError("Expected a MobileNet-like model with .features.")
        self.features = base_model.features
        self.classifier = base_model.classifier
        self.exit_indices = sorted(set(int(index) for index in exit_indices))
        self.exit_names = [f"exit_after_features_{index}" for index in self.exit_indices] + ["final"]
        self.side_branches = nn.ModuleDict()

        channels = self._infer_exit_channels(num_classes)
        for index in self.exit_indices:
            self.side_branches[str(index)] = ConvExitBranch(channels[index], num_classes)

    def _infer_exit_channels(self, num_classes):
        was_training = self.training
        self.eval()
        device = next(self.features.parameters()).device
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        channels = {}
        with torch.no_grad():
            x = dummy
            for index, layer in enumerate(self.features):
                x = layer(x)
                if index in self.exit_indices:
                    channels[index] = int(x.shape[1])
        if was_training:
            self.train()
        if set(channels) != set(self.exit_indices):
            raise RuntimeError(f"Could not infer all exit channels: {channels.keys()} vs {self.exit_indices}")
        return channels

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.exit_indices:
                outputs.append(self.side_branches[str(index)](x))
        final_logits = self.classifier(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
        outputs.append(final_logits)
        return outputs


def make_datasets(val_size, seed, max_train_samples=None, max_test_samples=None):
    train_base = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=True, download=True, transform=None)
    test_base = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    if max_train_samples is not None:
        train_base = Subset(train_base, range(min(max_train_samples, len(train_base))))
    if max_test_samples is not None:
        test_base = Subset(test_base, range(min(max_test_samples, len(test_base))))
    if val_size > 0:
        train_size = len(train_base) - min(val_size, len(train_base) // 2)
        train_set, val_set = random_split(
            train_base,
            [train_size, len(train_base) - train_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_set = train_base
        val_set = test_base
    return train_set, val_set, test_base


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, collate_fn=custom_collate)


def transform_train():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])


def transform_eval():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])


def weighted_exit_loss(outputs, labels, weights):
    losses = [F.cross_entropy(logits, labels) for logits in outputs]
    weighted = sum(weight * loss for weight, loss in zip(weights, losses))
    return weighted, [float(loss.detach().cpu().item()) for loss in losses]


def train_branchynet(model, loader, transform, device, epochs, learning_rate, weight_decay, loss_weights):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.99, 0.999))
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0
        correct = None
        loss_values = []
        exit_loss_values = []
        for images, labels in tqdm(loader, desc=f"BranchyNet training epoch {epoch + 1}/{epochs}"):
            labels = labels.to(device)
            batch = torch.stack([transform(image) for image in images]).to(device)
            outputs = model(batch)
            loss, exit_losses = weighted_exit_loss(outputs, labels, loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if correct is None:
                correct = np.zeros(len(outputs), dtype=np.int64)
            for idx, logits in enumerate(outputs):
                correct[idx] += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += int(labels.numel())
            loss_values.append(float(loss.detach().cpu().item()))
            exit_loss_values.append(exit_losses)
        avg_exit_losses = np.asarray(exit_loss_values, dtype=np.float32).mean(axis=0).tolist()
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(loss_values)),
            "exit_losses": avg_exit_losses,
            "exit_accuracy": (correct / total).astype(float).tolist(),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        history.append(row)
    return history


@torch.no_grad()
def collect_outputs(model, loader, transform, device, exit_names, costs):
    model.eval()
    labels_all = []
    pred_all = []
    correct_all = []
    entropy_all = []
    confidence_all = []
    for images, labels in tqdm(loader, desc="Collecting BranchyNet outputs"):
        labels = labels.to(device)
        batch = torch.stack([transform(image) for image in images]).to(device)
        outputs = model(batch)
        labels_all.append(labels.detach().cpu().numpy())
        pred_all.append(np.stack([torch.argmax(logits, dim=1).detach().cpu().numpy() for logits in outputs], axis=1))
        correct_all.append(np.stack([(torch.argmax(logits, dim=1) == labels).detach().cpu().numpy() for logits in outputs], axis=1))
        entropy_all.append(np.stack([entropy_from_logits(logits).detach().cpu().numpy() for logits in outputs], axis=1))
        confidence_all.append(np.stack([torch.softmax(logits, dim=1).max(dim=1).values.detach().cpu().numpy() for logits in outputs], axis=1))

    labels_np = np.concatenate(labels_all).astype(np.int16)
    pred_np = np.concatenate(pred_all).astype(np.int16)
    correct_np = np.concatenate(correct_all).astype(bool)
    entropy_np = np.concatenate(entropy_all).astype(np.float32)
    confidence_np = np.concatenate(confidence_all).astype(np.float32)
    return {
        "labels": labels_np,
        "pred": pred_np,
        "correct": correct_np,
        "entropy": entropy_np,
        "confidence": confidence_np,
        "exit_names": np.asarray(exit_names, dtype=object),
        "exit_costs": np.asarray(costs, dtype=np.float32),
    }


def infer_with_thresholds(entropy, pred, correct, costs, thresholds):
    sample_count, exit_count = entropy.shape
    final_idx = exit_count - 1
    chosen_exit = np.full(sample_count, final_idx, dtype=np.int16)
    chosen_pred = pred[:, final_idx].copy()
    chosen_correct = correct[:, final_idx].copy()
    for idx in range(final_idx):
        unresolved = chosen_exit == final_idx
        take = unresolved & (entropy[:, idx] < thresholds[idx])
        chosen_exit[take] = idx
        chosen_pred[take] = pred[take, idx]
        chosen_correct[take] = correct[take, idx]
    energy = costs[chosen_exit]
    counts = Counter(chosen_exit.tolist())
    return {
        "thresholds": [float(x) for x in thresholds],
        "accuracy": float(chosen_correct.mean()),
        "avg_cost": float(energy.mean()),
        "relative_cost": float(energy.mean() / costs[-1]),
        "exit_counts": {str(i): int(counts.get(i, 0)) for i in range(exit_count)},
        "exit_rates": {str(i): float(counts.get(i, 0) / sample_count) for i in range(exit_count)},
    }


def threshold_candidates(entropy_column, count):
    quantiles = np.linspace(0.0, 1.0, num=count)
    values = np.quantile(entropy_column, quantiles)
    values = np.unique(values.astype(np.float32))
    return values.tolist()


def tune_thresholds(val_data, count_per_exit, max_combinations):
    entropy = val_data["entropy"]
    pred = val_data["pred"]
    correct = val_data["correct"]
    costs = val_data["exit_costs"]
    final_idx = entropy.shape[1] - 1
    candidate_lists = [threshold_candidates(entropy[:, idx], count_per_exit) for idx in range(final_idx)]
    combinations = list(itertools.product(*candidate_lists))
    if len(combinations) > max_combinations:
        rng = np.random.default_rng(123)
        keep = rng.choice(len(combinations), size=max_combinations, replace=False)
        combinations = [combinations[int(i)] for i in keep]

    results = [infer_with_thresholds(entropy, pred, correct, costs, combo) for combo in combinations]
    baseline_acc = float(correct[:, -1].mean())
    knee_candidates = [row for row in results if row["accuracy"] >= baseline_acc - 0.01]
    knee = min(knee_candidates, key=lambda row: row["relative_cost"]) if knee_candidates else max(results, key=lambda row: (row["accuracy"], -row["relative_cost"]))
    best_energy = min(results, key=lambda row: row["relative_cost"])
    best_accuracy = max(results, key=lambda row: (row["accuracy"], -row["relative_cost"]))
    frontier = sorted(results, key=lambda row: (-row["accuracy"], row["relative_cost"]))[:40]
    return {
        "baseline_final_accuracy": baseline_acc,
        "candidate_count": len(results),
        "knee_within_one_point": knee,
        "best_energy": best_energy,
        "best_accuracy": best_accuracy,
        "frontier_top40": frontier,
    }


def level_counts(correct):
    levels = np.full(correct.shape[0], correct.shape[1], dtype=np.int16)
    for idx, row in enumerate(correct):
        hits = np.flatnonzero(row)
        if len(hits):
            levels[idx] = int(hits[0])
    counts = Counter(levels.tolist())
    return {str(level): int(counts.get(level, 0)) for level in range(correct.shape[1] + 1)}


def parse_exit_indices(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_loss_weights(text, exit_count):
    if text:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    else:
        values = [1.0 for _ in range(exit_count)]
    if len(values) != exit_count:
        raise ValueError(f"Expected {exit_count} loss weights, got {len(values)}")
    total = sum(values)
    return [value / total for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="branchynet_mobilenet_cifar100_001")
    parser.add_argument("--exit-indices", default="5,12")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-weights", default="")
    parser.add_argument("--threshold-candidates", type=int, default=9)
    parser.add_argument("--max-threshold-combinations", type=int, default=2000)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    train_set, val_set, test_set = make_datasets(args.val_size, seed=123, max_train_samples=args.max_train_samples, max_test_samples=args.max_test_samples)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False)

    print("Loading pretrained baseline MobileNetV2 x0_5...", flush=True)
    base_model = load_low_model(device)
    exit_indices = parse_exit_indices(args.exit_indices)
    model = BranchyMobileNetV2(base_model, exit_indices).to(device)
    exit_count = len(model.exit_names)
    loss_weights = parse_loss_weights(args.loss_weights, exit_count)
    feature_count = len(model.features)
    costs = [(index + 1) / feature_count for index in exit_indices] + [1.0]

    history = train_branchynet(model, train_loader, transform_train(), device, args.epochs, args.learning_rate, args.weight_decay, loss_weights)
    val_data = collect_outputs(model, val_loader, transform_eval(), device, model.exit_names, costs)
    test_data = collect_outputs(model, test_loader, transform_eval(), device, model.exit_names, costs)
    tuning = tune_thresholds(val_data, args.threshold_candidates, args.max_threshold_combinations)
    selected_thresholds = tuning["knee_within_one_point"]["thresholds"]
    test_selected = infer_with_thresholds(test_data["entropy"], test_data["pred"], test_data["correct"], test_data["exit_costs"], selected_thresholds)

    result_npz = RESULTS_DIR / f"{args.output_name}.npz"
    np.savez_compressed(result_npz, **test_data)
    summary = {
        "status": "ok",
        "purpose": "BranchyNet-style early-exit baseline: side branches, joint weighted loss, entropy threshold fast inference.",
        "paper_alignment": [
            "pretrained baseline initialization",
            "side branch classifiers at intermediate layers",
            "joint weighted cross-entropy over all exits",
            "Adam optimizer",
            "entropy-threshold early-exit inference",
            "threshold sweep on validation data",
        ],
        "model": "cifar100_mobilenetv2_x0_5",
        "exit_indices": exit_indices,
        "exit_names": model.exit_names,
        "exit_costs": [float(x) for x in costs],
        "loss_weights": loss_weights,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "train_history": history,
        "validation_tuning": tuning,
        "test_final_only": {
            "accuracy": float(test_data["correct"][:, -1].mean()),
            "avg_cost": 1.0,
            "relative_cost": 1.0,
        },
        "test_branchynet_selected_thresholds": test_selected,
        "test_earliest_correct_level_counts": level_counts(test_data["correct"]),
        "result_npz": str(result_npz),
    }
    output_path = RESULTS_DIR / f"{args.output_name}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
