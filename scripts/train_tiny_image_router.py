import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW
from src.experiment_paths import DATA_DIR, DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs


ROUTER_FLOPS_ESTIMATE = 0.003
PARALLEL_ALPHA = 0.10


class DifficultyDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.cifar = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        record = self.records[item]
        image, _ = self.cifar[record["index"]]
        x_value = self.transform(image)
        y_value = torch.tensor(1.0 if record["low_correct"] else 0.0, dtype=torch.float32)
        return x_value, y_value


class TinyImageRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x_value):
        x_value = self.features(x_value).flatten(1)
        return self.classifier(x_value).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(max_samples):
    records = json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))
    return records[:max_samples] if max_samples else records


def train_fold(model, train_loader, device, epochs, learning_rate, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0
        for x_value, y_value in train_loader:
            x_value = x_value.to(device, non_blocking=True)
            y_value = y_value.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_value)
            loss = criterion(logits, y_value)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y_value)
            total_items += len(y_value)
        print(f"epoch={epoch + 1} loss={total_loss / max(1, total_items):.6f}", flush=True)


def predict_fold(model, val_loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x_value, _ in val_loader:
            x_value = x_value.to(device, non_blocking=True)
            logits = model(x_value)
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


def evaluate_routing(records, confidences, router_flops):
    high_accuracy = 100 * sum(item["high_correct"] for item in records) / len(records)
    target_accuracy = high_accuracy - 1.0
    best = None
    for threshold in np.linspace(0, 1.0, 201):
        to_low = [item for item, conf in zip(records, confidences) if conf >= threshold]
        to_high = [item for item, conf in zip(records, confidences) if conf < threshold]
        avg_cost = (len(records) * router_flops + len(to_low) * FLOPS_LOW + len(to_high) * FLOPS_HIGH) / len(records)
        correct = sum(item["low_correct"] for item in to_low) + sum(item["high_correct"] for item in to_high)
        accuracy = 100 * correct / len(records)
        if accuracy >= target_accuracy and (best is None or avg_cost < best["avg_cost"]):
            best = {
                "threshold": float(threshold),
                "avg_cost": float(avg_cost),
                "accuracy": float(accuracy),
                "to_low": len(to_low),
                "to_high": len(to_high),
                "easy_saved_ratio": float(100 * sum(item["low_correct"] for item in to_low) / max(1, sum(item["low_correct"] for item in records))),
            }
    return high_accuracy, target_accuracy, best


def baseline_costs(records, target_accuracy):
    labels = [item["low_correct"] for item in records]
    high_correct = [item["high_correct"] for item in records]
    best = {}
    for rescue_rate in np.linspace(0, 1, 1001):
        # Cost-only break-even reference; actual cascade/parallel quality is measured separately.
        router_cost = rescue_rate * FLOPS_LOW + (1 - rescue_rate) * FLOPS_HIGH + ROUTER_FLOPS_ESTIMATE
        cascade_cost = FLOPS_LOW + (1 - rescue_rate) * FLOPS_HIGH
        parallel_cost = FLOPS_LOW + rescue_rate * PARALLEL_ALPHA * FLOPS_HIGH + (1 - rescue_rate) * FLOPS_HIGH
        best[float(rescue_rate)] = (float(router_cost), float(cascade_cost), float(parallel_cost))
    return {
        "target_accuracy": target_accuracy,
        "oracle_low_correct_rate": float(sum(labels) / len(labels)),
        "oracle_high_correct_rate": float(sum(high_correct) / len(high_correct)),
        "router_flops_estimate": ROUTER_FLOPS_ESTIMATE,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    records = load_records(args.max_samples)
    labels = np.asarray([1 if item["low_correct"] else 0 for item in records], dtype=np.int64)
    dataset = DifficultyDataset(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} samples={len(records)} positives={int(labels.sum())}", flush=True)

    confidences = np.zeros(len(records), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        print(f"=== fold {fold_index}/{args.folds} train={len(train_idx)} val={len(val_idx)} ===", flush=True)
        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        positives = float(labels[train_idx].sum())
        negatives = float(len(train_idx) - positives)
        pos_weight = torch.tensor([negatives / max(1.0, positives)], device=device)
        model = TinyImageRouter().to(device)
        train_fold(model, train_loader, device, args.epochs, args.learning_rate, pos_weight)
        confidences[val_idx] = predict_fold(model, val_loader, device)

    high_accuracy, target_accuracy, best = evaluate_routing(records, confidences, ROUTER_FLOPS_ESTIMATE)
    output = {
        "status": "ok",
        "model": "TinyImageRouter",
        "protocol": f"{args.folds}-fold stratified CV",
        "samples": len(records),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "router_flops_estimate": ROUTER_FLOPS_ESTIMATE,
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
        "references": baseline_costs(records, target_accuracy),
    }
    output_path = RESULTS_DIR / "tiny_image_router_validation.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
