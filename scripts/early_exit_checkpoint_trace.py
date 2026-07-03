import argparse
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.prepare_difficulty_labels import load_low_model
from src.experiment_paths import DATA_DIR, RESULTS_DIR, ensure_dirs


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def choose_checkpoint_modules(model, checkpoint_count):
    if not hasattr(model, "features"):
        raise RuntimeError("Expected the model to expose a .features Sequential module.")
    children = list(model.features.named_children())
    positions = np.linspace(0, len(children) - 1, num=min(checkpoint_count, len(children)), dtype=int)
    checkpoints = []
    seen = set()
    for position in positions.tolist():
        name, module = children[position]
        full_name = f"features.{name}"
        if full_name not in seen:
            checkpoints.append((full_name, module, position + 1, len(children)))
            seen.add(full_name)
    return checkpoints


def safe_name(name):
    return name.replace(".", "_").replace(":", "_")


def pooled_features(tensor):
    if tensor.ndim == 4:
        return F.adaptive_avg_pool2d(tensor, 1).flatten(1)
    if tensor.ndim == 3:
        return tensor.mean(dim=1)
    return tensor.flatten(1)


def output_features(logits):
    probs = torch.softmax(logits, dim=1)
    top_probs, _ = torch.topk(probs, k=2, dim=1)
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(probs.shape[1])
    pred = torch.argmax(logits, dim=1)
    return pred, top_probs[:, 0], margin, entropy


class FrozenBackboneExitHeads:
    def __init__(self, model, checkpoints, device):
        self.model = model
        self.checkpoints = checkpoints
        self.device = device
        self.outputs = {}
        self.handles = []
        for name, module, _units, _total_units in checkpoints:
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            self.outputs[name] = output.detach()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def extract(self, batch):
        self.outputs.clear()
        final_logits = self.model(batch)
        features = {name: pooled_features(self.outputs[name]) for name, _module, _units, _total_units in self.checkpoints}
        return features, final_logits.detach()


def make_loader(train, transform, batch_size, max_samples=None, shuffle=False):
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=train, download=True, transform=None)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate, num_workers=2, pin_memory=True)


def initialize_heads(backbone, loader, transform, device, num_classes):
    images, _labels = next(iter(loader))
    batch = torch.stack([transform(image) for image in images]).to(device)
    features, _final_logits = backbone.extract(batch)
    heads = nn.ModuleDict()
    feature_dims = {}
    for name, values in features.items():
        key = safe_name(name)
        feature_dims[name] = int(values.shape[1])
        heads[key] = nn.Linear(values.shape[1], num_classes)
    return heads.to(device), feature_dims


def train_exit_heads(backbone, heads, train_loader, transform, device, epochs, learning_rate, weight_decay):
    heads.train()
    optimizer = torch.optim.AdamW(heads.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        losses = []
        correct = Counter()
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Training exit heads epoch {epoch + 1}/{epochs}"):
            labels = labels.to(device)
            batch = torch.stack([transform(image) for image in images]).to(device)
            features, _final_logits = backbone.extract(batch)
            loss = 0.0
            for name, values in features.items():
                logits = heads[safe_name(name)](values.detach())
                loss = loss + F.cross_entropy(logits, labels)
                correct[name] += int((torch.argmax(logits, dim=1) == labels).sum().item())
            loss = loss / len(features)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            total += int(labels.numel())
        history.append({
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "checkpoint_accuracy": {name: correct[name] / total for name in correct},
        })
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    return history


def evaluate_trace(backbone, heads, test_loader, transform, device, checkpoints, max_test_samples):
    checkpoint_names = [name for name, _module, _units, _total_units in checkpoints] + ["final"]
    checkpoint_costs = [units / total_units for _name, _module, units, total_units in checkpoints] + [1.0]
    rows = []
    heads.eval()
    index = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating checkpoint trace"):
            labels = labels.to(device)
            batch = torch.stack([transform(image) for image in images]).to(device)
            features, final_logits = backbone.extract(batch)

            batch_preds = []
            batch_correct = []
            batch_confidence = []
            batch_margin = []
            batch_entropy = []
            for name, _module, _units, _total_units in checkpoints:
                logits = heads[safe_name(name)](features[name])
                pred, confidence, margin, entropy = output_features(logits)
                batch_preds.append(pred.cpu().numpy())
                batch_correct.append((pred == labels).cpu().numpy())
                batch_confidence.append(confidence.cpu().numpy())
                batch_margin.append(margin.cpu().numpy())
                batch_entropy.append(entropy.cpu().numpy())

            pred, confidence, margin, entropy = output_features(final_logits)
            batch_preds.append(pred.cpu().numpy())
            batch_correct.append((pred == labels).cpu().numpy())
            batch_confidence.append(confidence.cpu().numpy())
            batch_margin.append(margin.cpu().numpy())
            batch_entropy.append(entropy.cpu().numpy())

            pred_arr = np.stack(batch_preds, axis=1)
            correct_arr = np.stack(batch_correct, axis=1).astype(bool)
            confidence_arr = np.stack(batch_confidence, axis=1)
            margin_arr = np.stack(batch_margin, axis=1)
            entropy_arr = np.stack(batch_entropy, axis=1)

            labels_np = labels.cpu().numpy()
            for row_idx in range(len(labels_np)):
                if max_test_samples is not None and index >= max_test_samples:
                    break
                rows.append({
                    "index": index,
                    "label": int(labels_np[row_idx]),
                    "pred": pred_arr[row_idx],
                    "correct": correct_arr[row_idx],
                    "confidence": confidence_arr[row_idx],
                    "margin": margin_arr[row_idx],
                    "entropy": entropy_arr[row_idx],
                })
                index += 1
            if max_test_samples is not None and index >= max_test_samples:
                break

    return checkpoint_names, np.asarray(checkpoint_costs, dtype=np.float32), rows


def summarize_levels(correct_matrix):
    level = np.full(correct_matrix.shape[0], correct_matrix.shape[1], dtype=np.int16)
    for idx, row in enumerate(correct_matrix):
        correct_indices = np.flatnonzero(row)
        if len(correct_indices):
            level[idx] = int(correct_indices[0])
    counts = Counter(level.tolist())
    return {str(key): int(counts.get(key, 0)) for key in range(correct_matrix.shape[1] + 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="early_exit_checkpoint_trace_001")
    parser.add_argument("--checkpoint-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    print("Loading medium backbone: cifar100_mobilenetv2_x0_5", flush=True)
    model = load_low_model(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    checkpoints = choose_checkpoint_modules(model, args.checkpoint_count)
    print(f"checkpoints={[name for name, _module, _units, _total_units in checkpoints]}", flush=True)

    train_loader = make_loader(True, transform, args.batch_size, args.max_train_samples, shuffle=True)
    test_loader = make_loader(False, transform, args.batch_size, args.max_test_samples, shuffle=False)

    backbone = FrozenBackboneExitHeads(model, checkpoints, device)
    try:
        heads, feature_dims = initialize_heads(backbone, train_loader, transform, device, num_classes=100)
        history = train_exit_heads(backbone, heads, train_loader, transform, device, args.epochs, args.learning_rate, args.weight_decay)
        checkpoint_names, checkpoint_costs, rows = evaluate_trace(backbone, heads, test_loader, transform, device, checkpoints, args.max_test_samples)
    finally:
        backbone.close()

    labels = np.asarray([row["label"] for row in rows], dtype=np.int16)
    pred = np.stack([row["pred"] for row in rows]).astype(np.int16)
    correct = np.stack([row["correct"] for row in rows]).astype(bool)
    confidence = np.stack([row["confidence"] for row in rows]).astype(np.float32)
    margin = np.stack([row["margin"] for row in rows]).astype(np.float32)
    entropy = np.stack([row["entropy"] for row in rows]).astype(np.float32)
    earliest_correct_level_counts = summarize_levels(correct)
    checkpoint_accuracy = correct.mean(axis=0)

    trace_path = Path("artifacts") / f"{args.output_name}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        labels=labels,
        pred=pred,
        correct=correct,
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        checkpoint_names=np.asarray(checkpoint_names, dtype=object),
        checkpoint_costs=checkpoint_costs,
    )

    summary = {
        "status": "ok",
        "purpose": "Build a multi-checkpoint early-exit trace using a frozen medium model and trained linear probe exits.",
        "model": "cifar100_mobilenetv2_x0_5",
        "exit_head": "frozen-backbone linear probe",
        "device": str(device),
        "samples": int(correct.shape[0]),
        "checkpoint_names": checkpoint_names,
        "checkpoint_costs": checkpoint_costs.astype(float).tolist(),
        "checkpoint_accuracy": checkpoint_accuracy.astype(float).tolist(),
        "earliest_correct_level_counts": earliest_correct_level_counts,
        "never_correct_count": int((~correct.any(axis=1)).sum()),
        "feature_dims": feature_dims,
        "train_history": history,
        "trace_file": str(trace_path),
    }
    output_path = RESULTS_DIR / f"{args.output_name}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
