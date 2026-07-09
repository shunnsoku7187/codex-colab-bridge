import argparse
import itertools
import json
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.checkpointing import atomic_write_json, checkpoint_dir, latest_checkpoint, save_torch_checkpoint, utc_now
from src.experiment_paths import DATA_DIR, RESULTS_DIR, ensure_dirs


CIFAR_STATS = {
    "cifar10": {
        "classes": 10,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2470, 0.2435, 0.2616],
    },
    "cifar100": {
        "classes": 100,
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761],
    },
}


DEFAULT_EXITS = {
    "mobilenetv2_x0_5": "features.5,features.12",
    # ResNet-110 for CIFAR has three residual groups. The paper places branches
    # early and around the first third of the network. These module-level exits
    # approximate that placement while remaining robust to the torch.hub module
    # naming used by chenyaofo/pytorch-cifar-models.
    "resnet110": "layer1.0,layer2.0",
}


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def entropy_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1) / math.log(probs.shape[1])


class ConvExitBranch(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_cap=256, conv_depth=1):
        super().__init__()
        hidden_channels = min(max(in_channels, 32), hidden_cap)
        layers = []
        current_channels = in_channels
        for _ in range(max(1, conv_depth)):
            layers.extend([
                nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = hidden_channels
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, num_classes),
        ])
        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        return self.branch(x)


class VectorExitBranch(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        if x.ndim > 2:
            x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.branch(x)


def get_module_by_name(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        candidates = [key for key in modules if key.endswith(name)]
        if len(candidates) == 1:
            return modules[candidates[0]]
        preview = ", ".join(list(modules.keys())[:80])
        raise KeyError(f"Module {name!r} not found. First module names: {preview}")
    return modules[name]


class BranchyHookedModel(nn.Module):
    def __init__(self, base_model, exit_modules, num_classes, branch_depths):
        super().__init__()
        self.base_model = base_model
        self.exit_modules = list(exit_modules)
        self.branch_depths = list(branch_depths)
        self.exit_names = [f"exit_after_{name}" for name in self.exit_modules] + ["final"]
        self.side_branches = nn.ModuleDict()
        self._hook_outputs = {}
        self._handles = []
        self._register_hooks()
        self._init_branches(num_classes)

    def _init_branches(self, num_classes):
        was_training = self.base_model.training
        self.base_model.eval()
        device = next(self.base_model.parameters()).device
        with torch.no_grad():
            self._hook_outputs.clear()
            _ = self.base_model(torch.zeros(2, 3, 32, 32, device=device))
            for name in self.exit_modules:
                tensor = self._hook_outputs.get(name)
                if tensor is None:
                    raise RuntimeError(f"Hook did not capture output for {name}")
                key = self._key(name)
                branch_depth = self.branch_depths[self.exit_modules.index(name)]
                if tensor.ndim == 4:
                    self.side_branches[key] = ConvExitBranch(int(tensor.shape[1]), num_classes, conv_depth=branch_depth)
                else:
                    self.side_branches[key] = VectorExitBranch(int(np.prod(tensor.shape[1:])), num_classes)
        if was_training:
            self.base_model.train()

    def _register_hooks(self):
        for name in self.exit_modules:
            module = get_module_by_name(self.base_model, name)
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            self._hook_outputs[name] = output
        return hook

    @staticmethod
    def _key(name):
        return name.replace(".", "_")

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def forward(self, x):
        self._hook_outputs.clear()
        final_logits = self.base_model(x)
        outputs = []
        for name in self.exit_modules:
            outputs.append(self.side_branches[self._key(name)](self._hook_outputs[name]))
        outputs.append(final_logits)
        return outputs


def load_cifar_model(arch, dataset, device):
    hub_name = f"{dataset}_{arch}"
    print(f"Loading pretrained model from torch.hub: {hub_name}", flush=True)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=True)
    return model.to(device)


def make_dataset(name, train):
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=train, download=True, transform=None)
    if name == "cifar100":
        return torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=train, download=True, transform=None)
    raise ValueError(f"Unknown dataset: {name}")


def make_datasets(dataset, threshold_val_size, seed):
    train_set = make_dataset(dataset, train=True)
    test_set = make_dataset(dataset, train=False)
    val_size = min(threshold_val_size, len(test_set) // 2)
    eval_size = len(test_set) - val_size
    threshold_set, eval_set = random_split(
        test_set,
        [val_size, eval_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_set, threshold_set, eval_set


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, collate_fn=custom_collate)


def transform_train(dataset):
    stats = CIFAR_STATS[dataset]
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])


def transform_eval(dataset):
    stats = CIFAR_STATS[dataset]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])


def weighted_exit_loss(outputs, labels, weights):
    losses = [F.cross_entropy(logits, labels) for logits in outputs]
    weighted = sum(weight * loss for weight, loss in zip(weights, losses))
    return weighted, [float(loss.detach().cpu().item()) for loss in losses]


def train_branchynet(
    model,
    loader,
    transform,
    device,
    epochs,
    learning_rate,
    weight_decay,
    loss_weights,
    checkpoint_path=None,
    output_name=None,
    checkpoint_metadata=None,
    resume=True,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.99, 0.999))
    history = []
    start_epoch = 0

    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        resume_checkpoint = latest_checkpoint(checkpoint_path) if resume else None
        if resume_checkpoint is not None:
            try:
                payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            except TypeError:
                payload = torch.load(resume_checkpoint, map_location=device)
            checkpoint_metadata = checkpoint_metadata or {}
            for key, expected in checkpoint_metadata.items():
                observed = payload.get(key)
                if observed != expected:
                    raise RuntimeError(
                        f"Checkpoint {resume_checkpoint} is incompatible for {key}: "
                        f"expected {expected!r}, observed {observed!r}."
                    )
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            history = payload.get("history", [])
            start_epoch = int(payload.get("epoch", 0))
            print(f"Resuming from checkpoint {resume_checkpoint} at completed_epoch={start_epoch}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0
        correct = None
        losses = []
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
            losses.append(float(loss.detach().cpu().item()))
            exit_loss_values.append(exit_losses)
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "exit_losses": np.asarray(exit_loss_values, dtype=np.float32).mean(axis=0).astype(float).tolist(),
            "exit_accuracy": (correct / total).astype(float).tolist(),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        history.append(row)

        if checkpoint_path is not None:
            checkpoint_payload = {
                "output_name": output_name,
                "epoch": epoch + 1,
                "saved_at": utc_now(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            }
            if checkpoint_metadata:
                checkpoint_payload.update(checkpoint_metadata)
            epoch_path = checkpoint_path / f"epoch_{epoch + 1:03d}.pt"
            latest_path = checkpoint_path / "latest.pt"
            save_torch_checkpoint(epoch_path, checkpoint_payload)
            save_torch_checkpoint(latest_path, checkpoint_payload)
            progress = {
                "output_name": output_name,
                "status": "training",
                "completed_epochs": epoch + 1,
                "target_epochs": epochs,
                "latest_checkpoint": str(latest_path),
                "epoch_checkpoint": str(epoch_path),
                "latest_metrics": row,
                "updated_at": utc_now(),
            }
            atomic_write_json(checkpoint_path / "progress.json", progress)
            atomic_write_json(RESULTS_DIR / f"{output_name}_progress.json", progress)
            print(f"Saved checkpoint: {latest_path}", flush=True)
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
    return {
        "labels": np.concatenate(labels_all).astype(np.int16),
        "pred": np.concatenate(pred_all).astype(np.int16),
        "correct": np.concatenate(correct_all).astype(bool),
        "entropy": np.concatenate(entropy_all).astype(np.float32),
        "confidence": np.concatenate(confidence_all).astype(np.float32),
        "exit_names": np.asarray(exit_names, dtype=object),
        "exit_costs": np.asarray(costs, dtype=np.float32),
    }


def infer_with_thresholds(entropy, pred, correct, costs, thresholds):
    sample_count, exit_count = entropy.shape
    final_idx = exit_count - 1
    chosen_exit = np.full(sample_count, final_idx, dtype=np.int16)
    chosen_correct = correct[:, final_idx].copy()
    for idx in range(final_idx):
        unresolved = chosen_exit == final_idx
        take = unresolved & (entropy[:, idx] < thresholds[idx])
        chosen_exit[take] = idx
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
    values = np.quantile(entropy_column, np.linspace(0.0, 1.0, num=count))
    return np.unique(values.astype(np.float32)).tolist()


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
    return {
        "baseline_final_accuracy": baseline_acc,
        "candidate_count": len(results),
        "knee_within_one_point": knee,
        "best_energy": min(results, key=lambda row: row["relative_cost"]),
        "best_accuracy": max(results, key=lambda row: (row["accuracy"], -row["relative_cost"])),
        "frontier_top40": sorted(results, key=lambda row: (-row["accuracy"], row["relative_cost"]))[:40],
    }


def level_counts(correct):
    levels = np.full(correct.shape[0], correct.shape[1], dtype=np.int16)
    for idx, row in enumerate(correct):
        hits = np.flatnonzero(row)
        if len(hits):
            levels[idx] = int(hits[0])
    counts = Counter(levels.tolist())
    return {str(level): int(counts.get(level, 0)) for level in range(correct.shape[1] + 1)}


def parse_csv_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text, expected):
    if text:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    else:
        values = [1.0 for _ in range(expected)]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)}")
    total = sum(values)
    return [value / total for value in values]


def estimate_costs(arch, exit_modules):
    if arch == "mobilenetv2_x0_5":
        total = 19
        return [(int(module.split(".")[-1]) + 1) / total for module in exit_modules] + [1.0]
    if arch == "resnet110":
        # CIFAR ResNet-110 has 3 groups of 18 residual blocks after the stem.
        # Use rough cumulative block positions for comparative energy theory.
        costs = []
        for module in exit_modules:
            if module.startswith("layer1."):
                block = int(module.split(".")[1])
                costs.append((1 + 2 * (block + 1)) / 109)
            elif module.startswith("layer2."):
                block = int(module.split(".")[1])
                costs.append((1 + 2 * 18 + 2 * (block + 1)) / 109)
            elif module.startswith("layer3."):
                block = int(module.split(".")[1])
                costs.append((1 + 2 * 36 + 2 * (block + 1)) / 109)
            else:
                costs.append(1.0)
        return costs + [1.0]
    return [1.0 for _ in exit_modules] + [1.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--dataset", choices=sorted(CIFAR_STATS), required=True)
    parser.add_argument("--arch", choices=sorted(DEFAULT_EXITS), required=True)
    parser.add_argument("--exit-modules", default="")
    parser.add_argument("--branch-depths", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-val-size", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-weights", default="")
    parser.add_argument("--threshold-candidates", type=int, default=9)
    parser.add_argument("--max-threshold-combinations", type=int, default=2000)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    train_set, val_set, test_set = make_datasets(args.dataset, args.threshold_val_size, seed=123)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False)

    base_model = load_cifar_model(args.arch, args.dataset, device)
    exit_modules = [item.strip() for item in (args.exit_modules or DEFAULT_EXITS[args.arch]).split(",") if item.strip()]
    branch_depths = parse_csv_ints(args.branch_depths) if args.branch_depths else [1 for _ in exit_modules]
    if len(branch_depths) != len(exit_modules):
        raise ValueError(f"Expected {len(exit_modules)} branch depths, got {len(branch_depths)}")
    model = BranchyHookedModel(base_model, exit_modules, CIFAR_STATS[args.dataset]["classes"], branch_depths).to(device)
    loss_weights = parse_csv_floats(args.loss_weights, len(model.exit_names))
    costs = estimate_costs(args.arch, exit_modules)
    print(f"exit_modules={exit_modules}", flush=True)
    print(f"branch_depths={branch_depths}", flush=True)
    print(f"exit_names={model.exit_names}", flush=True)
    print(f"costs={costs}", flush=True)

    checkpoint_path = checkpoint_dir(args.output_name)
    checkpoint_metadata = {
        "dataset": args.dataset,
        "arch": args.arch,
        "exit_modules": exit_modules,
        "branch_depths": branch_depths,
        "loss_weights": loss_weights,
    }
    history = train_branchynet(
        model,
        train_loader,
        transform_train(args.dataset),
        device,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        loss_weights,
        checkpoint_path=checkpoint_path,
        output_name=args.output_name,
        checkpoint_metadata=checkpoint_metadata,
        resume=not args.no_resume,
    )
    val_data = collect_outputs(model, val_loader, transform_eval(args.dataset), device, model.exit_names, costs)
    test_data = collect_outputs(model, test_loader, transform_eval(args.dataset), device, model.exit_names, costs)
    tuning = tune_thresholds(val_data, args.threshold_candidates, args.max_threshold_combinations)
    selected = infer_with_thresholds(test_data["entropy"], test_data["pred"], test_data["correct"], test_data["exit_costs"], tuning["knee_within_one_point"]["thresholds"])

    result_npz = RESULTS_DIR / f"{args.output_name}.npz"
    np.savez_compressed(result_npz, **test_data)
    summary = {
        "status": "ok",
        "purpose": "BranchyNet sweep with paper-aligned side branches, joint weighted loss, Adam, entropy threshold tuning.",
        "paper_reference": "Teerapittayanon, McDanel, Kung, BranchyNet, arXiv:1709.01686",
        "dataset": args.dataset,
        "arch": args.arch,
        "exit_modules": exit_modules,
        "branch_depths": branch_depths,
        "exit_names": model.exit_names,
        "exit_costs": [float(value) for value in costs],
        "loss_weights": loss_weights,
        "train_samples": len(train_set),
        "threshold_val_samples": len(val_set),
        "test_samples": len(test_set),
        "threshold_protocol": "Train side branches on CIFAR train; tune entropy thresholds on a held-out split of CIFAR test; evaluate on remaining CIFAR test split.",
        "train_history": history,
        "checkpoint_dir": str(checkpoint_path),
        "resume_enabled": not args.no_resume,
        "validation_tuning": tuning,
        "test_final_only": {
            "accuracy": float(test_data["correct"][:, -1].mean()),
            "avg_cost": 1.0,
            "relative_cost": 1.0,
        },
        "test_branchynet_selected_thresholds": selected,
        "test_earliest_correct_level_counts": level_counts(test_data["correct"]),
        "result_npz": str(result_npz),
    }
    output_path = RESULTS_DIR / f"{args.output_name}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atomic_write_json(
        checkpoint_path / "progress.json",
        {
            "output_name": args.output_name,
            "status": "complete",
            "completed_epochs": args.epochs,
            "target_epochs": args.epochs,
            "summary": str(output_path),
            "result_npz": str(result_npz),
            "updated_at": utc_now(),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    model.close()


if __name__ == "__main__":
    main()
