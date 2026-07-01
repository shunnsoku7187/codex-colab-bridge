import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs


FLOPS_LOW = 0.301
FLOPS_HIGH = 17.6


def load_labels(max_samples):
    data = json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))
    return data[:max_samples] if max_samples else data


def add_low_confidence(data, output_path, batch_size):
    if output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if len(cached) >= len(data):
            print(f"Using cached low-confidence artifact: {output_path}", flush=True)
            return cached[:len(data)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_mobilenetv2_x1_0",
        pretrained=True,
        trust_repo=True,
    )
    model = model.to(device).eval()
    transform_low = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)

    enriched = [dict(item) for item in data]
    with torch.no_grad():
        for start in tqdm(range(0, len(enriched), batch_size), desc="Low confidence inference"):
            batch = enriched[start:start + batch_size]
            images = [transform_low(dataset[item["index"]][0]) for item in batch]
            labels = [dataset[item["index"]][1] for item in batch]
            logits = model(torch.stack(images).to(device))
            probs = torch.nn.functional.softmax(logits, dim=1)
            confs, preds = torch.max(probs, 1)
            for item, label, conf, pred in zip(batch, labels, confs, preds):
                item["real_low_conf"] = float(conf.item())
                item["real_low_correct"] = bool(pred.item() == label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(enriched)} enriched records to {output_path}", flush=True)
    return enriched


def best_threshold(data, mode, target_accuracy, alpha_penalty):
    thresholds = np.linspace(0, 1.0, 201)
    best = None
    for threshold in thresholds:
        to_low = [item for item in data if item["real_low_conf"] >= threshold]
        to_high = [item for item in data if item["real_low_conf"] < threshold]
        if mode == "cascade":
            avg_cost = (len(data) * FLOPS_LOW + len(to_high) * FLOPS_HIGH) / len(data)
        elif mode == "parallel":
            avg_cost = (
                len(data) * FLOPS_LOW
                + len(to_low) * alpha_penalty * FLOPS_HIGH
                + len(to_high) * FLOPS_HIGH
            ) / len(data)
        else:
            raise ValueError(mode)
        correct = sum(item["real_low_correct"] for item in to_low) + sum(item["high_correct"] for item in to_high)
        accuracy = 100 * correct / len(data)
        if accuracy >= target_accuracy and (best is None or avg_cost < best["avg_cost"]):
            best = {
                "mode": mode,
                "threshold": float(threshold),
                "avg_cost": float(avg_cost),
                "accuracy": float(accuracy),
                "to_low": len(to_low),
                "to_high": len(to_high),
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha-penalty", type=float, default=0.10)
    args = parser.parse_args()

    ensure_dirs()
    data = load_labels(args.max_samples)
    enriched_path = ARTIFACT_DIR / "cifar100_low_confidence_x1_0.json"
    data = add_low_confidence(data, enriched_path, args.batch_size)
    high_accuracy = 100 * sum(item["high_correct"] for item in data) / len(data)
    target_accuracy = high_accuracy - 1.0
    results = {
        "samples": len(data),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "cascade": best_threshold(data, "cascade", target_accuracy, args.alpha_penalty),
        "parallel": best_threshold(data, "parallel", target_accuracy, args.alpha_penalty),
    }
    output_path = RESULTS_DIR / "architecture_eval.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
