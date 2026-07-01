import argparse
import json
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import ViTForImageClassification

from src.experiment_paths import DIFFICULTY_LABELS_PATH, ensure_dirs


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


def load_low_model(device):
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_mobilenetv2_x0_5",
        pretrained=True,
    )
    return model.to(device).eval()


def load_high_model(device):
    model = ViTForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
    return model.to(device).eval()


def run_inference(output_path, batch_size, max_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    transform_low = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_high = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print("Preparing CIFAR-100 test set...", flush=True)
    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    print("Loading Low model...", flush=True)
    low_model = load_low_model(device)
    print("Loading High model...", flush=True)
    high_model = load_high_model(device)

    results = []
    global_idx = 0
    print(f"Starting inference on {len(dataset)} images...", flush=True)
    with torch.no_grad():
        for images, labels in tqdm(loader):
            labels = labels.to(device)

            batch_low = torch.stack([transform_low(img) for img in images]).to(device)
            out_low = low_model(batch_low)
            pred_low = torch.argmax(out_low, dim=1)

            batch_high = torch.stack([transform_high(img) for img in images]).to(device)
            out_high = high_model(batch_high).logits
            pred_high = torch.argmax(out_high, dim=1)

            for i in range(len(labels)):
                label_val = labels[i].item()
                low_val = pred_low[i].item()
                high_val = pred_high[i].item()
                is_low_correct = low_val == label_val
                is_high_correct = high_val == label_val

                if is_low_correct and is_high_correct:
                    category = "Easy"
                elif not is_low_correct and is_high_correct:
                    category = "Hard"
                elif not is_low_correct and not is_high_correct:
                    category = "Impossible"
                else:
                    category = "Inverse"

                results.append({
                    "index": global_idx,
                    "label": label_val,
                    "low_pred": low_val,
                    "high_pred": high_val,
                    "low_correct": is_low_correct,
                    "high_correct": is_high_correct,
                    "category": category,
                })
                global_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} records to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DIFFICULTY_LABELS_PATH))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Using existing difficulty labels: {output_path}", flush=True)
        return
    run_inference(output_path, args.batch_size, args.max_samples)


if __name__ == "__main__":
    main()
