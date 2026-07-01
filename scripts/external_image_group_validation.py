import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW, FLOPS_ROUTER, build_dataset
from scripts.search_record_breakers import (
    FEATURE_NAMES_LIGHTWEIGHT,
    classifier,
    evaluate_scores,
    extract_hog_features,
    hog_feature_names,
    labels_for,
    regressor,
    route_details,
)
from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, RESULTS_DIR, ensure_dirs
from src.research_features import extract_lightweight_features


class IndexedCIFAR100(Dataset):
    def __init__(self, train, indices):
        self.dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=train, download=True, transform=None)
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        image, label = self.dataset[index]
        return image, label, index


def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    indices = [item[2] for item in batch]
    return images, labels, indices


def load_low_model(device):
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar100_mobilenetv2_x0_5",
        pretrained=True,
        trust_repo=True,
    )
    return model.to(device).eval()


def load_high_model(device):
    model = ViTForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
    return model.to(device).eval()


def sample_indices(train, max_samples, seed):
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=train, download=True, transform=None)
    count = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(len(dataset), size=count, replace=False).tolist())


def label_cache_path(split, max_samples, seed):
    suffix = "full" if max_samples is None else str(max_samples)
    return ARTIFACT_DIR / f"external_{split}_difficulty_labels_{suffix}_seed{seed}.json"


def generate_external_labels(split, max_samples, batch_size, seed, force=False):
    train = split == "train"
    output_path = label_cache_path(split, max_samples, seed)
    if output_path.exists() and not force:
        print(f"Using cached external labels: {output_path}", flush=True)
        return json.loads(output_path.read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = sample_indices(train=train, max_samples=max_samples, seed=seed)
    loader = DataLoader(
        IndexedCIFAR100(train=train, indices=indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )
    transform_low = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_high = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"Generating labels for CIFAR-100 {split} split: {len(indices)} images", flush=True)
    low_model = load_low_model(device)
    high_model = load_high_model(device)

    records = []
    with torch.no_grad():
        for images, labels, batch_indices in tqdm(loader, desc=f"External {split} inference"):
            labels = labels.to(device)
            batch_low = torch.stack([transform_low(image) for image in images]).to(device)
            pred_low = torch.argmax(low_model(batch_low), dim=1)
            batch_high = torch.stack([transform_high(image) for image in images]).to(device)
            pred_high = torch.argmax(high_model(batch_high).logits, dim=1)

            for i, index in enumerate(batch_indices):
                label_val = int(labels[i].item())
                low_val = int(pred_low[i].item())
                high_val = int(pred_high[i].item())
                low_correct = low_val == label_val
                high_correct = high_val == label_val
                if low_correct and high_correct:
                    category = "Easy"
                elif not low_correct and high_correct:
                    category = "Hard"
                elif not low_correct and not high_correct:
                    category = "Impossible"
                else:
                    category = "Inverse"
                records.append({
                    "index": int(index),
                    "label": label_val,
                    "low_pred": low_val,
                    "high_pred": high_val,
                    "low_correct": low_correct,
                    "high_correct": high_correct,
                    "category": category,
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote external labels: {output_path}", flush=True)
    return records


def external_feature_cache_path(split, max_samples, seed, feature_set):
    suffix = "full" if max_samples is None else str(max_samples)
    return ARTIFACT_DIR / f"external_{split}_{feature_set}_features_{suffix}_seed{seed}.npz"


def build_external_feature_sets(records, split, max_samples, seed):
    train = split == "train"
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=train, download=True, transform=None)

    light_path = external_feature_cache_path(split, max_samples, seed, "lightweight")
    if light_path.exists():
        print(f"Loading external lightweight cache: {light_path}", flush=True)
        lightweight_x = np.load(light_path)["x_values"]
    else:
        features = []
        for record in tqdm(records, desc=f"External {split} lightweight features"):
            image, _ = dataset[record["index"]]
            features.append(extract_lightweight_features(image))
        lightweight_x = np.asarray(features, dtype=np.float32)
        np.savez_compressed(light_path, x_values=lightweight_x)
        print(f"Wrote external lightweight cache: {light_path}", flush=True)

    hog_path = external_feature_cache_path(split, max_samples, seed, "hog4x4")
    if hog_path.exists():
        print(f"Loading external HOG cache: {hog_path}", flush=True)
        hog_x = np.load(hog_path)["x_values"]
    else:
        features = []
        for record in tqdm(records, desc=f"External {split} HOG features"):
            image, _ = dataset[record["index"]]
            features.append(extract_hog_features(image, cell_size=4, bins=9))
        hog_x = np.asarray(features, dtype=np.float32)
        np.savez_compressed(hog_path, x_values=hog_x)
        print(f"Wrote external HOG cache: {hog_path}", flush=True)

    hog_names = hog_feature_names(cell_size=4, bins=9)
    return {
        "lightweight": (lightweight_x, FEATURE_NAMES_LIGHTWEIGHT),
        "hog4x4": (hog_x, hog_names),
        "lightweight_plus_hog4x4": (
            np.concatenate([lightweight_x, hog_x], axis=1),
            FEATURE_NAMES_LIGHTWEIGHT + hog_names,
        ),
    }


def source_feature_sets():
    source_data, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    from scripts.search_record_breakers import load_hog_dataset

    feature_sets = {"lightweight": (lightweight_x, FEATURE_NAMES_LIGHTWEIGHT)}
    feature_sets.update(load_hog_dataset(source_data, lightweight_x))
    return source_data, feature_sets


def candidates():
    return [
        ("current_lgbm_cell7_record", "lightweight", classifier(), "low_correct", "classifier"),
        ("current_lgbm_less_constrained", "lightweight", classifier({"n_estimators": 500, "max_depth": -1, "num_leaves": 127, "learning_rate": 0.02}), "low_correct", "classifier"),
        ("current_lgbm_deeper_margin", "lightweight", classifier({"n_estimators": 300, "max_depth": 7, "num_leaves": 127, "learning_rate": 0.025}), "low_correct", "classifier"),
        ("hog4x4_lgbm", "hog4x4", classifier({"n_estimators": 300, "learning_rate": 0.03}), "low_correct", "classifier"),
        ("hog4x4_soft_category_regressor", "hog4x4", regressor({"n_estimators": 300, "learning_rate": 0.03}), "category_soft", "regressor"),
        ("lightweight_hog_lgbm", "lightweight_plus_hog4x4", classifier({"n_estimators": 300, "learning_rate": 0.03}), "low_correct", "classifier"),
        ("lightweight_hog_soft_category_regressor", "lightweight_plus_hog4x4", regressor({"n_estimators": 300, "learning_rate": 0.03}), "category_soft", "regressor"),
    ]


def oracle_report(records):
    total = len(records)
    easy = sum(1 for item in records if item["low_correct"] and item["high_correct"])
    hard = sum(1 for item in records if not item["low_correct"] and item["high_correct"])
    impossible = sum(1 for item in records if not item["low_correct"] and not item["high_correct"])
    inverse = sum(1 for item in records if item["low_correct"] and not item["high_correct"])
    high_accuracy = 100 * (easy + hard) / total
    target_correct = int(np.ceil((high_accuracy - 1.0) * total / 100.0))
    needed_hard_high = max(0, target_correct - easy - inverse)
    hard_low_allowed = max(0, hard - needed_hard_high)
    oracle_low = easy + inverse + impossible + hard_low_allowed
    oracle_high = total - oracle_low
    oracle_cost = (oracle_low * FLOPS_LOW + oracle_high * FLOPS_HIGH) / total
    return {
        "counts": {"Easy": easy, "Hard": hard, "Impossible": impossible, "Inverse": inverse},
        "high_accuracy": high_accuracy,
        "target_accuracy": high_accuracy - 1.0,
        "oracle_low": oracle_low,
        "oracle_high": oracle_high,
        "oracle_low_rate": oracle_low / total,
        "oracle_high_rate": oracle_high / total,
        "oracle_cost": float(oracle_cost),
        "hard_low_allowed_at_target": hard_low_allowed,
    }


def fit_and_external_eval(name, feature_set_name, model, target_name, score_kind, source_data, source_features, external_data, external_features):
    print(f"=== external image-group validation: {name} ===", flush=True)
    train_x, _ = source_features[feature_set_name]
    train_y = labels_for(source_data, target_name)
    test_x, _ = external_features[feature_set_name]
    model.fit(train_x, train_y)
    if score_kind == "classifier":
        scores = model.predict_proba(test_x)[:, 1]
    elif score_kind == "regressor":
        scores = np.clip(model.predict(test_x), 0.0, 1.0)
    else:
        raise ValueError(score_kind)

    high_accuracy, target_accuracy, best = evaluate_scores(external_data, scores)
    result = {
        "name": name,
        "protocol": "train router on original CIFAR-100 test labels; evaluate on separate CIFAR-100 split images",
        "feature_set": feature_set_name,
        "target": target_name,
        "score_kind": score_kind,
        "samples": len(external_data),
        "feature_count": int(test_x.shape[1]),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
        "routing_details": route_details(external_data, scores, best["threshold"]) if best else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force-labels", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    source_data, source_features = source_feature_sets()
    external_data = generate_external_labels(
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        force=args.force_labels,
    )
    external_features = build_external_feature_sets(external_data, args.split, args.max_samples, args.seed)

    results = [
        fit_and_external_eval(name, feature_set_name, model, target_name, score_kind, source_data, source_features, external_data, external_features)
        for name, feature_set_name, model, target_name, score_kind in candidates()
    ]
    ranked = sorted([item for item in results if item["best"] is not None], key=lambda item: item["best"]["avg_cost"])
    summary = {
        "status": "ok",
        "purpose": "External image-group validation for overfitting check.",
        "important_caveat": (
            "CIFAR-100 train split is a different image group from the router source labels, "
            "but it may overlap with the base classifiers' original training data. Treat this as an overfitting stress test, not a final external benchmark."
        ),
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "external_oracle": oracle_report(external_data),
        "results": results,
        "ranked_by_avg_cost": [item["name"] for item in ranked],
    }
    output_path = RESULTS_DIR / f"external_{args.split}_image_group_validation.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
