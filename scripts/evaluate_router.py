import argparse
import json

import lightgbm as lgb
import numpy as np
import torchvision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from src.experiment_paths import DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs
from src.research_features import extract_grid_features, extract_lightweight_features, extract_raw_pixel_features


FLOPS_LOW = 0.301
FLOPS_HIGH = 17.6
FLOPS_ROUTER = 0.0


def build_dataset(mode, max_samples):
    data = json.loads(DIFFICULTY_LABELS_PATH.read_text(encoding="utf-8"))
    if max_samples:
        data = data[:max_samples]
    cifar = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    features = []
    feature_names = None
    for item in tqdm(data, desc=f"Extracting {mode} features"):
        img_pil, _ = cifar[item["index"]]
        if mode == "lightweight_rf":
            feats = extract_lightweight_features(img_pil)
        elif mode == "ultimate_lgbm":
            feats, names = extract_grid_features(img_pil, grid_size=4)
            feature_names = feature_names or names
        elif mode == "robust_lgbm":
            feats, names = extract_grid_features(img_pil, grid_size=2)
            feature_names = feature_names or names
        elif mode == "raw_lgbm":
            feats = extract_raw_pixel_features(img_pil)
        else:
            raise ValueError(mode)
        features.append(feats)
    labels = [1 if item["low_correct"] else 0 for item in data]
    return data, np.array(features), np.array(labels), feature_names


def classifier_for(mode):
    if mode == "lightweight_rf":
        return RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    if mode == "ultimate_lgbm":
        return lgb.LGBMClassifier(n_estimators=200, max_depth=6, num_leaves=63, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    if mode == "robust_lgbm":
        return lgb.LGBMClassifier(n_estimators=300, max_depth=6, num_leaves=31, min_child_samples=50, colsample_bytree=0.8, subsample=0.8, learning_rate=0.03, reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1)
    if mode == "raw_lgbm":
        return lgb.LGBMClassifier(n_estimators=500, max_depth=8, num_leaves=127, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    raise ValueError(mode)


def evaluate_routing(data, confidences):
    high_accuracy = 100 * sum(item["high_correct"] for item in data) / len(data)
    target_accuracy = high_accuracy - 1.0
    best = None
    for threshold in np.linspace(0, 1.0, 201):
        to_low = [item for item, conf in zip(data, confidences) if conf >= threshold]
        to_high = [item for item, conf in zip(data, confidences) if conf < threshold]
        avg_cost = (len(data) * FLOPS_ROUTER + len(to_low) * FLOPS_LOW + len(to_high) * FLOPS_HIGH) / len(data)
        correct = sum(item["low_correct"] for item in to_low) + sum(item["high_correct"] for item in to_high)
        accuracy = 100 * correct / len(data)
        if accuracy >= target_accuracy and (best is None or avg_cost < best["avg_cost"]):
            best = {
                "threshold": float(threshold),
                "avg_cost": float(avg_cost),
                "accuracy": float(accuracy),
                "to_low": len(to_low),
                "to_high": len(to_high),
                "easy_saved_ratio": float(100 * sum(item["low_correct"] for item in to_low) / max(1, sum(item["low_correct"] for item in data))),
            }
    return high_accuracy, target_accuracy, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lightweight_rf", "ultimate_lgbm", "robust_lgbm", "raw_lgbm"], required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    data, x_values, y_values, feature_names = build_dataset(args.mode, args.max_samples)
    clf = classifier_for(args.mode)
    if args.mode == "lightweight_rf":
        clf.fit(x_values, y_values)
        confidences = clf.predict_proba(x_values)[:, 1]
    else:
        confidences = cross_val_predict(clf, x_values, y_values, cv=5, method="predict_proba", n_jobs=-1)[:, 1]

    high_accuracy, target_accuracy, best = evaluate_routing(data, confidences)
    output = {
        "mode": args.mode,
        "samples": len(data),
        "feature_count": int(x_values.shape[1]),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
    }
    if feature_names:
        output["feature_names"] = feature_names
    output_path = RESULTS_DIR / f"router_eval_{args.mode}.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
