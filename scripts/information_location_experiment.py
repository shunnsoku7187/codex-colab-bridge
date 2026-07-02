import argparse
import json

import cv2
import lightgbm as lgb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.difficulty_mechanism_decomposition import category_of, load_allowed_feature_space
from scripts.evaluate_architectures import add_low_confidence
from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW, FLOPS_ROUTER, build_dataset
from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, RESULTS_DIR, ensure_dirs


TARGET_K = {
    "current_strict_best_low_1389": 1389,
    "notebook_record_low_3749": 3749,
    "cascade_like_low_5248": 5248,
}
TARGET_MARGIN_PCT = 1.5


def record_category_counts(records):
    counts = {"Easy": 0, "Hard": 0, "Impossible": 0, "Inverse": 0}
    for record in records:
        counts[category_of(record)] += 1
    return counts


def strata_for(records):
    mapping = {"Easy": 0, "Hard": 1, "Impossible": 2, "Inverse": 3}
    return np.asarray([mapping[category_of(record)] for record in records], dtype=np.int64)


def safe_labels_for(records):
    return np.asarray([1 if category_of(record) != "Hard" else 0 for record in records], dtype=np.int64)


def class_labels_for(records):
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    labels = []
    for record in records:
        label = record.get("label")
        if label is None:
            _, label = dataset[int(record["index"])]
        labels.append(int(label))
    return np.asarray(labels, dtype=np.int64), dataset.classes


def lowres_pixel_cache_path(size):
    return ARTIFACT_DIR / f"info_location_lowres_rgb_{size}x{size}.npz"


def load_lowres_pixels(records, size=16):
    path = lowres_pixel_cache_path(size)
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        return cached["x_values"], cached["feature_names"].tolist()

    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    x_values = []
    for record in records:
        image, _ = dataset[int(record["index"])]
        resized = cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA)
        x_values.append((resized.astype(np.float32) / 255.0).reshape(-1))
    x_values = np.asarray(x_values, dtype=np.float32)
    names = [f"rgb{size}_{i}" for i in range(x_values.shape[1])]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x_values=x_values, feature_names=np.asarray(names, dtype=object))
    return x_values, names


def embedding_cache_path(model_name):
    return ARTIFACT_DIR / f"info_location_embedding_{model_name}.npz"


def load_resnet18_embeddings(records, batch_size=128):
    path = embedding_cache_path("resnet18_imagenet")
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        return cached["x_values"], cached["feature_names"].tolist(), None

    try:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
    except Exception as exc:
        return None, None, f"failed_to_load_resnet18_weights: {exc}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dataset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = [preprocess(dataset[int(record["index"])][0]) for record in batch]
            output = model(torch.stack(images).to(device)).detach().cpu().numpy()
            embeddings.append(output.astype(np.float32))
    x_values = np.concatenate(embeddings, axis=0)
    names = [f"resnet18_{i}" for i in range(x_values.shape[1])]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x_values=x_values, feature_names=np.asarray(names, dtype=object))
    return x_values, names, None


def model_for(feature_count):
    return lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.75 if feature_count > 64 else 1.0,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def oof_model_scores(x_values, labels, strata):
    scores = np.zeros(len(labels), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = model_for(x_values.shape[1])
    fold_rows = []
    for fold, (train_idx, eval_idx) in enumerate(splitter.split(np.zeros(len(strata)), strata), start=1):
        model = clone(base_model)
        model.fit(x_values[train_idx], labels[train_idx])
        fold_scores = model.predict_proba(x_values[eval_idx])[:, 1]
        scores[eval_idx] = fold_scores
        fold_rows.append({
            "fold": fold,
            "train_positive_rate": float(np.mean(labels[train_idx])),
            "eval_positive_rate": float(np.mean(labels[eval_idx])),
        })
    return scores, fold_rows


def oof_class_prior_scores(class_labels, labels, strata, smoothing=1.0):
    scores = np.zeros(len(labels), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    global_prior = float(np.mean(labels))
    fold_rows = []
    for fold, (train_idx, eval_idx) in enumerate(splitter.split(np.zeros(len(strata)), strata), start=1):
        train_classes = class_labels[train_idx]
        train_labels = labels[train_idx]
        per_class = {}
        for cls, label in zip(train_classes, train_labels):
            row = per_class.setdefault(int(cls), [0.0, 0.0])
            row[0] += float(label)
            row[1] += 1.0
        for idx in eval_idx:
            pos, total = per_class.get(int(class_labels[idx]), [global_prior * smoothing, 0.0])
            scores[idx] = (pos + global_prior * smoothing) / (total + smoothing)
        fold_rows.append({"fold": fold, "known_classes": len(per_class)})
    return scores, fold_rows


def accuracy_from_route(records, route_low):
    correct = 0
    for record, low_branch in zip(records, route_low):
        correct += bool(record["low_correct"]) if low_branch else bool(record["high_correct"])
    return 100.0 * correct / len(records)


def purity_at_k(records, scores):
    high_accuracy = 100.0 * sum(record["high_correct"] for record in records) / len(records)
    order = np.argsort(-np.asarray(scores, dtype=float))
    rows = []
    for name, k in TARGET_K.items():
        selected = order[:k]
        route_low = np.zeros(len(records), dtype=bool)
        route_low[selected] = True
        counts = {"Easy": 0, "Hard": 0, "Impossible": 0, "Inverse": 0}
        for idx in selected:
            counts[category_of(records[int(idx)])] += 1
        hard_minus_inverse = counts["Hard"] - counts["Inverse"]
        accuracy = accuracy_from_route(records, route_low)
        rows.append({
            "target": name,
            "to_low": int(k),
            "counts_in_low": counts,
            "hard_to_low_minus_inverse_to_low": int(hard_minus_inverse),
            "allowed_hard_minus_inverse_at_1p5pct": int(len(records) * TARGET_MARGIN_PCT / 100.0),
            "meets_1p5pct_margin": bool(hard_minus_inverse <= len(records) * TARGET_MARGIN_PCT / 100.0),
            "accuracy": float(accuracy),
            "accuracy_drop_from_high": float(high_accuracy - accuracy),
            "avg_cost": float((len(records) * FLOPS_ROUTER + k * FLOPS_LOW + (len(records) - k) * FLOPS_HIGH) / len(records)),
        })
    return rows


def summarize_scores(name, observation_level, records, labels, scores, fold_rows=None, notes=None):
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = None
    result = {
        "name": name,
        "observation_level": observation_level,
        "safe_vs_hard_auc": auc,
        "purity_at_k": purity_at_k(records, scores),
    }
    if fold_rows is not None:
        result["folds"] = fold_rows
    if notes:
        result["notes"] = notes
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-embedding", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    ensure_dirs()
    records, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    records = add_low_confidence(records, ARTIFACT_DIR / "cifar100_low_confidence_x1_0.json", batch_size=args.batch_size)
    labels = safe_labels_for(records)
    strata = strata_for(records)
    class_labels, class_names = class_labels_for(records)
    category_counts = record_category_counts(records)

    results = []

    feature_sets = load_allowed_feature_space(records, lightweight_x)
    allowed_x, _ = feature_sets["allowed_full"]
    allowed_scores, allowed_folds = oof_model_scores(allowed_x, labels, strata)
    results.append(summarize_scores(
        "allowed_full_streaming_statistics",
        "resource_bounded_streaming_statistics",
        records,
        labels,
        allowed_scores,
        allowed_folds,
        "Claimable-ish statistical observation class; no model logits/confidence at runtime.",
    ))

    class_scores, class_folds = oof_class_prior_scores(class_labels, labels, strata)
    results.append(summarize_scores(
        "true_class_label_prior",
        "semantic_oracle",
        records,
        labels,
        class_scores,
        class_folds,
        "Uses true CIFAR-100 class label as a semantic information upper-bound; not available to the runtime router.",
    ))

    low_conf_scores = np.asarray([record["real_low_conf"] for record in records], dtype=np.float32)
    results.append(summarize_scores(
        "low_model_confidence",
        "model_internal_signal",
        records,
        labels,
        low_conf_scores,
        None,
        "Cascade-like positive control. It is not allowed for a zero-latency pre-router because LOW has already run.",
    ))

    lowres_x, _ = load_lowres_pixels(records, size=16)
    lowres_scores, lowres_folds = oof_model_scores(lowres_x, labels, strata)
    results.append(summarize_scores(
        "lowres_16x16_raw_rgb",
        "image_content_upper_probe",
        records,
        labels,
        lowres_scores,
        lowres_folds,
        "Raw low-resolution image probe. Not a cheap streaming statistic; tests whether image content itself carries signal.",
    ))

    if args.include_embedding:
        embedding_x, _, embedding_error = load_resnet18_embeddings(records, batch_size=args.batch_size)
        if embedding_error:
            results.append({
                "name": "resnet18_imagenet_embedding",
                "observation_level": "pretrained_semantic_embedding",
                "status": "skipped",
                "error": embedding_error,
            })
        else:
            embedding_scores, embedding_folds = oof_model_scores(embedding_x, labels, strata)
            results.append(summarize_scores(
                "resnet18_imagenet_embedding",
                "pretrained_semantic_embedding",
                records,
                labels,
                embedding_scores,
                embedding_folds,
                "High-level pretrained representation probe; not runtime-claimable, used to locate information.",
            ))

    summary = {
        "status": "ok",
        "purpose": "Information-location experiment: test whether LOW/HIGH routing information is present in streaming statistics, semantic labels, model-internal signals, or richer image representations.",
        "samples": len(records),
        "category_counts": category_counts,
        "target_definition": "safe=not Hard; Hard means LOW wrong and HIGH correct.",
        "interpretation_rule": {
            "statistics_weak_semantic_or_model_strong": "The issue is likely information level: cheap statistics do not expose the needed semantic/model-capacity signal.",
            "statistics_and_positive_controls_weak": "The LOW/HIGH difference may be intrinsically difficult to predict before running the models.",
            "statistics_strong": "A zero-latency statistical router remains plausible.",
        },
        "results": results,
    }
    path = RESULTS_DIR / "information_location_experiment.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
