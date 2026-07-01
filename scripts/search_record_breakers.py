import json

import cv2
import lightgbm as lgb
import numpy as np
import torchvision
from tqdm import tqdm

from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW, FLOPS_ROUTER, build_dataset
from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, DIFFICULTY_LABELS_PATH, RESULTS_DIR, ensure_dirs


FEATURE_NAMES_LIGHTWEIGHT = [
    "Variance",
    "Edge Mean",
    "Edge Density",
    "Color Variance",
    "Extreme Pixels",
    "Total Variation",
    "Center-Surround",
    "Flatness",
]


def extract_hog_features(img_pil, cell_size=4, bins=9):
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    bin_index = np.floor(angle / (180.0 / bins)).astype(np.int32)
    bin_index = np.clip(bin_index, 0, bins - 1)

    h, w = gray.shape
    features = []
    for row in range(0, h, cell_size):
        for col in range(0, w, cell_size):
            cell_bins = bin_index[row:row + cell_size, col:col + cell_size]
            cell_mag = magnitude[row:row + cell_size, col:col + cell_size]
            hist = np.bincount(cell_bins.reshape(-1), weights=cell_mag.reshape(-1), minlength=bins)
            norm = np.linalg.norm(hist) + 1e-6
            features.extend((hist / norm).astype(float).tolist())
    return features


def hog_feature_names(cell_size=4, bins=9):
    cells = 32 // cell_size
    return [f"HOG_r{r}_c{c}_b{b}" for r in range(cells) for c in range(cells) for b in range(bins)]


def labels_for(data, target):
    if target == "low_correct":
        return np.asarray([1.0 if item["low_correct"] else 0.0 for item in data], dtype=np.float32)
    if target == "safe_low_binary":
        return np.asarray([1.0 if item["low_correct"] or not item["high_correct"] else 0.0 for item in data], dtype=np.float32)
    if target == "category_soft":
        values = []
        for item in data:
            if item["low_correct"] and item["high_correct"]:
                values.append(0.95)  # Easy
            elif item["low_correct"] and not item["high_correct"]:
                values.append(1.00)  # Inverse
            elif not item["low_correct"] and not item["high_correct"]:
                values.append(0.75)  # Impossible: low is cheaper and no less correct
            else:
                values.append(0.00)  # Hard
        return np.asarray(values, dtype=np.float32)
    if target == "category_soft_conservative":
        values = []
        for item in data:
            if item["low_correct"] and item["high_correct"]:
                values.append(0.85)
            elif item["low_correct"] and not item["high_correct"]:
                values.append(1.00)
            elif not item["low_correct"] and not item["high_correct"]:
                values.append(0.55)
            else:
                values.append(0.00)
        return np.asarray(values, dtype=np.float32)
    raise ValueError(target)


def feature_cache_path(name):
    return ARTIFACT_DIR / f"record_breaker_features_{name}_full.npz"


def load_hog_dataset(data, lightweight_values):
    cache_path = feature_cache_path("hog4x4")
    if cache_path.exists():
        print(f"Loading HOG feature cache: {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        hog_values = cached["x_values"]
        hog_names = cached["feature_names"].tolist()
    else:
        print(f"Extracting HOG features into cache: {cache_path}", flush=True)
        cifar = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
        hog_features = []
        for item in tqdm(data, desc="Extracting HOG 4x4"):
            img_pil, _ = cifar[item["index"]]
            hog_features.append(extract_hog_features(img_pil, cell_size=4, bins=9))
        hog_values = np.asarray(hog_features, dtype=np.float32)
        hog_names = hog_feature_names(cell_size=4, bins=9)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            x_values=hog_values,
            feature_names=np.asarray(hog_names, dtype=object),
        )
        print(f"Wrote HOG feature cache: {cache_path}", flush=True)

    return {
        "hog4x4": (hog_values, hog_names),
        "lightweight_plus_hog4x4": (
            np.concatenate([lightweight_values, hog_values], axis=1),
            FEATURE_NAMES_LIGHTWEIGHT + hog_names,
        ),
    }


def classifier(params=None):
    merged = {
        "n_estimators": 100,
        "max_depth": 6,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
    }
    if params:
        merged.update(params)
    return lgb.LGBMClassifier(**merged)


def regressor(params=None):
    merged = {
        "n_estimators": 200,
        "max_depth": 6,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "random_state": 42,
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
    }
    if params:
        merged.update(params)
    return lgb.LGBMRegressor(**merged)


def evaluate_scores(data, scores):
    scores = np.asarray(scores, dtype=float)
    high_accuracy = 100 * sum(item["high_correct"] for item in data) / len(data)
    target_accuracy = high_accuracy - 1.0
    best = None
    thresholds = np.linspace(float(np.nanmin(scores)), float(np.nanmax(scores)), 301)
    for threshold in thresholds:
        to_low = [item for item, score in zip(data, scores) if score >= threshold]
        to_high = [item for item, score in zip(data, scores) if score < threshold]
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


def route_details(data, scores, threshold):
    to_low = [(item, score) for item, score in zip(data, scores) if score >= threshold]
    to_high = [(item, score) for item, score in zip(data, scores) if score < threshold]
    return {
        "true_easy_to_low": sum(1 for item, _ in to_low if item["low_correct"] and item["high_correct"]),
        "true_inverse_to_low": sum(1 for item, _ in to_low if item["low_correct"] and not item["high_correct"]),
        "true_impossible_to_low": sum(1 for item, _ in to_low if not item["low_correct"] and not item["high_correct"]),
        "true_hard_to_low": sum(1 for item, _ in to_low if not item["low_correct"] and item["high_correct"]),
        "true_easy_to_high": sum(1 for item, _ in to_high if item["low_correct"] and item["high_correct"]),
        "true_hard_to_high": sum(1 for item, _ in to_high if not item["low_correct"] and item["high_correct"]),
    }


def importances_for(model, names, limit=20):
    if not hasattr(model, "feature_importances_"):
        return None
    values = np.asarray(model.feature_importances_, dtype=float)
    if values.sum() > 0:
        values = 100.0 * values / values.sum()
    return [
        {"name": name, "importance_pct": float(value)}
        for name, value in sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:limit]
    ]


def run_candidate(name, model, x_values, feature_names, y_values, data, score_kind):
    print(f"=== record-breaker candidate: {name} ===", flush=True)
    model.fit(x_values, y_values)
    if score_kind == "classifier":
        scores = model.predict_proba(x_values)[:, 1]
    elif score_kind == "regressor":
        scores = np.clip(model.predict(x_values), 0.0, 1.0)
    else:
        raise ValueError(score_kind)
    high_accuracy, target_accuracy, best = evaluate_scores(data, scores)
    result = {
        "name": name,
        "protocol": "notebook-compatible full-fit search; not cross-validation",
        "score_kind": score_kind,
        "samples": len(data),
        "feature_count": int(x_values.shape[1]),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
        "routing_details": route_details(data, scores, best["threshold"]) if best else None,
        "top_feature_importances": importances_for(model, feature_names),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ensure_dirs()
    data, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    feature_sets = {"lightweight": (lightweight_x, FEATURE_NAMES_LIGHTWEIGHT)}
    feature_sets.update(load_hog_dataset(data, lightweight_x))

    candidates = [
        ("current_lgbm_cell7_record", "lightweight", classifier(), "low_correct", "classifier"),
        ("current_lgbm_more_trees", "lightweight", classifier({"n_estimators": 300, "learning_rate": 0.025}), "low_correct", "classifier"),
        ("current_lgbm_deeper_margin", "lightweight", classifier({"n_estimators": 300, "max_depth": 7, "num_leaves": 127, "learning_rate": 0.025}), "low_correct", "classifier"),
        ("current_lgbm_less_constrained", "lightweight", classifier({"n_estimators": 500, "max_depth": -1, "num_leaves": 127, "learning_rate": 0.02}), "low_correct", "classifier"),
        ("soft_safe_binary_lgbm", "lightweight", classifier({"n_estimators": 200, "learning_rate": 0.03}), "safe_low_binary", "classifier"),
        ("soft_category_regressor", "lightweight", regressor(), "category_soft", "regressor"),
        ("soft_category_conservative_regressor", "lightweight", regressor({"min_child_samples": 20}), "category_soft_conservative", "regressor"),
        ("hog4x4_lgbm", "hog4x4", classifier({"n_estimators": 300, "learning_rate": 0.03}), "low_correct", "classifier"),
        ("hog4x4_soft_category_regressor", "hog4x4", regressor({"n_estimators": 300, "learning_rate": 0.03}), "category_soft", "regressor"),
        ("lightweight_hog_lgbm", "lightweight_plus_hog4x4", classifier({"n_estimators": 300, "learning_rate": 0.03}), "low_correct", "classifier"),
        ("lightweight_hog_soft_category_regressor", "lightweight_plus_hog4x4", regressor({"n_estimators": 300, "learning_rate": 0.03}), "category_soft", "regressor"),
    ]

    results = []
    for name, feature_set_name, model, target_name, score_kind in candidates:
        x_values, feature_names = feature_sets[feature_set_name]
        y_values = labels_for(data, target_name)
        result = run_candidate(name, model, x_values, feature_names, y_values, data, score_kind)
        result["feature_set"] = feature_set_name
        result["target"] = target_name
        results.append(result)

    ranked = sorted([item for item in results if item["best"] is not None], key=lambda item: item["best"]["avg_cost"])
    summary = {
        "status": "ok",
        "record_to_beat": {
            "name": "lightweight_lgbm",
            "avg_cost": 11.1146049,
            "accuracy": 88.95,
            "to_low": 3749,
            "easy_saved_ratio": 49.079409697821504,
        },
        "approaches": [
            "strengthen current 8-feature LightGBM",
            "Histogram of Oriented Gradients",
            "LightGBM soft-target learning",
        ],
        "protocol_note": "Notebook-compatible full-fit search; strict-CV should be run later only for finalists.",
        "results": results,
        "ranked_by_avg_cost": [item["name"] for item in ranked],
    }
    output_path = RESULTS_DIR / "record_breaker_search.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
