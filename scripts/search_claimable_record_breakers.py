import json

import cv2
import lightgbm as lgb
import numpy as np
import torchvision
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW, FLOPS_ROUTER, build_dataset
from scripts.routing_guardrails import guardrail_report
from scripts.search_record_breakers import (
    FEATURE_NAMES_LIGHTWEIGHT,
    extract_hog_features,
    hog_feature_names,
    labels_for,
    load_hog_dataset,
)
from src.experiment_paths import ARTIFACT_DIR, DATA_DIR, RESULTS_DIR, ensure_dirs


RECORD_TO_BEAT = 11.1146049


def category_strata(records):
    mapping = {
        (True, True): 0,
        (False, True): 1,
        (False, False): 2,
        (True, False): 3,
    }
    return np.asarray([mapping[(item["low_correct"], item["high_correct"])] for item in records], dtype=np.int64)


def cheap_spectrum_cache_path():
    return ARTIFACT_DIR / "claimable_features_cheap_spectrum_full.npz"


def extract_cheap_spectrum_features(img_pil):
    img = np.asarray(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_norm = gray / 255.0

    features = []
    names = []

    for channel, channel_name in enumerate(["r", "g", "b"]):
        hist, _ = np.histogram(img[:, :, channel], bins=8, range=(0, 256), density=True)
        features.extend(hist.astype(float).tolist())
        names.extend([f"{channel_name}_hist_{i}" for i in range(8)])

    gray_hist, _ = np.histogram(gray, bins=16, range=(0, 256), density=True)
    features.extend(gray_hist.astype(float).tolist())
    names.extend([f"gray_hist_{i}" for i in range(16)])

    dct = cv2.dct(gray_norm)
    for row in range(8):
        for col in range(8):
            if row == 0 and col == 0:
                continue
            features.append(float(abs(dct[row, col])))
            names.append(f"dct_abs_{row}_{col}")

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    orient_bins = np.floor(angle / 20.0).astype(np.int32)
    orient_bins = np.clip(orient_bins, 0, 8)
    orient_hist = np.bincount(orient_bins.reshape(-1), weights=magnitude.reshape(-1), minlength=9)
    orient_hist = orient_hist / (np.linalg.norm(orient_hist) + 1e-6)
    features.extend(orient_hist.astype(float).tolist())
    names.extend([f"grad_orient_{i}" for i in range(9)])

    for grid_size in [2, 4]:
        step = 32 // grid_size
        for row in range(grid_size):
            for col in range(grid_size):
                patch = gray[row * step:(row + 1) * step, col * step:(col + 1) * step]
                patch_mag = magnitude[row * step:(row + 1) * step, col * step:(col + 1) * step]
                prefix = f"g{grid_size}_{row}_{col}"
                features.extend([float(np.mean(patch)), float(np.var(patch)), float(np.mean(patch_mag))])
                names.extend([f"{prefix}_mean", f"{prefix}_var", f"{prefix}_edge"])

    return features, names


def load_cheap_spectrum_dataset(records):
    cache_path = cheap_spectrum_cache_path()
    if cache_path.exists():
        print(f"Loading cheap spectrum feature cache: {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=True)
        return cached["x_values"], cached["feature_names"].tolist()

    print(f"Extracting cheap spectrum features into cache: {cache_path}", flush=True)
    cifar = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=None)
    values = []
    names = None
    for item in records:
        img_pil, _ = cifar[item["index"]]
        feats, feature_names = extract_cheap_spectrum_features(img_pil)
        values.append(feats)
        names = names or feature_names
    x_values = np.asarray(values, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x_values=x_values, feature_names=np.asarray(names, dtype=object))
    print(f"Wrote cheap spectrum feature cache: {cache_path}", flush=True)
    return x_values, names


def classifier(params=None):
    merged = {
        "n_estimators": 300,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        merged.update(params)
    return lgb.LGBMClassifier(**merged)


def regressor(params=None):
    merged = {
        "n_estimators": 300,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        merged.update(params)
    return lgb.LGBMRegressor(**merged)


def sample_weights(records, mode):
    if mode == "none":
        return None
    weights = []
    for item in records:
        low_correct = item["low_correct"]
        high_correct = item["high_correct"]
        if mode == "penalize_hard" and (not low_correct and high_correct):
            weights.append(6.0)
        elif mode == "reward_safe_low" and (low_correct or not high_correct):
            weights.append(2.0)
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=np.float32)


def score_model(model, x_values, score_kind):
    if score_kind == "classifier":
        return model.predict_proba(x_values)[:, 1]
    if score_kind == "regressor":
        return np.clip(model.predict(x_values), 0.0, 1.0)
    raise ValueError(score_kind)


def best_threshold_on_calibration(records, scores, min_high_rate=0.05, min_low_rate=0.05):
    scores = np.asarray(scores, dtype=float)
    total = len(records)
    high_accuracy = 100 * sum(item["high_correct"] for item in records) / total
    target_accuracy = high_accuracy - 1.0
    best = None
    thresholds = np.linspace(float(np.nanmin(scores)), float(np.nanmax(scores)), 301)
    for threshold in thresholds:
        to_low = [item for item, score in zip(records, scores) if score >= threshold]
        to_high = [item for item, score in zip(records, scores) if score < threshold]
        if len(to_low) / total < min_low_rate or len(to_high) / total < min_high_rate:
            continue
        correct = sum(item["low_correct"] for item in to_low) + sum(item["high_correct"] for item in to_high)
        accuracy = 100 * correct / total
        avg_cost = (total * FLOPS_ROUTER + len(to_low) * FLOPS_LOW + len(to_high) * FLOPS_HIGH) / total
        if accuracy >= target_accuracy and (best is None or avg_cost < best["avg_cost"]):
            best = {
                "threshold": float(threshold),
                "avg_cost": float(avg_cost),
                "accuracy": float(accuracy),
                "to_low": len(to_low),
                "to_high": len(to_high),
            }
    return best


def apply_threshold(records, scores, threshold):
    total = len(records)
    to_low_mask = np.asarray(scores) >= threshold
    to_low = int(np.sum(to_low_mask))
    to_high = total - to_low
    correct = 0
    for item, low_branch in zip(records, to_low_mask):
        correct += bool(item["low_correct"]) if low_branch else bool(item["high_correct"])
    return {
        "avg_cost": float((total * FLOPS_ROUTER + to_low * FLOPS_LOW + to_high * FLOPS_HIGH) / total),
        "accuracy": float(100 * correct / total),
        "to_low": to_low,
        "to_high": to_high,
    }


def fit_with_optional_weights(model, x_values, y_values, records, weighting):
    weights = sample_weights(records, weighting)
    if weights is None:
        model.fit(x_values, y_values)
    else:
        model.fit(x_values, y_values, sample_weight=weights)
    return model


def claimable_cv_eval(name, model, feature_set_name, x_values, target_name, score_kind, weighting, records, strata):
    print(f"=== claimable record-breaker search: {name} ===", flush=True)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_outputs = []
    route_decisions = np.zeros(len(records), dtype=bool)

    for fold_id, (train_calib_idx, eval_idx) in enumerate(outer.split(np.zeros(len(strata)), strata), start=1):
        train_calib_strata = strata[train_calib_idx]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=100 + fold_id)
        train_sub, calib_sub = next(splitter.split(np.zeros(len(train_calib_idx)), train_calib_strata))
        train_idx = train_calib_idx[train_sub]
        calib_idx = train_calib_idx[calib_sub]

        fold_model = clone(model)
        train_records = [records[i] for i in train_idx]
        y_train = labels_for(train_records, target_name)
        fit_with_optional_weights(fold_model, x_values[train_idx], y_train, train_records, weighting)

        calib_records = [records[i] for i in calib_idx]
        calib_scores = score_model(fold_model, x_values[calib_idx], score_kind)
        threshold_result = best_threshold_on_calibration(calib_records, calib_scores)
        if threshold_result is None:
            fold_outputs.append({"fold": fold_id, "status": "no_feasible_calibration_threshold"})
            continue

        eval_records = [records[i] for i in eval_idx]
        eval_scores = score_model(fold_model, x_values[eval_idx], score_kind)
        eval_result = apply_threshold(eval_records, eval_scores, threshold_result["threshold"])
        route_decisions[eval_idx] = eval_scores >= threshold_result["threshold"]
        fold_outputs.append({
            "fold": fold_id,
            "status": "ok",
            "calibration_best": threshold_result,
            "eval": eval_result,
        })

    valid_folds = [fold for fold in fold_outputs if fold["status"] == "ok"]
    if len(valid_folds) != outer.get_n_splits():
        overall = None
        guardrails = guardrail_report(records, None, FLOPS_LOW, FLOPS_HIGH, FLOPS_ROUTER)
    else:
        total = len(records)
        to_low = int(np.sum(route_decisions))
        to_high = total - to_low
        correct = 0
        for item, low_branch in zip(records, route_decisions):
            correct += bool(item["low_correct"]) if low_branch else bool(item["high_correct"])
        overall = {
            "avg_cost": float((total * FLOPS_ROUTER + to_low * FLOPS_LOW + to_high * FLOPS_HIGH) / total),
            "accuracy": float(100 * correct / total),
            "to_low": to_low,
            "to_high": to_high,
            "target_accuracy": float(100 * sum(item["high_correct"] for item in records) / total - 1.0),
            "beats_record_to_beat": bool((total * FLOPS_ROUTER + to_low * FLOPS_LOW + to_high * FLOPS_HIGH) / total < RECORD_TO_BEAT),
        }
        guardrails = guardrail_report(records, overall, FLOPS_LOW, FLOPS_HIGH, FLOPS_ROUTER)

    result = {
        "name": name,
        "protocol": "5-fold outer CV; threshold selected only on inner calibration split; evaluation fold never used for fitting or threshold search",
        "feature_set": feature_set_name,
        "target": target_name,
        "score_kind": score_kind,
        "sample_weighting": weighting,
        "samples": len(records),
        "feature_count": int(x_values.shape[1]),
        "overall": overall,
        "guardrails": guardrails,
        "folds": fold_outputs,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ensure_dirs()
    records, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    hog_sets = load_hog_dataset(records, lightweight_x)
    spectrum_x, spectrum_names = load_cheap_spectrum_dataset(records)

    feature_sets = {
        "lightweight": (lightweight_x, FEATURE_NAMES_LIGHTWEIGHT),
        "hog4x4": hog_sets["hog4x4"],
        "lightweight_plus_hog4x4": hog_sets["lightweight_plus_hog4x4"],
        "cheap_spectrum": (spectrum_x, spectrum_names),
        "lightweight_plus_spectrum": (
            np.concatenate([lightweight_x, spectrum_x], axis=1),
            FEATURE_NAMES_LIGHTWEIGHT + spectrum_names,
        ),
    }
    strata = category_strata(records)

    candidates = [
        ("claim_lightweight_lgbm_regularized", "lightweight", classifier(), "low_correct", "classifier", "none"),
        ("claim_lightweight_lgbm_hard_penalty", "lightweight", classifier(), "low_correct", "classifier", "penalize_hard"),
        ("claim_lightweight_safe_low", "lightweight", classifier({"n_estimators": 200}), "safe_low_binary", "classifier", "penalize_hard"),
        ("claim_hog4x4_lgbm_hard_penalty", "hog4x4", classifier(), "low_correct", "classifier", "penalize_hard"),
        ("claim_lightweight_hog_lgbm_hard_penalty", "lightweight_plus_hog4x4", classifier(), "low_correct", "classifier", "penalize_hard"),
        ("claim_spectrum_lgbm_hard_penalty", "cheap_spectrum", classifier(), "low_correct", "classifier", "penalize_hard"),
        ("claim_lightweight_spectrum_lgbm_hard_penalty", "lightweight_plus_spectrum", classifier(), "low_correct", "classifier", "penalize_hard"),
        ("claim_spectrum_soft_category", "cheap_spectrum", regressor(), "category_soft_conservative", "regressor", "none"),
        ("claim_lightweight_spectrum_soft_category", "lightweight_plus_spectrum", regressor(), "category_soft_conservative", "regressor", "none"),
    ]

    results = []
    for name, feature_set_name, model, target_name, score_kind, weighting in candidates:
        x_values, _ = feature_sets[feature_set_name]
        results.append(
            claimable_cv_eval(name, model, feature_set_name, x_values, target_name, score_kind, weighting, records, strata)
        )

    feasible = [
        item for item in results
        if item["overall"] is not None
        and item["overall"]["accuracy"] >= item["overall"]["target_accuracy"]
        and item["guardrails"]["valid_for_claim"]
    ]
    ranked = sorted(feasible, key=lambda item: item["overall"]["avg_cost"])
    summary = {
        "status": "ok",
        "purpose": "Search for approaches that can beat the notebook-compatible 11.1146 GFLOPs record under claimable validation conditions.",
        "record_to_beat": {
            "name": "lightweight_lgbm_notebook_full_fit",
            "avg_cost": RECORD_TO_BEAT,
            "claimability": "reference only; original number used full-fit evaluation",
        },
        "claimable_conditions": [
            "outer evaluation fold is never used for model fitting",
            "outer evaluation fold is never used for threshold selection",
            "threshold is selected on an inner calibration split",
            "both LOW and HIGH branches must receive at least 5% of calibration samples",
            "all-low/all-high escape and degenerate benchmark conditions are rejected by guardrails",
        ],
        "results": results,
        "ranked_claimable_by_avg_cost": [item["name"] for item in ranked],
        "best_claimable": ranked[0] if ranked else None,
    }
    output_path = RESULTS_DIR / "claimable_record_breaker_search.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
