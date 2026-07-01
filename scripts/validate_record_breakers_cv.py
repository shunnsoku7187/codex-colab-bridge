import json

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from scripts.search_record_breakers import (
    FEATURE_NAMES_LIGHTWEIGHT,
    classifier,
    evaluate_scores,
    labels_for,
    load_hog_dataset,
    regressor,
    route_details,
)
from scripts.evaluate_router import build_dataset
from src.experiment_paths import RESULTS_DIR, ensure_dirs


def category_strata(data):
    mapping = {
        (True, True): 0,    # Easy
        (False, True): 1,   # Hard
        (False, False): 2,  # Impossible
        (True, False): 3,   # Inverse
    }
    return np.asarray([mapping[(item["low_correct"], item["high_correct"])] for item in data], dtype=np.int64)


def candidates():
    return [
        (
            "cv_current_lgbm_less_constrained",
            "lightweight",
            classifier({"n_estimators": 500, "max_depth": -1, "num_leaves": 127, "learning_rate": 0.02}),
            "low_correct",
            "classifier",
        ),
        (
            "cv_current_lgbm_deeper_margin",
            "lightweight",
            classifier({"n_estimators": 300, "max_depth": 7, "num_leaves": 127, "learning_rate": 0.025}),
            "low_correct",
            "classifier",
        ),
        (
            "cv_hog4x4_lgbm",
            "hog4x4",
            classifier({"n_estimators": 300, "learning_rate": 0.03}),
            "low_correct",
            "classifier",
        ),
        (
            "cv_hog4x4_soft_category_regressor",
            "hog4x4",
            regressor({"n_estimators": 300, "learning_rate": 0.03}),
            "category_soft",
            "regressor",
        ),
        (
            "cv_lightweight_hog_lgbm",
            "lightweight_plus_hog4x4",
            classifier({"n_estimators": 300, "learning_rate": 0.03}),
            "low_correct",
            "classifier",
        ),
        (
            "cv_lightweight_hog_soft_category_regressor",
            "lightweight_plus_hog4x4",
            regressor({"n_estimators": 300, "learning_rate": 0.03}),
            "category_soft",
            "regressor",
        ),
    ]


def predict_cv(model, x_values, y_values, strata, score_kind):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = cv.split(np.zeros(len(strata)), strata)
    if score_kind == "classifier":
        return cross_val_predict(model, x_values, y_values, cv=splits, method="predict_proba", n_jobs=-1)[:, 1]
    if score_kind == "regressor":
        scores = cross_val_predict(model, x_values, y_values, cv=splits, method="predict", n_jobs=-1)
        return np.clip(scores, 0.0, 1.0)
    raise ValueError(score_kind)


def run_candidate(name, feature_set_name, model, target_name, score_kind, feature_sets, data, strata):
    print(f"=== strict-CV record-breaker validation: {name} ===", flush=True)
    x_values, _ = feature_sets[feature_set_name]
    y_values = labels_for(data, target_name)
    scores = predict_cv(model, x_values, y_values, strata, score_kind)
    high_accuracy, target_accuracy, best = evaluate_scores(data, scores)
    result = {
        "name": name,
        "protocol": "5-fold stratified CV by difficulty category",
        "score_kind": score_kind,
        "target": target_name,
        "feature_set": feature_set_name,
        "samples": len(data),
        "feature_count": int(x_values.shape[1]),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
        "routing_details": route_details(data, scores, best["threshold"]) if best else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ensure_dirs()
    data, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    feature_sets = {"lightweight": (lightweight_x, FEATURE_NAMES_LIGHTWEIGHT)}
    feature_sets.update(load_hog_dataset(data, lightweight_x))
    strata = category_strata(data)

    results = [
        run_candidate(name, feature_set_name, model, target_name, score_kind, feature_sets, data, strata)
        for name, feature_set_name, model, target_name, score_kind in candidates()
    ]
    ranked = sorted([item for item in results if item["best"] is not None], key=lambda item: item["best"]["avg_cost"])
    summary = {
        "status": "ok",
        "purpose": "Credibility check for record-breaker full-fit results.",
        "oracle_cost_at_target_accuracy": 3.362923,
        "full_fit_best_cost": 3.3715725,
        "results": results,
        "ranked_by_avg_cost": [item["name"] for item in ranked],
    }
    output_path = RESULTS_DIR / "record_breaker_cv_validation.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
