import json

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW, build_dataset, evaluate_routing
from src.experiment_paths import RESULTS_DIR, ensure_dirs


PARALLEL_ALPHA = 0.10


def candidates():
    return {
        "lightweight_lgbm_cv_original": lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            num_leaves=63,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        "lightweight_lgbm_cv_regularized": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            num_leaves=15,
            min_child_samples=80,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        "lightweight_lgbm_cv_tiny": lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            num_leaves=7,
            min_child_samples=100,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        "lightweight_tree_cv_depth4": DecisionTreeClassifier(max_depth=4, min_samples_leaf=80, random_state=42),
        "lightweight_tree_cv_depth6": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=42),
    }


def same_split_comparison(best, samples):
    if best is None:
        return None
    to_low = best["to_low"]
    to_high = best["to_high"]
    cascade_same_split = FLOPS_LOW + (to_high / samples) * FLOPS_HIGH
    parallel_same_split = (
        FLOPS_LOW
        + (to_low / samples) * PARALLEL_ALPHA * FLOPS_HIGH
        + (to_high / samples) * FLOPS_HIGH
    )
    return {
        "cascade_same_split_cost": float(cascade_same_split),
        "parallel_same_split_cost_alpha_0_10": float(parallel_same_split),
        "router_margin_vs_cascade_same_split": float(cascade_same_split - best["avg_cost"]),
        "router_margin_vs_parallel_same_split": float(parallel_same_split - best["avg_cost"]),
    }


def run_candidate(name, clf, data, x_values, y_values):
    print(f"=== validating pruned candidate: {name} ===", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    confidences = cross_val_predict(clf, x_values, y_values, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    high_accuracy, target_accuracy, best = evaluate_routing(data, confidences)
    result = {
        "name": name,
        "samples": len(data),
        "feature_count": int(x_values.shape[1]),
        "protocol": "5-fold stratified cross_val_predict",
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
        "same_split_comparison": same_split_comparison(best, len(data)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ensure_dirs()
    data, x_values, y_values, _ = build_dataset("lightweight_lgbm", max_samples=None)
    results = [run_candidate(name, clf, data, x_values, y_values) for name, clf in candidates().items()]
    ranked = sorted(
        [item for item in results if item["best"] is not None],
        key=lambda item: item["best"]["avg_cost"],
    )
    summary = {
        "status": "ok",
        "decision_context": "Validate only candidates that remain plausible after pruning grid/raw/RF branches.",
        "results": results,
        "ranked_by_avg_cost": [item["name"] for item in ranked],
    }
    output_path = RESULTS_DIR / "pruned_candidate_validation.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
