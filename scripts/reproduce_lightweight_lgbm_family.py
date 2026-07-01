import json

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scripts.evaluate_router import build_dataset, evaluate_routing
from src.experiment_paths import RESULTS_DIR, ensure_dirs


FEATURE_NAMES = [
    "Variance",
    "Edge Mean",
    "Edge Density",
    "Color Variance",
    "Extreme Pixels",
    "Total Variation",
    "Center-Surround",
    "Flatness",
]


def candidates():
    return {
        "lightweight_lgbm_notebook_cell7": lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            num_leaves=63,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
        ),
        "lightweight_lgbm_200trees": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            num_leaves=63,
            learning_rate=0.03,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
        ),
        "lightweight_lgbm_depth5": lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            num_leaves=31,
            learning_rate=0.04,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
        ),
        "lightweight_lgbm_depth4_regularized": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            learning_rate=0.03,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
        ),
        "lightweight_rf_notebook_cell6_reference": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
        ),
    }


def route_details(data, confidences, threshold):
    to_low = [(item, conf) for item, conf in zip(data, confidences) if conf >= threshold]
    to_high = [(item, conf) for item, conf in zip(data, confidences) if conf < threshold]
    easy_total = sum(1 for item in data if item["low_correct"])
    hard_total = len(data) - easy_total
    easy_to_low = sum(1 for item, _ in to_low if item["low_correct"])
    easy_to_high = sum(1 for item, _ in to_high if item["low_correct"])
    hard_to_low = sum(1 for item, _ in to_low if not item["low_correct"])
    hard_to_high = sum(1 for item, _ in to_high if not item["low_correct"])
    return {
        "true_easy_total": easy_total,
        "true_hard_total": hard_total,
        "true_easy_to_low": easy_to_low,
        "true_easy_to_high": easy_to_high,
        "true_hard_to_low": hard_to_low,
        "true_hard_to_high": hard_to_high,
        "true_easy_saved_ratio": float(100 * easy_to_low / max(1, easy_total)),
        "true_hard_kept_on_high_ratio": float(100 * hard_to_high / max(1, hard_total)),
    }


def importances_for(clf):
    if not hasattr(clf, "feature_importances_"):
        return None
    values = np.asarray(clf.feature_importances_, dtype=float)
    if values.sum() > 0:
        values = 100.0 * values / values.sum()
    return [
        {"name": name, "importance_pct": float(value)}
        for name, value in sorted(zip(FEATURE_NAMES, values), key=lambda item: item[1], reverse=True)
    ]


def run_candidate(name, clf, data, x_values, y_values):
    print(f"=== notebook-compatible lightweight candidate: {name} ===", flush=True)
    clf.fit(x_values, y_values)
    confidences = clf.predict_proba(x_values)[:, 1]
    high_accuracy, target_accuracy, best = evaluate_routing(data, confidences)
    result = {
        "name": name,
        "protocol": "notebook-compatible full-fit evaluation; not cross-validation",
        "samples": len(data),
        "feature_count": int(x_values.shape[1]),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
        "routing_details": route_details(data, confidences, best["threshold"]) if best else None,
        "feature_importances": importances_for(clf),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ensure_dirs()
    data, x_values, y_values, _ = build_dataset("lightweight_lgbm", max_samples=None)
    results = [run_candidate(name, clf, data, x_values, y_values) for name, clf in candidates().items()]
    ranked = sorted(
        [result for result in results if result["best"] is not None],
        key=lambda result: result["best"]["avg_cost"],
    )
    summary = {
        "status": "ok",
        "current_candidate_family": "8 lightweight FPGA-streamable statistics + LightGBM/tree ensemble",
        "primary_candidate": "lightweight_lgbm_notebook_cell7",
        "protocol_note": (
            "These are original-notebook-compatible full-fit results. "
            "Use them to reproduce and extend the notebook's strongest direction; "
            "strict-CV results remain separate evidence."
        ),
        "results": results,
        "ranked_by_avg_cost": [result["name"] for result in ranked],
    }
    output_path = RESULTS_DIR / "lightweight_lgbm_family_reproduction.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
