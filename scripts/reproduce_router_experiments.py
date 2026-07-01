import json

from scripts.evaluate_router import build_dataset, classifier_for, evaluate_routing
from src.experiment_paths import RESULTS_DIR, ensure_dirs


MODES = [
    "lightweight_rf",
    "lightweight_lgbm",
    "ultimate_lgbm",
    "robust_lgbm",
    "raw_lgbm",
]


def run_mode(mode):
    import numpy as np
    from sklearn.model_selection import cross_val_predict

    print(f"=== start router mode: {mode} ===", flush=True)
    data, x_values, y_values, feature_names = build_dataset(mode, max_samples=None)
    clf = classifier_for(mode)
    if mode in {"lightweight_rf", "lightweight_lgbm"}:
        print(f"[{mode}] fitting in-sample classifier", flush=True)
        clf.fit(x_values, y_values)
        confidences = clf.predict_proba(x_values)[:, 1]
    else:
        print(f"[{mode}] running 5-fold cross-val prediction", flush=True)
        confidences = cross_val_predict(clf, x_values, y_values, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    print(f"[{mode}] evaluating routing thresholds", flush=True)
    high_accuracy, target_accuracy, best = evaluate_routing(data, confidences)
    result = {
        "mode": mode,
        "samples": len(data),
        "feature_count": int(x_values.shape[1]),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
    }
    if feature_names:
        result["feature_names"] = feature_names
    if hasattr(clf, "feature_importances_") and mode in {"lightweight_rf", "lightweight_lgbm"}:
        names = feature_names or [
            "Variance",
            "Edge Mean",
            "Edge Density",
            "Color Variance",
            "Extreme Pixels",
            "Total Variation",
            "Center-Surround",
            "Flatness",
        ]
        importances = np.asarray(clf.feature_importances_, dtype=float)
        if importances.sum() > 0:
            importances = 100.0 * importances / importances.sum()
        result["feature_importances"] = [
            {"name": name, "importance_pct": float(value)}
            for name, value in sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
        ]
    output_path = RESULTS_DIR / f"router_eval_{mode}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"=== finished router mode: {mode} ===", flush=True)
    return result


def main():
    ensure_dirs()
    results = [run_mode(mode) for mode in MODES]
    summary = {
        "status": "ok",
        "modes": MODES,
        "results": results,
    }
    output_path = RESULTS_DIR / "router_experiments_reproduction.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
