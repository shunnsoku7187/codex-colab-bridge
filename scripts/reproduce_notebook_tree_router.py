import json

import numpy as np

from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW, FLOPS_ROUTER, build_dataset, classifier_for
from src.experiment_paths import RESULTS_DIR, ensure_dirs


FEATURE_NAMES = [
    "Variance (Contrast)",
    "Edge Mean",
    "Edge Density",
    "Color Variance",
    "Extreme Pixels",
    "Total Variation",
    "Center-Surround Diff",
    "Flatness",
]


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
        "to_low": len(to_low),
        "to_high": len(to_high),
        "true_easy_total": easy_total,
        "true_hard_total": hard_total,
        "true_easy_to_low": easy_to_low,
        "true_easy_to_high": easy_to_high,
        "true_hard_to_low": hard_to_low,
        "true_hard_to_high": hard_to_high,
        "true_easy_saved_ratio": float(100 * easy_to_low / max(1, easy_total)),
        "true_hard_kept_on_high_ratio": float(100 * hard_to_high / max(1, hard_total)),
    }


def threshold_search(data, confidences):
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
            }
    return high_accuracy, target_accuracy, best


def main():
    ensure_dirs()
    data, x_values, y_values, _ = build_dataset("lightweight_rf", max_samples=None)

    clf = classifier_for("lightweight_rf")
    print("Training notebook-compatible RandomForest router on the full dataset.", flush=True)
    clf.fit(x_values, y_values)
    confidences = clf.predict_proba(x_values)[:, 1]

    high_accuracy, target_accuracy, best = threshold_search(data, confidences)
    details = route_details(data, confidences, best["threshold"]) if best else None

    importances = np.asarray(clf.feature_importances_, dtype=float)
    output = {
        "status": "ok",
        "source_notebook_cell": 6,
        "source_title": "8 lightweight statistics + decision-tree based router",
        "protocol": "notebook-compatible full-fit evaluation; not cross-validation",
        "model": "RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)",
        "samples": len(data),
        "feature_count": int(x_values.shape[1]),
        "high_accuracy": high_accuracy,
        "target_accuracy": target_accuracy,
        "best": best,
        "routing_details": details,
        "feature_importances": [
            {"name": name, "importance_pct": float(100 * value)}
            for name, value in sorted(zip(FEATURE_NAMES, importances), key=lambda item: item[1], reverse=True)
        ],
        "interpretation": (
            "This reproduces the original notebook's strongest tree-based router setting. "
            "It is useful as the notebook-result target, while strict-CV results should be "
            "reported separately when discussing generalization."
        ),
    }
    output_path = RESULTS_DIR / "notebook_tree_router_reproduction.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
