import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiment_paths import ARTIFACT_DIR, RESULTS_DIR, ensure_dirs


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def add_prefixed_scalars(features, names, prefix, values):
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            features.append(float(value))
            names.append(f"{prefix}.{key}")


def feature_matrix(rows, group):
    names = []
    matrix = []
    for row in rows:
        features = []
        row_names = []
        if group in ("low_output_runtime", "low_output_all"):
            low = row["low"]
            runtime = {
                "top1_prob": low["top1_prob"],
                "top2_prob": low["top2_prob"],
                "top1_top2_logit_margin": low["top1_top2_logit_margin"],
                "entropy_norm": low["entropy_norm"],
            }
            add_prefixed_scalars(features, row_names, "low", runtime)
        if group == "low_output_all":
            low = row["low"]
            oracle = {
                "true_rank": low["true_rank"],
                "true_prob": low["true_prob"],
                "true_logit": low["true_logit"],
                "pred_logit": low["pred_logit"],
            }
            add_prefixed_scalars(features, row_names, "low_oracle", oracle)
        if group == "high_output_runtime":
            high = row["high"]
            runtime = {
                "top1_prob": high["top1_prob"],
                "top2_prob": high["top2_prob"],
                "top1_top2_logit_margin": high["top1_top2_logit_margin"],
                "entropy_norm": high["entropy_norm"],
            }
            add_prefixed_scalars(features, row_names, "high", runtime)
        if group.startswith("low_layer:"):
            layer = group.split(":", 1)[1]
            add_prefixed_scalars(features, row_names, f"low_layers.{layer}", row["low_layers"].get(layer, {}))
        if group.startswith("high_layer:"):
            layer = group.split(":", 1)[1]
            add_prefixed_scalars(features, row_names, f"high_layers.{layer}", row["high_layers"].get(layer, {}))
        if group == "all_low_layers":
            for layer in sorted(row["low_layers"]):
                add_prefixed_scalars(features, row_names, f"low_layers.{layer}", row["low_layers"][layer])
        if group == "all_high_layers":
            for layer in sorted(row["high_layers"]):
                add_prefixed_scalars(features, row_names, f"high_layers.{layer}", row["high_layers"][layer])
        if group == "all_trace_runtime":
            low = row["low"]
            high = row["high"]
            add_prefixed_scalars(features, row_names, "low", {
                "top1_prob": low["top1_prob"],
                "top2_prob": low["top2_prob"],
                "top1_top2_logit_margin": low["top1_top2_logit_margin"],
                "entropy_norm": low["entropy_norm"],
            })
            add_prefixed_scalars(features, row_names, "high", {
                "top1_prob": high["top1_prob"],
                "top2_prob": high["top2_prob"],
                "top1_top2_logit_margin": high["top1_top2_logit_margin"],
                "entropy_norm": high["entropy_norm"],
            })
            for layer in sorted(row["low_layers"]):
                add_prefixed_scalars(features, row_names, f"low_layers.{layer}", row["low_layers"][layer])
            for layer in sorted(row["high_layers"]):
                add_prefixed_scalars(features, row_names, f"high_layers.{layer}", row["high_layers"][layer])
        if not names:
            names = row_names
        matrix.append(features)
    return np.asarray(matrix, dtype=np.float32), names


def oof_binary_scores(x_values, labels, strata):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    results = {}
    for model_name, model in models.items():
        scores = np.zeros(len(labels), dtype=np.float32)
        preds = np.zeros(len(labels), dtype=np.int64)
        for train_idx, eval_idx in splitter.split(np.zeros(len(strata)), strata):
            model.fit(x_values[train_idx], labels[train_idx])
            scores[eval_idx] = model.predict_proba(x_values[eval_idx])[:, 1]
            preds[eval_idx] = model.predict(x_values[eval_idx])
        results[model_name] = {
            "hard_auc": float(roc_auc_score(labels, scores)),
            "hard_accuracy": float(accuracy_score(labels, preds)),
        }
    return results


def group_summaries(rows):
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    result = {}
    for category, items in sorted(by_category.items()):
        result[category] = {
            "count": len(items),
            "low_margin_mean": float(np.mean([item["low"]["top1_top2_logit_margin"] for item in items])),
            "low_entropy_mean": float(np.mean([item["low"]["entropy_norm"] for item in items])),
            "low_top1_prob_mean": float(np.mean([item["low"]["top1_prob"] for item in items])),
            "high_margin_mean": float(np.mean([item["high"]["top1_top2_logit_margin"] for item in items])),
            "high_entropy_mean": float(np.mean([item["high"]["entropy_norm"] for item in items])),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", default="model_process_trace_balanced_002")
    parser.add_argument("--output-name", default="model_process_trace_balanced_002_analysis.json")
    args = parser.parse_args()

    ensure_dirs()
    rows_path = ARTIFACT_DIR / "model_process_traces" / f"{args.input_prefix}_rows.jsonl"
    rows = load_rows(rows_path)
    categories = [row["category"] for row in rows]
    labels = np.asarray([1 if category == "Hard" else 0 for category in categories], dtype=np.int64)
    strata = np.asarray(categories)

    first = rows[0]
    groups = [
        "low_output_runtime",
        "low_output_all",
        "high_output_runtime",
        "all_low_layers",
        "all_high_layers",
        "all_trace_runtime",
    ]
    groups.extend([f"low_layer:{layer}" for layer in sorted(first["low_layers"])])
    groups.extend([f"high_layer:{layer}" for layer in sorted(first["high_layers"])])

    results = []
    for group in groups:
        x_values, feature_names = feature_matrix(rows, group)
        if x_values.shape[1] == 0:
            continue
        model_results = oof_binary_scores(x_values, labels, strata)
        results.append({
            "group": group,
            "feature_count": int(x_values.shape[1]),
            **model_results,
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    summary = {
        "status": "ok",
        "purpose": "Quantify where Hard-vs-rest signal appears in the saved LOW/HIGH process trace.",
        "rows_path": str(rows_path),
        "samples": len(rows),
        "category_counts": {category: int(categories.count(category)) for category in sorted(set(categories))},
        "category_metric_summary": group_summaries(rows),
        "feature_group_results": sorted(
            results,
            key=lambda item: item["random_forest"]["hard_auc"],
            reverse=True,
        ),
        "notes": [
            "LOW/HIGH output and layer features are diagnostic signals, not claimable zero-latency pre-router inputs.",
            "The key question is whether Hard becomes visible only after substantial LOW/HIGH processing.",
        ],
    }
    output_path = RESULTS_DIR / args.output_name
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
