import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analyze_model_process_trace import feature_matrix, load_rows
from src.experiment_paths import ARTIFACT_DIR, RESULTS_DIR, ensure_dirs


RETENTION_TARGETS = [0.995, 0.99, 0.98, 0.95]


def category_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    return dict(sorted(counts.items()))


def candidate_groups(rows):
    first = rows[0]
    groups = [
        {
            "group": "low_output_runtime",
            "runtime_stage": "after_low_output",
            "claim": "LOWを実行した後に使えるconfidence/margin/entropy",
        },
        {
            "group": "all_low_layers",
            "runtime_stage": "during_low",
            "claim": "LOW内部の単純統計",
        },
        {
            "group": "all_high_layers",
            "runtime_stage": "partial_or_full_high_hidden",
            "claim": "HIGH内部特徴の単純統計",
        },
        {
            "group": "all_trace_runtime",
            "runtime_stage": "diagnostic_mixed_runtime",
            "claim": "LOW/HIGH出力と層統計を混ぜた診断用上限",
        },
        {
            "group": "low_output_all",
            "runtime_stage": "oracle_positive_control",
            "claim": "真値依存情報を含む診断用上限",
        },
    ]
    for layer in sorted(first["low_layers"]):
        groups.append({
            "group": f"low_layer:{layer}",
            "runtime_stage": "during_low",
            "claim": f"LOW {layer} の単純統計",
        })
    for layer in sorted(first["high_layers"]):
        groups.append({
            "group": f"high_layer:{layer}",
            "runtime_stage": "partial_high_hidden",
            "claim": f"HIGH {layer} の単純統計",
        })
    return groups


def oof_scores(x_values, labels, strata, seed):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    output = {}
    for model_name, model in models.items():
        scores = np.zeros(len(labels), dtype=np.float32)
        for train_idx, eval_idx in splitter.split(x_values, strata):
            model.fit(x_values[train_idx], labels[train_idx])
            scores[eval_idx] = model.predict_proba(x_values[eval_idx])[:, 1]
        output[model_name] = scores
    return output


def evaluate_conservative_frontier(scores, rows, labels):
    hard = labels.astype(bool)
    low_correct = np.asarray([row["low"]["correct"] for row in rows], dtype=bool)
    high_correct = np.asarray([row["high"]["correct"] for row in rows], dtype=bool)
    easy = np.asarray([row["category"] == "Easy" for row in rows], dtype=bool)
    inverse = np.asarray([row["category"] == "Inverse" for row in rows], dtype=bool)
    impossible = np.asarray([row["category"] == "Impossible" for row in rows], dtype=bool)

    thresholds = np.unique(scores)
    candidates = []
    for threshold in thresholds:
        selected = scores >= threshold
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        low_correct_selected = int((selected & low_correct).sum())
        low_correct_total = int(low_correct.sum())
        low_correct_retention = 1.0 - (low_correct_selected / low_correct_total if low_correct_total else 0.0)
        hard_selected = int((selected & hard).sum())
        hard_total = int(hard.sum())
        hard_recovery = hard_selected / hard_total if hard_total else 0.0

        # If selected samples were sent to a perfect correction path for Hard
        # but could damage LOW-correct samples, this is the key conservative
        # feasibility accounting.
        selected_high_correct = int((selected & high_correct).sum())
        selected_inverse = int((selected & inverse).sum())
        selected_impossible = int((selected & impossible).sum())
        selected_easy = int((selected & easy).sum())
        candidates.append({
            "threshold": float(threshold),
            "selected": selected_count,
            "selected_rate": float(selected.mean()),
            "low_correct_selected": low_correct_selected,
            "low_correct_retention": float(low_correct_retention),
            "hard_selected": hard_selected,
            "hard_recovery": float(hard_recovery),
            "selected_easy": selected_easy,
            "selected_hard": hard_selected,
            "selected_impossible": selected_impossible,
            "selected_inverse": selected_inverse,
            "selected_high_correct": selected_high_correct,
            "precision_hard_among_selected": float(hard_selected / selected_count),
        })

    by_retention = {}
    for target in RETENTION_TARGETS:
        feasible = [row for row in candidates if row["low_correct_retention"] >= target]
        if feasible:
            best = max(feasible, key=lambda row: (row["hard_recovery"], row["selected"], row["precision_hard_among_selected"]))
            by_retention[str(target)] = best
        else:
            by_retention[str(target)] = None
    best_any = max(candidates, key=lambda row: row["hard_recovery"]) if candidates else None
    return by_retention, best_any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-prefix", default="model_process_trace_balanced_002")
    parser.add_argument("--output-name", default="deadline_mid_feasibility_001.json")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    ensure_dirs()
    rows_path = ARTIFACT_DIR / "model_process_traces" / f"{args.input_prefix}_rows.jsonl"
    rows = load_rows(rows_path)
    categories = np.asarray([row["category"] for row in rows])
    labels = np.asarray([1 if category == "Hard" else 0 for category in categories], dtype=np.int64)
    strata = categories

    group_results = []
    for group_meta in candidate_groups(rows):
        group = group_meta["group"]
        x_values, feature_names = feature_matrix(rows, group)
        if x_values.shape[1] == 0:
            continue
        model_scores = oof_scores(x_values, labels, strata, args.seed)
        model_results = []
        for model_name, scores in model_scores.items():
            auc = float(roc_auc_score(labels, scores))
            by_retention, best_any = evaluate_conservative_frontier(scores, rows, labels)
            model_results.append({
                "model": model_name,
                "hard_auc": auc,
                "frontier_by_low_correct_retention": by_retention,
                "best_any_retention_unconstrained": best_any,
            })
        best_99 = None
        for result in model_results:
            row = result["frontier_by_low_correct_retention"].get("0.99")
            if row and (best_99 is None or row["hard_recovery"] > best_99["hard_recovery"]):
                best_99 = {
                    "model": result["model"],
                    "hard_auc": result["hard_auc"],
                    **row,
                }
        group_results.append({
            **group_meta,
            "feature_count": int(x_values.shape[1]),
            "feature_names": feature_names,
            "model_results": model_results,
            "best_at_low_correct_retention_0.99": best_99,
        })
        print(json.dumps({
            "group": group,
            "feature_count": int(x_values.shape[1]),
            "best_at_0.99": best_99,
        }, ensure_ascii=False), flush=True)

    sorted_groups = sorted(
        group_results,
        key=lambda item: (
            -1 if item["best_at_low_correct_retention_0.99"] is None else item["best_at_low_correct_retention_0.99"]["hard_recovery"]
        ),
        reverse=True,
    )
    summary = {
        "status": "ok",
        "purpose": "Judge whether real observed LOW/HIGH process signals can support a conservative MID stage for the deadline-aware proposal.",
        "rows_path": str(rows_path),
        "samples": len(rows),
        "category_counts": category_counts(rows),
        "retention_targets": RETENTION_TARGETS,
        "interpretation_rule": [
            "A candidate is promising only if it can recover a meaningful fraction of Hard while retaining LOW-correct samples near 99%.",
            "This is a signal-feasibility test, not a trained final MID classifier.",
            "Groups using HIGH hidden/output features imply partial-HIGH or diagnostic upper-bound cost, not zero-latency routing.",
        ],
        "group_results": sorted_groups,
    }
    output_path = RESULTS_DIR / args.output_name
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
