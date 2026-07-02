import json

import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.difficulty_mechanism_decomposition import category_of, load_allowed_feature_space
from scripts.evaluate_architectures import add_low_confidence
from scripts.evaluate_router import FLOPS_HIGH, FLOPS_LOW, FLOPS_ROUTER, build_dataset
from scripts.routing_guardrails import guardrail_report
from scripts.search_claimable_record_breakers import (
    CASCADE_BASELINE_COST,
    PARALLEL_BASELINE_COST,
    RECORD_TO_BEAT,
    TARGET_MARGIN_PCTS,
    apply_threshold,
    best_threshold_on_calibration,
)
from src.experiment_paths import ARTIFACT_DIR, RESULTS_DIR, ensure_dirs


def strata_for(records):
    mapping = {"Easy": 0, "Hard": 1, "Impossible": 2, "Inverse": 3}
    return np.asarray([mapping[category_of(record)] for record in records], dtype=np.int64)


def cascade_teacher_threshold(records, target_margin_pct):
    high_accuracy = 100.0 * sum(record["high_correct"] for record in records) / len(records)
    target_accuracy = high_accuracy - target_margin_pct
    best = None
    for threshold in np.linspace(0.0, 1.0, 401):
        to_low = [record for record in records if record["real_low_conf"] >= threshold]
        to_high = [record for record in records if record["real_low_conf"] < threshold]
        correct = sum(record["real_low_correct"] for record in to_low) + sum(record["high_correct"] for record in to_high)
        accuracy = 100.0 * correct / len(records)
        avg_cost = (len(records) * FLOPS_LOW + len(to_high) * FLOPS_HIGH) / len(records)
        if accuracy >= target_accuracy and (best is None or avg_cost < best["avg_cost"]):
            best = {
                "threshold": float(threshold),
                "avg_cost": float(avg_cost),
                "accuracy": float(accuracy),
                "to_low": len(to_low),
                "to_high": len(to_high),
                "target_accuracy": float(target_accuracy),
            }
    return best


def target_values(records, target_name):
    if target_name == "safe_low_actual":
        return np.asarray([1 if category_of(record) != "Hard" else 0 for record in records], dtype=np.int64)
    if target_name == "low_correct":
        return np.asarray([1 if record["low_correct"] else 0 for record in records], dtype=np.int64)
    if target_name.startswith("cascade_teacher_margin_"):
        margin = float(target_name.rsplit("_", 1)[-1])
        teacher = cascade_teacher_threshold(records, margin)
        if teacher is None:
            raise RuntimeError(f"No cascade teacher threshold for margin {margin}")
        return np.asarray([1 if record["real_low_conf"] >= teacher["threshold"] else 0 for record in records], dtype=np.int64)
    raise ValueError(target_name)


def model_zoo():
    return {
        "logistic_l2_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        ),
        "sgd_log_loss_balanced": make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=1e-4, l1_ratio=0.15, class_weight="balanced", random_state=42),
        ),
        "random_forest_deep": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_deep": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "lightgbm_upper_bound": lgb.LGBMClassifier(
            n_estimators=700,
            max_depth=-1,
            num_leaves=127,
            learning_rate=0.02,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "mlp_small": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(96, 32),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=250,
                early_stopping=True,
                random_state=42,
            ),
        ),
    }


def predict_positive(model, x_values):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_values)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_values)
        raw = np.asarray(raw, dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError(f"Model has no score method: {type(model)}")


def fit_model(model, x_values, y_values):
    model.fit(x_values, y_values)
    return model


def evaluate_model_target(name, model, x_values, records, strata, target_name):
    print(f"=== allowed-feature upper bound: {name} / {target_name} ===", flush=True)
    y_all = target_values(records, target_name)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_all = np.zeros(len(records), dtype=float)
    fold_success_by_margin = {margin: 0 for margin in TARGET_MARGIN_PCTS}
    route_decisions_by_margin = {margin: np.zeros(len(records), dtype=bool) for margin in TARGET_MARGIN_PCTS}
    folds = []

    for fold_id, (train_calib_idx, eval_idx) in enumerate(outer.split(np.zeros(len(strata)), strata), start=1):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=500 + fold_id)
        train_sub, calib_sub = next(splitter.split(np.zeros(len(train_calib_idx)), strata[train_calib_idx]))
        train_idx = train_calib_idx[train_sub]
        calib_idx = train_calib_idx[calib_sub]

        fold_model = clone(model)
        fit_model(fold_model, x_values[train_idx], y_all[train_idx])

        calib_scores = predict_positive(fold_model, x_values[calib_idx])
        eval_scores = predict_positive(fold_model, x_values[eval_idx])
        scores_all[eval_idx] = eval_scores
        calib_records = [records[i] for i in calib_idx]
        eval_records = [records[i] for i in eval_idx]

        margin_rows = []
        for margin in TARGET_MARGIN_PCTS:
            best = best_threshold_on_calibration(calib_records, calib_scores, margin)
            if best is None:
                margin_rows.append({"target_margin_pct": margin, "status": "no_feasible_calibration_threshold"})
                continue
            eval_result = apply_threshold(eval_records, eval_scores, best["threshold"])
            route_decisions_by_margin[margin][eval_idx] = eval_scores >= best["threshold"]
            fold_success_by_margin[margin] += 1
            margin_rows.append({
                "target_margin_pct": margin,
                "status": "ok",
                "calibration_best": best,
                "eval": eval_result,
            })

        folds.append({"fold": fold_id, "margins": margin_rows})

    try:
        auc = float(roc_auc_score(y_all, scores_all))
        auc = max(auc, 1.0 - auc)
    except ValueError:
        auc = None

    high_accuracy = 100.0 * sum(record["high_correct"] for record in records) / len(records)
    overall_by_margin = []
    for margin in TARGET_MARGIN_PCTS:
        if fold_success_by_margin[margin] != outer.get_n_splits():
            overall_by_margin.append({
                "target_margin_pct": margin,
                "overall": None,
                "guardrails": guardrail_report(records, None, FLOPS_LOW, FLOPS_HIGH, FLOPS_ROUTER, target_margin_pct=margin),
            })
            continue

        route_decisions = route_decisions_by_margin[margin]
        to_low = int(np.sum(route_decisions))
        to_high = len(records) - to_low
        correct = 0
        for record, low_branch in zip(records, route_decisions):
            correct += bool(record["low_correct"]) if low_branch else bool(record["high_correct"])
        avg_cost = (len(records) * FLOPS_ROUTER + to_low * FLOPS_LOW + to_high * FLOPS_HIGH) / len(records)
        overall = {
            "avg_cost": float(avg_cost),
            "accuracy": float(100.0 * correct / len(records)),
            "to_low": to_low,
            "to_high": to_high,
            "high_accuracy": float(high_accuracy),
            "target_margin_pct": margin,
            "target_accuracy": float(high_accuracy - margin),
            "cost_gap_vs_cascade": float(avg_cost - CASCADE_BASELINE_COST),
            "cost_gap_vs_parallel": float(avg_cost - PARALLEL_BASELINE_COST),
            "cost_gap_vs_record_to_beat": float(avg_cost - RECORD_TO_BEAT),
            "beats_cascade_baseline": bool(avg_cost < CASCADE_BASELINE_COST),
            "beats_parallel_baseline": bool(avg_cost < PARALLEL_BASELINE_COST),
            "beats_record_to_beat": bool(avg_cost < RECORD_TO_BEAT),
        }
        overall_by_margin.append({
            "target_margin_pct": margin,
            "overall": overall,
            "guardrails": guardrail_report(records, overall, FLOPS_LOW, FLOPS_HIGH, FLOPS_ROUTER, target_margin_pct=margin),
        })

    result = {
        "model": name,
        "target": target_name,
        "target_positive_rate": float(np.mean(y_all)),
        "auc_for_target": auc,
        "overall_by_target_margin": overall_by_margin,
        "folds": folds,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    ensure_dirs()
    records, lightweight_x, _, _ = build_dataset("lightweight_lgbm", max_samples=None)
    records = add_low_confidence(records, ARTIFACT_DIR / "cifar100_low_confidence_x1_0.json", batch_size=128)
    feature_sets = load_allowed_feature_space(records, lightweight_x)
    x_values, feature_names = feature_sets["allowed_full"]
    strata = strata_for(records)
    teacher_thresholds = {
        str(margin): cascade_teacher_threshold(records, margin)
        for margin in TARGET_MARGIN_PCTS
    }

    targets = ["safe_low_actual", "low_correct"] + [
        f"cascade_teacher_margin_{margin}" for margin in TARGET_MARGIN_PCTS
    ]
    results = []
    models = model_zoo()
    for target_name in targets:
        for model_name, model in models.items():
            results.append(evaluate_model_target(model_name, model, x_values, records, strata, target_name))

    feasible = []
    for result in results:
        for margin_row in result["overall_by_target_margin"]:
            overall = margin_row["overall"]
            if (
                overall is not None
                and overall["accuracy"] >= overall["target_accuracy"]
                and margin_row["guardrails"]["valid_for_claim"]
            ):
                feasible.append({
                    "model": result["model"],
                    "target": result["target"],
                    "target_margin_pct": margin_row["target_margin_pct"],
                    "auc_for_target": result["auc_for_target"],
                    "overall": overall,
                    "guardrails": margin_row["guardrails"],
                })

    ranked = sorted(feasible, key=lambda item: item["overall"]["avg_cost"])
    best_by_margin = {}
    for margin in TARGET_MARGIN_PCTS:
        subset = [item for item in feasible if item["target_margin_pct"] == margin]
        best_by_margin[str(margin)] = min(subset, key=lambda item: item["overall"]["avg_cost"]) if subset else None

    summary = {
        "status": "ok",
        "purpose": "Upper-bound feasibility check for the allowed zero-latency feature space using strong models and offline teacher labels.",
        "samples": len(records),
        "feature_set": "allowed_full",
        "feature_count": int(x_values.shape[1]),
        "feature_name_count": len(feature_names),
        "targets": targets,
        "models": list(models.keys()),
        "teacher_cascade_thresholds": teacher_thresholds,
        "baselines": {
            "record_to_beat": RECORD_TO_BEAT,
            "cascade": CASCADE_BASELINE_COST,
            "parallel": PARALLEL_BASELINE_COST,
        },
        "results": results,
        "ranked_feasible_by_avg_cost": [
            {
                "model": item["model"],
                "target": item["target"],
                "target_margin_pct": item["target_margin_pct"],
                "auc_for_target": item["auc_for_target"],
                "avg_cost": item["overall"]["avg_cost"],
                "accuracy": item["overall"]["accuracy"],
                "to_low": item["overall"]["to_low"],
                "cost_gap_vs_cascade": item["overall"]["cost_gap_vs_cascade"],
                "cost_gap_vs_record_to_beat": item["overall"]["cost_gap_vs_record_to_beat"],
            }
            for item in ranked[:30]
        ],
        "best_by_target_margin": best_by_margin,
        "best_overall": ranked[0] if ranked else None,
        "interpretation_rule": {
            "positive": "If any strong model routes roughly 37.5-52% to LOW while meeting target accuracy, the allowed feature space contains enough information and the next problem is hardware-friendly compression.",
            "negative": "If all strong models remain far below 37.5% LOW, the allowed feature space likely lacks the information needed to match the notebook record, parallel, or cascade baselines.",
        },
    }
    output_path = RESULTS_DIR / "allowed_feature_upper_bound.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
