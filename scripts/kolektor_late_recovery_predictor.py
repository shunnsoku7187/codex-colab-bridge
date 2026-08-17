"""Test a late-recovery predictor on KolektorSDD.

The predictor asks a different question from direct defect classification:

    If this sample goes to the final stage, will it become a reliable good pass?

If the answer is likely no, the sample can be rejected early.  This script
compares that learned rejector with final-only selective classification and
ordinary upper-only BranchyNet under validation-selected inspection constraints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.kolektor_dual_exit_significance import (
    ResNet18Branchy,
    apply_policy,
    better,
    candidates,
    collect_outputs,
    eval_feasible,
    feasible,
    metric,
    simulate_bn,
    simulate_final,
    train_model,
)
from scripts.train_kolektor_strong_final import (
    class_weight,
    download_and_extract,
    find_samples,
    make_loader,
    round_float,
    set_seed,
    split_by_item,
)
from src.experiment_paths import ensure_dirs


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(1, min(n - 1, int(round(n * 0.55))))
    return idx[:cut], idx[cut:]


def make_features(data: dict[str, np.ndarray], exit_idx: int, feature_set: str) -> np.ndarray:
    p = np.asarray(data["p_defect"], dtype=np.float32)
    p_good = 1.0 - p
    eps = 1e-8
    entropy = -(p * np.log2(p + eps) + p_good * np.log2(p_good + eps))
    cols = [
        p[:, exit_idx],
        p_good[:, exit_idx],
        entropy[:, exit_idx],
        p[:, exit_idx] * p[:, exit_idx],
        p_good[:, exit_idx] * p_good[:, exit_idx],
    ]
    if feature_set in {"trace", "trace_delta"} and exit_idx >= 1:
        cols += [
            p[:, 0],
            p_good[:, 0],
            entropy[:, 0],
            p[:, exit_idx] - p[:, 0],
            p_good[:, exit_idx] - p_good[:, 0],
            entropy[:, exit_idx] - entropy[:, 0],
        ]
    return np.stack(cols, axis=1).astype(np.float32)


def model_specs(seed: int) -> list[tuple[str, Any, str]]:
    return [
        ("logistic_l2", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)), "fixed-point linear score"),
        ("tree_depth2", DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, class_weight="balanced", random_state=seed), "small comparator tree"),
        ("tree_depth3", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=seed), "small comparator tree"),
        ("mlp_8", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), alpha=1e-3, max_iter=800, random_state=seed)), "tiny nonlinear head"),
    ]


def non_recovery_score(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    idx = classes.index(False) if False in classes else classes.index(0)
    return np.asarray(proba[:, idx], dtype=np.float32)


def simulate_predictor_policy(
    data: dict[str, np.ndarray],
    score: np.ndarray,
    exit_idx: int,
    pass_threshold: float,
    reject_score_threshold: float,
    final_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = data["p_defect"]
    costs_ref = data["exit_costs"]
    decisions = np.ones(len(p), dtype=np.int64)
    costs = np.full(len(p), costs_ref[-1], dtype=np.float32)
    early_pass = p[:, exit_idx] <= pass_threshold
    early_reject = (~early_pass) & (score >= reject_score_threshold)
    early = early_pass | early_reject
    decisions[early_pass] = 0
    decisions[early_reject] = 1
    costs[early] = costs_ref[exit_idx]
    final_active = ~early
    decisions[final_active] = (p[final_active, -1] >= final_threshold).astype(np.int64)
    return decisions, costs


def choose_final_threshold(data: dict[str, np.ndarray], max_false_pass: float, min_good_pass: float, max_threshold_candidates: int) -> dict | None:
    best = None
    for tf in candidates(data["p_defect"][:, -1], max_threshold_candidates):
        decisions, costs = simulate_final(data, tf)
        row = metric(data["labels"], decisions, costs)
        if feasible(row, max_false_pass, min_good_pass):
            cand = {"policy": "final_selective", "params": {"final_reject_threshold": tf}, "val_metric": row}
            if better(cand, best):
                best = cand
    return best


def choose_bn(data: dict[str, np.ndarray], max_false_pass: float, min_good_pass: float, max_threshold_candidates: int) -> dict | None:
    best = None
    cand0 = candidates(data["p_defect"][:, 0], max_threshold_candidates)
    cand1 = candidates(data["p_defect"][:, 1], max_threshold_candidates)
    candf = candidates(data["p_defect"][:, 2], max_threshold_candidates)
    for t0 in cand0:
        for t1 in cand1:
            for tf in candf:
                decisions, costs = simulate_bn(data, t0, t1, tf)
                row = metric(data["labels"], decisions, costs)
                if feasible(row, max_false_pass, min_good_pass):
                    cand = {"policy": "branchynet_upper_only", "params": {"exit0_pass_threshold": t0, "exit1_pass_threshold": t1, "final_reject_threshold": tf}, "val_metric": row}
                    if better(cand, best):
                        best = cand
    return best


def select_predictor_policy(
    train_data: dict[str, np.ndarray],
    cal_data: dict[str, np.ndarray],
    max_false_pass: float,
    min_good_pass: float,
    max_threshold_candidates: int,
    seed: int,
) -> list[dict]:
    rows = []
    final = choose_final_threshold(cal_data, max_false_pass, min_good_pass, max_threshold_candidates)
    if final is None:
        return rows
    final_threshold = final["params"]["final_reject_threshold"]
    y_train_final_good = (train_data["labels"] == 0) & (train_data["p_defect"][:, -1] < final_threshold)

    for exit_idx in [0, 1]:
        pass_candidates = candidates(cal_data["p_defect"][:, exit_idx], max_threshold_candidates)
        feature_sets = ["scalar"] if exit_idx == 0 else ["scalar", "trace"]
        for feature_set in feature_sets:
            x_train = make_features(train_data, exit_idx, feature_set)
            x_cal = make_features(cal_data, exit_idx, feature_set)
            for name, model, note in model_specs(seed):
                try:
                    model.fit(x_train, y_train_final_good)
                    score_cal = non_recovery_score(model, x_cal)
                except Exception as exc:  # noqa: BLE001
                    rows.append({"policy": "late_recovery_predictor", "valid": False, "reason": repr(exc), "exit_idx": exit_idx, "feature_set": feature_set, "predictor": name})
                    continue
                score_candidates = candidates(score_cal, max_threshold_candidates)
                best = None
                for pass_t in pass_candidates:
                    for reject_t in score_candidates:
                        decisions, costs = simulate_predictor_policy(cal_data, score_cal, exit_idx, pass_t, reject_t, final_threshold)
                        row = metric(cal_data["labels"], decisions, costs)
                        if feasible(row, max_false_pass, min_good_pass):
                            cand = {
                                "policy": "late_recovery_predictor",
                                "params": {"exit_idx": exit_idx, "pass_threshold": pass_t, "reject_score_threshold": reject_t, "final_reject_threshold": final_threshold},
                                "val_metric": row,
                                "feature_set": feature_set,
                                "predictor": name,
                                "fpga_note": note,
                                "model": model,
                            }
                            if better(cand, best):
                                best = cand
                if best is not None:
                    rows.append(best)
    return rows


def apply_predictor_policy(data: dict[str, np.ndarray], row: dict) -> dict:
    p = row["params"]
    x = make_features(data, int(p["exit_idx"]), row["feature_set"])
    score = non_recovery_score(row["model"], x)
    decisions, costs = simulate_predictor_policy(
        data,
        score,
        int(p["exit_idx"]),
        float(p["pass_threshold"]),
        float(p["reject_score_threshold"]),
        float(p["final_reject_threshold"]),
    )
    return metric(data["labels"], decisions, costs)


def strip_model(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k != "model"}
    return out


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KolektorSDD late-recovery predictor experiment",
        "",
        "Thresholds are selected on a calibration subset and fixed on evaluation data.",
        "",
        "| max false pass | min good pass | policy | predictor | eval feasible | eval good pass | eval false pass | eval avg cost | speedup |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["policy_rows"]:
        e = row["eval_metric"]
        pred = row.get("predictor", "-")
        lines.append(
            f"| {100 * row['constraint']['max_false_pass_rate_defect']:.1f}% | "
            f"{100 * row['constraint']['min_good_pass_rate_good']:.1f}% | {row['policy']} | {pred} | "
            f"{'yes' if row.get('eval_feasible') else 'no'} | {100 * e['good_pass_rate_good']:.2f}% | "
            f"{100 * e['false_pass_rate_defect']:.2f}% | {e['avg_cost']:.4f} | {e['speedup_vs_final_only']:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/kolektor_late_recovery_predictor_001_summary.json")
    parser.add_argument("--markdown", default="docs/kolektor_late_recovery_predictor_001.md")
    parser.add_argument("--checkpoint", default="artifacts/kolektor_late_recovery_predictor_001/resnet18_branchy.pt")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.1])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.95, 0.98])
    parser.add_argument("--max-threshold-candidates", type=int, default=13)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    cache_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    samples = find_samples(download_and_extract(cache_root / "kolektor_sdd"))
    split = split_by_item(samples, args.seed)
    image_size = (args.image_height, args.image_width)
    train_loader = make_loader(split["train"], image_size, args.batch_size, train=True)
    val_loader = make_loader(split["val"], image_size, args.batch_size, train=False)
    eval_loader = make_loader(split["eval"], image_size, args.batch_size, train=False)

    model = ResNet18Branchy(pretrained=True).to(device)
    training = train_model(model, train_loader, val_loader, class_weight(split["train"], device), args.epochs, device)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "image_size": list(image_size), "exit_costs": model.exit_costs.tolist()}, args.checkpoint)

    val_data_full = collect_outputs(model, val_loader, device)
    eval_data = collect_outputs(model, eval_loader, device)
    pred_train_idx, cal_idx = split_indices(len(val_data_full["labels"]), args.seed + 1000)
    train_data = {k: (v[pred_train_idx] if isinstance(v, np.ndarray) and len(v) == len(val_data_full["labels"]) else v) for k, v in val_data_full.items()}
    cal_data = {k: (v[cal_idx] if isinstance(v, np.ndarray) and len(v) == len(val_data_full["labels"]) else v) for k, v in val_data_full.items()}

    policy_rows = []
    for max_fp in args.max_false_pass_rates:
        for min_gp in args.min_good_pass_rates:
            constraint = {"max_false_pass_rate_defect": max_fp, "min_good_pass_rate_good": min_gp}
            for base in [choose_final_threshold(cal_data, max_fp, min_gp, args.max_threshold_candidates), choose_bn(cal_data, max_fp, min_gp, args.max_threshold_candidates)]:
                if base is None:
                    continue
                row_out = {"constraint": constraint, **base, "eval_metric": apply_policy(eval_data, base)}
                row_out["eval_feasible"] = eval_feasible(row_out)
                policy_rows.append(row_out)
            pred_rows = select_predictor_policy(train_data, cal_data, max_fp, min_gp, args.max_threshold_candidates, args.seed)
            for row in pred_rows:
                if row.get("valid") is False or "val_metric" not in row:
                    continue
                eval_metric = apply_predictor_policy(eval_data, row)
                row_out = {"constraint": constraint, **strip_model(row), "eval_metric": eval_metric}
                row_out["eval_feasible"] = eval_feasible(row_out)
                policy_rows.append(row_out)

    payload = {
        "purpose": "Test whether a learned late-recovery predictor can make dual-sided early exit useful under inspection constraints.",
        "dataset": {
            "name": "KolektorSDD",
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
            "split_counts": {
                key: {"samples": len(value), "defects": int(sum(s.label for s in value)), "good": int(sum(1 - s.label for s in value))}
                for key, value in split.items()
            },
            "predictor_train_count": int(len(pred_train_idx)),
            "calibration_count": int(len(cal_idx)),
        },
        "model": {"arch": "ResNet18Branchy", "checkpoint": args.checkpoint, "training": training},
        "policy_rows": policy_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "rows": len(policy_rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
