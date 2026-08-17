"""Use already-computed intermediate activations for late-recovery prediction.

This keeps the predictor inside the existing BranchyNet data path: no extra CNN
is evaluated.  The predictors use exit probabilities plus summaries of the
feature map that already exists at the selected exit.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from scripts.kolektor_dual_exit_significance import (
    ResNet18Branchy,
    better,
    candidates,
    eval_feasible,
    feasible,
    metric,
    simulate_bn,
    simulate_final,
    train_model,
)
from scripts.kolektor_late_recovery_predictor import (
    choose_bn,
    choose_final_threshold,
    non_recovery_score,
    split_indices,
    strip_model,
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


def scalar_exit_features(p_defect: np.ndarray, exit_idx: int) -> tuple[np.ndarray, list[str]]:
    p = np.asarray(p_defect, dtype=np.float32)
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
    names = [
        f"exit{exit_idx}_p_defect",
        f"exit{exit_idx}_p_good",
        f"exit{exit_idx}_entropy",
        f"exit{exit_idx}_p_defect_sq",
        f"exit{exit_idx}_p_good_sq",
    ]
    return np.stack(cols, axis=1).astype(np.float32), names


def activation_global_stats(tensor: torch.Tensor) -> np.ndarray:
    flat = tensor.flatten(1)
    stats = [
        flat.mean(dim=1),
        flat.std(dim=1, unbiased=False),
        flat.amax(dim=1),
        flat.amin(dim=1),
        flat.abs().mean(dim=1),
        (flat <= 0).float().mean(dim=1),
    ]
    return torch.stack(stats, dim=1).detach().cpu().numpy().astype(np.float32)


def activation_channel_pool(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    mean = tensor.mean(dim=(2, 3)).detach().cpu().numpy().astype(np.float32)
    std = tensor.std(dim=(2, 3), unbiased=False).detach().cpu().numpy().astype(np.float32)
    return mean, std


@torch.no_grad()
def collect_outputs_and_features(model: ResNet18Branchy, loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    layer1_global, layer2_global = [], []
    layer1_mean, layer1_std = [], []
    layer2_mean, layer2_std = [], []

    for x, y in tqdm(loader, desc="collect features", leave=False):
        x = x.to(device)
        h = model.stem(x)
        h1 = model.layer1(h)
        out0 = model.exit0(h1)
        h2 = model.layer2(h1)
        out1 = model.exit1(h2)
        h = model.layer3(h2)
        h = model.layer4(h)
        out2 = model.final(h)

        probs = [torch.softmax(logits, dim=1).detach().cpu().numpy() for logits in [out0, out1, out2]]
        labels_all.append(y.numpy())
        probs_all.append(np.stack(probs, axis=1))

        layer1_global.append(activation_global_stats(h1))
        layer2_global.append(activation_global_stats(h2))
        m1, s1 = activation_channel_pool(h1)
        m2, s2 = activation_channel_pool(h2)
        layer1_mean.append(m1)
        layer1_std.append(s1)
        layer2_mean.append(m2)
        layer2_std.append(s2)

    labels = np.concatenate(labels_all).astype(np.int64)
    probs = np.concatenate(probs_all).astype(np.float32)
    return {
        "labels": labels,
        "p_defect": probs[:, :, 1],
        "exit_costs": model.exit_costs.copy(),
        "exit_names": np.asarray(model.exit_names, dtype=object),
        "layer1_global": np.concatenate(layer1_global).astype(np.float32),
        "layer2_global": np.concatenate(layer2_global).astype(np.float32),
        "layer1_mean": np.concatenate(layer1_mean).astype(np.float32),
        "layer1_std": np.concatenate(layer1_std).astype(np.float32),
        "layer2_mean": np.concatenate(layer2_mean).astype(np.float32),
        "layer2_std": np.concatenate(layer2_std).astype(np.float32),
    }


def feature_matrix(data: dict[str, np.ndarray], exit_idx: int, feature_set: str) -> tuple[np.ndarray, list[str]]:
    scalar, scalar_names = scalar_exit_features(data["p_defect"], exit_idx)
    if exit_idx == 0:
        prefix = "layer1"
    elif exit_idx == 1:
        prefix = "layer2"
    else:
        raise ValueError(f"unsupported exit_idx={exit_idx}")

    global_values = data[f"{prefix}_global"]
    global_names = [f"{prefix}_{name}" for name in ["mean", "std", "max", "min", "abs_mean", "zero_fraction"]]
    mean_values = data[f"{prefix}_mean"]
    std_values = data[f"{prefix}_std"]
    mean_names = [f"{prefix}_ch{i}_mean" for i in range(mean_values.shape[1])]
    std_names = [f"{prefix}_ch{i}_std" for i in range(std_values.shape[1])]

    if feature_set == "scalar":
        return scalar, scalar_names
    if feature_set == "global_stats":
        return np.concatenate([scalar, global_values], axis=1), scalar_names + global_names
    if feature_set == "channel_mean":
        return np.concatenate([scalar, global_values, mean_values], axis=1), scalar_names + global_names + mean_names
    if feature_set == "channel_mean_std":
        return (
            np.concatenate([scalar, global_values, mean_values, std_values], axis=1),
            scalar_names + global_names + mean_names + std_names,
        )
    raise ValueError(feature_set)


def model_specs(seed: int) -> list[tuple[str, Any, str]]:
    return [
        (
            "logistic_l1_sparse",
            make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)),
            "sparse fixed-point linear score",
        ),
        (
            "logistic_l2",
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
            "fixed-point linear score",
        ),
        ("tree_depth2", DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, class_weight="balanced", random_state=seed), "small comparator tree"),
        ("tree_depth3", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=seed), "small comparator tree"),
    ]


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


def split_data(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    n = len(data["labels"])
    return {k: (v[indices] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in data.items()}


def apply_base_policy(data: dict[str, np.ndarray], row: dict) -> dict:
    p = row["params"]
    if row["policy"] == "final_selective":
        decisions, costs = simulate_final(data, p["final_reject_threshold"])
    elif row["policy"] == "branchynet_upper_only":
        decisions, costs = simulate_bn(data, p["exit0_pass_threshold"], p["exit1_pass_threshold"], p["final_reject_threshold"])
    else:
        raise ValueError(row["policy"])
    return metric(data["labels"], decisions, costs)


def apply_predictor_policy(data: dict[str, np.ndarray], row: dict) -> dict:
    p = row["params"]
    x, _names = feature_matrix(data, int(p["exit_idx"]), row["feature_set"])
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


def select_predictor_policies(
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
        for feature_set in ["scalar", "global_stats", "channel_mean", "channel_mean_std"]:
            x_train, feature_names = feature_matrix(train_data, exit_idx, feature_set)
            x_cal, _ = feature_matrix(cal_data, exit_idx, feature_set)
            for name, model, note in model_specs(seed):
                try:
                    model.fit(x_train, y_train_final_good)
                    score_cal = non_recovery_score(model, x_cal)
                except Exception as exc:  # noqa: BLE001
                    rows.append({"valid": False, "reason": repr(exc), "policy": "intermediate_recovery_predictor", "exit_idx": exit_idx, "feature_set": feature_set, "predictor": name})
                    continue
                best = None
                for pass_t in pass_candidates:
                    for reject_t in candidates(score_cal, max_threshold_candidates):
                        decisions, costs = simulate_predictor_policy(cal_data, score_cal, exit_idx, pass_t, reject_t, final_threshold)
                        row = metric(cal_data["labels"], decisions, costs)
                        if feasible(row, max_false_pass, min_good_pass):
                            cand = {
                                "policy": "intermediate_recovery_predictor",
                                "params": {
                                    "exit_idx": exit_idx,
                                    "pass_threshold": pass_t,
                                    "reject_score_threshold": reject_t,
                                    "final_reject_threshold": final_threshold,
                                },
                                "val_metric": row,
                                "feature_set": feature_set,
                                "feature_count": int(x_train.shape[1]),
                                "predictor": name,
                                "fpga_note": note,
                                "feature_names_head": feature_names[:12],
                                "model": model,
                            }
                            if better(cand, best):
                                best = cand
                if best is not None:
                    rows.append(best)
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KolektorSDD intermediate-feature recovery predictor",
        "",
        "This experiment uses already-computed feature maps at each early exit. No extra CNN is evaluated.",
        "",
        "| max false pass | min good pass | policy | feature set | predictor | features | eval feasible | eval good pass | eval false pass | eval avg cost | speedup |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["policy_rows"]:
        e = row["eval_metric"]
        lines.append(
            f"| {100 * row['constraint']['max_false_pass_rate_defect']:.1f}% | "
            f"{100 * row['constraint']['min_good_pass_rate_good']:.1f}% | "
            f"{row['policy']} | {row.get('feature_set', '-')} | {row.get('predictor', '-')} | "
            f"{row.get('feature_count', '-')} | {'yes' if row.get('eval_feasible') else 'no'} | "
            f"{100 * e['good_pass_rate_good']:.2f}% | {100 * e['false_pass_rate_defect']:.2f}% | "
            f"{e['avg_cost']:.4f} | {e['speedup_vs_final_only']:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_eval_summary(rows: list[dict]) -> list[dict]:
    out = []
    by_constraint: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        c = row["constraint"]
        key = (float(c["max_false_pass_rate_defect"]), float(c["min_good_pass_rate_good"]))
        by_constraint.setdefault(key, []).append(row)
    for key, items in sorted(by_constraint.items()):
        feasible_items = [r for r in items if r.get("eval_feasible")]
        pool = feasible_items or items
        best = sorted(
            pool,
            key=lambda r: (
                int(bool(r.get("eval_feasible"))),
                float(r["eval_metric"]["speedup_vs_final_only"]),
                -float(r["eval_metric"]["false_pass_rate_defect"]),
                float(r["eval_metric"]["good_pass_rate_good"]),
            ),
            reverse=True,
        )[0]
        m = best["eval_metric"]
        out.append(
            {
                "constraint": {"max_false_pass_rate_defect": key[0], "min_good_pass_rate_good": key[1]},
                "policy": best["policy"],
                "feature_set": best.get("feature_set"),
                "predictor": best.get("predictor"),
                "eval_feasible": bool(best.get("eval_feasible")),
                "eval_good_pass_rate_good": m["good_pass_rate_good"],
                "eval_false_pass_rate_defect": m["false_pass_rate_defect"],
                "eval_avg_cost": m["avg_cost"],
                "eval_speedup": m["speedup_vs_final_only"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/kolektor_intermediate_recovery_predictor_001_summary.json")
    parser.add_argument("--markdown", default="docs/kolektor_intermediate_recovery_predictor_001.md")
    parser.add_argument("--checkpoint", default="artifacts/kolektor_intermediate_recovery_predictor_001/resnet18_branchy.pt")
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

    val_data_full = collect_outputs_and_features(model, val_loader, device)
    eval_data = collect_outputs_and_features(model, eval_loader, device)
    pred_train_idx, cal_idx = split_indices(len(val_data_full["labels"]), args.seed + 1000)
    train_data = split_data(val_data_full, pred_train_idx)
    cal_data = split_data(val_data_full, cal_idx)

    policy_rows = []
    for max_fp in args.max_false_pass_rates:
        for min_gp in args.min_good_pass_rates:
            constraint = {"max_false_pass_rate_defect": max_fp, "min_good_pass_rate_good": min_gp}
            for base in [
                choose_final_threshold(cal_data, max_fp, min_gp, args.max_threshold_candidates),
                choose_bn(cal_data, max_fp, min_gp, args.max_threshold_candidates),
            ]:
                if base is None:
                    continue
                row_out = {"constraint": constraint, **base, "eval_metric": apply_base_policy(eval_data, base)}
                row_out["eval_feasible"] = eval_feasible(row_out)
                policy_rows.append(row_out)
            for row in select_predictor_policies(train_data, cal_data, max_fp, min_gp, args.max_threshold_candidates, args.seed):
                if row.get("valid") is False or "val_metric" not in row:
                    continue
                eval_metric = apply_predictor_policy(eval_data, row)
                row_out = {"constraint": constraint, **strip_model(row), "eval_metric": eval_metric}
                row_out["eval_feasible"] = eval_feasible(row_out)
                policy_rows.append(row_out)

    payload = {
        "purpose": "Check whether intermediate activations already computed at early exits can predict late recovery without adding another CNN.",
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
        "feature_sets": {
            "scalar": "exit probabilities only",
            "global_stats": "exit probabilities plus global activation summaries",
            "channel_mean": "global stats plus per-channel mean",
            "channel_mean_std": "per-channel mean and standard deviation",
        },
        "policy_rows": policy_rows,
        "best_eval_summary": best_eval_summary(policy_rows),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "rows": len(policy_rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
