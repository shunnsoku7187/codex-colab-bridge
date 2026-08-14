"""Train tiny predictors for whether a sample is worth sending deeper.

The lower-side exit should answer a different question from classification:

    Will the final exit become a reliable accepted label?

If the answer is likely no, the sample can be rejected early and the final
stage can be skipped.  This script trains small FPGA-friendly predictors from
early-exit signals and compares them with a raw confidence lower threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from scripts.branchynet_cifar_sweep import (
    CIFAR_STATS,
    DEFAULT_EXITS,
    estimate_costs,
    make_datasets,
    parse_csv_ints,
    transform_eval,
)
from scripts.compare_final_threshold_vs_dual_exit import calibrate_final_threshold
from scripts.dual_exit_reliability_shift_experiment import (
    collect_scenario,
    concat_data,
    load_branchynet,
    make_loader,
)
from scripts.dual_exit_speed_tradeoff_experiment import mixture_specs, required_scenarios
from src.experiment_paths import ensure_dirs


def round_float(value: float | np.floating | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def split_indices(n: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_n = int(round(n * train_fraction))
    return indices[:train_n], indices[train_n:]


def one_hot(values: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((len(values), classes), dtype=np.float32)
    out[np.arange(len(values)), values.astype(int)] = 1.0
    return out


def make_features(data: dict[str, np.ndarray], exit_idx: int, feature_set: str, num_classes: int) -> tuple[np.ndarray, list[str]]:
    conf = np.asarray(data["confidence"], dtype=np.float32)
    entropy = np.asarray(data["entropy"], dtype=np.float32)
    pred = np.asarray(data["pred"], dtype=np.int16)

    cols: list[np.ndarray] = [
        conf[:, exit_idx],
        entropy[:, exit_idx],
        conf[:, exit_idx] - entropy[:, exit_idx],
        conf[:, exit_idx] * conf[:, exit_idx],
        entropy[:, exit_idx] * entropy[:, exit_idx],
    ]
    names = [
        f"exit{exit_idx}_confidence",
        f"exit{exit_idx}_entropy",
        f"exit{exit_idx}_conf_minus_entropy",
        f"exit{exit_idx}_confidence_sq",
        f"exit{exit_idx}_entropy_sq",
    ]

    if feature_set in {"trace", "class_aware"} and exit_idx >= 1:
        cols += [
            conf[:, 0],
            entropy[:, 0],
            conf[:, exit_idx] - conf[:, 0],
            entropy[:, exit_idx] - entropy[:, 0],
            (pred[:, exit_idx] == pred[:, 0]).astype(np.float32),
        ]
        names += [
            "exit0_confidence",
            "exit0_entropy",
            f"exit{exit_idx}_confidence_gain",
            f"exit{exit_idx}_entropy_gain",
            f"exit0_exit{exit_idx}_prediction_agree",
        ]

    x = np.stack(cols, axis=1).astype(np.float32)
    if feature_set == "class_aware":
        x = np.concatenate([x, one_hot(pred[:, exit_idx], num_classes)], axis=1)
        names += [f"exit{exit_idx}_pred_is_class_{idx}" for idx in range(num_classes)]
    return x, names


def final_reliable_labels(data: dict[str, np.ndarray], final_threshold: float) -> np.ndarray:
    correct = np.asarray(data["correct"], dtype=bool)
    conf = np.asarray(data["confidence"], dtype=np.float32)
    return correct[:, -1] & (conf[:, -1] >= final_threshold)


def model_specs(seed: int) -> list[tuple[str, Any, str]]:
    return [
        (
            "logistic_l2",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
            "linear score; cheap to map to fixed-point MAC/comparator",
        ),
        (
            "tree_depth2",
            DecisionTreeClassifier(max_depth=2, min_samples_leaf=50, class_weight="balanced", random_state=seed),
            "few comparisons; LUT/comparator friendly",
        ),
        (
            "tree_depth3",
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=seed),
            "small comparison tree",
        ),
        (
            "mlp_8",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(8,), alpha=1e-3, max_iter=600, random_state=seed),
            ),
            "tiny nonlinear predictor; upper bound for a small learned head",
        ),
    ]


def non_reliable_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
    else:
        proba = model[-1].predict_proba(model[:-1].transform(x))  # pragma: no cover
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    idx = classes.index(False) if False in classes else classes.index(0)
    return np.asarray(proba[:, idx], dtype=np.float32)


def choose_reject_threshold(score: np.ndarray, y_final_reliable: np.ndarray, max_lost: float) -> dict[str, Any] | None:
    candidates = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 301))).astype(np.float32)
    best = None
    total = len(score)
    for threshold in candidates:
        reject = score >= threshold
        lost = float((reject & y_final_reliable).sum() / total)
        if lost > max_lost:
            continue
        row = {
            "threshold": float(threshold),
            "early_reject_rate": float(reject.mean()),
            "lost_final_reliable_rate": lost,
            "reject_precision_non_reliable": None
            if not reject.any()
            else float((reject & ~y_final_reliable).sum() / reject.sum()),
        }
        if best is None or (
            row["early_reject_rate"],
            -row["lost_final_reliable_rate"],
            row["reject_precision_non_reliable"] or 0.0,
        ) > (
            best["early_reject_rate"],
            -best["lost_final_reliable_rate"],
            best["reject_precision_non_reliable"] or 0.0,
        ):
            best = row
    return best


def evaluate_reject_policy(
    score: np.ndarray,
    y_final_reliable: np.ndarray,
    threshold: float,
    exit_cost: float,
    final_cost: float,
) -> dict[str, Any]:
    reject = score >= threshold
    total = len(score)
    early_reject_rate = float(reject.mean())
    lost = float((reject & y_final_reliable).sum() / total)
    avg_cost = float(early_reject_rate * exit_cost + (1.0 - early_reject_rate) * final_cost)
    return {
        "early_reject_rate": round_float(early_reject_rate),
        "final_execution_rate": round_float(1.0 - early_reject_rate),
        "avg_cost": round_float(avg_cost),
        "speedup_vs_final_only": round_float(final_cost / avg_cost if avg_cost else None),
        "cost_reduction_vs_final_only": round_float(final_cost - avg_cost),
        "lost_final_reliable_rate": round_float(lost),
        "reject_precision_non_reliable": round_float(float((reject & ~y_final_reliable).sum() / reject.sum()) if reject.any() else None),
        "rejected_count": int(reject.sum()),
        "lost_final_reliable_count": int((reject & y_final_reliable).sum()),
    }


def score_quality(score: np.ndarray, y_final_reliable: np.ndarray) -> dict[str, Any]:
    y_non_reliable = ~y_final_reliable
    out = {}
    if len(np.unique(y_non_reliable)) > 1:
        out["auroc_non_reliable"] = round_float(roc_auc_score(y_non_reliable, score))
        out["average_precision_non_reliable"] = round_float(average_precision_score(y_non_reliable, score))
    else:
        out["auroc_non_reliable"] = None
        out["average_precision_non_reliable"] = None
    out["final_reliable_rate"] = round_float(float(y_final_reliable.mean()))
    return out


def make_baseline_score(data: dict[str, np.ndarray], exit_idx: int) -> np.ndarray:
    conf = np.asarray(data["confidence"], dtype=np.float32)
    return 1.0 - conf[:, exit_idx]


def tree_summary(model: Any, feature_names: list[str]) -> str | None:
    if isinstance(model, DecisionTreeClassifier):
        return export_text(model, feature_names=feature_names, max_depth=3)
    return None


def run_one(
    scenario: str,
    target_accuracy: float,
    max_lost: float,
    exit_idx: int,
    feature_set: str,
    val_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    costs: np.ndarray,
    num_classes: int,
    seed: int,
) -> list[dict[str, Any]]:
    val_correct = np.asarray(val_data["correct"], dtype=bool)
    val_conf = np.asarray(val_data["confidence"], dtype=np.float32)
    final_threshold = calibrate_final_threshold(val_correct[:, -1], val_conf[:, -1], target_accuracy)
    y_val = final_reliable_labels(val_data, final_threshold["threshold"])
    y_eval = final_reliable_labels(eval_data, final_threshold["threshold"])

    train_idx, cal_idx = split_indices(len(y_val), 0.6, seed)
    x_val, feature_names = make_features(val_data, exit_idx, feature_set, num_classes)
    x_eval, _ = make_features(eval_data, exit_idx, feature_set, num_classes)
    rows = []

    base_cal_score = make_baseline_score(val_data, exit_idx)[cal_idx]
    base_eval_score = make_baseline_score(eval_data, exit_idx)
    base_threshold = choose_reject_threshold(base_cal_score, y_val[cal_idx], max_lost)
    if base_threshold is not None:
        eval_metrics = evaluate_reject_policy(base_eval_score, y_eval, base_threshold["threshold"], float(costs[exit_idx]), float(costs[-1]))
        rows.append({
            "scenario": scenario,
            "target_accuracy": target_accuracy,
            "max_lost_final_reliable_rate": max_lost,
            "exit_idx": exit_idx,
            "feature_set": "confidence_only",
            "predictor": "raw_lower_confidence",
            "valid": True,
            "validation_final_threshold": final_threshold,
            "selected_threshold": {key: round_float(value) if isinstance(value, float) else value for key, value in base_threshold.items()},
            "score_quality_eval": score_quality(base_eval_score, y_eval),
            **eval_metrics,
            "fpga_note": "single confidence comparator",
        })

    for name, model, note in model_specs(seed):
        try:
            model.fit(x_val[train_idx], y_val[train_idx])
            cal_score = non_reliable_score(model, x_val[cal_idx])
            threshold = choose_reject_threshold(cal_score, y_val[cal_idx], max_lost)
            if threshold is None:
                rows.append({
                    "scenario": scenario,
                    "target_accuracy": target_accuracy,
                    "max_lost_final_reliable_rate": max_lost,
                    "exit_idx": exit_idx,
                    "feature_set": feature_set,
                    "predictor": name,
                    "valid": False,
                    "reason": "no threshold satisfies max_lost on calibration split",
                    "fpga_note": note,
                })
                continue
            eval_score = non_reliable_score(model, x_eval)
            eval_metrics = evaluate_reject_policy(eval_score, y_eval, threshold["threshold"], float(costs[exit_idx]), float(costs[-1]))
            rows.append({
                "scenario": scenario,
                "target_accuracy": target_accuracy,
                "max_lost_final_reliable_rate": max_lost,
                "exit_idx": exit_idx,
                "feature_set": feature_set,
                "predictor": name,
                "valid": True,
                "validation_final_threshold": final_threshold,
                "selected_threshold": {key: round_float(value) if isinstance(value, float) else value for key, value in threshold.items()},
                "score_quality_eval": score_quality(eval_score, y_eval),
                "tree_rules": tree_summary(model, feature_names),
                **eval_metrics,
                "fpga_note": note,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "scenario": scenario,
                "target_accuracy": target_accuracy,
                "max_lost_final_reliable_rate": max_lost,
                "exit_idx": exit_idx,
                "feature_set": feature_set,
                "predictor": name,
                "valid": False,
                "reason": repr(exc),
                "fpga_note": note,
            })
    return rows


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("valid")]
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in valid:
        key = (row["scenario"], row["target_accuracy"], row["max_lost_final_reliable_rate"])
        current = grouped.get(key)
        if current is None or (
            row["avg_cost"],
            -row["early_reject_rate"],
            row["lost_final_reliable_rate"],
        ) < (
            current["avg_cost"],
            -current["early_reject_rate"],
            current["lost_final_reliable_rate"],
        ):
            grouped[key] = row
    return sorted(grouped.values(), key=lambda row: (row["target_accuracy"], row["max_lost_final_reliable_rate"], row["scenario"]))


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = best_rows(payload["rows"])
    lines = [
        "# 後段改善見込み判定器の予備実験",
        "",
        "## 目的",
        "",
        "早い出口の低信頼度だけで棄却するのではなく、早い出口の情報から「finalまで進めても信頼ある通過にならないか」を直接予測できるかを調べた。",
        "棄却しなかった画像はすべてfinalまで進める前提にし、下側出口専用判定器そのものの省計算効果を測る。",
        "",
        "## 指標",
        "",
        "- 信頼ある通過: finalが正解し、final信頼度が要求通過精度を満たす閾値以上",
        "- 良品ロス: finalなら信頼ある通過になった画像を早期棄却した割合",
        "- 早期棄却率: finalまで進めずに下側出口で止めた割合",
        "- 平均計算量: finalのみを1.0とした相対計算量",
        "",
        "## 最良結果",
        "",
        "| 条件 | 要求通過精度 | 良品ロス上限 | 出口 | 特徴 | 判定器 | 早期棄却率 | final実行率 | 平均計算量 | 速度換算 | 実測良品ロス |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scenario"],
                    f"{100 * row['target_accuracy']:.1f}%",
                    f"{100 * row['max_lost_final_reliable_rate']:.1f}%",
                    f"exit{row['exit_idx']}",
                    row["feature_set"],
                    row["predictor"],
                    f"{100 * row['early_reject_rate']:.2f}%",
                    f"{100 * row['final_execution_rate']:.2f}%",
                    f"{row['avg_cost']:.4f}",
                    f"{row['speedup_vs_final_only']:.2f}x",
                    f"{100 * row['lost_final_reliable_rate']:.2f}%",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## 読み方",
        "",
        "raw_lower_confidenceが勝つなら、単純な低信頼度閾値で十分という意味になる。",
        "treeやlinearが勝つなら、出口信頼度だけでなく、エントロピー・出口間の変化・予測クラスなどを使う小型判定器に価値がある。",
        "特にtree_depth2/3で効果が出る場合、FPGAでは比較器とLUTに近い小回路として下側出口へ置ける可能性がある。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(payload: dict[str, Any], path: Path) -> None:
    rows = [row for row in best_rows(payload["rows"]) if abs(row["target_accuracy"] - 0.99) < 1e-9 and abs(row["max_lost_final_reliable_rate"] - 0.02) < 1e-9]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows[:8]
    width = 1180
    top = 76
    row_h = 42
    left = 340
    bar_w = 520
    height = top + row_h * len(rows) + 72
    max_gain = max(row["cost_reduction_vs_final_only"] for row in rows) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#0f172a;font-size:14px}.title{font-size:22px;font-weight:700}.small{font-size:12px;fill:#475569}</style>',
        '<text x="34" y="36" class="title">Late-recovery predictor: early reject gain</text>',
        '<text x="34" y="58" class="small">Target accepted accuracy 99%, lost final-reliable samples <= 2%. Bars show compute reduction vs final-only.</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        gain = row["cost_reduction_vs_final_only"]
        bw = bar_w * gain / max_gain
        label = row["scenario"].replace("_", " ")
        parts += [
            f'<text x="34" y="{y + 24}">{label}</text>',
            f'<rect x="{left}" y="{y + 8}" width="{bar_w}" height="22" fill="#e2e8f0"/>',
            f'<rect x="{left}" y="{y + 8}" width="{bw:.1f}" height="22" fill="#2563eb"/>',
            f'<text x="{left + bar_w + 18}" y="{y + 24}">-{100 * gain:.1f}% cost</text>',
            f'<text x="{left + bar_w + 138}" y="{y + 24}" class="small">{row["predictor"]}, reject {100 * row["early_reject_rate"]:.1f}%, lost {100 * row["lost_final_reliable_rate"]:.2f}%</text>',
        ]
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/late_recovery_predictor_001_summary.json")
    parser.add_argument("--markdown", default="docs/late_recovery_predictor_001.md")
    parser.add_argument("--svg", default="results/late_recovery_predictor_001.svg")
    parser.add_argument("--model-output-name", default="0000b_branchynet_reproduce_resnet56_cifar10")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--arch", default="resnet56", choices=["resnet56"])
    parser.add_argument("--exit-modules", default="")
    parser.add_argument("--branch-depths", default="3,2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-val-size", type=int, default=5000)
    parser.add_argument("--target-accuracies", nargs="*", type=float, default=[0.99, 0.995])
    parser.add_argument("--max-lost-rates", nargs="*", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--feature-sets", nargs="*", default=["scalar", "trace", "class_aware"])
    parser.add_argument("--exits", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    specs = mixture_specs()
    scenarios = required_scenarios(specs)
    exit_modules = [item.strip() for item in (args.exit_modules or DEFAULT_EXITS[args.arch]).split(",") if item.strip()]
    branch_depths = parse_csv_ints(args.branch_depths)
    costs = np.asarray(estimate_costs(args.arch, exit_modules), dtype=np.float32)
    model, checkpoint_payload = load_branchynet(args.model_output_name, args.arch, args.dataset, exit_modules, branch_depths, device)
    _, val_set, eval_set = make_datasets(args.dataset, args.threshold_val_size, seed=args.seed)
    val_loader = make_loader(val_set, args.batch_size)
    eval_loader = make_loader(eval_set, args.batch_size)
    base_transform = transform_eval(args.dataset)

    val_by_scenario = {}
    eval_by_scenario = {}
    for scenario in scenarios:
        val_by_scenario[scenario] = collect_scenario(model, val_loader, base_transform, scenario, device, model.exit_names, costs)
        eval_by_scenario[scenario] = collect_scenario(model, eval_loader, base_transform, scenario, device, model.exit_names, costs)

    rows: list[dict[str, Any]] = []
    num_classes = CIFAR_STATS[args.dataset]["classes"]
    for scenario_name, spec in specs.items():
        print(f"Training late-recovery predictors scenario={scenario_name}", flush=True)
        val_mix = concat_data([val_by_scenario[key] for key, _ in spec], [weight for _, weight in spec])
        eval_mix = concat_data([eval_by_scenario[key] for key, _ in spec], [weight for _, weight in spec])
        for target in args.target_accuracies:
            for max_lost in args.max_lost_rates:
                for exit_idx in args.exits:
                    for feature_set in args.feature_sets:
                        if exit_idx == 0 and feature_set == "trace":
                            continue
                        rows.extend(
                            run_one(
                                scenario_name,
                                target,
                                max_lost,
                                exit_idx,
                                feature_set,
                                val_mix,
                                eval_mix,
                                costs,
                                num_classes,
                                args.seed,
                            )
                        )

    payload = {
        "purpose": "Train tiny lower-exit predictors for whether a sample will recover to a reliable final accepted label.",
        "model": {
            "checkpoint_output_name": args.model_output_name,
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "dataset": args.dataset,
            "arch": args.arch,
            "exit_modules": exit_modules,
            "branch_depths": branch_depths,
            "exit_names": model.exit_names,
            "exit_costs": [float(x) for x in costs],
        },
        "definitions": {
            "final_reliable": "final exit is correct and final confidence passes the threshold calibrated for target accepted accuracy",
            "lost_final_reliable_rate": "fraction of all samples that final-only would reliably accept but the early rejector discards",
            "early_reject_rate": "fraction of samples stopped at the lower exit before final execution",
            "avg_cost": "final-only normalized compute cost is 1.0",
        },
        "mixture_specs": specs,
        "rows": rows,
        "best_rows": best_rows(rows),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    write_svg(payload, Path(args.svg))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "svg": args.svg, "rows": len(rows)}, ensure_ascii=False, indent=2), flush=True)
    model.close()


if __name__ == "__main__":
    main()
