"""Conservative safe-exit training for KolektorSDD.

This version fixes the previous auxiliary-safe experiment's weak supervision.
Instead of teaching early exits "good label == safe", it teaches:

    safe = ground-truth good AND final classifier is highly confident good.

Ambiguous good samples are treated as unsafe for the early exit.  This is a
stricter inspection-oriented objective: an early pass should mean "safe to pass
now", not merely "probably normal".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from tqdm import tqdm

from scripts.kolektor_dual_exit_significance import (
    better,
    candidates,
    eval_feasible,
    feasible,
    metric,
    simulate_bn,
    simulate_final,
)
from scripts.kolektor_late_recovery_predictor import choose_bn, choose_final_threshold
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


class ResNet18ConservativeSafeBranchy(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.exit0 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))
        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2))
        self.safe0 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))
        self.safe1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 2))
        self.final = nn.Sequential(base.avgpool, nn.Flatten(), nn.Linear(base.fc.in_features, 2))
        self.exit_costs = np.asarray([0.28, 0.58, 1.0], dtype=np.float32)
        self.exit_names = ["exit0", "exit1", "final"]

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        h = self.stem(x)
        h1 = self.layer1(h)
        out0 = self.exit0(h1)
        safe0 = self.safe0(h1)
        h2 = self.layer2(h1)
        out1 = self.exit1(h2)
        safe1 = self.safe1(h2)
        h = self.layer3(h2)
        h = self.layer4(h)
        out2 = self.final(h)
        return [out0, out1, out2], [safe0, safe1]


@torch.no_grad()
def collect_outputs(model: ResNet18ConservativeSafeBranchy, loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    labels_all, probs_all, safe_all = [], [], []
    for x, y in tqdm(loader, desc="collect", leave=False):
        x = x.to(device)
        outputs, safe_outputs = model(x)
        probs = [torch.softmax(logits, dim=1).detach().cpu().numpy() for logits in outputs]
        safe_probs = [torch.softmax(logits, dim=1).detach().cpu().numpy() for logits in safe_outputs]
        labels_all.append(y.numpy())
        probs_all.append(np.stack(probs, axis=1))
        safe_all.append(np.stack(safe_probs, axis=1))
    labels = np.concatenate(labels_all).astype(np.int64)
    probs = np.concatenate(probs_all).astype(np.float32)
    safe_probs = np.concatenate(safe_all).astype(np.float32)
    return {
        "labels": labels,
        "p_defect": probs[:, :, 1],
        "p_safe_good": safe_probs[:, :, 1],
        "exit_costs": model.exit_costs.copy(),
        "exit_names": np.asarray(model.exit_names, dtype=object),
    }


def conservative_safe_target(y: torch.Tensor, final_logits: torch.Tensor, final_good_threshold: float) -> torch.Tensor:
    final_prob = torch.softmax(final_logits.detach(), dim=1)
    p_good = final_prob[:, 0]
    return ((y == 0) & (p_good >= final_good_threshold)).long()


def train_model(
    model: ResNet18ConservativeSafeBranchy,
    train_loader,
    val_loader,
    cls_weights,
    epochs: int,
    device: torch.device,
    safe_loss_weight: float,
    unsafe_loss_weight: float,
    final_good_threshold: float,
    safe_warmup_epochs: int,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best_state = None
    best_score = -1.0
    history = []
    class_loss_weights = [0.35, 0.65, 1.0]
    safe_weights = torch.tensor([unsafe_loss_weight, 1.0], dtype=torch.float32, device=device)
    for epoch in range(epochs):
        model.train()
        losses = []
        safe_positive = 0
        safe_total = 0
        for x, y in tqdm(train_loader, desc=f"conservative safe epoch {epoch + 1}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            outputs, safe_outputs = model(x)
            class_loss = sum(
                w * F.cross_entropy(logits, y, weight=cls_weights, label_smoothing=0.02)
                for w, logits in zip(class_loss_weights, outputs)
            )
            if epoch + 1 <= safe_warmup_epochs:
                safe_loss = torch.zeros((), device=device)
            else:
                y_safe = conservative_safe_target(y, outputs[-1], final_good_threshold)
                safe_positive += int(y_safe.sum().detach().cpu())
                safe_total += int(y_safe.numel())
                safe_loss = sum(F.cross_entropy(logits, y_safe, weight=safe_weights, label_smoothing=0.0) for logits in safe_outputs)
            loss = class_loss + safe_loss_weight * safe_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val = collect_outputs(model, val_loader, device)
        final_decisions, final_costs = simulate_final(val, 0.5)
        m = metric(val["labels"], final_decisions, final_costs)
        score = float(m["good_pass_rate_good"] or 0.0) + float(m["defect_recall"] or 0.0)
        row = {
            "epoch": epoch + 1,
            "loss": round_float(np.mean(losses)),
            "safe_positive_rate_train": round_float(safe_positive / safe_total if safe_total else 0.0),
            **m,
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_score": round_float(best_score)}


def simulate_safe_exit(
    data: dict[str, np.ndarray],
    pass0: float,
    reject0: float,
    pass1: float,
    reject1: float,
    final_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    p_safe = data["p_safe_good"]
    p_defect = data["p_defect"]
    costs_ref = data["exit_costs"]
    decisions = np.ones(len(p_defect), dtype=np.int64)
    costs = np.full(len(p_defect), costs_ref[-1], dtype=np.float32)

    pass_exit0 = p_safe[:, 0] >= pass0
    reject_exit0 = p_safe[:, 0] <= reject0
    done0 = pass_exit0 | reject_exit0
    decisions[pass_exit0] = 0
    decisions[reject_exit0] = 1
    costs[done0] = costs_ref[0]

    active = ~done0
    pass_exit1 = active & (p_safe[:, 1] >= pass1)
    reject_exit1 = active & (p_safe[:, 1] <= reject1)
    done1 = pass_exit1 | reject_exit1
    decisions[pass_exit1] = 0
    decisions[reject_exit1] = 1
    costs[done1] = costs_ref[1]

    final_active = ~(done0 | done1)
    decisions[final_active] = (p_defect[final_active, -1] >= final_threshold).astype(np.int64)
    return decisions, costs


def choose_safe_exit_policy(data: dict[str, np.ndarray], max_false_pass: float, min_good_pass: float, max_threshold_candidates: int) -> dict | None:
    final = choose_final_threshold(data, max_false_pass, min_good_pass, max_threshold_candidates)
    if final is None:
        return None
    final_threshold = final["params"]["final_reject_threshold"]
    cand0 = candidates(data["p_safe_good"][:, 0], max_threshold_candidates)
    cand1 = candidates(data["p_safe_good"][:, 1], max_threshold_candidates)
    best = None
    for pass0 in cand0:
        for reject0 in cand0:
            if reject0 > pass0:
                continue
            for pass1 in cand1:
                for reject1 in cand1:
                    if reject1 > pass1:
                        continue
                    decisions, costs = simulate_safe_exit(data, pass0, reject0, pass1, reject1, final_threshold)
                    row = metric(data["labels"], decisions, costs)
                    if feasible(row, max_false_pass, min_good_pass):
                        cand = {
                            "policy": "conservative_safe_dual_exit",
                            "params": {
                                "exit0_safe_pass_threshold": pass0,
                                "exit0_safe_reject_threshold": reject0,
                                "exit1_safe_pass_threshold": pass1,
                                "exit1_safe_reject_threshold": reject1,
                                "final_reject_threshold": final_threshold,
                            },
                            "val_metric": row,
                        }
                        if better(cand, best):
                            best = cand
    return best


def apply_policy(data: dict[str, np.ndarray], row: dict) -> dict:
    p = row["params"]
    if row["policy"] == "final_selective":
        decisions, costs = simulate_final(data, p["final_reject_threshold"])
    elif row["policy"] == "branchynet_upper_only":
        decisions, costs = simulate_bn(data, p["exit0_pass_threshold"], p["exit1_pass_threshold"], p["final_reject_threshold"])
    elif row["policy"] == "conservative_safe_dual_exit":
        decisions, costs = simulate_safe_exit(
            data,
            p["exit0_safe_pass_threshold"],
            p["exit0_safe_reject_threshold"],
            p["exit1_safe_pass_threshold"],
            p["exit1_safe_reject_threshold"],
            p["final_reject_threshold"],
        )
    else:
        raise ValueError(row["policy"])
    return metric(data["labels"], decisions, costs)


def run_one_seed(args: argparse.Namespace, samples, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    split = split_by_item(samples, seed)
    image_size = (args.image_height, args.image_width)
    train_loader = make_loader(split["train"], image_size, args.batch_size, train=True)
    val_loader = make_loader(split["val"], image_size, args.batch_size, train=False)
    eval_loader = make_loader(split["eval"], image_size, args.batch_size, train=False)
    model = ResNet18ConservativeSafeBranchy(pretrained=True).to(device)
    training = train_model(
        model,
        train_loader,
        val_loader,
        class_weight(split["train"], device),
        args.epochs,
        device,
        args.safe_loss_weight,
        args.unsafe_loss_weight,
        args.final_good_threshold,
        args.safe_warmup_epochs,
    )
    checkpoint = Path(args.checkpoint_dir) / f"seed_{seed}" / "resnet18_conservative_safe_branchy.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "image_size": list(image_size), "exit_costs": model.exit_costs.tolist()}, checkpoint)

    val_data = collect_outputs(model, val_loader, device)
    eval_data = collect_outputs(model, eval_loader, device)
    policy_rows = []
    for max_fp in args.max_false_pass_rates:
        for min_gp in args.min_good_pass_rates:
            constraint = {"max_false_pass_rate_defect": max_fp, "min_good_pass_rate_good": min_gp}
            selected = [
                choose_final_threshold(val_data, max_fp, min_gp, args.max_threshold_candidates),
                choose_bn(val_data, max_fp, min_gp, args.max_threshold_candidates),
                choose_safe_exit_policy(val_data, max_fp, min_gp, args.max_threshold_candidates),
            ]
            for row in selected:
                if row is None:
                    continue
                row_out = {"seed": seed, "constraint": constraint, **row, "eval_metric": apply_policy(eval_data, row)}
                row_out["eval_feasible"] = eval_feasible(row_out)
                policy_rows.append(row_out)
    return {
        "seed": seed,
        "split_counts": {
            key: {"samples": len(value), "defects": int(sum(s.label for s in value)), "good": int(sum(1 - s.label for s in value))}
            for key, value in split.items()
        },
        "checkpoint": str(checkpoint),
        "training": training,
        "policy_rows": policy_rows,
    }


def aggregate_rows(seed_results: list[dict]) -> list[dict]:
    rows = [row for result in seed_results for row in result["policy_rows"]]
    groups: dict[tuple[float, float, str], list[dict]] = {}
    for row in rows:
        c = row["constraint"]
        key = (float(c["max_false_pass_rate_defect"]), float(c["min_good_pass_rate_good"]), row["policy"])
        groups.setdefault(key, []).append(row)
    out = []
    for (max_fp, min_gp, policy), items in sorted(groups.items()):
        speeds = [float(r["eval_metric"]["speedup_vs_final_only"]) for r in items]
        false_pass = [float(r["eval_metric"]["false_pass_rate_defect"]) for r in items]
        good_pass = [float(r["eval_metric"]["good_pass_rate_good"]) for r in items]
        out.append(
            {
                "max_false_pass_rate_defect": max_fp,
                "min_good_pass_rate_good": min_gp,
                "policy": policy,
                "seeds": len(items),
                "eval_feasible_seeds": int(sum(bool(r.get("eval_feasible")) for r in items)),
                "mean_speedup": round_float(np.mean(speeds)),
                "mean_good_pass_rate_good": round_float(np.mean(good_pass)),
                "mean_false_pass_rate_defect": round_float(np.mean(false_pass)),
                "worst_false_pass_rate_defect": round_float(np.max(false_pass)),
                "worst_good_pass_rate_good": round_float(np.min(good_pass)),
            }
        )
    return out


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KolektorSDD conservative safe-exit training",
        "",
        "Safe labels are conservative: only ground-truth good samples that the final classifier also marks as high-confidence good are safe.",
        "",
        "## Aggregate",
        "",
        "| max false pass | min good pass | policy | feasible seeds | mean good pass | mean false pass | worst false pass | mean speedup |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_rows"]:
        lines.append(
            f"| {100 * row['max_false_pass_rate_defect']:.1f}% | {100 * row['min_good_pass_rate_good']:.1f}% | "
            f"{row['policy']} | {row['eval_feasible_seeds']}/{row['seeds']} | "
            f"{100 * row['mean_good_pass_rate_good']:.2f}% | {100 * row['mean_false_pass_rate_defect']:.2f}% | "
            f"{100 * row['worst_false_pass_rate_defect']:.2f}% | {row['mean_speedup']:.2f}x |"
        )
    lines += [
        "",
        "## Per-seed rows",
        "",
        "| seed | max false pass | min good pass | policy | feasible | good pass | false pass | avg cost | speedup |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["seed_results"]:
        for row in result["policy_rows"]:
            e = row["eval_metric"]
            lines.append(
                f"| {row['seed']} | {100 * row['constraint']['max_false_pass_rate_defect']:.1f}% | "
                f"{100 * row['constraint']['min_good_pass_rate_good']:.1f}% | {row['policy']} | "
                f"{'yes' if row.get('eval_feasible') else 'no'} | {100 * e['good_pass_rate_good']:.2f}% | "
                f"{100 * e['false_pass_rate_defect']:.2f}% | {e['avg_cost']:.4f} | {e['speedup_vs_final_only']:.2f}x |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/kolektor_conservative_safe_exit_training_001_summary.json")
    parser.add_argument("--markdown", default="docs/kolektor_conservative_safe_exit_training_001.md")
    parser.add_argument("--checkpoint-dir", default="artifacts/kolektor_conservative_safe_exit_training_001")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seeds", nargs="*", type=int, default=[123, 456, 789])
    parser.add_argument("--safe-loss-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-loss-weight", type=float, default=4.0)
    parser.add_argument("--final-good-threshold", type=float, default=0.9)
    parser.add_argument("--safe-warmup-epochs", type=int, default=5)
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.1])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.90, 0.95])
    parser.add_argument("--max-threshold-candidates", type=int, default=13)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    cache_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    samples = find_samples(download_and_extract(cache_root / "kolektor_sdd"))
    seed_results = [run_one_seed(args, samples, seed, device) for seed in args.seeds]
    payload = {
        "purpose": "Reduce overfitting by conservative safe labels and multi-seed evaluation.",
        "dataset": {
            "name": "KolektorSDD",
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
        },
        "training_config": {
            "epochs": args.epochs,
            "safe_loss_weight": args.safe_loss_weight,
            "unsafe_loss_weight": args.unsafe_loss_weight,
            "final_good_threshold": args.final_good_threshold,
            "safe_warmup_epochs": args.safe_warmup_epochs,
            "seeds": args.seeds,
        },
        "seed_results": seed_results,
        "aggregate_rows": aggregate_rows(seed_results),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "seeds": args.seeds}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
