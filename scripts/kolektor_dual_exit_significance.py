"""Compare dual-sided early exit against inspection baselines on KolektorSDD.

The key question is not whether early rejection can be fast.  It is whether it
is faster under the same inspection constraints:

* false pass among defects must stay below a target;
* good pass among normal samples must stay above a target.

Thresholds are selected on validation data and then fixed on evaluation data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from tqdm import tqdm

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


class ResNet18Branchy(nn.Module):
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
        self.final = nn.Sequential(base.avgpool, nn.Flatten(), nn.Linear(base.fc.in_features, 2))
        self.exit_costs = np.asarray([0.28, 0.58, 1.0], dtype=np.float32)
        self.exit_names = ["exit0", "exit1", "final"]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.stem(x)
        h = self.layer1(h)
        out0 = self.exit0(h)
        h = self.layer2(h)
        out1 = self.exit1(h)
        h = self.layer3(h)
        h = self.layer4(h)
        out2 = self.final(h)
        return [out0, out1, out2]


@torch.no_grad()
def collect_outputs(model: ResNet18Branchy, loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    labels_all, probs_all = [], []
    for x, y in tqdm(loader, desc="collect", leave=False):
        x = x.to(device)
        outputs = model(x)
        probs = [torch.softmax(logits, dim=1).detach().cpu().numpy() for logits in outputs]
        labels_all.append(y.numpy())
        probs_all.append(np.stack(probs, axis=1))
    labels = np.concatenate(labels_all).astype(np.int64)
    probs = np.concatenate(probs_all).astype(np.float32)
    return {
        "labels": labels,
        "p_defect": probs[:, :, 1],
        "exit_costs": model.exit_costs.copy(),
        "exit_names": np.asarray(model.exit_names, dtype=object),
    }


def metric(labels: np.ndarray, decisions: np.ndarray, costs: np.ndarray) -> dict[str, float | int | None]:
    # decisions: 0=pass as good, 1=reject as defect/unsafe
    good = labels == 0
    defect = labels == 1
    passed = decisions == 0
    rejected = decisions == 1
    false_pass = defect & passed
    good_loss = good & rejected
    avg_cost = float(np.mean(costs))
    return {
        "samples": int(len(labels)),
        "good_count": int(good.sum()),
        "defect_count": int(defect.sum()),
        "pass_rate_all": round_float(float(passed.mean())),
        "reject_rate_all": round_float(float(rejected.mean())),
        "good_pass_rate_good": round_float(float((good & passed).sum() / max(good.sum(), 1))),
        "good_loss_rate_good": round_float(float(good_loss.sum() / max(good.sum(), 1))),
        "false_pass_rate_defect": round_float(float(false_pass.sum() / max(defect.sum(), 1))),
        "defect_recall": round_float(float((defect & rejected).sum() / max(defect.sum(), 1))),
        "avg_cost": round_float(avg_cost),
        "speedup_vs_final_only": round_float(1.0 / avg_cost if avg_cost else None),
        "false_pass_count": int(false_pass.sum()),
        "good_loss_count": int(good_loss.sum()),
    }


def simulate_final(data: dict[str, np.ndarray], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    p = data["p_defect"][:, -1]
    decisions = (p >= threshold).astype(np.int64)
    costs = np.ones(len(p), dtype=np.float32)
    return decisions, costs


def simulate_bn(data: dict[str, np.ndarray], t0: float, t1: float, tf: float) -> tuple[np.ndarray, np.ndarray]:
    p = data["p_defect"]
    costs_ref = data["exit_costs"]
    decisions = np.ones(len(p), dtype=np.int64)
    costs = np.full(len(p), costs_ref[-1], dtype=np.float32)
    done = p[:, 0] <= t0
    decisions[done] = 0
    costs[done] = costs_ref[0]
    active = ~done
    done1 = active & (p[:, 1] <= t1)
    decisions[done1] = 0
    costs[done1] = costs_ref[1]
    final_active = ~(done | done1)
    decisions[final_active] = (p[final_active, -1] >= tf).astype(np.int64)
    return decisions, costs


def simulate_dual(data: dict[str, np.ndarray], pass0: float, rej0: float, pass1: float, rej1: float, tf: float) -> tuple[np.ndarray, np.ndarray]:
    p = data["p_defect"]
    costs_ref = data["exit_costs"]
    decisions = np.ones(len(p), dtype=np.int64)
    costs = np.full(len(p), costs_ref[-1], dtype=np.float32)

    pass_exit0 = p[:, 0] <= pass0
    reject_exit0 = p[:, 0] >= rej0
    done0 = pass_exit0 | reject_exit0
    decisions[pass_exit0] = 0
    decisions[reject_exit0] = 1
    costs[done0] = costs_ref[0]

    active = ~done0
    pass_exit1 = active & (p[:, 1] <= pass1)
    reject_exit1 = active & (p[:, 1] >= rej1)
    done1 = pass_exit1 | reject_exit1
    decisions[pass_exit1] = 0
    decisions[reject_exit1] = 1
    costs[done1] = costs_ref[1]

    final_active = ~(done0 | done1)
    decisions[final_active] = (p[final_active, -1] >= tf).astype(np.int64)
    return decisions, costs


def candidates(values: np.ndarray) -> list[float]:
    qs = np.asarray([0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0])
    out = sorted(set(float(x) for x in np.quantile(values, qs)))
    return out


def feasible(row: dict[str, float | int | None], max_false_pass: float, min_good_pass: float) -> bool:
    return (
        float(row["false_pass_rate_defect"] or 0.0) <= max_false_pass + 1e-12
        and float(row["good_pass_rate_good"] or 0.0) >= min_good_pass - 1e-12
    )


def better(candidate: dict, current: dict | None) -> bool:
    if current is None:
        return True
    c = candidate["val_metric"]
    b = current["val_metric"]
    return (
        float(c["avg_cost"]),
        -float(c["good_pass_rate_good"]),
        float(c["false_pass_rate_defect"]),
    ) < (
        float(b["avg_cost"]),
        -float(b["good_pass_rate_good"]),
        float(b["false_pass_rate_defect"]),
    )


def select_policies(val_data: dict[str, np.ndarray], max_false_pass: float, min_good_pass: float) -> list[dict]:
    rows: list[dict] = []
    cand0 = candidates(val_data["p_defect"][:, 0])
    cand1 = candidates(val_data["p_defect"][:, 1])
    candf = candidates(val_data["p_defect"][:, 2])

    best = None
    for tf in candf:
        decisions, costs = simulate_final(val_data, tf)
        row = metric(val_data["labels"], decisions, costs)
        if feasible(row, max_false_pass, min_good_pass):
            candidate = {"policy": "final_selective", "params": {"final_reject_threshold": tf}, "val_metric": row}
            if better(candidate, best):
                best = candidate
    if best is not None:
        rows.append(best)

    best = None
    for t0 in cand0:
        for t1 in cand1:
            for tf in candf:
                decisions, costs = simulate_bn(val_data, t0, t1, tf)
                row = metric(val_data["labels"], decisions, costs)
                if feasible(row, max_false_pass, min_good_pass):
                    candidate = {
                        "policy": "branchynet_upper_only",
                        "params": {"exit0_pass_threshold": t0, "exit1_pass_threshold": t1, "final_reject_threshold": tf},
                        "val_metric": row,
                    }
                    if better(candidate, best):
                        best = candidate
    if best is not None:
        rows.append(best)

    best = None
    for pass0 in cand0:
        for rej0 in cand0:
            if rej0 < pass0:
                continue
            for pass1 in cand1:
                for rej1 in cand1:
                    if rej1 < pass1:
                        continue
                    for tf in candf:
                        decisions, costs = simulate_dual(val_data, pass0, rej0, pass1, rej1, tf)
                        row = metric(val_data["labels"], decisions, costs)
                        if feasible(row, max_false_pass, min_good_pass):
                            candidate = {
                                "policy": "dual_sided_early_exit",
                                "params": {
                                    "exit0_pass_threshold": pass0,
                                    "exit0_reject_threshold": rej0,
                                    "exit1_pass_threshold": pass1,
                                    "exit1_reject_threshold": rej1,
                                    "final_reject_threshold": tf,
                                },
                                "val_metric": row,
                            }
                            if better(candidate, best):
                                best = candidate
    if best is not None:
        rows.append(best)
    return rows


def apply_policy(data: dict[str, np.ndarray], row: dict) -> dict:
    p = row["params"]
    if row["policy"] == "final_selective":
        decisions, costs = simulate_final(data, p["final_reject_threshold"])
    elif row["policy"] == "branchynet_upper_only":
        decisions, costs = simulate_bn(data, p["exit0_pass_threshold"], p["exit1_pass_threshold"], p["final_reject_threshold"])
    elif row["policy"] == "dual_sided_early_exit":
        decisions, costs = simulate_dual(
            data,
            p["exit0_pass_threshold"],
            p["exit0_reject_threshold"],
            p["exit1_pass_threshold"],
            p["exit1_reject_threshold"],
            p["final_reject_threshold"],
        )
    else:
        raise ValueError(row["policy"])
    return metric(data["labels"], decisions, costs)


def train_model(model: ResNet18Branchy, train_loader, val_loader, weights, epochs: int, device: torch.device) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best_state = None
    best_score = -1.0
    history = []
    loss_weights = [0.35, 0.65, 1.0]
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"branchy epoch {epoch + 1}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = sum(w * F.cross_entropy(logits, y, weight=weights, label_smoothing=0.02) for w, logits in zip(loss_weights, outputs))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val = collect_outputs(model, val_loader, device)
        final_decisions, final_costs = simulate_final(val, 0.5)
        m = metric(val["labels"], final_decisions, final_costs)
        score = float(m["good_pass_rate_good"] or 0.0) + float(m["defect_recall"] or 0.0)
        row = {"epoch": epoch + 1, "loss": round_float(np.mean(losses)), **m}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_score": round_float(best_score)}


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KolektorSDD dual-exit significance experiment",
        "",
        "## Purpose",
        "",
        "Compare final-only selective classification, ordinary upper-only BranchyNet, and the proposed dual-sided early exit under the same inspection constraints.",
        "",
        "Thresholds are selected on validation data and fixed on evaluation data.",
        "",
        "## Eval comparison",
        "",
        "| max false pass | min good pass | policy | eval good pass | eval false pass | eval avg cost | speedup |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["policy_rows"]:
        e = row["eval_metric"]
        lines.append(
            f"| {100 * row['constraint']['max_false_pass_rate_defect']:.1f}% | "
            f"{100 * row['constraint']['min_good_pass_rate_good']:.1f}% | "
            f"{row['policy']} | {100 * e['good_pass_rate_good']:.2f}% | "
            f"{100 * e['false_pass_rate_defect']:.2f}% | {e['avg_cost']:.4f} | {e['speedup_vs_final_only']:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(payload: dict, path: Path) -> None:
    rows = payload["policy_rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1120
    row_h = 34
    height = 92 + row_h * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827;font-size:13px}.title{font-size:22px;font-weight:700}.small{fill:#4b5563;font-size:12px}</style>',
        '<text x="34" y="36" class="title">KolektorSDD policy comparison</text>',
        '<text x="34" y="58" class="small">Lower cost is better; all policies use validation-selected thresholds.</text>',
    ]
    max_speed = max(float(r["eval_metric"]["speedup_vs_final_only"]) for r in rows) if rows else 1.0
    for i, row in enumerate(rows):
        y = 86 + i * row_h
        speed = float(row["eval_metric"]["speedup_vs_final_only"])
        bw = 360 * speed / max_speed
        label = f"{row['policy']}  FP<={100*row['constraint']['max_false_pass_rate_defect']:.0f}% GP>={100*row['constraint']['min_good_pass_rate_good']:.0f}%"
        parts += [
            f'<text x="34" y="{y + 18}">{label}</text>',
            f'<rect x="500" y="{y}" width="360" height="20" fill="#e5e7eb"/>',
            f'<rect x="500" y="{y}" width="{bw:.1f}" height="20" fill="#2563eb"/>',
            f'<text x="880" y="{y + 16}">{speed:.2f}x</text>',
            f'<text x="950" y="{y + 16}" class="small">good {100*row["eval_metric"]["good_pass_rate_good"]:.1f}%, false pass {100*row["eval_metric"]["false_pass_rate_defect"]:.1f}%</text>',
        ]
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/kolektor_dual_exit_significance_001_summary.json")
    parser.add_argument("--markdown", default="docs/kolektor_dual_exit_significance_001.md")
    parser.add_argument("--svg", default="results/kolektor_dual_exit_significance_001.svg")
    parser.add_argument("--checkpoint", default="artifacts/kolektor_dual_exit_significance_001/resnet18_branchy.pt")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.1])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.95, 0.98])
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

    val_data = collect_outputs(model, val_loader, device)
    eval_data = collect_outputs(model, eval_loader, device)
    policy_rows = []
    for max_fp in args.max_false_pass_rates:
        for min_gp in args.min_good_pass_rates:
            selected = select_policies(val_data, max_fp, min_gp)
            for row in selected:
                policy_rows.append(
                    {
                        "constraint": {"max_false_pass_rate_defect": max_fp, "min_good_pass_rate_good": min_gp},
                        **row,
                        "eval_metric": apply_policy(eval_data, row),
                    }
                )

    payload = {
        "purpose": "Show whether proposed dual-sided early exit is useful under fixed inspection constraints.",
        "dataset": {
            "name": "KolektorSDD",
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
            "split_counts": {
                key: {"samples": len(value), "defects": int(sum(s.label for s in value)), "good": int(sum(1 - s.label for s in value))}
                for key, value in split.items()
            },
        },
        "model": {"arch": "ResNet18Branchy", "checkpoint": args.checkpoint, "exit_costs": model.exit_costs.tolist(), "training": training},
        "policy_rows": policy_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    write_svg(payload, Path(args.svg))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "svg": args.svg, "rows": len(policy_rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
