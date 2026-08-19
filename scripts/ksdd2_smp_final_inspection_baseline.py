"""KSDD2 final inspection baseline with common pretrained segmentation models.

This is not meant to be the proposed method.  It uses widely adopted
segmentation-models-pytorch architectures to build a stronger final inspection
model quickly, so early-exit experiments can later use a credible base model.
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

from scripts.ksdd2_conservative_safe_exit_training import download_and_extract, find_samples
from scripts.ksdd2_unet_inspection_baseline import (
    aggregate,
    collect_scores,
    evaluate_thresholds,
    make_curve_rows,
    make_loader,
    maybe_pixel_metrics,
    plot_curve,
    score_auc,
    split_counts,
    train_one_epoch,
    write_markdown,
)
from scripts.ksdd2_conservative_safe_exit_training import make_split
from scripts.train_kolektor_strong_final import round_float, set_seed
from src.experiment_paths import ensure_dirs


def make_smp_model(architecture: str, encoder: str, encoder_weights: str | None) -> nn.Module:
    import segmentation_models_pytorch as smp

    if architecture == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
    if architecture == "fpn":
        return smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
    raise ValueError(f"Unsupported architecture: {architecture}")


def run_one_seed(args: argparse.Namespace, samples, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    split = make_split(samples, seed)
    image_size = (args.image_height, args.image_width)
    train_loader = make_loader(split["train"], image_size, args.batch_size, train=True)
    val_loader = make_loader(split["val"], image_size, args.batch_size, train=False)
    test_loader = make_loader(split["eval"], image_size, args.batch_size, train=False)

    model = make_smp_model(args.architecture, args.encoder, args.encoder_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_state = None
    best_score = -math.inf
    history = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, opt, device, args.positive_pixel_weight)
        scheduler.step()
        val_scores = collect_scores(model, val_loader, device, args.topk_fraction)
        auc = score_auc(val_scores["labels"], val_scores["topk_score"])
        row = {"epoch": epoch + 1, "loss": round_float(loss), **auc}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        history.append(row)
        score = float(auc["image_auroc"] or 0.0) + float(auc["image_aupr"] or 0.0)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = Path(args.checkpoint_dir) / f"{args.architecture}_{args.encoder}" / f"seed_{seed}" / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": list(image_size),
            "architecture": args.architecture,
            "encoder": args.encoder,
            "encoder_weights": args.encoder_weights,
        },
        checkpoint,
    )

    val_scores = collect_scores(model, val_loader, device, args.topk_fraction)
    test_scores = collect_scores(model, test_loader, device, args.topk_fraction)
    curve_rows = {name: make_curve_rows(test_scores, name, args.curve_points) for name in ["max_score", "topk_score"]}
    if seed == args.seeds[0]:
        plot_curve(curve_rows, Path(args.curve_png))
    return {
        "seed": seed,
        "split_counts": split_counts(split),
        "checkpoint": str(checkpoint),
        "history": history,
        "val_auc": {name: score_auc(val_scores["labels"], val_scores[name]) for name in ["max_score", "topk_score"]},
        "test_auc": {name: score_auc(test_scores["labels"], test_scores[name]) for name in ["max_score", "topk_score"]},
        "pixel_metrics_sampled": maybe_pixel_metrics(test_scores),
        "threshold_rows": evaluate_thresholds(val_scores, test_scores, args),
        "test_curve_rows": curve_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ksdd2_smp_final_inspection_baseline_001_summary.json")
    parser.add_argument("--markdown", default="docs/ksdd2_smp_final_inspection_baseline_001.md")
    parser.add_argument("--checkpoint-dir", default="artifacts/ksdd2_smp_final_inspection_baseline_001")
    parser.add_argument("--curve-png", default="results/ksdd2_smp_final_inspection_baseline_001_tradeoff.png")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--architecture", default="unetplusplus", choices=["unet", "unetplusplus", "fpn"])
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=704)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-pixel-weight", type=float, default=60.0)
    parser.add_argument("--topk-fraction", type=float, default=0.001)
    parser.add_argument("--seeds", nargs="*", type=int, default=[123, 456])
    parser.add_argument("--max-false-pass-rates", nargs="*", type=float, default=[0.0, 0.01, 0.05])
    parser.add_argument("--min-good-pass-rates", nargs="*", type=float, default=[0.90, 0.95])
    parser.add_argument("--max-threshold-candidates", type=int, default=101)
    parser.add_argument("--curve-points", type=int, default=120)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required. In Colab, select a GPU runtime before running.")

    cache_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    samples = find_samples(download_and_extract(cache_root / "kolektor_sdd2"))
    seed_results = [run_one_seed(args, samples, seed, device) for seed in args.seeds]
    payload = {
        "purpose": "Build a strong off-the-shelf final inspection baseline before early-exit research.",
        "dataset": {
            "name": "KolektorSDD2",
            "sample_count": len(samples),
            "defect_count": int(sum(s.label for s in samples)),
            "good_count": int(sum(1 - s.label for s in samples)),
        },
        "model": {
            "library": "segmentation-models-pytorch",
            "architecture": args.architecture,
            "encoder": args.encoder,
            "encoder_weights": args.encoder_weights,
        },
        "training_config": vars(args),
        "seed_results": seed_results,
        "aggregate_rows": aggregate(seed_results),
        "curve_png": args.curve_png,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "curve_png": args.curve_png}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
