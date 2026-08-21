"""Deep KSDD2 split/seed bias audit.

This joins saved U-Net/ResNet50 scores with KSDD2 image paths and mask-area
statistics.  The goal is to explain whether the weak seed is likely caused by
harder defect samples, validation threshold instability, or a broader model
failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.ksdd2_conservative_safe_exit_training import download_and_extract, find_samples, make_split


def round_float(value, ndigits: int = 6):
    if value is None:
        return None
    return round(float(value), ndigits)


def mask_area_ratio(mask_path: Path | None) -> float:
    if mask_path is None or not mask_path.exists():
        return 0.0
    with Image.open(mask_path) as image:
        arr = np.asarray(image.convert("L"))
    return float((arr > 0).sum() / max(arr.size, 1))


def series_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"[_-]?\d+$", "", stem)
    parent = path.parent.name
    grand = path.parent.parent.name if path.parent.parent else ""
    return f"{grand}/{parent}/{stem}"


def load_scores(scores_dir: Path, seed: int) -> dict[str, np.ndarray]:
    path = scores_dir / f"seed_{seed}_scores.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path))


def metric(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    good = labels == 0
    defect = labels == 1
    reject = scores >= threshold
    false_pass = defect & ~reject
    return {
        "threshold": round_float(threshold),
        "good_pass_rate": round_float(float((good & ~reject).sum() / max(good.sum(), 1))),
        "false_pass_rate": round_float(float(false_pass.sum() / max(defect.sum(), 1))),
        "false_pass_count": int(false_pass.sum()),
        "good_count": int(good.sum()),
        "defect_count": int(defect.sum()),
    }


def threshold_candidates(scores: np.ndarray, count: int) -> np.ndarray:
    unique = np.unique(scores.astype(np.float64))
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) >= 2 else np.asarray([], dtype=np.float64)
    pooled = np.unique(np.concatenate([unique, mids]))
    if len(pooled) > count:
        pooled = np.unique(np.quantile(pooled, np.linspace(0, 1, count)))
    return pooled


def best_safe(labels: np.ndarray, scores: np.ndarray, max_false_pass: float, count: int) -> dict:
    best = None
    for threshold in threshold_candidates(scores, count):
        row = metric(labels, scores, float(threshold))
        if row["false_pass_rate"] <= max_false_pass + 1e-12:
            if best is None or row["good_pass_rate"] > best["good_pass_rate"]:
                best = row
    if best is None:
        best = metric(labels, scores, float(np.max(scores) + 1e-6))
    return best


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round_float(arr.mean()),
        "median": round_float(np.quantile(arr, 0.5)),
        "q10": round_float(np.quantile(arr, 0.1)),
        "q90": round_float(np.quantile(arr, 0.9)),
        "min": round_float(arr.min()),
        "max": round_float(arr.max()),
    }


def make_gallery(rows: list[dict], output: Path, max_images: int) -> str | None:
    selected = rows[:max_images]
    if not selected:
        return None
    thumb_w, thumb_h = 260, 120
    cols = 2
    rows_count = int(np.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + 42)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, row in enumerate(selected):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 42)
        try:
            with Image.open(row["image_path"]) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_w, thumb_h))
                canvas.paste(img, (x, y))
        except Exception:
            pass
        caption = f"seed {row['seed']} {row['score_name']} score={row['score']:.4f} area={row['mask_area_ratio']:.5f}"
        draw.text((x + 4, y + thumb_h + 3), caption[:62], fill="black", font=font)
        draw.text((x + 4, y + thumb_h + 18), Path(row["image_path"]).name[:62], fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return str(output)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "score_name",
        "score",
        "label",
        "mask_area_ratio",
        "series_key",
        "image_path",
        "mask_path",
        "is_false_pass_at_safe5",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KSDD2 split bias deep audit",
        "",
        "Purpose: explain whether seed 456 is a likely split/difficulty outlier.",
        "",
        "## Seed summary",
        "",
        "| seed | score | good pass at <=5% false-pass | false-pass | false-pass count | defect score q10 | defect mask area median | false-pass area median |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seed_summaries"]:
        lines.append(
            f"| {row['seed']} | {row['score_name']} | {100*row['safe5']['good_pass_rate']:.2f}% | "
            f"{100*row['safe5']['false_pass_rate']:.2f}% | {row['safe5']['false_pass_count']} | "
            f"{row['defect_score_summary'].get('q10')} | {row['defect_area_summary'].get('median')} | "
            f"{row['false_pass_area_summary'].get('median')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If seed 456 has lower defect-score q10 or a different false-pass area profile, it supports the split/difficulty-bias hypothesis.",
        "- If false-pass samples are concentrated in a few series keys, image-series grouping should be added to the evaluation protocol.",
        "- If seed 456 is not explainable by area or series, the foundation model itself is unstable and should not be treated as completed.",
        "",
        "## Output files",
        "",
        f"- CSV: `{payload['csv']}`",
    ]
    if payload.get("gallery"):
        lines.append(f"- Gallery: `{payload['gallery']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="results/ksdd2_unet_resnet50_foundation_recheck_caviar9_001_scores")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output", default="results/ksdd2_split_bias_deep_audit_001.json")
    parser.add_argument("--markdown", default="docs/ksdd2_split_bias_deep_audit_001.md")
    parser.add_argument("--csv", default="results/ksdd2_split_bias_deep_audit_001_samples.csv")
    parser.add_argument("--gallery", default="results/ksdd2_split_bias_deep_audit_001_false_pass_gallery.jpg")
    parser.add_argument("--seeds", nargs="*", type=int, default=[123, 456, 789])
    parser.add_argument("--score-names", nargs="*", default=["max_score", "topk_score"])
    parser.add_argument("--max-threshold-candidates", type=int, default=300)
    parser.add_argument("--max-gallery-images", type=int, default=18)
    args = parser.parse_args()

    cache_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    samples = find_samples(download_and_extract(cache_root / "kolektor_sdd2"))
    all_rows = []
    seed_summaries = []
    for seed in args.seeds:
        split = make_split(samples, seed)
        eval_samples = split["eval"]
        score_data = load_scores(Path(args.scores_dir), seed)
        labels = score_data["test_labels"].astype(np.int64)
        if len(eval_samples) != len(labels):
            raise RuntimeError(f"sample/score length mismatch for seed {seed}: {len(eval_samples)} vs {len(labels)}")

        area_cache = [mask_area_ratio(sample.mask_path) for sample in eval_samples]
        for score_name in args.score_names:
            scores = score_data[f"test_{score_name}"].astype(np.float32)
            safe5 = best_safe(labels, scores, 0.05, args.max_threshold_candidates)
            false_pass_mask = (labels == 1) & (scores < safe5["threshold"])
            defect_scores = scores[labels == 1].tolist()
            defect_areas = [area_cache[i] for i, label in enumerate(labels) if label == 1]
            fp_areas = [area_cache[i] for i, is_fp in enumerate(false_pass_mask) if is_fp]
            seed_summaries.append(
                {
                    "seed": seed,
                    "score_name": score_name,
                    "safe5": safe5,
                    "defect_score_summary": summarize_numeric(defect_scores),
                    "defect_area_summary": summarize_numeric(defect_areas),
                    "false_pass_area_summary": summarize_numeric(fp_areas),
                }
            )
            for i, sample in enumerate(eval_samples):
                if labels[i] != 1:
                    continue
                row = {
                    "seed": seed,
                    "score_name": score_name,
                    "score": float(scores[i]),
                    "label": int(labels[i]),
                    "mask_area_ratio": float(area_cache[i]),
                    "series_key": series_key(sample.image_path),
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path or ""),
                    "is_false_pass_at_safe5": bool(false_pass_mask[i]),
                }
                all_rows.append(row)

    false_pass_rows = sorted(
        [row for row in all_rows if row["is_false_pass_at_safe5"]],
        key=lambda row: (row["seed"] != 456, row["score"]),
    )
    gallery = make_gallery(false_pass_rows, Path(args.gallery), args.max_gallery_images)
    write_csv(all_rows, Path(args.csv))
    payload = {
        "purpose": "Join KSDD2 foundation scores with image paths and mask-area statistics.",
        "seed_summaries": seed_summaries,
        "csv": args.csv,
        "gallery": gallery,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "csv": args.csv, "gallery": gallery}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
