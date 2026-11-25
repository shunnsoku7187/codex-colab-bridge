"""Evaluate every labelled test image of one category against its exact K=3000 bank."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import torch
from pytorch_nndct.apis import torch_quantizer

from evaluate_quantized_pilot import member_bytes, patches, preprocess, score
from quantize_wrn_features import FeatureExtractor

SPEC = Path("/work/evaluation_spec.json")
QUANT_DIR = Path("/out/quantize_result_packed_20pc")


def auc(rows: list[dict], name: str) -> float:
    good = [row[name] for row in rows if row["label"] == 0]
    defect = [row[name] for row in rows if row["label"] == 1]
    if not good or not defect:
        raise RuntimeError("Both labels required for AUROC")
    return sum((d > g) + 0.5 * (d == g) for d in defect for g in good) / (len(good) * len(defect))


def quality(rows: list[dict], name: str, threshold: float) -> dict:
    good = [row for row in rows if row["label"] == 0]
    defect = [row for row in rows if row["label"] == 1]
    return {
        "image_auroc": auc(rows, name),
        "good_pass_rate": sum(row[name] < threshold for row in good) / len(good),
        "false_pass_rate_defect": sum(row[name] < threshold for row in defect) / len(defect),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--output-dir", default="/out/full_eval_20pc")
    args = parser.parse_args()
    category = args.category
    spec = next((row for row in json.loads(SPEC.read_text()) if row["category"] == category), None)
    if spec is None:
        raise RuntimeError(f"Unknown category: {category}")
    archive_path = Path(args.archive or f"/eval/{category}.tar")
    out = Path(args.output_dir) / f"{category}.json"
    model = FeatureExtractor().eval()
    with tarfile.open(archive_path, "r") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if manifest["category"] != category:
            raise RuntimeError("Archive category mismatch")
        bank_entry = next(row for row in manifest["files"] if row["type"] == "bank")
        images = [row for row in manifest["files"] if row["type"] == "image"]
        if not images:
            raise RuntimeError("No test images")
        bank_bytes = member_bytes(archive, bank_entry["archive_path"], bank_entry["sha256"])
        bank_np = np.load(io.BytesIO(bank_bytes), allow_pickle=False)
        if hashlib.sha256(np.ascontiguousarray(bank_np).tobytes()).hexdigest() != spec["bank_raw_sha256"]:
            raise RuntimeError("Selected research bank hash mismatch")
        dimension = 512 if spec["out_indices"] == [2] else 768
        if bank_np.shape != (3000, dimension):
            raise RuntimeError(f"Unexpected bank shape: {bank_np.shape}")
        bank = torch.from_numpy(bank_np.astype(np.float32, copy=False))
        first = preprocess(member_bytes(archive, images[0]["archive_path"], images[0]["sha256"]))
        quantizer = torch_quantizer("test", model, (first,), device=torch.device("cpu"), output_dir=str(QUANT_DIR))
        quant_model = quantizer.quant_model.eval()
        rows = []
        with torch.no_grad():
            for index, entry in enumerate(images):
                image = first if index == 0 else preprocess(member_bytes(archive, entry["archive_path"], entry["sha256"]))
                float_patches = patches(model(image), spec["out_indices"])
                quant_patches = patches(quant_model(image), spec["out_indices"])
                float_topk, _ = score(float_patches, bank, spec["topk_fraction"])
                quant_topk, _ = score(quant_patches, bank, spec["topk_fraction"])
                delta = (float_patches - quant_patches).abs()
                values = (float_topk, quant_topk, delta.mean().item(), delta.max().item())
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f"Non-finite result: {entry['archive_path']}")
                rows.append({
                    "image": entry["archive_path"], "image_sha256": entry["sha256"],
                    "label": entry["label"], "float_topk": float_topk,
                    "quant_topk": quant_topk, "patch_mean_abs_delta": values[2],
                    "patch_max_abs_delta": values[3],
                })
    threshold = spec["frozen_threshold"]
    report = {
        "category": category, "scope": "all materialized test images, selected K=3000 float bank, 20pc-quantized CNN, unchanged frozen threshold",
        "bank_raw_sha256": spec["bank_raw_sha256"], "threshold": threshold,
        "image_count": len(rows), "good_count": sum(row["label"] == 0 for row in rows),
        "defect_count": sum(row["label"] == 1 for row in rows),
        "float_quality": quality(rows, "float_topk", threshold),
        "quant_quality": quality(rows, "quant_topk", threshold),
        "score_mean_abs_delta": sum(abs(row["float_topk"] - row["quant_topk"]) for row in rows) / len(rows),
        "decision_disagreements": sum((row["float_topk"] >= threshold) != (row["quant_topk"] >= threshold) for row in rows),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(out)
    print(f"FULL_CATEGORY_REPORT_PASS category={category} images={len(rows)} good={report['good_count']} defect={report['defect_count']} output={out}")


if __name__ == "__main__":
    main()
