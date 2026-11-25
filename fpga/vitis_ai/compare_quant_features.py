"""Quantify packed WRN feature drift on one normal image per MVTec category."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from pytorch_nndct.apis import torch_quantizer

from quantize_wrn_features import FeatureExtractor, calibration_images

OUT = Path("/out/feature_compare.json")
QUANT_DIR = Path("/out/quantize_result_packed")


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    ref = reference.double().reshape(-1)
    got = actual.double().reshape(-1)
    diff = ref - got
    result = {
        "mae": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(ref, got, dim=0).item(),
    }
    if not all(math.isfinite(x) for x in result.values()):
        raise RuntimeError("Non-finite feature comparison")
    return result


def main() -> None:
    images = calibration_images()[::2]
    if len(images) != 15:
        raise RuntimeError(f"Expected one image per category, got {len(images)}")
    model = FeatureExtractor().eval()
    quantizer = torch_quantizer("test", model, (images[0],), device=torch.device("cpu"), output_dir=str(QUANT_DIR))
    quant_model = quantizer.quant_model.eval()
    rows = []
    with torch.no_grad():
        for category_index, image in enumerate(images):
            reference = model(image)
            actual = quant_model(image)
            if tuple(reference.shape) != (1, 768, 56, 56) or actual.shape != reference.shape:
                raise RuntimeError(f"Unexpected feature shapes: {reference.shape}, {actual.shape}")
            rows.append({
                "category_index": category_index,
                "layer1": metrics(reference[:, :256], actual[:, :256]),
                "layer2_expanded": metrics(reference[:, 256:], actual[:, 256:]),
            })
        report = {
            "scope": "15 normal training images, one per category; float-vs-quantized feature drift only",
            "note": "Not a detection-accuracy test or a DPU-runtime result",
            "rows": rows,
            "summary": {
                layer: {
                    "mean_mae": sum(row[layer]["mae"] for row in rows) / len(rows),
                    "worst_rmse": max(row[layer]["rmse"] for row in rows),
                    "lowest_cosine": min(row[layer]["cosine"] for row in rows),
                }
                for layer in ("layer1", "layer2_expanded")
            },
        }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FEATURE_COMPARE_REPORT_PASS images={len(rows)} output={OUT}")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
