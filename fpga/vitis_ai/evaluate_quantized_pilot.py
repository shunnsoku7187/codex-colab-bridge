"""Compare float and Vitis-quantized image scores with the selected K=3000 banks.

One good and one defect test image per category: a pipeline pilot, not a quality
acceptance experiment. The original research uses 224x224 ImageNet normalization,
14x14 bilinear layer resizing, channel concatenation and L2 normalization.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_nndct.apis import torch_quantizer
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from quantize_wrn_features import FeatureExtractor, MEAN, STD

ARCHIVE = Path("/weights/evaluation_pilot_1good_1defect_per_category.tar.gz")
SPEC = Path("/work/evaluation_spec.json")
OUT = Path("/out/evaluation_pilot_scores.json")
QUANT_DIR = Path("/out/quantize_result_packed")


def member_bytes(archive: tarfile.TarFile, name: str, expected_hash: str) -> bytes:
    item = archive.extractfile(name)
    if item is None:
        raise RuntimeError(f"Missing archive member: {name}")
    data = item.read()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise RuntimeError(f"Archive member SHA256 mismatch: {name}")
    return data


def preprocess(data: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        image = TF.resize(image, [224, 224], interpolation=TF.InterpolationMode.BILINEAR)
        return ((TF.to_tensor(image) - MEAN) / STD).unsqueeze(0)


def patches(packed: torch.Tensor, indices: list[int]) -> torch.Tensor:
    maps = []
    if 1 in indices:
        maps.append(F.interpolate(packed[:, :256], size=(14, 14), mode="bilinear", align_corners=False))
    if 2 in indices:
        maps.append(F.interpolate(packed[:, 256:, ::2, ::2], size=(14, 14), mode="bilinear", align_corners=False))
    merged = F.normalize(torch.cat(maps, dim=1), dim=1)
    return merged.permute(0, 2, 3, 1).reshape(196, -1)


def score(patch_vectors: torch.Tensor, bank: torch.Tensor, topk_fraction: float) -> tuple[float, float]:
    nearest = torch.cdist(patch_vectors, bank).min(dim=1).values
    k = max(1, round(196 * topk_fraction))
    return nearest.topk(k).values.mean().item(), nearest.max().item()


def main() -> None:
    specs = {row["category"]: row for row in json.loads(SPEC.read_text())}
    if len(specs) != 15:
        raise RuntimeError("Expected 15 category profiles")
    model = FeatureExtractor().eval()
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        by_category: dict[str, list[dict]] = {}
        for entry in manifest["files"]:
            by_category.setdefault(entry["category"], []).append(entry)
        first_image = next(item for item in manifest["files"] if item["type"] == "image")
        example = preprocess(member_bytes(archive, first_image["archive_path"], first_image["sha256"]))
        quantizer = torch_quantizer("test", model, (example,), device=torch.device("cpu"), output_dir=str(QUANT_DIR))
        quant_model = quantizer.quant_model.eval()
        rows = []
        with torch.no_grad():
            for category, spec in specs.items():
                entries = by_category.get(category, [])
                bank_entry = next((x for x in entries if x["type"] == "bank"), None)
                images = [x for x in entries if x["type"] == "image"]
                if bank_entry is None or sorted(x["label"] for x in images) != [0, 1]:
                    raise RuntimeError(f"Expected bank, good and defect for {category}")
                bank_data = member_bytes(archive, bank_entry["archive_path"], bank_entry["sha256"])
                bank_np = np.load(io.BytesIO(bank_data), allow_pickle=False)
                raw_sha = hashlib.sha256(np.ascontiguousarray(bank_np).tobytes()).hexdigest()
                if raw_sha != spec["bank_raw_sha256"]:
                    raise RuntimeError(f"Selected research bank hash mismatch: {category}")
                if bank_np.shape != (3000, 512 if spec["out_indices"] == [2] else 768):
                    raise RuntimeError(f"Unexpected bank shape: {category} {bank_np.shape}")
                bank = torch.from_numpy(bank_np.astype(np.float32, copy=False))
                for entry in images:
                    image = preprocess(member_bytes(archive, entry["archive_path"], entry["sha256"]))
                    floating = model(image)
                    quantized = quant_model(image)
                    float_topk, float_max = score(patches(floating, spec["out_indices"]), bank, spec["topk_fraction"])
                    quant_topk, quant_max = score(patches(quantized, spec["out_indices"]), bank, spec["topk_fraction"])
                    if not all(math.isfinite(v) for v in (float_topk, quant_topk, float_max, quant_max)):
                        raise RuntimeError(f"Non-finite score: {category}")
                    rows.append({
                        "category": category, "label": entry["label"], "image_sha256": entry["sha256"],
                        "float_topk": float_topk, "quant_topk": quant_topk,
                        "float_max": float_max, "quant_max": quant_max,
                        "frozen_threshold": spec["frozen_threshold"],
                        "float_flag": float_topk >= spec["frozen_threshold"],
                        "quant_flag": quant_topk >= spec["frozen_threshold"],
                    })
    report = {
        "scope": "one first-sorted good and defect test image per category, selected K=3000 banks, float-vs-quantized scores; no DPU runtime",
        "rows": rows,
        "summary": {
            "images": len(rows),
            "mean_abs_topk_delta": sum(abs(x["float_topk"] - x["quant_topk"]) for x in rows) / len(rows),
            "max_abs_topk_delta": max(abs(x["float_topk"] - x["quant_topk"]) for x in rows),
            "frozen_threshold_disagreements": sum(x["float_flag"] != x["quant_flag"] for x in rows),
        },
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"EVALUATION_PILOT_REPORT_PASS images={len(rows)} output={OUT}")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
