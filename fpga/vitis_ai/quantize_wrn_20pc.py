"""Prespecified 20-normal-images-per-category calibration, isolated from smoke run."""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import tarfile
from pathlib import Path

import torch
from PIL import Image
from pytorch_nndct.apis import torch_quantizer
from torchvision.transforms import functional as TF

from quantize_wrn_features import FeatureExtractor, MEAN, STD

ARCHIVE = Path("/weights/calibration_train_good_20percategory.tar.gz")
OUT = Path("/out/quantize_result_packed_20pc")


def images():
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        rows = manifest["images"]
        if len(rows) != 300 or len({x["category"] for x in rows}) != 15:
            raise RuntimeError("Expected 300 normal training images in 15 categories")
        for entry in rows:
            data = archive.extractfile(entry["archive_path"]).read()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Image SHA256 mismatch: {entry['archive_path']}")
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert("RGB")
                image = TF.resize(image, [224, 224], interpolation=TF.InterpolationMode.BILINEAR)
                yield ((TF.to_tensor(image) - MEAN) / STD).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calib", "test"), required=True)
    args = parser.parse_args()
    image_iter = images()
    first = next(image_iter)
    model = FeatureExtractor().eval()
    OUT.mkdir(parents=True, exist_ok=True)
    quantizer = torch_quantizer(args.mode, model, (first,), device=torch.device("cpu"), output_dir=str(OUT))
    quant_model = quantizer.quant_model.eval()
    count = 0
    with torch.no_grad():
        selected = itertools.chain((first,), image_iter) if args.mode == "calib" else (first,)
        for image in selected:
            output = quant_model(image)
            if tuple(output.shape) != (1, 768, 56, 56):
                raise RuntimeError(f"Unexpected packed feature shape: {output.shape}")
            count += 1
    if args.mode == "calib":
        if count != 300:
            raise RuntimeError(f"Calibration did not consume 300 images: {count}")
        quantizer.export_quant_config()
    else:
        quantizer.export_xmodel(output_dir=str(OUT))
    print(f"WRN_20PC_{args.mode.upper()}_PASS images={count}")


if __name__ == "__main__":
    main()
