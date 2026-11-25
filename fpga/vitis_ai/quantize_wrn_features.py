"""Vitis AI 3.5 smoke quantization for the shared WRN layer1/layer2 features."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import timm
import torch
from PIL import Image
from pytorch_nndct.apis import torch_quantizer
from torch import nn
from torchvision.transforms import functional as TF

WEIGHTS = Path("/weights/wrn50_2_l12_timm_1.0.28.pth")
CALIBRATION = Path("/weights/calibration_train_good_2percategory.tar.gz")
OUT = Path("/out/quantize_result_packed")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "wide_resnet50_2", pretrained=False, features_only=True, out_indices=(1, 2)
        )
        self.backbone.load_state_dict(torch.load(WEIGHTS, map_location="cpu"), strict=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        layer1, layer2 = self.backbone(image)
        # Keep both feature maps in one XIR output. Nearest-neighbor expansion is
        # reversible: taking every second pixel recovers layer2's 28x28 grid.
        layer2_expanded = torch.nn.functional.interpolate(layer2, scale_factor=2, mode="nearest")
        return torch.cat((layer1, layer2_expanded), dim=1)


def calibration_images() -> list[torch.Tensor]:
    images = []
    with tarfile.open(CALIBRATION, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        for item in manifest["images"]:
            data = archive.extractfile(item["archive_path"]).read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Image hash mismatch: {item['archive_path']}")
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert("RGB")
                image = TF.resize(image, [224, 224], interpolation=TF.InterpolationMode.BILINEAR)
                tensor = (TF.to_tensor(image) - MEAN) / STD
            images.append(tensor.unsqueeze(0))
    if len(images) != 30:
        raise RuntimeError(f"Expected 30 normal training images, got {len(images)}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calib", "test"), required=True)
    args = parser.parse_args()
    images = calibration_images()
    model = FeatureExtractor().eval()
    OUT.mkdir(parents=True, exist_ok=True)
    quantizer = torch_quantizer(
        args.mode, model, (images[0],), device=torch.device("cpu"), output_dir=str(OUT)
    )
    quant_model = quantizer.quant_model
    selected = images if args.mode == "calib" else images[:1]
    with torch.no_grad():
        for image in selected:
            outputs = quant_model(image)
    if args.mode == "calib":
        quantizer.export_quant_config()
    else:
        quantizer.export_xmodel(output_dir=str(OUT))
    print(f"WRN_QUANT_{args.mode.upper()}_PASS images={len(selected)} shape={tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
