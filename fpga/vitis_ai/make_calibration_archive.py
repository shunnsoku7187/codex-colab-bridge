"""Run on caviar9: package a small, normal-only MVTec calibration smoke set."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

DATA = Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2")
OUTPUT = Path("/home/shunya/codex-gpu-work/data/fpga_dpu_model_prep_001/calibration_train_good_2percategory.tar.gz")
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for category in CATEGORIES:
            candidates = sorted((DATA / category / "train" / "good").glob("*.png"))
            if len(candidates) < 2:
                raise RuntimeError(f"Not enough training images: {category}")
            for path in candidates[:2]:
                name = f"{category}/{path.name}"
                archive.add(path, arcname=name, recursive=False)
                manifest.append({"category": category, "archive_path": name, "sha256": file_hash(path)})
        payload = json.dumps({"source": str(DATA), "selection": "first two sorted train/good PNGs per category", "images": manifest}, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"path": str(OUTPUT), "images": len(manifest), "bytes": OUTPUT.stat().st_size, "sha256": file_hash(OUTPUT)}))


if __name__ == "__main__":
    main()
