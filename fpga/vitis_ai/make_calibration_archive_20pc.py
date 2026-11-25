"""Run on caviar9: package 20 sorted normal training images per category."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

DATA = Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2")
OUT = Path("/home/shunya/codex-gpu-work/data/fpga_dpu_model_prep_001/calibration_train_good_20percategory.tar.gz")
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with tarfile.open(OUT, "w:gz") as archive:
        for category in CATEGORIES:
            images = sorted((DATA / category / "train" / "good").glob("*.png"))
            if len(images) < 20:
                raise RuntimeError(f"Fewer than 20 normal training images: {category}")
            for path in images[:20]:
                name = f"{category}/{path.name}"
                archive.add(path, arcname=name, recursive=False)
                files.append({"category": category, "archive_path": name, "sha256": digest(path)})
        manifest = {"source": str(DATA), "selection": "first 20 sorted train/good PNGs per category", "images": files}
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"path": str(OUT), "images": len(files), "bytes": OUT.stat().st_size, "sha256": digest(OUT)}))


if __name__ == "__main__":
    main()
