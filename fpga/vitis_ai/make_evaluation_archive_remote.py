"""Run on caviar9 with SPEC_JSON supplied before this script on stdin.

Packages a small, labelled test-image pilot and the exact selected float banks.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

DATA = Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2")
OUT = Path("/home/shunya/codex-gpu-work/data/fpga_dpu_model_prep_001/evaluation_pilot_1good_1defect_per_category.tar.gz")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    specs = json.loads(SPEC_JSON)
    if len(specs) != 15:
        raise RuntimeError("Expected 15 selected categories")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tarfile.open(OUT, "w:gz") as archive:
        for spec in specs:
            category = spec["category"]
            test_root = DATA / category / "test"
            good = sorted((test_root / "good").glob("*.png"))
            defect = sorted(path for path in test_root.glob("*/*.png") if path.parent.name != "good")
            if not good or not defect:
                raise RuntimeError(f"Missing good/defect test image for {category}")
            for label, path in ((0, good[0]), (1, defect[0])):
                name = f"images/{category}/{path.parent.name}/{path.name}"
                archive.add(path, arcname=name, recursive=False)
                manifest.append({"type": "image", "category": category, "label": label, "archive_path": name, "sha256": digest(path)})
            bank = Path(spec["bank_source"])
            if not bank.is_file():
                raise RuntimeError(f"Missing bank: {bank}")
            name = f"banks/{category}.npy"
            archive.add(bank, arcname=name, recursive=False)
            manifest.append({"type": "bank", "category": category, "archive_path": name, "sha256": digest(bank)})
        payload = json.dumps({"selection": "first sorted good and defect test PNG per category; exact selected float banks", "specs": specs, "files": manifest}, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(json.dumps({"path": str(OUT), "files": len(manifest), "bytes": OUT.stat().st_size, "sha256": digest(OUT)}))


if __name__ == "__main__":
    main()
