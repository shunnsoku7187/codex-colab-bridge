"""Stream one category's complete labelled test set and selected bank as tar.

The launcher inserts CATEGORY and SPEC_JSON after the future import before
sending this script to caviar9 over SSH. stdout is binary tar; errors use stderr.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

DATA = Path("/home/shunya/codex-gpu-work/data/mvtec_ad_materialized_v2")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    spec = next((row for row in json.loads(SPEC_JSON) if row["category"] == CATEGORY), None)
    if spec is None:
        raise RuntimeError(f"Unknown category: {CATEGORY}")
    test_root = DATA / CATEGORY / "test"
    images = sorted(test_root.glob("*/*.png"))
    if not images or not any(x.parent.name == "good" for x in images) or not any(x.parent.name != "good" for x in images):
        raise RuntimeError(f"No complete good/defect test split: {CATEGORY}")
    bank = Path(spec["bank_source"])
    if not bank.is_file():
        raise RuntimeError(f"Missing selected bank: {bank}")
    files = []
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        archive.add(bank, arcname="bank.npy", recursive=False)
        files.append({"type": "bank", "archive_path": "bank.npy", "sha256": digest(bank)})
        for path in images:
            name = f"images/{path.parent.name}/{path.name}"
            archive.add(path, arcname=name, recursive=False)
            files.append({"type": "image", "archive_path": name, "label": int(path.parent.name != "good"), "sha256": digest(path)})
        payload = json.dumps({"category": CATEGORY, "files": files}, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(f"FULL_CATEGORY_STREAM_PASS category={CATEGORY} images={len(images)}", file=sys.stderr)


if __name__ == "__main__":
    main()
