"""Copy recorded model and normal-only calibration assets through the lab route."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REMOTE_DIR = "/home/shunya/codex-gpu-work/data/fpga_dpu_model_prep_001"
LOCAL_DIR = Path(r"C:\AMDDesignTools\Downloads\fpga_dpu_model_prep_001")
ASSETS = {
    "wrn50_2_l12_timm_1.0.28.pth": "94cee3586e3f6b4a6c83148e4f07e2c73e56da2df2c00ba6077135c26182536c",
    "calibration_train_good_2percategory.tar.gz": "e737d0e91f15b3830a6ce79b3b06a4a1a2b9eba9572e898c6cd9afa2fcc43731",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        dest = LOCAL_DIR / name
        if dest.exists() and sha256(dest) == expected:
            print(f"ASSET_READY {dest} SHA256={expected}")
            continue
        partial = dest.with_name(dest.name + ".part")
        command = [
            "ssh", "-A", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-J", "shunya@ssh.arch.info.mie-u.ac.jp",
            "shunya@konbu.arch.info.mie-u.ac.jp",
            f"ssh -T -o BatchMode=yes -o ConnectTimeout=15 caviar9 cat {REMOTE_DIR}/{name}",
        ]
        with partial.open("wb") as stream:
            subprocess.run(command, stdout=stream, check=True)
        actual = sha256(partial)
        if actual != expected:
            raise RuntimeError(f"Asset hash mismatch for {name}: {actual}")
        partial.replace(dest)
        print(f"ASSET_READY {dest} SHA256={actual}")


if __name__ == "__main__":
    main()
