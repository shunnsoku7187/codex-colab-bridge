"""Audit availability of larger inspection datasets on the remote runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path


KNOWN_DATASETS = {
    "mvtec_ad": {
        "candidate_dirs": ["mvtec_ad", "MVTecAD", "mvtec"],
        "categories": [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ],
        "network_checks": {
            "hf_mirror_api": "https://huggingface.co/api/datasets/TheoM55/mvtec_all_objects_split",
            "official_page": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        },
    },
    "visa": {
        "candidate_dirs": ["visa", "VisA", "visual_anomaly"],
        "categories": [],
        "network_checks": {
            "aws_registry": "https://registry.opendata.aws/visa/",
            "spot_diff_repo": "https://github.com/amazon-research/spot-diff",
        },
    },
    "mvtec_ad_2": {
        "candidate_dirs": ["mvtec_ad_2", "mvtec_ad2", "MVTecAD2", "MVTec_AD_2"],
        "categories": ["can", "fabric", "fruit_jelly", "sheet_metal", "vial", "wallplugs", "walnuts", "woriceod"],
        "network_checks": {
            "official_page": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2",
            "benchmark_page": "https://benchmark.mvtec.com/",
        },
    },
}


def dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def count_images(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for child in path.rglob("*") if child.suffix.lower() in exts)


def network_check(url: str, timeout: int) -> dict:
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "codex-dataset-audit"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            sample = response.read(256)
            return {
                "url": url,
                "ok": True,
                "status": getattr(response, "status", None),
                "content_type": response.headers.get("content-type"),
                "sample_bytes": len(sample),
            }
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc)}


def audit_dataset(data_root: Path, name: str, spec: dict, timeout: int) -> dict:
    found = []
    for dirname in spec["candidate_dirs"]:
        path = data_root / dirname
        if path.exists():
            found.append(
                {
                    "path": str(path),
                    "size_bytes": dir_size(path),
                    "image_count": count_images(path),
                    "present_categories": [category for category in spec["categories"] if (path / category).exists()],
                }
            )
    return {
        "name": name,
        "found": found,
        "network": {key: network_check(url, timeout) for key, url in spec["network_checks"].items()},
    }


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset availability audit",
        "",
        f"Data root: `{payload['data_root']}`",
        "",
        "## Local availability",
        "",
        "| dataset | found dirs | total images | total size GiB | network checks |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in payload["datasets"]:
        found_dirs = len(dataset["found"])
        image_count = sum(item["image_count"] for item in dataset["found"])
        size_gib = sum(item["size_bytes"] for item in dataset["found"]) / (1024**3)
        net = ", ".join(f"{key}:{'ok' if value['ok'] else 'fail'}" for key, value in dataset["network"].items())
        lines.append(f"| {dataset['name']} | {found_dirs} | {image_count} | {size_gib:.3f} | {net} |")
    lines += [
        "",
        "## Tool availability",
        "",
    ]
    for name, value in payload["tools"].items():
        lines.append(f"- `{name}`: `{value or 'not found'}`")
    lines += [
        "",
        "## Next action",
        "",
        "- If MVTec AD is already present, run the 3-category probe.",
        "- If not present but the Hugging Face mirror is reachable, prepare an automated download job.",
        "- If only official gated download is usable, download manually once to the data root and keep it there.",
        "- If VisA network access is reachable, prepare a separate download/format audit because the archive is large.",
        "- If MVTec AD 2 is reachable only through the official form/evaluation server, treat it as a medium-term dataset and record the manual setup steps.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output", default="results/dataset_availability_audit_001.json")
    parser.add_argument("--markdown", default="docs/dataset_availability_audit_001.md")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    payload = {
        "purpose": "Check whether larger inspection datasets are already available or reachable from the runner.",
        "data_root": str(data_root),
        "datasets": [audit_dataset(data_root, name, spec, args.timeout) for name, spec in KNOWN_DATASETS.items()],
        "tools": {
            "aws": shutil.which("aws"),
            "git": shutil.which("git"),
            "python": shutil.which("python"),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
