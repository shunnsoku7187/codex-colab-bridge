"""Download MVTec AD from a Hugging Face mirror and audit its structure.

The official MVTec AD download can be gated by a form.  For automated lab
experiments we use the previously-audited Hugging Face mirror as a practical
cached copy, then keep the downloaded files under the persistent caviar9 data
root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


MVTEC_CATEGORIES = [
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
]


def count_files(path: Path, suffixes: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.rglob("*") if child.is_file() and child.suffix.lower() in suffixes)


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def find_dataset_root(download_root: Path) -> Path:
    candidates = [download_root]
    candidates.extend(path for path in download_root.rglob("*") if path.is_dir())
    for candidate in candidates:
        present = sum(1 for category in MVTEC_CATEGORIES if (candidate / category).exists())
        if present >= 5:
            return candidate
    return download_root


def audit_structure(dataset_root: Path) -> list[dict]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    rows = []
    for category in MVTEC_CATEGORIES:
        root = dataset_root / category
        train = root / "train"
        test = root / "test"
        gt = root / "ground_truth"
        rows.append(
            {
                "category": category,
                "present": root.exists(),
                "train_images": count_files(train, image_suffixes),
                "test_images": count_files(test, image_suffixes),
                "ground_truth_images": count_files(gt, image_suffixes),
                "size_bytes": dir_size(root),
                "defect_types": sorted([child.name for child in test.iterdir() if child.is_dir()]) if test.exists() else [],
            }
        )
    return rows


def write_markdown(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MVTec AD Hugging Face download audit",
        "",
        f"Repo: `{payload['repo_id']}`",
        f"Download root: `{payload['download_root']}`",
        f"Dataset root: `{payload['dataset_root']}`",
        f"Total size: `{payload['total_size_gib']:.3f} GiB`",
        "",
        "## Category structure",
        "",
        "| category | present | train images | test images | ground truth images | defect types | size GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["categories"]:
        lines.append(
            f"| {row['category']} | {'yes' if row['present'] else 'no'} | {row['train_images']} | "
            f"{row['test_images']} | {row['ground_truth_images']} | {len(row['defect_types'])} | "
            f"{row['size_bytes'] / (1024**3):.3f} |"
        )
    lines += [
        "",
        "## Next action",
        "",
        "- If all categories are present, run a 3-category anomaly baseline probe first.",
        "- Keep this dataset under the persistent data root; do not commit the raw dataset to Git.",
        "- Use category-level results rather than a single random split, because MVTec AD already defines train/test structure.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="TheoM55/mvtec_all_objects_split")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--local-dir-name", default="mvtec_ad")
    parser.add_argument("--output", default="results/mvtec_ad_hf_download_001.json")
    parser.add_argument("--markdown", default="docs/mvtec_ad_hf_download_001.md")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    data_root = Path(args.data_root or os.environ.get("CODEX_COLAB_DATA_DIR", "artifacts/research_experiment/data"))
    data_root.mkdir(parents=True, exist_ok=True)
    local_dir = data_root / args.local_dir_name
    local_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=args.resume,
    )
    download_root = Path(path)
    dataset_root = find_dataset_root(download_root)
    categories = audit_structure(dataset_root)
    payload = {
        "purpose": "Download and audit MVTec AD from a Hugging Face mirror.",
        "repo_id": args.repo_id,
        "data_root": str(data_root),
        "download_root": str(download_root),
        "dataset_root": str(dataset_root),
        "total_size_bytes": dir_size(download_root),
        "total_size_gib": dir_size(download_root) / (1024**3),
        "categories": categories,
        "summary": {
            "present_categories": int(sum(1 for row in categories if row["present"])),
            "train_images": int(sum(row["train_images"] for row in categories)),
            "test_images": int(sum(row["test_images"] for row in categories)),
            "ground_truth_images": int(sum(row["ground_truth_images"] for row in categories)),
        },
        "tools": {
            "git": shutil.which("git"),
            "python": shutil.which("python"),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({"wrote": args.output, "markdown": args.markdown, "dataset_root": str(dataset_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
