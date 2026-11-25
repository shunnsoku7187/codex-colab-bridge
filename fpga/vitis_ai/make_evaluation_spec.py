"""Derive the selected 15-category evaluation contract from int8 preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/patchcore_int8_preparation_001/profiles_and_results.json"
OUT = Path(__file__).with_name("evaluation_spec.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    specs = [
        {
            "category": row["category"],
            "out_indices": row["profile"]["out_indices"],
            "patch_grid": row["profile"]["patch_grid"],
            "topk_fraction": row["profile"]["topk_fraction"],
            "bank_source": row["bank_source"],
            "bank_raw_sha256": row["bank_raw_sha256"],
            "frozen_threshold": row["frozen_threshold"],
        }
        for row in rows
    ]
    if len(specs) != 15 or len({x["category"] for x in specs}) != 15:
        raise RuntimeError("Expected 15 unique categories")
    payload = json.dumps(specs, indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != payload:
            raise RuntimeError("evaluation_spec.json is stale")
        print("EVALUATION_SPEC_CHECK_PASS categories=15")
    else:
        OUT.write_text(payload, encoding="utf-8")
        print(f"EVALUATION_SPEC_WRITE_PASS categories=15 path={OUT}")


if __name__ == "__main__":
    main()
