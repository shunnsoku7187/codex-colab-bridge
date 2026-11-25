"""Verify the shared CNN output covers every selected PatchCore profile."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/patchcore_int8_preparation_001/profiles_and_results.json"
OUT = ROOT / "fpga/vivado/build/profile_coverage.json"


def main() -> int:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(rows) != 15 or len({row["category"] for row in rows}) != 15:
        raise RuntimeError("Expected 15 unique MVTec categories")
    counts = Counter(tuple(row["profile"]["out_indices"]) for row in rows)
    bad = [row["category"] for row in rows if tuple(row["profile"]["out_indices"]) not in {(2,), (1, 2)} or row["profile"]["patch_grid"] != 14 or row["profile"]["feature_dim"] != (512 if row["profile"]["out_indices"] == [2] else 768)]
    result = {
        "profiles": {"+".join(map(str, key)): count for key, count in sorted(counts.items())},
        "packed_output": "56x56x768: first 256 channels index 1; remaining 512 channels index 2",
        "invalid_categories": bad,
        "scope": "profile layer/grid/dimension coverage only; feature values and scoring are not checked",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if bad:
        print(f"PROFILE_COVERAGE_FAIL categories={bad}")
        return 1
    print(f"PROFILE_COVERAGE_PASS categories=15 one_layer={counts[(2,)]} two_layers={counts[(1, 2)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
