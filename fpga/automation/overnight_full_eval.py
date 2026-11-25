"""Resume-safe overnight transfer/evaluation of all 15 MVTec test categories.

Retries only SSH transfer failures at 15-minute intervals. A Docker evaluation
failure stops immediately and preserves logs. No chat-side polling is needed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "fpga/vitis_ai"
SPEC = AI / "evaluation_spec.json"
EXPECTED_SOURCE = ROOT / "results/patchcore_int8_preparation_001/profiles_and_results.json"
REMOTE = AI / "stream_full_category_remote.py"
EVALUATOR = AI / "evaluate_full_category.py"
BASE = Path(r"C:\AMDDesignTools\Downloads\fpga_dpu_model_prep_001")
CACHE = BASE / "full_eval_archive_20pc"
QUANT = BASE / "quantized"
RESULTS = QUANT / "full_eval_20pc"
BUILD = ROOT / "fpga/vivado/build"
STATE = BUILD / "full_eval_20pc_state.json"
IMAGE = "ultra96-vitis-ai-pytorch:3.5-timm1.0.28"
WAIT_SECONDS = 15 * 60
DEADLINE_SECONDS = 10 * 60 * 60


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_state(state: dict) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATE)


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"categories": {}}


def remote_command(category: str, specs: str) -> bytes:
    script = REMOTE.read_text(encoding="utf-8")
    header = f"from __future__ import annotations\nCATEGORY = {category!r}\nSPEC_JSON = {specs!r}"
    return script.replace("from __future__ import annotations", header, 1).encode()


def transfer(category: str, specs: str, target: Path, error_log: Path, expected_images: int) -> bool:
    command = [
        "ssh", "-A", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        "-J", "shunya@ssh.arch.info.mie-u.ac.jp", "shunya@konbu.arch.info.mie-u.ac.jp",
        "ssh -T -o BatchMode=yes -o ConnectTimeout=20 caviar9 /home/shunya/miniconda3/envs/cuda/bin/python -",
    ]
    partial = target.with_suffix(".part")
    with partial.open("wb") as output, error_log.open("w", encoding="utf-8") as errors:
        try:
            result = subprocess.run(command, input=remote_command(category, specs), stdout=output, stderr=errors, timeout=45 * 60, check=False)
        except subprocess.TimeoutExpired:
            return False
    if result.returncode != 0 or "FULL_CATEGORY_STREAM_PASS" not in error_log.read_text(encoding="utf-8", errors="replace"):
        return False
    try:
        with tarfile.open(partial, "r") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            images = [row for row in manifest["files"] if row["type"] == "image"]
            if manifest["category"] != category or len(images) != expected_images or not any(x["label"] == 0 for x in images) or not any(x["label"] == 1 for x in images):
                return False
    except (tarfile.TarError, OSError, ValueError, KeyError, TypeError):
        return False
    partial.replace(target)
    return True


def evaluate(category: str, log: Path) -> bool:
    command = [
        "docker", "run", "--rm",
        "--mount", f"type=bind,source={AI},target=/work,readonly",
        "--mount", f"type=bind,source={BASE},target=/weights,readonly",
        "--mount", f"type=bind,source={QUANT},target=/out",
        "--mount", f"type=bind,source={CACHE},target=/eval,readonly",
        "--entrypoint", "bash", IMAGE, "-lc",
        "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch && python /work/evaluate_full_category.py --category " + category,
    ]
    with log.open("w", encoding="utf-8", errors="replace") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    return result.returncode == 0 and f"FULL_CATEGORY_REPORT_PASS category={category}" in log.read_text(encoding="utf-8", errors="replace")


def run() -> int:
    specs = SPEC.read_text(encoding="utf-8")
    categories = [row["category"] for row in json.loads(specs)]
    expected = {
        row["category"]: row["test_values"] // (196 * row["profile"]["feature_dim"])
        for row in json.loads(EXPECTED_SOURCE.read_text(encoding="utf-8"))
    }
    if len(categories) != 15 or len(set(categories)) != 15:
        raise RuntimeError("Expected 15 unique categories")
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    state = read_state()
    started = time.monotonic()
    for index, category in enumerate(categories, 1):
        record = state["categories"].get(category, {})
        target = CACHE / f"{category}.tar"
        output = RESULTS / f"{category}.json"
        signature = None
        if target.exists():
            signature = hashlib.sha256((sha256(target) + sha256(EVALUATOR) + sha256(SPEC) + sha256(QUANT / "quantize_result_packed_20pc/quant_info.json")).encode()).hexdigest()
        if record.get("status") == "pass" and record.get("signature") == signature and output.exists():
            print(f"{index}/15 {category}: SKIP verified result", flush=True)
            continue
        if not target.exists():
            attempt = 0
            while not target.exists():
                attempt += 1
                state["categories"][category] = {"status": "transfer", "attempt": attempt, "updated_utc": datetime.now(timezone.utc).isoformat()}
                write_state(state)
                print(f"{index}/15 {category}: transfer attempt {attempt}", flush=True)
                if transfer(category, specs, target, BUILD / f"full_eval_transfer_{category}.log", expected[category]):
                    break
                error_text = (BUILD / f"full_eval_transfer_{category}.log").read_text(encoding="utf-8", errors="replace")
                if "Traceback (most recent call last)" in error_text or "RuntimeError:" in error_text:
                    state["categories"][category]["status"] = "failed_source"
                    write_state(state)
                    print(f"FULL_EVAL_FAILED_SOURCE category={category}; see saved log", flush=True)
                    return 1
                if time.monotonic() - started >= DEADLINE_SECONDS:
                    state["categories"][category]["status"] = "blocked_network"
                    write_state(state)
                    print(f"FULL_EVAL_BLOCKED_NETWORK category={category}", flush=True)
                    return 2
                print(f"{category}: network unavailable; next attempt in 15 minutes", flush=True)
                time.sleep(WAIT_SECONDS)
        signature = hashlib.sha256((sha256(target) + sha256(EVALUATOR) + sha256(SPEC) + sha256(QUANT / "quantize_result_packed_20pc/quant_info.json")).encode()).hexdigest()
        state["categories"][category] = {"status": "evaluating", "signature": signature, "archive_sha256": sha256(target), "updated_utc": datetime.now(timezone.utc).isoformat()}
        write_state(state)
        print(f"{index}/15 {category}: evaluating", flush=True)
        if not evaluate(category, BUILD / f"full_eval_{category}.log"):
            state["categories"][category]["status"] = "failed"
            write_state(state)
            print(f"FULL_EVAL_FAILED category={category}; see saved log", flush=True)
            return 1
        state["categories"][category]["status"] = "pass"
        state["categories"][category]["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_state(state)
        print(f"{index}/15 {category}: PASS", flush=True)
    print("FULL_EVAL_COMPLETE categories=15", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "status":
        state = read_state()["categories"]
        print(f"full_eval: {sum(row.get('status') == 'pass' for row in state.values())}/15 categories passed")
        for category, row in state.items():
            if row.get("status") != "pass":
                print(f"{category}: {row.get('status')} attempt={row.get('attempt', '-')}" )
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
