"""One-shot, resumable boardless tool checks. No log polling is required."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "fpga" / "vivado" / "build"
STATE = BUILD / "boardless_pipeline_state.json"
EXISTING_LOGS = {
    "bd": BUILD / "ps_dpu_rebuild_2023_1.log",
    "synth": BUILD / "ps_dpu_resynth_2023_1.log",
    "route": BUILD / "ps_dpu_route_2023_1.log",
    "calib": BUILD / "wrn_quant_packed_calib.log",
    "quant_test": BUILD / "wrn_quant_packed_test.log",
    "compile": BUILD / "wrn_dpu_compile_packed.log",
}
VIVADO = Path(r"C:\Xilinx\Vivado\2023.1\bin\vivado.bat")
AI = ROOT / "fpga" / "vitis_ai"
WEIGHTS = Path(r"C:\AMDDesignTools\Downloads\fpga_dpu_model_prep_001")
QUANT = WEIGHTS / "quantized"
IMAGE = "ultra96-vitis-ai-pytorch:3.5-timm1.0.28"


def docker(command: str, writable: bool = True) -> list[str]:
    mounts = [
        "--mount", f"type=bind,source={AI},target=/work,readonly",
        "--mount", f"type=bind,source={WEIGHTS},target=/weights,readonly",
        "--mount", f"type=bind,source={QUANT},target=/out" + ("" if writable else ",readonly"),
    ]
    script = (
        "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && "
        "conda activate vitis-ai-pytorch && " + command
    )
    return ["docker", "run", "--rm", *mounts, "--entrypoint", "bash", IMAGE, "-lc", script]


def stages() -> dict[str, dict]:
    def vivado(name: str, script: str, marker: str, inputs: list[str]) -> dict:
        return {
            "command": [str(VIVADO), "-mode", "batch", "-source", str(ROOT / "fpga" / "vivado" / script), "-nolog", "-nojournal"],
            "marker": marker,
            "inputs": [ROOT / "fpga" / "vivado" / p for p in inputs],
            "log": BUILD / f"pipeline_{name}.log",
        }

    return {
        "bd": vivado("bd", "create_ultra96_ps_dpu_base.tcl", "ULTRA96_PS_DPU_BASE_PASS", ["create_ultra96_ps_dpu_base.tcl", "prepare_dpu_ip.ps1"]),
        "axi_wiring": {
            "command": [sys.executable, str(ROOT / "fpga" / "automation" / "check_axi_wiring.py")],
            "marker": "AXI_STATIC_ID_LOCK_PASS",
            "inputs": [ROOT / "fpga" / "automation" / "check_axi_wiring.py", BUILD / "ultra96_ps_dpu_base_2023_1" / "ultra96_ps_dpu_base.gen" / "sources_1" / "bd" / "system" / "synth" / "system.v", BUILD / ("pipeline_bd.log" if (BUILD / "pipeline_bd.log").exists() else "ps_dpu_rebuild_2023_1.log")],
            "log": BUILD / "pipeline_axi_wiring.log",
        },
        "profile_coverage": {
            "command": [sys.executable, str(ROOT / "fpga" / "automation" / "check_profile_coverage.py")],
            "marker": "PROFILE_COVERAGE_PASS",
            "inputs": [ROOT / "fpga" / "automation" / "check_profile_coverage.py", ROOT / "results" / "patchcore_int8_preparation_001" / "profiles_and_results.json"],
            "log": BUILD / "pipeline_profile_coverage.log",
        },
        "synth": vivado("synth", "synth_ultra96_ps_dpu_base.tcl", "ULTRA96_PS_DPU_SYNTH_PASS", ["create_ultra96_ps_dpu_base.tcl", "synth_ultra96_ps_dpu_base.tcl"]),
        "route": vivado("route", "route_ultra96_ps_dpu_base.tcl", "ULTRA96_PS_DPU_ROUTE_PASS", ["create_ultra96_ps_dpu_base.tcl", "synth_ultra96_ps_dpu_base.tcl", "route_ultra96_ps_dpu_base.tcl"]),
        "calib": {
            "command": docker("python /work/quantize_wrn_features.py --mode calib"),
            "marker": "WRN_QUANT_CALIB_PASS", "inputs": [AI / "quantize_wrn_features.py", AI / "Dockerfile", WEIGHTS / "wrn50_2_l12_timm_1.0.28.pth", WEIGHTS / "calibration_train_good_2percategory.tar.gz"],
            "log": BUILD / "pipeline_calib.log",
        },
        "quant_test": {
            "command": docker("python /work/quantize_wrn_features.py --mode test"),
            "marker": "WRN_QUANT_TEST_PASS", "inputs": [AI / "quantize_wrn_features.py", AI / "Dockerfile", QUANT / "quantize_result_packed" / "quant_info.json"],
            "log": BUILD / "pipeline_quant_test.log",
        },
        "feature_compare": {
            "command": docker("python /work/compare_quant_features.py"),
            "marker": "FEATURE_COMPARE_REPORT_PASS",
            "inputs": [AI / "compare_quant_features.py", AI / "quantize_wrn_features.py", WEIGHTS / "wrn50_2_l12_timm_1.0.28.pth", WEIGHTS / "calibration_train_good_2percategory.tar.gz", QUANT / "quantize_result_packed" / "quant_info.json"],
            "log": BUILD / "pipeline_feature_compare.log",
        },
        "calib_20pc": {
            "command": docker("python /work/quantize_wrn_20pc.py --mode calib"),
            "marker": "WRN_20PC_CALIB_PASS",
            "inputs": [AI / "quantize_wrn_20pc.py", AI / "quantize_wrn_features.py", AI / "Dockerfile", WEIGHTS / "wrn50_2_l12_timm_1.0.28.pth", WEIGHTS / "calibration_train_good_20percategory.tar.gz"],
            "log": BUILD / "pipeline_calib_20pc.log",
        },
        "quant_test_20pc": {
            "command": docker("python /work/quantize_wrn_20pc.py --mode test"),
            "marker": "WRN_20PC_TEST_PASS",
            "inputs": [AI / "quantize_wrn_20pc.py", AI / "quantize_wrn_features.py", QUANT / "quantize_result_packed_20pc" / "quant_info.json"],
            "log": BUILD / "pipeline_quant_test_20pc.log",
        },
        "evaluation_pilot": {
            "command": docker("python /work/evaluate_quantized_pilot.py"),
            "marker": "EVALUATION_PILOT_REPORT_PASS",
            "inputs": [AI / "evaluate_quantized_pilot.py", AI / "evaluation_spec.json", AI / "quantize_wrn_features.py", WEIGHTS / "evaluation_pilot_1good_1defect_per_category.tar.gz", WEIGHTS / "wrn50_2_l12_timm_1.0.28.pth", QUANT / "quantize_result_packed" / "quant_info.json"],
            "log": BUILD / "pipeline_evaluation_pilot.log",
        },
        "evaluation_pilot_20pc": {
            "command": docker("python /work/evaluate_quantized_pilot_20pc.py"),
            "marker": "EVALUATION_PILOT_REPORT_PASS",
            "inputs": [AI / "evaluate_quantized_pilot_20pc.py", AI / "evaluate_quantized_pilot.py", AI / "evaluation_spec.json", AI / "quantize_wrn_features.py", WEIGHTS / "evaluation_pilot_1good_1defect_per_category.tar.gz", WEIGHTS / "wrn50_2_l12_timm_1.0.28.pth", QUANT / "quantize_result_packed_20pc" / "quant_info.json"],
            "log": BUILD / "pipeline_evaluation_pilot_20pc.log",
        },
        "full_eval": {
            "command": [sys.executable, str(ROOT / "fpga" / "automation" / "overnight_full_eval.py")],
            "marker": "FULL_EVAL_COMPLETE",
            "inputs": [ROOT / "fpga" / "automation" / "overnight_full_eval.py", AI / "stream_full_category_remote.py", AI / "evaluate_full_category.py", AI / "evaluate_quantized_pilot.py", AI / "evaluation_spec.json", ROOT / "results" / "patchcore_int8_preparation_001" / "profiles_and_results.json", QUANT / "quantize_result_packed_20pc" / "quant_info.json"],
            "log": BUILD / "pipeline_full_eval.log",
        },
        "compile": {
            "command": docker("vai_c_xir -x /out/quantize_result_packed/FeatureExtractor_int.xmodel -a /work/arch_b512_ultra96.json -o /out/compiled_packed -n wrn_l12_packed_b512"),
            "marker": "Compile done.", "inputs": [AI / "quantize_wrn_features.py", AI / "arch_b512_ultra96.json", QUANT / "quantize_result_packed" / "FeatureExtractor_int.xmodel"],
            "log": BUILD / "pipeline_compile.log",
        },
    }


def digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode())
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(state: dict) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATE)


def run(names: list[str], force: bool) -> int:
    state = read_state()
    all_stages = stages()
    for name in names:
        item = all_stages[name]
        missing = [path for path in item["inputs"] if not path.exists()]
        if missing:
            print(f"{name}: BLOCKED (missing {missing[0]})")
            return 2
        signature = digest(item["inputs"])
        prior = state.get(name, {})
        if prior.get("status") == "pass" and prior.get("input_sha256") == signature and not force:
            print(f"{name}: SKIP (same inputs, previously passed)")
            continue
        if name in {"calib", "quant_test", "compile"} and not WEIGHTS.exists():
            print(f"{name}: BLOCKED (missing {WEIGHTS})")
            return 2
        BUILD.mkdir(parents=True, exist_ok=True)
        state[name] = {"status": "running", "input_sha256": signature, "started_utc": datetime.now(timezone.utc).isoformat(), "log": str(item["log"])}
        save_state(state)
        print(f"{name}: RUNNING; log={item['log']}", flush=True)
        with item["log"].open("w", encoding="utf-8", errors="replace") as log:
            completed = subprocess.run(item["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        output = item["log"].read_text(encoding="utf-8", errors="replace")
        passed = completed.returncode == 0 and item["marker"] in output
        state[name].update({"status": "pass" if passed else "fail", "exit_code": completed.returncode, "finished_utc": datetime.now(timezone.utc).isoformat(), "marker_found": item["marker"] in output})
        save_state(state)
        print(f"{name}: {state[name]['status'].upper()} exit={completed.returncode}; log={item['log']}", flush=True)
        if not passed:
            print("See the saved log for the actual error; later stages were not started.")
            return 1
    return 0


def adopt(names: list[str]) -> int:
    """Record already verified local logs so a later run will skip unchanged inputs."""
    state = read_state()
    for name in names:
        item = stages()[name]
        old_log = EXISTING_LOGS.get(name)
        missing = [path for path in item["inputs"] if not path.exists()]
        if missing or old_log is None or not old_log.exists():
            print(f"{name}: NOT ADOPTED (missing input or log)")
            continue
        output = old_log.read_text(encoding="utf-8", errors="replace")
        if item["marker"] not in output:
            print(f"{name}: NOT ADOPTED (success marker absent)")
            continue
        state[name] = {"status": "pass", "source": "existing-log", "input_sha256": digest(item["inputs"]), "log": str(old_log), "marker_found": True}
        save_state(state)
        print(f"{name}: ADOPTED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "status", "adopt", "run"))
    parser.add_argument("--only", choices=list(stages()))
    parser.add_argument("--force", action="store_true", help="Explicitly rerun a previously passed stage")
    parser.add_argument("--verbose", action="store_true", help="Include log paths in status")
    args = parser.parse_args()
    names = [args.only] if args.only else list(stages())
    if args.action == "plan":
        for name in names:
            print(f"{name}: {stages()[name]['marker']}")
        print("Full-image PatchCore simulation is not yet implemented; this runner does not mark it passed.")
        return 0
    if args.action == "status":
        state = read_state()
        for name in names:
            item = state.get(name, {})
            if item:
                log = f"; log={item['log']}" if args.verbose and item.get("log") else ""
                print(f"{name}: {item['status']}{log}")
            else:
                old_log = EXISTING_LOGS.get(name)
                if old_log is not None and old_log.exists() and stages()[name]["marker"] in old_log.read_text(encoding="utf-8", errors="replace"):
                    log = f"; log={old_log}" if args.verbose else ""
                    print(f"{name}: existing pass (outside runner){log}")
                else:
                    print(f"{name}: not-run by runner")
        return 0
    if args.action == "adopt":
        return adopt(names)
    return run(names, args.force)


if __name__ == "__main__":
    sys.exit(main())
