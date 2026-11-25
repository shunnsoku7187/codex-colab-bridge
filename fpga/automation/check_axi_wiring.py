"""Check known ID/LOCK wiring in generated Ultra96-V2 PS+DPU HDL.

This is structural evidence only. It does not replace AXI transaction simulation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HDL = ROOT / "fpga/vivado/build/ultra96_ps_dpu_base_2023_1/ultra96_ps_dpu_base.gen/sources_1/bd/system/synth/system.v"
PIPELINE_LOG = ROOT / "fpga/vivado/build/pipeline_bd.log"
WARNINGS = PIPELINE_LOG if PIPELINE_LOG.exists() else ROOT / "fpga/vivado/build/ps_dpu_rebuild_2023_1.log"
OUTPUT = ROOT / "fpga/vivado/build/axi_wiring_check.json"

CHECKS = {
    "instruction_read_id_zero_extended": r"\.dpu0_m_axi_instr_rid\(\{(?:1'b0,){4}dpu_DPU0_M_AXI_INSTR_RID\}\)",
    "instruction_write_id_zero_extended": r"\.dpu0_m_axi_instr_bid\(\{(?:1'b0,){4}dpu_DPU0_M_AXI_INSTR_BID\}\)",
    "control_read_lock_zero_extended": r"\.s_axi_arlock\(\{1'b0,ctrl_interconnect_M00_AXI_ARLOCK\}\)",
    "control_write_lock_zero_extended": r"\.s_axi_awlock\(\{1'b0,ctrl_interconnect_M00_AXI_AWLOCK\}\)",
    "data_ids_padded_into_xbar": r"s01_couplers_to_xbar_ARID,1'b0,1'b0,s00_couplers_to_xbar_ARID",
}


def main() -> int:
    if not HDL.exists() or not WARNINGS.exists():
        print("AXI_STATIC_BLOCKED: generated HDL or Vivado warning log is missing")
        return 2
    hdl = HDL.read_text(encoding="utf-8", errors="replace")
    warning_lines = WARNINGS.read_text(encoding="utf-8", errors="replace").splitlines()
    results = {name: bool(re.search(pattern, hdl)) for name, pattern in CHECKS.items()}
    width = sorted(set(line for line in warning_lines if "Width mismatch when connecting pin" in line))
    user = sorted(set(line for line in warning_lines if "AWUSER_WIDTH" in line or "ARUSER_WIDTH" in line))
    report = {
        "scope": "generated HDL ID and LOCK mapping only; no AXI transaction proof",
        "checks": results,
        "unique_width_warnings": len(width),
        "unique_user_width_warnings": len(user),
        "unresolved": ["AXI USER propagation", "overlapping-read/write response simulation", "DPU runtime"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if all(results.values()):
        print(f"AXI_STATIC_ID_LOCK_PASS checks={len(results)} unique_width_warnings={len(width)} unique_user_warnings={len(user)}")
        print("AXI_USER_AND_TRANSACTION_UNRESOLVED")
        return 0
    failed = ", ".join(name for name, passed in results.items() if not passed)
    print(f"AXI_STATIC_FAIL: {failed}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
