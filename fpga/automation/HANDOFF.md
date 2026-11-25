# Resume this Ultra96-V2 work after a chat break

Repository: `C:\Users\shun7\Documents\ChatGPT\vitis\research` on Windows Home. Target: Avnet Ultra96-V2, shared WideResNet50-2 CNN on DPUCZDX8G B512 plus separate NN logic. The board is not yet available.

At the start of a new chat or after a break, run `fpga\automation\resume.ps1` once. Read `fpga/ULTRA96V2_BOARDLESS_STATUS.md` and `fpga/BOARDLESS_GAPS.md` for scope. The runner's state is in ignored `fpga/vivado/build/boardless_pipeline_state.json`; full logs are also under that build folder. Do not infer a pass from a process starting or from a stale log. Do not rerun passed stages with unchanged inputs.

Current verified milestones (2026-11-26): PS+DPU block design, synthesis, and route completed; route WNS +4.706 ns. Packed WRN feature model compiled for B512 with one DPU subgraph and a 56x56x768 output. AXI width warnings and 98 warning-level DRC findings remain. NN and memory-bank integration, feature numerical equivalence, and full-image PatchCore decision simulation are incomplete.

The next run added a static AXI ID/LOCK wiring check (five assertions passed, USER and transaction behavior unresolved) and a 15-normal-image float/quantized feature comparison. Lowest cosine similarity was 0.9369 for layer 1 and 0.8442 for layer 2; these are not accuracy acceptance results. See `fpga/evidence/ps_dpu_wrn_2023_1/` for concise JSON results. The pipeline now covers eight stages.

If a new long tool job is needed, start it through `fpga\automation\start.ps1` and close the chat if desired. On return, use `resume.ps1` once. Keep new checks as scripts with a small pass/fail state and raw local logs, then update the Git status documents. Do not poll a long-running tool continuously.
