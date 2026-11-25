# Ultra96-V2 PS+DPU and shared CNN toolchain, 2026-11-26

Target: Vivado/Vitis 2023.1, Vitis AI 3.5, Ultra96-V2 `xczu3eg-sbva484-1-i`, DPUCZDX8G 4.1 B512.

## Passed

- Official board preset and patched vendor DPU IP generate, validate, close, and reopen as a Vivado block design. The local IP patch changes the invalid default `ARCH=144` to `ARCH=512`; source archive SHA256: `93032A7F39487EF7531E1BA6AA32F3911757FE931412BD95E72767130DE2292F`.
- Revised PS+DPU integrated synthesis: 34,443 CLB LUTs, 42,776 registers, 72 BRAM tiles, 134 DSPs. This excludes the NN and memory-bank implementation. Route completed with WNS +4.706 ns, TNS 0, 32,407 LUT, 41,328 registers, 72 BRAM tiles and 134 DSP. The DRC report has 98 warnings (32 AVAL-318, one DPOP-3, one DPOP-4, 64 REQP-1731). No error-level DRC finding was shown. This is PS+DPU only.
- Shared timm WideResNet50-2 layers 1 and 2 quantized on a 30-image normal-training smoke set. The packed model has one 56x56x768 output: channels 0:256 are layer 1; channels 256:768 are nearest-neighbor expanded layer 2. Every second spatial sample of the latter corresponds to its 28x28 grid in the quantized graph. The B512 compiler completed with one DPU subgraph and that output shape (`xmodel_inspect_packed.json`).
- Quantized xmodel SHA256: `337C71FF19FCD9666F0A580395BF6D556BD27CCC9AE29B7E9D5AACA25C1EB6F5`. Compiled xmodel SHA256: `1E618D0C5FD9E2B3C111C7B096AC5A74D3E34D38658F2E7237B72978ABA5AA17`. Large binaries are kept outside Git under `C:\AMDDesignTools\Downloads\fpga_dpu_model_prep_001\quantized`.
- A generated-HDL structural check passed five ID/LOCK wiring assertions. The instruction response IDs and control lock bits are zero-extended; the data IDs are padded into the interconnect. Eight unique width warnings and four unique USER-width warnings remain (`axi_wiring_check.json`). This does not prove AXI transaction behavior.
- On 15 normal training images (one per category), float versus quantized feature comparison produced lowest cosine similarity 0.9369 for layer 1 and 0.8442 for expanded layer 2. Mean absolute errors were 0.5962 and 0.7040 respectively (`feature_compare_15_normal.json`). These are descriptive results, not pass thresholds or detection accuracy.

## Open before a complete boardless system claim

- The PS+DPU block design emits AXI ID, USER, and lock-width warnings. Validate the transaction behavior, not just Vivado's block-diagram checker.
- Confirm the packed quantized features numerically against the research preprocessing and float reference across full images. Thirty normal training images establish only a toolchain smoke test.
- Investigate the observed feature drift with the exact research preprocessing and a representative calibration/evaluation split before choosing a final quantized model.
- Connect DPU output, feature processing, NN, DDR bank transport, profile control, and image decision. Simulate complete image transactions and failures; then route, check timing/DRC, and build the boot/software package.

None of these checks measures physical-board behavior or proves detection accuracy.

Full local logs and utilization reports are in `fpga/vivado/build/`; they are excluded from Git because they are generated artifacts. Reproduction scripts are in `fpga/vivado/` and `fpga/vitis_ai/`.

The one-shot, resumable runner is documented in `fpga/automation/README.md`. It records stage status locally and skips passed stages with unchanged inputs.
