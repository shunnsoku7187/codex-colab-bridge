# Mode-switching PatchCore FPGA analysis

Purpose: evaluate a product/category mode-switching architecture instead of per-image dynamic routing.

## Architecture idea

Before inspection starts, the FPGA enters the mode for the current product category.
During inspection, the configuration is fixed, so per-image routing mistakes and variable control latency are avoided.

Switchable mode contents:

- memory bank
- anomaly threshold
- top-k score setting
- patch grid / feature profile setting
- optional feature-layer selection

## Storage summary

- categories: `15`
- full baseline banks for all categories: `131.660 MiB`
- selected banks for all categories: `3.288 MiB`
- all-category bank ratio: `0.0250x`
- all-category bank reduction: `97.50%`
- largest selected active bank: `1.099 MiB`
- median selected active bank: `0.061 MiB`

## Architecture options

KNN latency below is estimated with `512` parallel distance lanes.

| option | resident bank | active bank | mean KNN | worst KNN | strength | weakness |
|---|---:|---:|---:|---:|---|---|
| full_baseline_mode | 131.660 MiB | 8.789 MiB | 17.6165 ms | 17.6400 ms | highest reference capacity | large bank storage and KNN latency |
| profiled_active_load | 1.099 MiB | 1.099 MiB | 0.1927 ms | 1.4700 ms | minimum on-chip active memory and deterministic during inspection | mode switch must load the next category bank |
| profiled_all_resident | 3.288 MiB | 1.099 MiB | 0.1927 ms | 1.4700 ms | near-zero mode switch latency after initialization | requires enough on-chip/off-chip storage for every selected bank |

## Per-category mode table

| category | selected config | good-pass | false-pass | bank | NN ops | total proxy | KNN ms | load time @ 100 MiB/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | `res18_l23_g5_b250_topk0p005` | 94.00% | 5.16% | 0.0052x | 0.0007x | 0.0958x | 0.0117 ms | 0.4578 ms |
| cable | `res18_l23_g7_b1500_topk0p05` | 33.79% | 2.61% | 0.0312x | 0.0078x | 0.0995x | 0.1378 ms | 2.7466 ms |
| capsule | `wrn_l2_g14_b125_topk0p05` | 21.82% | 5.45% | 0.0035x | 0.0035x | 0.2035x | 0.0612 ms | 0.3052 ms |
| carpet | `wrn_l3_g8_b125_topk0p005` | 68.57% | 4.89% | 0.0069x | 0.0023x | 0.4797x | 0.0400 ms | 0.6104 ms |
| grid | `res18_l23_g7_b6000_topk0p005` | 38.18% | 4.14% | 0.1250x | 0.0312x | 0.1117x | 0.5513 ms | 10.9863 ms |
| hazelnut | `res18_l23_g14_b500_topk0p02` | 92.00% | 1.14% | 0.0104x | 0.0104x | 0.1008x | 0.1837 ms | 0.9155 ms |
| leather | `wrn_l3_g10_b125_topk0p005` | 93.75% | 3.91% | 0.0069x | 0.0035x | 0.4804x | 0.0625 ms | 0.6104 ms |
| metal_nut | `wrn_l3_g6_b1000_topk0p005` | 40.00% | 6.38% | 0.0556x | 0.0102x | 0.4839x | 0.1800 ms | 4.8828 ms |
| pill | `res18_l23_g5_b125_topk0p005` | 18.46% | 2.82% | 0.0026x | 0.0003x | 0.0956x | 0.0059 ms | 0.2289 ms |
| screw | `res18_l23_g14_b125_topk0p05` | 3.81% | 5.76% | 0.0026x | 0.0026x | 0.0968x | 0.0459 ms | 0.2289 ms |
| tile | `wrn_l3_g7_b125_topk0p005` | 77.65% | 3.81% | 0.0069x | 0.0017x | 0.4795x | 0.0306 ms | 0.6104 ms |
| toothbrush | `wrn_l2_g5_b125_topk0p005` | 43.33% | 6.67% | 0.0035x | 0.0005x | 0.2041x | 0.0078 ms | 0.3052 ms |
| transistor | `wrn_l3_g6_b500_topk0p005` | 47.33% | 11.00% | 0.0278x | 0.0051x | 0.4812x | 0.0900 ms | 2.4414 ms |
| wood | `res18_l23_g7_b125_topk0p005` | 71.11% | 2.00% | 0.0026x | 0.0007x | 0.0957x | 0.0115 ms | 0.2289 ms |
| zipper | `wrn_l3_g14_b1500_topk0p05` | 75.00% | 2.71% | 0.0833x | 0.0833x | 0.5220x | 1.4700 ms | 7.3242 ms |

## Interpretation

- This supports a mode-switching design, not a per-image dynamic routing design.
- The large reduction comes from loading only the category-specific normal bank and feature profile needed for the current product.
- The strongest hardware story is `profiled_all_resident` if all selected banks fit, because mode switching becomes a pointer/config change.
- The safer first implementation is `profiled_active_load`, because it only needs one selected bank on the FPGA at a time.
- Quality is still the limiting issue.  The best first FPGA target should balance false-pass, good-pass, bank size, and KNN latency.

Figure: `results/mvtec_patchcore_mode_switch_analysis_003.png`
