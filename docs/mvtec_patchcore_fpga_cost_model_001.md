# FPGA cost model for profiled PatchCore

Purpose: translate the current PatchCore reduction evidence into FPGA-facing resource and latency proxies.

## Scope

This is a pre-RTL estimate.  It does not claim final FPGA power or timing.
It separates the parts that must be implemented and measured next:

- CNN feature extraction MACs
- PatchCore nearest-neighbor distance operations
- Memory-bank storage
- Memory-bank read traffic
- KNN latency under several parallel distance-lane counts

## Aggregate ratios

- mean CNN MAC ratio: `0.5492x`
- mean NN operation ratio: `0.0109x`
- mean total proxy ratio: `0.2687x`
- mean memory-bank ratio: `0.0249x`
- mean streamed-bank traffic ratio: `0.0109x`

## Category table

| category | selected config | good-pass | false-pass | CNN MAC | NN ops | total proxy | bank | streamed traffic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | `res18_l23_g5_b250_topk0p005` | 94.00% | 5.16% | 0.1994x | 0.0007x | 0.0958x | 0.0052x | 0.0007x |
| cable | `res18_l23_g7_b1500_topk0p05` | 33.79% | 2.61% | 0.1994x | 0.0078x | 0.0995x | 0.0312x | 0.0078x |
| capsule | `wrn_l2_g14_b125_topk0p05` | 21.82% | 5.45% | 0.4215x | 0.0035x | 0.2035x | 0.0035x | 0.0035x |
| carpet | `wrn_l3_g8_b125_topk0p005` | 68.57% | 4.89% | 1.0000x | 0.0023x | 0.4797x | 0.0069x | 0.0023x |
| grid | `res18_l23_g7_b6000_topk0p005` | 38.18% | 4.14% | 0.1994x | 0.0312x | 0.1117x | 0.1250x | 0.0312x |
| hazelnut | `res18_l23_g14_b500_topk0p02` | 92.00% | 1.14% | 0.1994x | 0.0104x | 0.1008x | 0.0104x | 0.0104x |
| leather | `wrn_l3_g10_b125_topk0p005` | 93.75% | 3.91% | 1.0000x | 0.0035x | 0.4804x | 0.0069x | 0.0035x |
| metal_nut | `wrn_l3_g6_b1000_topk0p005` | 40.00% | 6.38% | 1.0000x | 0.0102x | 0.4839x | 0.0556x | 0.0102x |
| pill | `res18_l23_g5_b125_topk0p005` | 18.46% | 2.82% | 0.1994x | 0.0003x | 0.0956x | 0.0026x | 0.0003x |
| screw | `res18_l23_g14_b125_topk0p05` | 3.81% | 5.76% | 0.1994x | 0.0026x | 0.0968x | 0.0026x | 0.0026x |
| tile | `wrn_l3_g7_b125_topk0p005` | 77.65% | 3.81% | 1.0000x | 0.0017x | 0.4795x | 0.0069x | 0.0017x |
| toothbrush | `wrn_l2_g5_b125_topk0p005` | 43.33% | 6.67% | 0.4215x | 0.0005x | 0.2041x | 0.0035x | 0.0005x |
| transistor | `wrn_l3_g6_b500_topk0p005` | 47.33% | 11.00% | 1.0000x | 0.0051x | 0.4812x | 0.0278x | 0.0051x |
| wood | `res18_l23_g7_b125_topk0p005` | 71.11% | 2.00% | 0.1994x | 0.0007x | 0.0957x | 0.0026x | 0.0007x |
| zipper | `wrn_l3_g14_b1500_topk0p05` | 75.00% | 2.71% | 1.0000x | 0.0833x | 0.5220x | 0.0833x | 0.0833x |

## Interpretation for thesis lock

- The nearest-neighbor search reduction is mathematically explained by patch count, bank size, and feature dimension.
- After the NN search is reduced, CNN feature extraction becomes the dominant remaining compute block.
- Therefore the FPGA thesis should not claim only `PatchCore is 98% lighter`.
- The defensible claim is: category profiling can shrink the memory-bank search engine dramatically, and the final FPGA implementation must measure how much of that reduction survives after CNN and memory-system costs are included.

CSV: `results/mvtec_patchcore_fpga_cost_model_001.csv`
Figure: `results/mvtec_patchcore_fpga_cost_model_001.png`
