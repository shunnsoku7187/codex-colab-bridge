# PatchCore cost credibility audit

Purpose: verify whether the reported ~98% NN-search reduction is a transparent formula result and whether it also appears in measured runtime.

## What is counted

The NN-search cost is counted as:

`test images x patches per image x memory-bank patches x feature dimension`

For one image, this reduces to:

`patches per image x memory-bank patches x feature dimension`

This is not yet total FPGA system power.  The measured online total adds feature extraction, nearest-neighbor search, and score aggregation.

## Aggregate

- mean relative NN ops: `0.0109x`
- mean NN ops reduction: `98.91%`
- mean measured NN time: `0.0278x`
- mean measured online total time: `0.7489x`
- median measured online total time: `0.7673x`

## Decomposition by category

| category | selected config | patch ratio | bank ratio | feature-dim ratio | formula NN ratio | measured NN time | measured total time |
|---|---|---:|---:|---:|---:|---:|---:|
| bottle | `res18_l23_g5_b250_topk0p005` | 0.1276x | 0.0208x | 0.2500x | 0.0007x | 0.0054x | 0.6966x |
| cable | `res18_l23_g7_b1500_topk0p05` | 0.2500x | 0.1250x | 0.2500x | 0.0078x | 0.0172x | 0.8216x |
| capsule | `wrn_l2_g14_b125_topk0p05` | 1.0000x | 0.0104x | 0.3333x | 0.0035x | 0.0296x | 0.8111x |
| carpet | `wrn_l3_g8_b125_topk0p005` | 0.3265x | 0.0104x | 0.6667x | 0.0023x | 0.0223x | 0.8176x |
| grid | `res18_l23_g7_b6000_topk0p005` | 0.2500x | 0.5000x | 0.2500x | 0.0312x | 0.0579x | 0.7293x |
| hazelnut | `res18_l23_g14_b500_topk0p02` | 1.0000x | 0.0417x | 0.2500x | 0.0104x | 0.0369x | 0.7866x |
| leather | `wrn_l3_g10_b125_topk0p005` | 0.5102x | 0.0104x | 0.6667x | 0.0035x | 0.0290x | 0.7708x |
| metal_nut | `wrn_l3_g6_b1000_topk0p005` | 0.1837x | 0.0833x | 0.6667x | 0.0102x | 0.0247x | 0.6433x |
| pill | `res18_l23_g5_b125_topk0p005` | 0.1276x | 0.0104x | 0.2500x | 0.0003x | 0.0032x | 0.6754x |
| screw | `res18_l23_g14_b125_topk0p05` | 1.0000x | 0.0104x | 0.2500x | 0.0026x | 0.0245x | 0.7093x |
| tile | `wrn_l3_g7_b125_topk0p005` | 0.2500x | 0.0104x | 0.6667x | 0.0017x | 0.0164x | 0.7024x |
| toothbrush | `wrn_l2_g5_b125_topk0p005` | 0.1276x | 0.0106x | 0.3333x | 0.0005x | 0.0079x | 0.7783x |
| transistor | `wrn_l3_g6_b500_topk0p005` | 0.1837x | 0.0417x | 0.6667x | 0.0051x | 0.0175x | 0.7880x |
| wood | `res18_l23_g7_b125_topk0p005` | 0.2500x | 0.0104x | 0.2500x | 0.0007x | 0.0079x | 0.7673x |
| zipper | `wrn_l3_g14_b1500_topk0p05` | 1.0000x | 0.1250x | 0.6667x | 0.0833x | 0.1167x | 0.7364x |

## Interpretation

- If formula NN ratio and measured NN time ratio move together, the ~98% reduction is not a table artifact.
- If measured total time is much larger than the NN ratio, the remaining cost is mainly feature extraction and framework overhead.
- For FPGA claims, the next step is to replace Python/GPU/CPU wall time with hardware-estimated CNN MAC, memory bandwidth, and distance-engine throughput.

Figure: `results/mvtec_patchcore_cost_credibility_audit_002.png`
