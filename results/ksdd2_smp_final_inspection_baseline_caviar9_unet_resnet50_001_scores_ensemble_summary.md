# KSDD2 score ensemble check

Seed別スコアとseed平均アンサンブルを、同じvalidation-selected thresholdで比較する。

Note: Score files use different validation/test ordering or splits; sample-wise seed averaging is not valid.

| score source | target false-pass | target good-pass | val feasible | test feasible | test AUROC | test AUPR | test false-pass | test good-pass | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_123/max_score | 0.00% | 90.00% | no | no | 0.9874 | 0.9545 | - | - | - |
| seed_123/max_score | 0.00% | 95.00% | no | no | 0.9874 | 0.9545 | - | - | - |
| seed_123/max_score | 1.00% | 90.00% | no | no | 0.9874 | 0.9545 | - | - | - |
| seed_123/max_score | 1.00% | 95.00% | no | no | 0.9874 | 0.9545 | - | - | - |
| seed_123/max_score | 5.00% | 90.00% | yes | no | 0.9874 | 0.9545 | 9.09% | 98.21% | 0.989555 |
| seed_123/max_score | 5.00% | 95.00% | yes | no | 0.9874 | 0.9545 | 9.09% | 98.21% | 0.989555 |
| seed_123/topk_score | 0.00% | 90.00% | no | no | 0.9901 | 0.9605 | - | - | - |
| seed_123/topk_score | 0.00% | 95.00% | no | no | 0.9901 | 0.9605 | - | - | - |
| seed_123/topk_score | 1.00% | 90.00% | no | no | 0.9901 | 0.9605 | - | - | - |
| seed_123/topk_score | 1.00% | 95.00% | no | no | 0.9901 | 0.9605 | - | - | - |
| seed_123/topk_score | 5.00% | 90.00% | yes | no | 0.9901 | 0.9605 | 10.00% | 99.44% | 0.944229 |
| seed_123/topk_score | 5.00% | 95.00% | yes | no | 0.9901 | 0.9605 | 10.00% | 99.44% | 0.944229 |
| seed_456/max_score | 0.00% | 90.00% | no | no | 0.9892 | 0.9467 | - | - | - |
| seed_456/max_score | 0.00% | 95.00% | no | no | 0.9892 | 0.9467 | - | - | - |
| seed_456/max_score | 1.00% | 90.00% | no | no | 0.9892 | 0.9467 | - | - | - |
| seed_456/max_score | 1.00% | 95.00% | no | no | 0.9892 | 0.9467 | - | - | - |
| seed_456/max_score | 5.00% | 90.00% | yes | yes | 0.9892 | 0.9467 | 3.64% | 94.07% | 0.943197 |
| seed_456/max_score | 5.00% | 95.00% | no | no | 0.9892 | 0.9467 | - | - | - |
| seed_456/topk_score | 0.00% | 90.00% | no | no | 0.9863 | 0.9535 | - | - | - |
| seed_456/topk_score | 0.00% | 95.00% | no | no | 0.9863 | 0.9535 | - | - | - |
| seed_456/topk_score | 1.00% | 90.00% | no | no | 0.9863 | 0.9535 | - | - | - |
| seed_456/topk_score | 1.00% | 95.00% | no | no | 0.9863 | 0.9535 | - | - | - |
| seed_456/topk_score | 5.00% | 90.00% | yes | yes | 0.9863 | 0.9535 | 3.64% | 94.52% | 0.645489 |
| seed_456/topk_score | 5.00% | 95.00% | no | no | 0.9863 | 0.9535 | - | - | - |
