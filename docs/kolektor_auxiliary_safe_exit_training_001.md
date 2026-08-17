# KolektorSDD auxiliary safe-exit training

Early exits are explicitly trained to predict safe good pass. No extra CNN is evaluated at inference.

| max false pass | min good pass | policy | eval feasible | eval good pass | eval false pass | eval avg cost | speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 10.0% | 95.0% | final_selective | no | 94.29% | 0.00% | 1.0000 | 1.00x |
| 10.0% | 95.0% | branchynet_upper_only | no | 97.14% | 30.00% | 0.9002 | 1.11x |
| 10.0% | 95.0% | auxiliary_safe_dual_exit | no | 98.57% | 90.00% | 0.3880 | 2.58x |
