# KolektorSDD non-CIFAR inspection experiment

## Purpose

This experiment checks whether the lower-side early reject idea still works on a real industrial surface-defect dataset, not CIFAR with synthetic corruption.

## Dataset

- Kolektor Surface-Defect Dataset
- Binary inspection task: good surface vs visible defect
- Pass means accepting a good-surface decision
- False pass means a defective surface is accepted as good

## Best strict-evaluation rows

| target pass precision | max good loss | predictor | exit | early reject | final rate | avg cost | speedup | measured good loss |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 95.0% | 2.0% | trace logistic_l2 | exit1 | 10.00% | 90.00% | 0.9550 | 1.05x | 1.25% |
| 95.0% | 5.0% | prob_good_only raw_prob_good_threshold | exit0 | 12.50% | 87.50% | 0.9025 | 1.11x | 3.75% |
| 95.0% | 10.0% | prob_good_only raw_prob_good_threshold | exit0 | 18.75% | 81.25% | 0.8538 | 1.17x | 5.00% |
| 98.0% | 2.0% | trace logistic_l2 | exit1 | 10.00% | 90.00% | 0.9550 | 1.05x | 1.25% |
| 98.0% | 5.0% | prob_good_only raw_prob_good_threshold | exit0 | 18.75% | 81.25% | 0.8538 | 1.17x | 3.75% |
| 98.0% | 10.0% | prob_good_only raw_prob_good_threshold | exit0 | 18.75% | 81.25% | 0.8538 | 1.17x | 3.75% |
| 99.0% | 2.0% | trace logistic_l2 | exit1 | 10.00% | 90.00% | 0.9550 | 1.05x | 1.25% |
| 99.0% | 5.0% | prob_good_only raw_prob_good_threshold | exit0 | 18.75% | 81.25% | 0.8538 | 1.17x | 3.75% |
| 99.0% | 10.0% | prob_good_only raw_prob_good_threshold | exit0 | 18.75% | 81.25% | 0.8538 | 1.17x | 3.75% |
