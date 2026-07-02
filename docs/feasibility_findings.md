# Feasibility Findings

## Experiment 1: Difficulty Mechanism Decomposition

Job:

- `difficulty_mechanism_decomposition_001`

Status:

- Completed successfully.
- Result file: `results/difficulty_mechanism_decomposition.json`

### Main Result

The first mechanism check currently supports the semantic/model-capacity
difficulty hypothesis more than the low-level observable difficulty hypothesis.

Category counts:

| Category | Definition | Count |
| --- | --- | ---: |
| Easy | LOW correct, HIGH correct | 6894 |
| Hard | LOW wrong, HIGH correct | 2091 |
| Impossible | LOW wrong, HIGH wrong | 794 |
| Inverse | LOW correct, HIGH wrong | 221 |

Model accuracies implied by these categories:

| Model | Accuracy |
| --- | ---: |
| LOW | 71.15% |
| HIGH | 89.85% |

### Zero-Latency Feature Separability

The strongest single zero-latency feature was weak:

| Metric | Best observed value |
| --- | ---: |
| Hard-vs-rest AUROC | 0.547 |
| Mutual information | 0.0081 |
| Absolute effect size vs Easy | 0.201 |

Interpretation:

- No individual streaming image statistic strongly separates Hard samples from
  the rest.
- This does not prove impossibility, but it weakens the hypothesis that simple
  low-level image quality factors are the main cause of LOW/HIGH disagreement.

Top single features were mostly flatness, block brightness means, blue-channel
histogram bins, edge density, and local HOG bins. Their AUROC values were all
close to chance.

### Class Concentration

Hard samples are not uniformly distributed across classes.

Summary:

| Statistic | Value |
| --- | ---: |
| Max class Hard rate | 40.0% |
| Min class Hard rate among top count table | 29.0% |
| Mean Hard rate among listed classes | 32.95% |
| Share of all Hard samples in top 10 classes | 16.93% |
| Share of all Hard samples in top 20 classes | 31.52% |

Interpretation:

- There is a noticeable class/semantic component.
- Because CIFAR-100 has balanced classes, class concentration is a meaningful
  sign that LOW/HIGH disagreement is not explained only by generic image quality.

### Feature Collision Evidence

Nearest-neighbor analysis in the full allowed feature space found Hard/Easy
pairs with opposite routing labels but close zero-latency feature values.

Summary:

| Metric | Value |
| --- | ---: |
| Hard samples | 2091 |
| Easy samples | 6894 |
| Median nearest Easy distance from Hard | 31.84 |
| 10th percentile distance | 29.75 |
| 90th percentile distance | 34.36 |
| Same-class rate among top collisions | 81.25% |

Interpretation:

- The strongest collision examples are often within the same class.
- This is important: even holding class constant, allowed low-level statistics
  can be similar while LOW/HIGH outcome differs.
- That points toward model-capacity, fine-detail, or representation effects that
  are not easily captured by image-load statistics.

### Representative Artifacts

The job generated representative image grids in the Colab Drive artifact folder:

| Artifact | Path |
| --- | --- |
| Hard samples with extreme feature values | `/content/drive/MyDrive/Research_Experiment/difficulty_mechanism_decomposition/hard_feature_extreme_grid.png` |
| Hard samples near feature-space center | `/content/drive/MyDrive/Research_Experiment/difficulty_mechanism_decomposition/hard_feature_normal_grid.png` |
| Nearest Hard/Easy feature-collision pairs | `/content/drive/MyDrive/Research_Experiment/difficulty_mechanism_decomposition/hard_easy_collision_pairs_grid.png` |

These grids should be visually inspected before making a final claim.

### Current Feasibility Interpretation

This experiment does not yet prove that a fixed zero-latency router is impossible.
However, it gives the first mechanistic warning:

- The best individual observable features barely separate Hard from non-Hard.
- Hard cases show class/semantic concentration.
- Hard/Easy feature collisions appear even within the same class.

Therefore the next experiment should not be another candidate-router search. It
should estimate the upper bound of the entire allowed feature space using strong
models. If strong upper-bound models also fail to route enough samples to LOW,
then the limitation is likely observability, not model tuning.

Next planned job:

- `allowed_feature_upper_bound_001`
