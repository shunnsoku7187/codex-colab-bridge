# Request to konbu Codex CLI: run dataset replan audit jobs

## Goal

Run the first experiments from the dataset-expansion replan, including the
light audits for VisA and MVTec AD 2.

These are light audit jobs, not long training jobs.

## Target jobs

Run both:

1. `ksdd2_split_bias_deep_audit_001`
2. `dataset_availability_audit_001`

Job files:

```text
jobs/ksdd2_split_bias_deep_audit_001.json
jobs/dataset_availability_audit_001.json
```

## Why

The KSDD2 U-Net/ResNet50 recheck showed large seed variation.  The next step is
not more model tweaking, but checking whether the variation is caused by
split/difficulty bias and whether larger inspection datasets can be introduced.

The dataset availability audit should cover:

- MVTec AD,
- VisA,
- MVTec AD 2.

## Requested action

1. Pull latest `origin/main`.
2. Run the two target jobs on caviar9 using the existing caviar9 Git runner.
3. Since both jobs are short, it is acceptable to run them to completion.
4. Publish all generated logs, docs, and results to GitHub.
5. Reply under `coordination/inbox/local/`.

## Expected outputs

For `ksdd2_split_bias_deep_audit_001`:

```text
docs/ksdd2_split_bias_deep_audit_001.md
results/ksdd2_split_bias_deep_audit_001.json
results/ksdd2_split_bias_deep_audit_001_samples.csv
results/ksdd2_split_bias_deep_audit_001_false_pass_gallery.jpg
logs/ksdd2_split_bias_deep_audit_001*.log
```

For `dataset_availability_audit_001`:

```text
docs/dataset_availability_audit_001.md
results/dataset_availability_audit_001.json
logs/dataset_availability_audit_001*.log
```

## Reply requested

Include:

- whether both jobs completed,
- pushed commit hash,
- short interpretation of seed 456 from the deep audit,
- whether MVTec AD, VisA, or MVTec AD 2 are already present,
- whether network/download routes look usable,
- final `git status --short --branch`.

## Stop and report immediately if

- KSDD2 data is missing and cannot be reused/downloaded,
- saved score files are missing,
- dataset availability audit cannot access the data root,
- GitHub push fails.
