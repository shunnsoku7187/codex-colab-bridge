# Experiment Retention Manifest

Date: 2026-07-03

This repository uses GitHub as a Codex-Colab bridge, so old job files and verbose
runner logs can accumulate quickly. The policy is to keep only files that are
needed to reproduce the latest experimental conclusions or run the next planned
job.

## Keep

Core negative-feasibility evidence:

- `docs/rigorous_router_feasibility_protocol.md`
- `docs/information_location_findings.md`
- `docs/model_process_trace_findings.md`
- `jobs/difficulty_mechanism_decomposition_001.json`
- `jobs/information_location_experiment_001.json`
- `jobs/feature_collision_impossibility_001.json`
- `jobs/model_process_tracing_balanced_002.json`
- `jobs/analyze_model_process_trace_003.json`
- corresponding latest `results/` files

Strict-router and validation evidence:

- `jobs/allowed_upper_low_correct_lgbm_002.json`
- `jobs/allowed_upper_low_correct_mlp_002.json`
- `jobs/allowed_upper_teacher_linear_002.json`
- `jobs/external_image_group_validation_train_002.json`
- `jobs/search_claimable_record_breakers_002.json`
- `jobs/validate_record_breakers_cv_001.json`
- corresponding latest `results/` files

Deadline-aware direction:

- `docs/virtual_deadline_experiment_plan.md`
- `docs/virtual_deadline_simulation_findings.md`
- `jobs/virtual_deadline_simulation_001.json`
- `jobs/virtual_deadline_sensitivity_001.json`
- corresponding latest `results/` files when available

Infrastructure still useful for future Colab runs:

- `tools/colab_github_runner_cell.py`
- `tools/bootstrap_colab_runner_cell.py`
- current scripts under `scripts/` and shared code under `src/`

## Delete

The cleanup removes:

- earlier numbered runs when a later run exists,
- smoke tests and roundtrip connectivity jobs,
- failed or superseded notebook execution jobs,
- template jobs that are no longer used,
- verbose logs for obsolete runs,
- duplicate alias result files where a numbered latest result exists,
- old local PowerPoint outputs superseded by the latest generated deck.

The intent is not to erase the research trail. The reasoning is preserved in
the docs above, while the bridge repository stays small enough to keep using as
a practical Colab handoff mechanism.
