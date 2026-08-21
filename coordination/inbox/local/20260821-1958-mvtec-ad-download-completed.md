# Reply from konbu Codex CLI: MVTec AD download completed

## Completion

- Job: `mvtec_ad_hf_download_001`
- Status: `done`
- Return code: `0`
- Runtime: `210.572 sec`
- Finished: 2026-08-21 19:55:49 JST
- Finish commit: `bcfec1b caviar9: finish mvtec_ad_hf_download_001`
- Final runner-log commit: `dc64f82 complete mvtec ad download runner log`
- Download/authentication problems: none
- Final Git state before this reply: `## main...origin/main`

## Dataset location and size

- Root: `/home/shunya/codex-gpu-work/data/mvtec_ad`
- Size reported by the audit: `4.907 GiB`
- Filesystem size from `du`: about `5.0G`
- Raw files were not committed to Git.

## Downloaded content

The Hugging Face mirror completed all 33 repository files. Its README metadata
describes all 15 MVTec AD categories:

`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`,
`metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, and
`zipper`.

README split metadata reports:

- train examples: `3,629`
- test examples: `1,725`
- total examples: `5,354`
- category coverage: `15/15`

The schema includes `image_path`, `split`, `object`, `defect`, `label`, and
`mask_path`. Ground-truth masks are embedded in the Parquet `mask_path` column,
not stored in a conventional `ground_truth/` directory. A standalone
ground-truth image count was not verified because a Parquet reader is not yet
installed in the caviar9 Python environment.

## Probe readiness

The download is complete, but it is not yet ready for the existing
directory-oriented 3-category probe.

The mirror stores category train/test splits as 31 Parquet shards under
`mvtec_ad/data/`, rather than the conventional structure:

```text
<category>/train/
<category>/test/
<category>/ground_truth/
```

This is why the original structure audit reported zero directory-form
categories even though all 15 category Parquet splits are present. Before the
3-category probe, add a Parquet reader/conversion step or teach the probe loader
to read the Hugging Face Parquet schema directly. The raw 5GB download should
be retained and reused; it does not need to be downloaded again.
