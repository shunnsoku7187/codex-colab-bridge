# Reply from konbu Codex CLI: dataset replan audits completed

## Completion status

Both requested light audit jobs completed and pushed their outputs.

### KSDD2 split bias deep audit

- Job: `ksdd2_split_bias_deep_audit_001`
- Status: `done`
- Return code: `0`
- Runtime: `108.894 sec`
- Finish commit: `8a3a8d8 caviar9: finish ksdd2_split_bias_deep_audit_001`

Generated the requested JSON, Markdown, sample CSV, false-pass gallery, and log
files.

### Dataset availability audit

- Job: `dataset_availability_audit_001`
- Status: `done`
- Return code: `0`
- Runtime: `19.624 sec`
- Finish commit: `7521c2e caviar9: finish dataset_availability_audit_001`

Generated the requested JSON, Markdown, and log files.

## Seed 456 interpretation

At a test-oracle defect false-pass budget of 5%, all seeds had five false-pass
defects per score, but seed 456 preserved much less good-pass:

- seed 123: `90.94%` max-score / `92.84%` top-k
- seed 456: `75.28%` max-score / `75.62%` top-k
- seed 789: `95.08%` max-score / `95.41%` top-k

Seed 456 has a much lower defect-score lower tail:

- max-score q10: `0.957245`, versus `0.999034` and `0.999427`
- top-k q10: `0.935642`, versus `0.987058` and `0.906824`

Its false-pass defects also have a smaller median mask-area ratio (`0.002518`)
than seeds 123 and 789 (`0.006072`). This supports the interpretation that the
seed-456 model/calibration is more vulnerable to small defects rather than the
model completely losing score separation.

The current `series_key` extraction collapses all false-pass samples to the
same `kolektor_sdd2/test/` key, so this audit cannot yet determine whether the
errors concentrate in a real image series. A filename/group parser correction
is needed before making a series-bias claim.

## Dataset availability

None of the three larger datasets is currently present under
`/home/shunya/codex-gpu-work/data`:

- MVTec AD: absent
- VisA: absent
- MVTec AD 2: absent

Network routes were reachable during the audit:

- MVTec AD: Hugging Face mirror API and official page reachable
- VisA: AWS registry and Spot-the-Difference repository reachable
- MVTec AD 2: official and benchmark pages reachable

The `aws` CLI is not installed, while Git and Python are available. An automated
MVTec AD download via the reachable Hugging Face mirror looks practical. VisA
needs a dedicated archive/download-format step. MVTec AD 2 may still require
manual official-form setup despite its pages being reachable.

## Final Git state

Before creating this reply:

```text
## main...origin/main
```
