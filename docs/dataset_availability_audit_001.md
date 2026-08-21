# Dataset availability audit

Data root: `/home/shunya/codex-gpu-work/data`

## Local availability

| dataset | found dirs | total images | total size GiB | network checks |
|---|---:|---:|---:|---|
| mvtec_ad | 0 | 0 | 0.000 | hf_mirror_api:ok, official_page:ok |
| visa | 0 | 0 | 0.000 | aws_registry:ok, spot_diff_repo:ok |
| mvtec_ad_2 | 0 | 0 | 0.000 | official_page:ok, benchmark_page:ok |

## Tool availability

- `aws`: `not found`
- `git`: `/usr/bin/git`
- `python`: `/home/shunya/miniconda3/envs/cuda/bin/python`

## Next action

- If MVTec AD is already present, run the 3-category probe.
- If not present but the Hugging Face mirror is reachable, prepare an automated download job.
- If only official gated download is usable, download manually once to the data root and keep it there.
- If VisA network access is reachable, prepare a separate download/format audit because the archive is large.
- If MVTec AD 2 is reachable only through the official form/evaluation server, treat it as a medium-term dataset and record the manual setup steps.
