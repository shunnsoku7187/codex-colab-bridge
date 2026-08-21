# To konbu Codex CLI: universal GPU start button added

Local Codex added:

```text
tools/konbu_gpu_start_button.sh
docs/konbu_gpu_start_button.md
```

Purpose:

- Make konbu behave like the Colab run-once button.
- If local SSH is unstable, the user only has to run one command on konbu.
- The script pulls Git, finds or accepts a pending caviar9/GPU job, starts it
  detached on caviar9, prints a startup check, opens a live log terminal when
  possible, and then lets the caviar9 runner return logs/results through Git.

Please pull latest `main` on konbu and confirm the script is present:

```bash
cd ~/codex-gpu-work/colab-github-bridge
git pull --ff-only origin main
bash -n tools/konbu_gpu_start_button.sh
bash tools/konbu_gpu_start_button.sh --list
```

No GPU job needs to be started by this message alone unless there is already a
pending job that the user explicitly wants to run.

Minimal user-facing start command:

```bash
cd ~/codex-gpu-work/colab-github-bridge && git pull --ff-only origin main && bash tools/konbu_gpu_start_button.sh
```
