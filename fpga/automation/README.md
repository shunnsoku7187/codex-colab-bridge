# Boardless checks without chat log monitoring

From the repository root on this Windows PC:

```powershell
python fpga/automation/boardless_pipeline.py status
python fpga/automation/boardless_pipeline.py plan
python fpga/automation/boardless_pipeline.py run
```

To leave the chat while a tool job continues, use the detached launcher:

```powershell
& fpga/automation/start.ps1
& fpga/automation/resume.ps1
```

`start.ps1` starts the runner in a hidden independent Windows process and returns immediately. `resume.ps1` reports whether that process is still alive and prints the saved stage states once. It does not restart anything. After closing or changing chats, ask Codex to read `fpga/automation/HANDOFF.md` and run `resume.ps1` once. If Windows is shut down, the process stops; its stage state and logs remain for a later explicit `run`. The current six stages are already passed, so starting now only verifies that they are skipped.

`run` executes each supported stage once in order and waits locally. It writes the full output to `fpga/vivado/build/pipeline_<stage>.log` and a small state file to `fpga/vivado/build/boardless_pipeline_state.json`. It prints only a start/result line. A successful stage with unchanged input files is skipped on later runs. On failure, the pipeline stops and preserves the log; a later `run` retries that stage. Use `--only route` to run one stage or `--force --only route` to deliberately repeat a passed stage. Do not start a second copy while one is running.

The existing successful runs on this PC were recorded with `adopt`, so the first `run` now skips them. `adopt` reads previously completed logs; it does not execute the tools. The state file is local evidence, not a Git artifact. For a concise update, use `status` once after the job finishes. There is no need for chat-side polling.

Stages currently automated: PS+DPU block design, synthesis, route, CNN calibration, quantized model export and DPU compilation. The script does **not** claim AXI transaction correctness, numerical PatchCore accuracy, or a full-image decision simulation. Those require additional implementations and test oracles before they can be added as passing stages.
