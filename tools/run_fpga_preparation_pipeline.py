"""One-shot unattended remote preparation -> real-vector RTL -> report archive."""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = 'patchcore_int8_preparation_001'
BUILD = ROOT/'fpga/vivado/build'
EVIDENCE = ROOT/'fpga/evidence/implementation_001'
STATUS = ROOT/'fpga/sim/build/preparation_status.json'


def command(args, **kwargs):
    print('RUN', ' '.join(map(str,args)), flush=True)
    return subprocess.run(args, cwd=ROOT, check=True, timeout=kwargs.pop('timeout',180), **kwargs)


def state(stage, **extra):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'stage':stage,**extra},indent=2))
    print('STAGE',stage,flush=True)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    vivado = os.environ.get('VIVADO_BAT','C:/AMDDesignTools/2025.2/Vivado/bin/vivado.bat')
    state('starting')
    # The local implementation is independent of remote feature processing.
    v_log = (EVIDENCE/'console.log').open('w')
    vproc = subprocess.Popen([vivado,'-mode','batch','-source',str(ROOT/'fpga/vivado/implement_profiled_nn.tcl'),
                              '-log','vivado.log','-journal','vivado.jou'],cwd=BUILD,stdout=v_log,stderr=subprocess.STDOUT)
    state('remote_launch',vivado_pid=vproc.pid)
    command(['ssh','-A','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15',
             '-J','shunya@ssh.arch.info.mie-u.ac.jp','shunya@konbu.arch.info.mie-u.ac.jp',
             'cd ~/codex-gpu-work/colab-github-bridge && git pull --ff-only origin main && '
             f'bash tools/konbu_gpu_start_button.sh --job {JOB} --no-terminal'],timeout=240)
    state('remote_running',vivado_pid=vproc.pid)
    deadline=time.monotonic()+26*3600
    while time.monotonic()<deadline:
        try:
            command(['git','fetch','origin'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            job=json.loads(subprocess.check_output(['git','show',f'origin/main:jobs/{JOB}.json'],cwd=ROOT))
            if job['status'] in ('done','failed'):
                break
        except (subprocess.SubprocessError,ValueError) as exc:
            print('STATUS RETRY',type(exc).__name__,flush=True)
        time.sleep(180)
    else:
        raise TimeoutError('Remote completion not received after 26h')
    command(['git','merge','--ff-only','origin/main'])
    rtl='skipped_remote_failure'
    if job['status']=='done':
        state('real_feature_simulation')
        try:
            command([sys.executable,'tools/run_real_feature_sim.py'],timeout=3*3600)
            rtl='pass'
        except subprocess.SubprocessError:
            rtl='failed'
    state('collecting_implementation')
    try:
        vrc=vproc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        vrc='still_running'
    v_log.close()
    for filename in ('utilization.rpt','timing.rpt','vivado.log','vivado.jou'):
        src=BUILD/filename
        if src.exists(): shutil.copy2(src,EVIDENCE/filename)
    result={'gpu_job':job['status'],'rtl':rtl,'vivado_returncode':vrc,
            'rtl_scope':'Selected-profile signed-int8 squared-L2 for two real patches per category; not full-image RTL or CNN.',
            'rtl_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
            'completed_at':datetime.now(timezone.utc).isoformat()}
    out=ROOT/'results/fpga_preparation_pipeline_001.json'
    out.write_text(json.dumps(result,indent=2))
    command(['git','add','fpga/evidence','results/fpga_preparation_pipeline_001.json'])
    command(['git','commit','-m','results: archive real-vector RTL and scalar NN implementation evidence'])
    try:
        command(['git','push','origin','main'])
    except subprocess.CalledProcessError:
        command(['git','fetch','origin'])
        command(['git','merge','--no-edit','origin/main'])
        command(['git','push','origin','main'])
    state('complete',**result)


if __name__=='__main__':
    os.chdir(ROOT)
    try:
        main()
    except Exception as exc:
        state('failed',error=repr(exc))
        raise
