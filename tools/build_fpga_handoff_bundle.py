"""Create a portable handoff ZIP from versioned sources, fixtures and evidence."""
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('fpga/', 'results/patchcore_int8_preparation_001/')
EXPLICIT = {'docs/FPGA_HARDWARE_HANDOFF.md', 'docs/FPGA_PREPARATION_PIPELINE.md',
            'docs/RESEARCH_HANDOFF.md', 'docs/ARCHIVE_GAP_AUDIT.md',
            'tools/run_real_feature_sim.py', 'tools/check_int8_preparation_contract.py',
            'scripts/prepare_patchcore_int8.py', 'tools/build_fpga_handoff_bundle.py',
            'results/fpga_preparation_pipeline_001.json', 'requirements-colab.txt', '.gitattributes'}

if __name__ == '__main__':
    files = subprocess.check_output(['git','ls-files'],cwd=ROOT).decode().splitlines()
    files = [name for name in files if name.startswith(PREFIXES) or name in EXPLICIT]
    out = ROOT/'output/handoff/patchcore_fpga_hardware_handoff.zip'
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest = {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                'read_first':'docs/FPGA_HARDWARE_HANDOFF.md','files':[]}
    with ZipFile(out,'w',ZIP_DEFLATED) as archive:
        archive.writestr('READ_FIRST.txt','Start with docs/FPGA_HARDWARE_HANDOFF.md.\nSource repository: https://github.com/shunnsoku7187/codex-colab-bridge\nThis bundle contains real int8 NN fixtures and scalar implementation evidence, not a complete PatchCore bitstream.\n')
        for name in files:
            data = (ROOT/name).read_bytes()
            archive.writestr(name,data)
            manifest['files'].append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        archive.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
    with ZipFile(out) as archive:
        if archive.testzip() is not None: raise RuntimeError('ZIP integrity failure')
        for item in manifest['files']:
            if hashlib.sha256(archive.read(item['path'])).hexdigest()!=item['sha256']:
                raise RuntimeError('Bundle hash mismatch: '+item['path'])
    print(f'{out}\n{len(files)} files, {out.stat().st_size} bytes; all ZIP hashes checked')
