"""Convert versioned int8 fixtures to hex and run WSL Icarus, preserving logs."""
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'fpga/sim/build/real'
EVIDENCE = ROOT / 'fpga/evidence/real_features_001'


def linux(path):
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]


def run():
    BUILD.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    exe = BUILD / 'real.vvp'
    subprocess.run(['wsl', '-e', 'iverilog', '-g2012', '-s', 'tb_real_features', '-o', linux(exe),
                    linux(ROOT/'fpga/rtl/profiled_nn_scalar.v'), linux(ROOT/'fpga/sim/tb_real_features.v')], check=True)
    results = []
    fixtures = sorted((ROOT/'results/patchcore_int8_preparation_001').glob('*_rtl.npz'))
    if len(fixtures) != 15:
        raise RuntimeError(f'Expected 15 category fixtures, found {len(fixtures)}')
    for path in fixtures:
        data = np.load(path, allow_pickle=False)
        for name in ('bank', 'queries', 'expected'):
            a = data[name].reshape(-1)
            (BUILD/f'{name}.hex').write_text(''.join(f'{int(v)&(0xffffffff if name=="expected" else 255):0{8 if name=="expected" else 2}x}\n' for v in a))
        bank = data['bank']
        with (EVIDENCE/f'{path.stem}.log').open('w') as log:
            subprocess.run(['wsl','-e','vvp',linux(exe),f'+DIM={bank.shape[1]}',f'+POINTS={bank.shape[0]}',
                            '+BANK='+linux(BUILD/'bank.hex'),'+QUERY='+linux(BUILD/'queries.hex'),
                            '+EXPECTED='+linux(BUILD/'expected.hex')], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        results.append({'category':path.stem.removesuffix('_rtl'),'status':'pass'})
        (EVIDENCE/'summary.json').write_text(json.dumps(results,indent=2))
        print('RTL PASS',path.stem,flush=True)


if __name__ == '__main__':
    run()
