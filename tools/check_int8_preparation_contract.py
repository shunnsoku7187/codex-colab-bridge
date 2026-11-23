"""Small preflight for rounding, saturation and the actual RTL's worst distance."""
import subprocess
import numpy as np
from scripts.prepare_patchcore_int8 import quantize
from tools.run_real_feature_sim import ROOT, BUILD, linux

assert quantize(np.array([-200.,-1.5,-0.5,0.5,1.5,200.]),1.).tolist()==[-127,-2,0,0,2,127]
BUILD.mkdir(parents=True,exist_ok=True)
exe=BUILD/'contract.vvp'
subprocess.run(['wsl','-e','iverilog','-g2012','-s','tb_real_features','-o',linux(exe),
                linux(ROOT/'fpga/rtl/profiled_nn_scalar.v'),linux(ROOT/'fpga/sim/tb_real_features.v')],check=True)
for dim in (512,768):
    (BUILD/'contract_bank.hex').write_text('7f\n'*dim)
    (BUILD/'contract_query.hex').write_text('81\n'*dim+'7f\n'*dim)
    (BUILD/'contract_expected.hex').write_text(f'{dim*254**2:08x}\n00000000\n')
    subprocess.run(['wsl','-e','vvp',linux(exe),f'+DIM={dim}','+POINTS=1',
                    '+BANK='+linux(BUILD/'contract_bank.hex'),'+QUERY='+linux(BUILD/'contract_query.hex'),
                    '+EXPECTED='+linux(BUILD/'contract_expected.hex')],check=True)
print('INT8_CONTRACT_PREFLIGHT_PASS')
