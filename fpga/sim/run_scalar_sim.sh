#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="$root/fpga/sim/build"
mkdir -p "$build"
iverilog -g2012 -s tb_profiled_nn_scalar -o "$build/profiled_nn.vvp" \
  "$root/fpga/rtl/profiled_nn_scalar.v" "$root/fpga/sim/tb_profiled_nn_scalar.v"
vvp "$build/profiled_nn.vvp"
