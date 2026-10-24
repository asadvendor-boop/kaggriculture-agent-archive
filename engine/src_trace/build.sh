#!/bin/bash
# kagsim_trace build — mirrors the pinned production build flags
# (setup.py Pybind11Extension: cxx_std=17, -O3; pybind11 headers offline from
# the torch-vendored copy, version 3.0.1 — phase0 rebuild used 3.1.0; parity
# is proven behaviorally, not by flag identity).
set -euo pipefail
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/Users/asad.ali/Downloads/Kaggriculture-1327-Recovery/.venv/bin/python
PYINC="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PB11="/Users/asad.ali/Downloads/annual reports pildat all/venv_gpu/lib/python3.11/site-packages/torch/include"
EXT="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
c++ -O3 -std=c++17 -fvisibility=hidden -DNDEBUG -fPIC \
  -I "$PB11" \
  -I "$PYINC" \
  -I "$WT/engine/kaggriculture-cppsim/sim" \
  -shared -undefined dynamic_lookup \
  "$WT/e315/engine_trace/kagsim_trace.cpp" \
  -o "$WT/e315/engine_trace/kagsim_trace$EXT"
echo "built: $WT/e315/engine_trace/kagsim_trace$EXT"
shasum -a 256 "$WT/e315/engine_trace/kagsim_trace$EXT"
