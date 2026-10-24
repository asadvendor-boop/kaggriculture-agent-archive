#!/bin/bash
# Verifier builds of the reference engine (server only). Never touches engine/ : sources are read in place.
#   cov/  : -O0 --coverage  (gcov line + branch coverage of sim_trace.hpp / pyrandom.hpp / kagsim_trace.cpp)
#   opt/  : -O3             (same wrapper, for fast scenario runs)
# Both add Game.set_state(obs0, obs1) and market_price(item, inv) (see kagsim_cov.cpp); rules code unmodified.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=/Users/asad.ali/kaggriculture/engine/src_v2
PY=/venv/main/bin/python
PYINC="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PB11="$($PY -c 'import pybind11; print(pybind11.get_include())')"
EXT="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
mkdir -p "$HERE/cov" "$HERE/opt"
cd "$HERE/cov"
g++ -O0 --coverage -fprofile-update=atomic -std=c++17 -fvisibility=hidden -fPIC -I "$PB11" -I "$PYINC" -I "$SRC" \
    -shared "$HERE/kagsim_cov.cpp" -o "$HERE/cov/kagsim_trace$EXT"
cd "$HERE/opt"
g++ -O3 -DNDEBUG -std=c++17 -fvisibility=hidden -fPIC -I "$PB11" -I "$PYINC" -I "$SRC" \
    -shared "$HERE/kagsim_cov.cpp" -o "$HERE/opt/kagsim_trace$EXT"
ls -la "$HERE/cov" "$HERE/opt"
md5sum "$SRC"/*.hpp "$SRC"/*.cpp
