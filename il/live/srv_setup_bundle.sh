#!/bin/bash
# Run on a NEW server after extracting the upload bundle into / (tar xf kag_server_bundle_sep26.tar -C /):
# Python packages, the C++ engine built for this machine's Python, the loss-tape folder for RL, the CPU-governor markers,
# then the engine parity check on 30 recorded games and an import check.   Usage: bash srv_setup_bundle.sh
K=/Users/asad.ali/kaggriculture; PY=/venv/main/bin/python
$PY -m pip install -q orjson pybind11 scipy gdown 2>&1 | tail -1
command -v c++ >/dev/null || (apt-get update -qq && apt-get install -y -qq g++ >/dev/null)
mkdir -p $K/engine/v2 $K/data/tapes_nlive $K/il/logs /root/Downloads
cd $K/engine/src_v2 || exit 1
EXT=$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
c++ -O3 -std=c++17 -fvisibility=hidden -DNDEBUG -fPIC $($PY -m pybind11 --includes) -I. -shared kagsim_trace.cpp -o ../v2/kagsim_trace$EXT \
  && echo "engine built: $K/engine/v2/kagsim_trace$EXT"
for d in tapes_n10m tapes_n4m tapes_n10md tapes_n10mr; do cp -n $K/data/$d/*.json.gz $K/data/tapes_nlive/ 2>/dev/null; done
echo "loss tapes for RL: $(ls $K/data/tapes_nlive | wc -l)"
grep -q 'FRESH7 READY' $K/il/logs/fresh_data.log 2>/dev/null || printf 'FRESH READY (old server)\nFRESH7 READY (old server)\n' >> $K/il/logs/fresh_data.log
rm -f $K/il/logs/fr_pending
cd $K/il
$PY live/engine_parity.py $K/data/tapes_top12 30
$PY -c 'import features, live_rollout, ppo_live, torch; print("imports OK, torch", torch.__version__, "cuda", torch.cuda.is_available())'
echo "SETUP DONE"
