#!/bin/bash
# Fresh server setup (Sep 26: the old A100's GPU was taken by another renter, the instance cannot start). Copies from the
# Mac only what today's three chains need (~1 GB: engine source, code, 3 checkpoints, recorded games, 2 opponent bots),
# builds the C++ engine for the server's Python, checks it bit-exact on recorded games, installs orjson/pybind11/scipy.
# If the operator copied the old disk (/Users/) with Vast's copy button, existing files are kept (rsync -u).
# Usage: srv_bootstrap.sh <ssh port> <host>
PORT=$1; HOST=$2
K=/Users/asad.ali/kaggriculture; PY=/venv/main/bin/python
S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT root@$HOST"
RS() { rsync -azu -e "ssh -o StrictHostKeyChecking=accept-new -p $PORT" "$@"; }
set -e
$S "nproc; free -g | head -2; df -h / | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; $PY -V; \
    mkdir -p $K/il/ckpt $K/il/logs $K/il/data $K/il/live $K/tools $K/engine/src_v2 $K/engine/v2 $K/agents/proxy \
             $K/package/STACK1_RACE5 $K/package/STACK1_RACE4 $K/data /root/Downloads"
RS $K/engine/src_v2/ root@$HOST:$K/engine/src_v2/
RS --exclude='__pycache__' --exclude='*.so' $K/tools/ root@$HOST:$K/tools/
RS $K/il/*.py root@$HOST:$K/il/
RS --include='*.py' --include='*.sh' --include='*.json' --exclude='*' $K/il/live/ root@$HOST:$K/il/live/
RS $K/il/data/team_vocab.json root@$HOST:$K/il/data/
RS $K/il/ckpt/fr_t_s0.pt $K/il/ckpt/fr_t_dsm_s0.pt $K/il/ckpt/rl2_it24.pt root@$HOST:$K/il/ckpt/
RS $K/agents/proxy/FORK_M.py root@$HOST:$K/agents/proxy/
RS $K/package/STACK1_RACE5/main.py root@$HOST:$K/package/STACK1_RACE5/
RS $K/package/STACK1_RACE4/main.py root@$HOST:$K/package/STACK1_RACE4/
for d in tapes_top12 tapes_kd tapes_n10md tapes_n10mr tapes_n10m tapes_n4m tapes_neural tapes_race tapes_stack1 tapes_ladder; do
  RS $K/data/$d root@$HOST:$K/data/
done
RS $K/data/rows_*.json root@$HOST:$K/data/
# baselines for the paired comparisons (the Mac copies of last night's server results)
for f in ladder_fr_t_dsm_s0_t10_r3_tau0.5_test.json ladder_rl2_it24_t10_r3_tau0.5_test.json top_fresh_fr_t_dsm_s0_t10.json; do
  src=$K/il/logs/$f; [ -f $src ] || src=$K/il/logs/server_logs/$f
  [ -f $src ] && RS $src root@$HOST:$K/il/logs/
done
set +e
# losses of all four neural submissions for RL (LOSS_DIR), the governor's data-build markers, engine build + checks
$S "mkdir -p $K/data/tapes_nlive && for d in tapes_n10m tapes_n4m tapes_n10md tapes_n10mr; do cp -n $K/data/\$d/*.json.gz $K/data/tapes_nlive/ 2>/dev/null; done; \
    ls $K/data/tapes_nlive | wc -l; \
    grep -q 'FRESH7 READY' $K/il/logs/fresh_data.log 2>/dev/null || printf 'FRESH READY (old server)\nFRESH7 READY (old server)\n' >> $K/il/logs/fresh_data.log; \
    rm -f $K/il/logs/fr_pending; \
    $PY -m pip install -q orjson pybind11 scipy 2>&1 | tail -1; \
    command -v c++ >/dev/null || (apt-get update -qq && apt-get install -y -qq g++ >/dev/null); \
    cd $K/engine/src_v2 && EXT=\$($PY -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') && \
    ls ../v2/kagsim_trace\$EXT >/dev/null 2>&1 || c++ -O3 -std=c++17 -fvisibility=hidden -DNDEBUG -fPIC \$($PY -m pybind11 --includes) -I. -shared \
       kagsim_trace.cpp -o ../v2/kagsim_trace\$EXT; ls -la ../v2/; \
    cd $K/il && $PY live/engine_parity.py $K/data/tapes_top12 30 && \
    $PY -c 'import features, live_rollout, ppo_live, torch; print(\"imports OK, cuda\", torch.cuda.is_available())'"
echo "BOOTSTRAP DONE"
