#!/bin/bash
# Sprint machine setup (Sep 27; 8x H100 NVL rental, 2.5-3 h): copies from the Mac only what the JAX RL sprint and our
# rulers need (~250 MB: engine source, code incl. kagjax without logs/caches, the 10M warm start + its multi-style
# parent, the 1,051 ruler tapes, leader + specialist tapes, baseline result files), installs JAX-CUDA (own venv) and
# kaggle-environments, builds the C++ engine, and runs the quick checks (C++ engine on 30 recorded games, JAX sees all
# GPUs). Usage: srv_bootstrap2.sh <ssh port> <host>
PORT=$1; HOST=$2; K=/Users/asad.ali/kaggriculture
S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=10 -p $PORT root@$HOST"
RS() { for a in 1 2 3 4 5 6; do rsync -az --partial-dir=.rsync-partial --timeout=120 -e "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT" "$@" && return 0; sleep 5; done; echo "RSYNC FAILED: $*"; return 1; }
T0=$(date +%s); echo "[$(date -u +%H:%M:%S) UTC] bootstrap starts"
B0=$($S 'ls /venv/main/bin/python 2>/dev/null || command -v python3' 2>/dev/null | tail -1); B0=${B0:-/venv/main/bin/python}
echo "base python on the server: $B0"
$S "nproc; free -g | head -2; df -h / | tail -1; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; $B0 -V; \
    mkdir -p $K/il/ckpt $K/il/logs $K/il/data $K/il/live $K/tools $K/engine/src_v2 $K/engine/v2 $K/agents/ref $K/agents/proxy $K/data $K/kagjax /root/Downloads"
# the JAX + kaggle-environments install runs on the server while the files are being copied
$S "setsid nohup bash -c '$B0 -m venv --system-site-packages /opt/jaxenv && /opt/jaxenv/bin/pip install -q -U \"jax[cuda12]\" && /opt/jaxenv/bin/pip install -q --no-deps kaggle-environments==1.32.7 && /opt/jaxenv/bin/pip install -q jsonschema requests; echo PIP EXIT \$?' > /root/pip.log 2>&1 < /dev/null &"
RS $K/engine/src_v2/ root@$HOST:$K/engine/src_v2/
RS --exclude='__pycache__' --exclude='*.so' $K/tools/ root@$HOST:$K/tools/
RS $K/il/*.py root@$HOST:$K/il/
RS --include='*.py' --include='*.sh' --include='*.json' --exclude='*' $K/il/live/ root@$HOST:$K/il/live/
RS $K/il/data/team_vocab.json root@$HOST:$K/il/data/
RS $K/il/ckpt/rf_w_dsm_s0.pt $K/il/ckpt/rf_w_s0.pt root@$HOST:$K/il/ckpt/
RS $K/agents/ref/V94_2945_main.py root@$HOST:$K/agents/ref/
RS $K/agents/proxy/FORK_M.py root@$HOST:$K/agents/proxy/
RS --exclude='__pycache__' --exclude='.jax_cache' --exclude='logs_server' --exclude='runs' --exclude='cov_build' \
   --exclude='results/*.npz' --exclude='*.npz.tmp' --exclude='e2e_out' $K/kagjax/ root@$HOST:$K/kagjax/
for d in tapes_top12 tapes_ladder tapes_race tapes_stack1 tapes_neural tapes_spec; do RS $K/data/$d root@$HOST:$K/data/; done
[ -d $K/data/tapes_kimi ] && RS $K/data/tapes_kimi root@$HOST:$K/data/
# baselines for the paired comparisons: NEURAL_20MW5_SE (with SE) and the 10M warm start (without SE)
for f in ladder_g20_w5_dsm_s0_t10_r3_tau0.5_se_test.json top_fresh_g20_w5_dsm_s0_t10_se.json top_fresh2_g20_w5_dsm_s0_t10_se.json \
         ladder_rf_w_dsm_s0_t10_r3_tau0.5_test.json top_fresh_rf_w_dsm_s0_t10.json top_fresh2_rf_w_dsm_s0_t10.json; do
  RS $K/il/logs/server2/$f root@$HOST:$K/il/logs/
done
echo "[$(date -u +%H:%M:%S) UTC] files copied ($(( $(date +%s) - T0 )) s)"
$S "command -v c++ >/dev/null || (apt-get update -qq && apt-get install -y -qq g++ >/dev/null); $B0 -m pip install -q orjson pybind11 scipy numpy torch 2>&1 | tail -1; \
    cd $K/engine/src_v2 && EXT=\$($B0 -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') && \
    c++ -O3 -std=c++17 -fvisibility=hidden -DNDEBUG -fPIC \$($B0 -m pybind11 --includes) -I. -shared kagsim_trace.cpp -o ../v2/kagsim_trace\$EXT && ls -la ../v2/ && \
    cd $K/il && $B0 live/engine_parity.py $K/data/tapes_top12 30"
echo "[$(date -u +%H:%M:%S) UTC] waiting for the JAX install"
for i in $(seq 1 60); do $S "grep -q 'PIP EXIT' /root/pip.log" 2>/dev/null && break; sleep 10; done
cat > /tmp/claude_gputest.py <<'PYX'
import jax, jax.numpy as jnp, time
d = jax.devices(); print("jax", jax.__version__, len(d), "devices:", d[0].device_kind)
k = jax.random.PRNGKey(0); x = jax.random.normal(k, (1024, 10, 10, 256)); w = jax.random.normal(k, (3, 3, 256, 256)) * 0.02
@jax.jit
def f(x, w):
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return jnp.sum(jax.nn.relu(y) ** 2)
g = jax.jit(jax.grad(f, argnums=1))
t = time.time(); v = float(f(x, w)); gw = g(x, w).block_until_ready(); print("conv fwd+grad OK", round(time.time() - t, 1), "s (incl. compile)")
t = time.time()
for _ in range(20): gw = g(x, w)
gw.block_until_ready(); print("20 conv grads", round(time.time() - t, 3), "s")
PYX
rsync -az -e "ssh -o ConnectTimeout=20 -p $PORT" /tmp/claude_gputest.py root@$HOST:/root/gputest.py
$S "tail -2 /root/pip.log; cd $K && XLA_PYTHON_CLIENT_PREALLOCATE=false /opt/jaxenv/bin/python /root/gputest.py"
echo "[$(date -u +%H:%M:%S) UTC] BOOTSTRAP DONE in $(( $(date +%s) - T0 )) s"
