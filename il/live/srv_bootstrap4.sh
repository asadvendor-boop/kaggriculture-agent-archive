#!/bin/bash
# Sep 28 evening box (256-thread EPYC 7B13 + RTX 3090, CPU tests; srv_bootstrap3.sh trimmed): copies from the Mac what
# tonight's batch needs (~0.5 GB): engine source, tools, il code + live scripts, 3 checkpoints (the live pick's soup3,
# newplan seed 0, v13_p3), the live NEURAL_20MW8_SE package (h2h opponent), the ruler tapes (464 = tapes_ladder / race /
# stack1 / neural, leader tests = tapes_top12, live losses = tapes_nps / n20mw8 / n20mw5), data/*.json and the baseline
# result files of the pick (soup3nose) -> builds the C++ engine, engine parity on 30 recorded games, kaggle-environments
# for the official package check. Usage: srv_bootstrap4.sh <port> <host>
PORT=$1; HOST=$2; K=/Users/asad.ali/kaggriculture
S="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=10 -p $PORT root@$HOST"
RS() { for a in 1 2 3 4 5 6; do rsync -az --partial-dir=.rsync-partial --timeout=120 -e "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $PORT" "$@" && return 0; sleep 5; done; echo "RSYNC FAILED: $*"; return 1; }
T0=$(date +%s); echo "[$(date -u +%H:%M:%S) UTC] bootstrap starts"
B0=$($S 'ls /venv/main/bin/python 2>/dev/null || command -v python3' 2>/dev/null | tail -1); B0=${B0:-python3}
echo "base python on the server: $B0"
$S "nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null; free -g | head -2; df -h / | tail -1; \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; $B0 -V; \
    mkdir -p $K/il/ckpt $K/il/logs $K/il/data $K/il/live $K/tools $K/engine/src_v2 $K/engine/v2 $K/agents/ref $K/data $K/package/NEURAL_20MW8_SE"
# official-check environment (kaggle-environments) installs while the files are being copied (detached: nohup hangs ssh)
$S "setsid -f bash -c '$B0 -m venv --system-site-packages /opt/jaxenv && /opt/jaxenv/bin/pip install -q --no-deps kaggle-environments==1.32.7 && /opt/jaxenv/bin/pip install -q jsonschema requests; echo PIP EXIT \$?' > /root/pip.log 2>&1 < /dev/null"
RS $K/engine/src_v2/ root@$HOST:$K/engine/src_v2/
RS --exclude='__pycache__' --exclude='*.so' $K/tools/ root@$HOST:$K/tools/
RS $K/il/*.py root@$HOST:$K/il/
RS --include='*.py' --include='*.sh' --include='*.json' --exclude='*' $K/il/live/ root@$HOST:$K/il/live/
RS $K/il/data/team_vocab.json root@$HOST:$K/il/data/
RS $K/il/ckpt/ilsp_v13_newplan_soup3.pt $K/il/ckpt/ilsp_v13_newplan.pt $K/il/ckpt/ilsp_v13_p3.pt root@$HOST:$K/il/ckpt/
RS --include='*.json' --exclude='*' $K/data/ root@$HOST:$K/data/
RS $K/agents/ref/V94_2945_main.py root@$HOST:$K/agents/ref/
for d in tapes_top12 tapes_ladder tapes_race tapes_stack1 tapes_neural tapes_npse tapes_soh tapes_nps tapes_n20mw8 tapes_n20mw5; do RS $K/data/$d root@$HOST:$K/data/; done
# baselines: the live pick NEURAL_20MW8NPS_SOH (soup3, SE off, held) and v13_p3 with SE (shipped NEURAL_20MW8_SE)
for f in top_fresh_ilsp_v13_newplan_soup3_t10_soup3nose.json top_fresh2_ilsp_v13_newplan_soup3_t10_soup3nose.json \
         ladder_ilsp_v13_newplan_soup3_t10_r3_tau0.5_soup3nose_test.json ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json \
         top_fresh_ilsp_v13_p3_t10_se.json top_fresh2_ilsp_v13_p3_t10_se.json \
         top_fresh_ilsp_v13_p3_t10_soh.json top_fresh2_ilsp_v13_p3_t10_soh.json ladder_ilsp_v13_p3_t10_r3_tau0.5_soh_test.json \
         top_fresh_ilsp_v13_newplan_t10_nph.json top_fresh2_ilsp_v13_newplan_t10_nph.json ladder_ilsp_v13_newplan_t10_r3_tau0.5_nph_test.json; do
  RS $K/backup_a100_sep28/il_logs/$f root@$HOST:$K/il/logs/
done
tar -xzOf $K/package/NEURAL_20MW8_SE.tar.gz main.py | $S "cat > $K/package/NEURAL_20MW8_SE/main.py"
echo "[$(date -u +%H:%M:%S) UTC] files copied ($(( $(date +%s) - T0 )) s)"
$S "command -v c++ >/dev/null || (apt-get update -qq && apt-get install -y -qq g++ >/dev/null); $B0 -m pip install -q orjson pybind11 scipy numpy torch 2>&1 | tail -1; \
    cd $K/engine/src_v2 && EXT=\$($B0 -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') && \
    c++ -O3 -std=c++17 -fvisibility=hidden -DNDEBUG -fPIC \$($B0 -m pybind11 --includes) -I. -shared kagsim_trace.cpp -o ../v2/kagsim_trace\$EXT && ls -la ../v2/ && \
    sha256sum $K/package/NEURAL_20MW8_SE/main.py | cut -c1-16 && cd $K/il && $B0 live/engine_parity.py $K/data/tapes_top12 30"
for i in $(seq 1 60); do $S "grep -q 'PIP EXIT' /root/pip.log" 2>/dev/null && break; sleep 10; done
$S "tail -1 /root/pip.log; ln -sf $B0 /venv_main_python 2>/dev/null; [ -x /venv/main/bin/python ] || (mkdir -p /venv/main/bin && ln -sf \$(command -v $B0) /venv/main/bin/python); ls -la /venv/main/bin/python"
echo "[$(date -u +%H:%M:%S) UTC] bootstrap done ($(( $(date +%s) - T0 )) s)"
