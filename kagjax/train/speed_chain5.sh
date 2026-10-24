#!/bin/bash
# 10M rollout modes again, timed round-robin (Sep 26 ~23:30 UTC: the first 10M rollout benchmark measured the modes
# one after another while the shared GPU load changed; bf16 opponents looked 3.4x faster).  Same rules.  Launch ONCE:
#   setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/train/speed_chain5.sh > /Users/asad.ali/kaggriculture/kagjax/logs_server/D/speed/chain5.out 2>&1 < /dev/null &
exec 9>/tmp/kagjax_speed_chain5.lock
flock -n 9 || { echo "already running"; exit 0; }
R=/Users/asad.ali/kaggriculture
S=$R/kagjax/logs_server/D/speed
T=$R/kagjax/train
cd $R
say() { echo "[$(date -u '+%H:%M:%S') UTC] $*"; }
STOP=$(date -u -d "$(date -u -d '+1 day' +%Y-%m-%d) 02:10" +%s)
[ "$(date -u +%H)" -lt 12 ] && STOP=$(date -u -d "$(date -u +%Y-%m-%d) 02:10" +%s)
if [ "$(date -u +%s)" -ge "$STOP" ]; then say "skip bench_rollout_10M_R (past 02:10 UTC)"; exit 0; fi
say "queue bench_rollout_10M_R"
bash $R/kagjax/srv_jax.sh --gpu env PYTHONPATH=$R INIT=$R/il/ckpt/rf_w_dsm_s0.pt IL=$R/il/ckpt/rf_w_s0.pt BENCH=rollout BENCH_OUT=$S BENCH_TAG=10M_R BENCH_B=1024 /opt/jaxenv/bin/python $T/bench_update.py > $S/bench_rollout_10M_R.out 2>&1
say "done  bench_rollout_10M_R (exit $?)"
say "CHAIN5 DONE"
