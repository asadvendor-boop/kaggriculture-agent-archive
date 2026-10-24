#!/bin/bash
# 10M measurements (rf_w_dsm_s0, 256x8) for the dedicated-GPU projection; same rules as speed_chain.sh.  Launch ONCE:
#   setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/train/speed_chain2.sh > /Users/asad.ali/kaggriculture/kagjax/logs_server/D/speed/chain2.out 2>&1 < /dev/null &
exec 9>/tmp/kagjax_speed_chain2.lock
flock -n 9 || { echo "already running"; exit 0; }
R=/Users/asad.ali/kaggriculture
S=$R/kagjax/logs_server/D/speed
T=$R/kagjax/train
cd $R
say() { echo "[$(date -u '+%H:%M:%S') UTC] $*"; }
STOP=$(date -u -d "$(date -u -d '+1 day' +%Y-%m-%d) 02:10" +%s)
[ "$(date -u +%H)" -lt 12 ] && STOP=$(date -u -d "$(date -u +%Y-%m-%d) 02:10" +%s)
gpu() {
  local name=$1; shift
  if [ "$(date -u +%s)" -ge "$STOP" ]; then say "skip $name (past 02:10 UTC)"; return; fi
  say "queue $name"
  bash $R/kagjax/srv_jax.sh --gpu env PYTHONPATH=$R "$@" > $S/$name.out 2>&1
  say "done  $name (exit $?)"
}
M10="INIT=$R/il/ckpt/rf_w_dsm_s0.pt IL=$R/il/ckpt/rf_w_s0.pt"
gpu bench_update_10M $M10 BENCH=update BENCH_OUT=$S BENCH_TAG=10M BENCH_VARIANTS=gate_E2_MB256,E1_MB4096_ref25,E1_MB4096_ref25_head,E1_MB4096_ref25_head_bf16 /opt/jaxenv/bin/python $T/bench_update.py
gpu bench_rollout_10M $M10 BENCH=rollout BENCH_OUT=$S BENCH_TAG=10M BENCH_B=1024 /opt/jaxenv/bin/python $T/bench_update.py
say "CHAIN2 DONE"
