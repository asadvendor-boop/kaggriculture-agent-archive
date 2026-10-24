#!/bin/bash
# (3b: the 20M part moved into speed_chain1b.sh; MICRO=512 - MICRO=1024 ran out of memory under the 8 GB cap.)
# UPDATE_PRECISION=bf16grad (TF32 forward values - ratios / clip / KL as with TF32 - and bf16 gradients), added Sep 26
# ~22:40 UTC after the bf16 update failed the LR-0 criterion.  Same rules as speed_chain.sh.  Launch ONCE:
#   setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/train/speed_chain3b.sh > /Users/asad.ali/kaggriculture/kagjax/logs_server/D/speed/chain3b.out 2>&1 < /dev/null &
exec 9>/tmp/kagjax_speed_chain3b.lock
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
V=E1_MB4096_ref25_head,E1_MB4096_ref25_head_bf16grad
gpu bench_update_C10M INIT=$R/il/ckpt/rf_w_dsm_s0.pt IL=$R/il/ckpt/rf_w_s0.pt BENCH=update BENCH_OUT=$S BENCH_TAG=C10M BENCH_VARIANTS=$V /opt/jaxenv/bin/python $T/bench_update.py
# 3-iteration sanity run with bf16grad (= san_S2_fast_head + UPDATE_PRECISION=bf16grad), after san_S0_gate's iteration-0 eval
until [ -f $S/san_S0_gate/eval_0.npz ] || [ "$(date -u +%s)" -ge "$STOP" ]; do sleep 60; done
SAN="KAGJAX_BACKEND=real INIT=$R/il/ckpt/g20_w_dsm_s0.pt IL=$R/il/ckpt/g20_w_s0.pt TEAM=10 B=128 K=4 SEG=720 T=0.2 KSHARP=10 TAU=0.5 BETA=0.05 VWARM=1
 OPPS=LATEST=2,SELF=2,IL=4 IL_TEAMS=10:3,5:2,3:2,4:2,0:2 SNAP_EVERY=1 SELF_MAX=4 EVAL_EVERY=3 EVAL_N=128
 EVAL_OPPS=SNAP0,IL@10 CKPT_EVERY=0 PT_EVERY=0 ITERS=3 KL_MAX=1.0 DIAG_EVERY=0 SEED=0"
gpu san_S4_fast_head_bf16grad $SAN EVAL_BASE=$S/san_S0_gate/eval_0.npz EPOCHS=1 MB=4096 MICRO=512 LR=2e-5 VLR=4e-4 REF_FRAC=0.25 VALUE_INIT=head VALUE_TARGET=final UPDATE_PRECISION=bf16grad RUN_DIR=$S/san_S4_fast_head_bf16grad /opt/jaxenv/bin/python $T/ppo.py
find $S/san_S4_fast_head_bf16grad -name "*.npz" -size +1M -delete 2>/dev/null; find $S/san_S4_fast_head_bf16grad -name "*.pt" -delete 2>/dev/null
say "CHAIN3B DONE"
