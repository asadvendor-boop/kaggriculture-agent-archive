#!/bin/bash
# Update-speed measurements (coordinator, Sep 26 night).  Launch ONCE, detached:
#   setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/train/speed_chain.sh > /Users/asad.ali/kaggriculture/kagjax/logs_server/D/speed/chain.out 2>&1 < /dev/null &
# Every step is its own short GPU job through the shared lock (kagjax/srv_jax.sh --gpu: cores 56-63, nice 10, 8 GB,
# one JAX GPU job at a time), so other agents' jobs interleave.  No step starts after 02:10 UTC; the saved rollouts
# (~5 GB) are deleted at the end.
exec 9>/tmp/kagjax_speed_chain.lock
flock -n 9 || { echo "already running"; exit 0; }
R=/Users/asad.ali/kaggriculture
S=$R/kagjax/logs_server/D/speed
T=$R/kagjax/train
mkdir -p $S/exp
cd $R
say() { echo "[$(date -u '+%H:%M:%S') UTC] $*"; }
STOP=$(date -u -d "$(date -u -d '+1 day' +%Y-%m-%d) 02:10" +%s)
[ "$(date -u +%H)" -lt 12 ] && STOP=$(date -u -d "$(date -u +%Y-%m-%d) 02:10" +%s)   # started after midnight
gpu() {   # gpu <name> [VAR=value ...] -- <script>
  local name=$1; shift
  if [ "$(date -u +%s)" -ge "$STOP" ]; then say "skip $name (past 02:10 UTC)"; return; fi
  say "queue $name"
  bash $R/kagjax/srv_jax.sh --gpu env PYTHONPATH=$R "$@" > $S/$name.out 2>&1
  say "done  $name (exit $?)"
}
INIT=$R/il/ckpt/g20_w_dsm_s0.pt; IL=$R/il/ckpt/g20_w_s0.pt
# 1. rollouts shared by the value / KL experiments
gpu exp_collect EXP=collect EXP_DIR=$S/exp /opt/jaxenv/bin/python $T/exp_value_kl.py
# 2. update throughput (+ LR-0 ratio noise of the first minibatch), part A (separate value net) and B (value head, bf16)
gpu bench_update_A BENCH=update BENCH_OUT=$S BENCH_TAG=A BENCH_VARIANTS=gate_E2_MB256,E1_MB256,E1_MB4096,E1_MB4096_ref25 /opt/jaxenv/bin/python $T/bench_update.py
gpu bench_update_B BENCH=update BENCH_OUT=$S BENCH_TAG=B BENCH_VARIANTS=E1_MB4096_ref25_head,E1_MB4096_head,E2_MB4096_ref25_head,E1_MB4096_ref25_small,E1_MB4096_ref25_head_bf16,E1_MB4096_ref25_head_bf16_recompute,E1_MB4096_ref25_head_m2048 /opt/jaxenv/bin/python $T/bench_update.py
# 3. KL-anchor subsample: bias / noise / gradient
gpu exp_kl EXP=kl EXP_DIR=$S/exp /opt/jaxenv/bin/python $T/exp_value_kl.py
# 4. value options on the same rollouts (held-out explained variance, advantage noise)
gpu exp_value_A EXP=value EXP_DIR=$S/exp EXP_OPTIONS=head_final,head_togo,head_final_vlr1e3,small_final,small_togo /opt/jaxenv/bin/python $T/exp_value_kl.py
gpu exp_value_B EXP=value EXP_DIR=$S/exp EXP_OPTIONS=full_togo_gateopt,full_togo,full_final /opt/jaxenv/bin/python $T/exp_value_kl.py
# 5. rollout: opponent forwards (map over 4 blocks vs 2 / 1 param groups with per-row style ids), B = 1024
gpu bench_rollout BENCH=rollout BENCH_OUT=$S BENCH_TAG=r1 BENCH_B=1024 /opt/jaxenv/bin/python $T/bench_update.py
# 6. 3-iteration sanity runs from g20_w_dsm_s0 (128 games; greedy fp32 eval on 128 fixed worlds vs the frozen start and
#    the DSM clone; the first run's iteration-0 evaluation is reused by the others)
SAN="KAGJAX_BACKEND=real INIT=$INIT IL=$IL TEAM=10 B=128 K=4 SEG=720 T=0.2 KSHARP=10 TAU=0.5 BETA=0.05 VWARM=1
 OPPS=LATEST=2,SELF=2,IL=4 IL_TEAMS=10:3,5:2,3:2,4:2,0:2 SNAP_EVERY=1 SELF_MAX=4 EVAL_EVERY=3 EVAL_N=128
 EVAL_OPPS=SNAP0,IL@10 CKPT_EVERY=0 PT_EVERY=0 ITERS=3 KL_MAX=1.0 DIAG_EVERY=0 SEED=0"
gpu san_S0_gate $SAN EPOCHS=2 MB=256 LR=5e-6 VLR=1e-4 RUN_DIR=$S/san_S0_gate /opt/jaxenv/bin/python $T/ppo.py
BASE_EVAL=$S/san_S0_gate/eval_0.npz
gpu san_S1_fast_fullv $SAN EVAL_BASE=$BASE_EVAL EPOCHS=1 MB=4096 MICRO=512 LR=2e-5 VLR=4e-4 REF_FRAC=0.25 RUN_DIR=$S/san_S1_fast_fullv /opt/jaxenv/bin/python $T/ppo.py
gpu san_S2_fast_head $SAN EVAL_BASE=$BASE_EVAL EPOCHS=1 MB=4096 MICRO=1024 LR=2e-5 VLR=4e-4 REF_FRAC=0.25 VALUE_INIT=head VALUE_TARGET=final RUN_DIR=$S/san_S2_fast_head /opt/jaxenv/bin/python $T/ppo.py
gpu san_S3_fast_head_groups $SAN EVAL_BASE=$BASE_EVAL EPOCHS=1 MB=4096 MICRO=1024 LR=2e-5 VLR=4e-4 REF_FRAC=0.25 VALUE_INIT=head VALUE_TARGET=final OPP_GROUPS=2 RUN_DIR=$S/san_S3_fast_head_groups /opt/jaxenv/bin/python $T/ppo.py
# 7. repeat of the key throughput comparison (the GPU is shared: noise check)
gpu bench_update_R BENCH=update BENCH_OUT=$S BENCH_TAG=R BENCH_VARIANTS=gate_E2_MB256,E1_MB4096_ref25_head,E1_MB4096_ref25_head_bf16 /opt/jaxenv/bin/python $T/bench_update.py
gpu bench_rollout_R BENCH=rollout BENCH_OUT=$S BENCH_TAG=r2 BENCH_B=1024 /opt/jaxenv/bin/python $T/bench_update.py
rm -f $S/exp/roll_*.npz
find $S -name "*.npz" -size +1M -delete; find $S -name "*.pt" -delete
say "CHAIN DONE"
