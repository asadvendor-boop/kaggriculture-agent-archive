#!/bin/bash
# Remaining update-speed steps after bench_update_B ran out of memory (Sep 26 22:48 UTC: MICRO=1024 on the 20M asks
# for a 4.2 GiB temp buffer; with the 8 GB cap and the GPU nearly full, MICRO=512 for the value-head runs).  Replaces
# speed_chain.sh from exp_value_A on (exp_kl was already queued by it).  Same rules.  Launch ONCE:
#   setsid nohup bash /Users/asad.ali/kaggriculture/kagjax/train/speed_chain1b.sh > /Users/asad.ali/kaggriculture/kagjax/logs_server/D/speed/chain1b.out 2>&1 < /dev/null &
exec 9>/tmp/kagjax_speed_chain1b.lock
flock -n 9 || { echo "already running"; exit 0; }
R=/Users/asad.ali/kaggriculture
S=$R/kagjax/logs_server/D/speed
T=$R/kagjax/train
mkdir -p $S/exp
cd $R
say() { echo "[$(date -u '+%H:%M:%S') UTC] $*"; }
STOP=$(date -u -d "$(date -u -d '+1 day' +%Y-%m-%d) 02:10" +%s)
[ "$(date -u +%H)" -lt 12 ] && STOP=$(date -u -d "$(date -u +%Y-%m-%d) 02:10" +%s)
gpu() {   # gpu <name> [VAR=value ...] -- <script>
  local name=$1; shift
  if [ "$(date -u +%s)" -ge "$STOP" ]; then say "skip $name (past 02:10 UTC)"; return; fi
  say "queue $name"
  bash $R/kagjax/srv_jax.sh --gpu env PYTHONPATH=$R "$@" > $S/$name.out 2>&1
  say "done  $name (exit $?)"
}
INIT=$R/il/ckpt/g20_w_dsm_s0.pt; IL=$R/il/ckpt/g20_w_s0.pt
# 1. update throughput with the policy's value head (+ LR-0 ratio noise), bf16 variants; MICRO 1024 last (may not fit)
gpu bench_update_B2 BENCH=update BENCH_OUT=$S BENCH_TAG=B2 BENCH_VARIANTS=E1_MB4096_ref25_head,E1_MB4096_ref25_head_bf16grad,E1_MB4096_ref25_head_bf16,E1_MB4096_ref25_head_bf16_recompute,E1_MB4096_head,E2_MB4096_ref25_head,E1_MB4096_ref25_small,E1_MB4096_ref25_head_m1024 /opt/jaxenv/bin/python $T/bench_update.py
# 2. value options on the same rollouts: value head / small value net
gpu exp_value_A EXP=value EXP_DIR=$S/exp EXP_OPTIONS=head_final,head_togo,head_final_vlr1e3,small_final,small_togo /opt/jaxenv/bin/python $T/exp_value_kl.py
# 3. rollout: opponent forwards (4 blocks / 2 / 1 param groups), bf16 opponents, B = 1024 and 2048
gpu bench_rollout BENCH=rollout BENCH_OUT=$S BENCH_TAG=r1 BENCH_B=1024 /opt/jaxenv/bin/python $T/bench_update.py
# 4. 3-iteration sanity runs from g20_w_dsm_s0
SAN="KAGJAX_BACKEND=real INIT=$INIT IL=$IL TEAM=10 B=128 K=4 SEG=720 T=0.2 KSHARP=10 TAU=0.5 BETA=0.05 VWARM=1
 OPPS=LATEST=2,SELF=2,IL=4 IL_TEAMS=10:3,5:2,3:2,4:2,0:2 SNAP_EVERY=1 SELF_MAX=4 EVAL_EVERY=3 EVAL_N=128
 EVAL_OPPS=SNAP0,IL@10 CKPT_EVERY=0 PT_EVERY=0 ITERS=3 KL_MAX=1.0 DIAG_EVERY=0 SEED=0"
gpu san_S0_gate $SAN EPOCHS=2 MB=256 LR=5e-6 VLR=1e-4 RUN_DIR=$S/san_S0_gate /opt/jaxenv/bin/python $T/ppo.py
BASE_EVAL=$S/san_S0_gate/eval_0.npz
gpu san_S2_fast_head $SAN EVAL_BASE=$BASE_EVAL EPOCHS=1 MB=4096 MICRO=512 LR=2e-5 VLR=4e-4 REF_FRAC=0.25 VALUE_INIT=head VALUE_TARGET=final RUN_DIR=$S/san_S2_fast_head /opt/jaxenv/bin/python $T/ppo.py
gpu san_S1_fast_fullv $SAN EVAL_BASE=$BASE_EVAL EPOCHS=1 MB=4096 MICRO=512 LR=2e-5 VLR=4e-4 REF_FRAC=0.25 RUN_DIR=$S/san_S1_fast_fullv /opt/jaxenv/bin/python $T/ppo.py
gpu san_S3_fast_head_groups $SAN EVAL_BASE=$BASE_EVAL EPOCHS=1 MB=4096 MICRO=512 LR=2e-5 VLR=4e-4 REF_FRAC=0.25 VALUE_INIT=head VALUE_TARGET=final OPP_GROUPS=2 RUN_DIR=$S/san_S3_fast_head_groups /opt/jaxenv/bin/python $T/ppo.py
# 5. value options with the full-size value network (reference for the EV comparison)
gpu exp_value_B EXP=value EXP_DIR=$S/exp EXP_OPTIONS=full_togo_gateopt,full_togo,full_final /opt/jaxenv/bin/python $T/exp_value_kl.py
# 6. repeat of the key throughput comparison back-to-back (the GPU is shared: noise check)
gpu bench_update_R BENCH=update BENCH_OUT=$S BENCH_TAG=R BENCH_VARIANTS=gate_E2_MB256,E1_MB4096_ref25,E1_MB4096_ref25_head,E1_MB4096_ref25_head_bf16grad /opt/jaxenv/bin/python $T/bench_update.py
gpu bench_rollout_R BENCH=rollout BENCH_OUT=$S BENCH_TAG=r2 BENCH_B=1024 /opt/jaxenv/bin/python $T/bench_update.py
# the saved rollouts are deleted once exp_kl (queued by speed_chain.sh) and the value runs are done
until ! ps -eo args | grep -q "^/opt/jaxenv/bin/python $T/exp_value_kl.py" && grep -q '"exp": "done"' $S/exp_kl.out 2>/dev/null \
      || [ "$(date -u +%s)" -ge "$STOP" ]; do sleep 60; done
rm -f $S/exp/roll_*.npz
find $S -name "*.npz" -size +1M -delete; find $S -name "*.pt" -delete
say "CHAIN1B DONE"
