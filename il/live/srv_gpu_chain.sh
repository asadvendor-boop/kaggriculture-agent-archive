#!/bin/bash
# Sep 28 ~16:30 PKT: one variant on ALL rulers with the policy's forward on the RTX 3070 (ROLLOUT_DEVICE=cuda; checked: 50/50
# games identical to the CPU runs, 6.1 s/game in one process vs ~150 s/game per CPU core under load). One process per ruler
# (LOCKSTEP=100 games in lockstep), rulers in sequence: live losses, fresh, fresh2, fresh3 (today's top-7 games), the 464.
# Usage: srv_gpu_chain.sh <ckpt> <base tag> "tag|VAR=v VAR=v" ["tag2|..."]   (base env SE_GUARD=1 SELL_ORDER=held)
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOCKSTEP=100 ROLLOUT_DEVICE=cuda
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
T0=$(stat -c %Y /root/tn.out)
done_tonight() { [ -f "$1" ] && [ $(stat -c %Y "$1") -gt $T0 ]; }
COMMON="N=300 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1"
CK=$1; BT=$2; shift 2
for spec in "$@"; do
  t=${spec%%|*}; envs=${spec#*|}; B=/tmp/tn_gblk_${BT}_$t.txt
  echo "[$(date -u +%H:%M) UTC] GPU $BT + $t ($envs) starts" >> $L
  set -- $envs
  for d in llall fresh fresh2 fresh3; do
    done_tonight logs/top_${d}_${CK}_t10_${BT}_$t.json && continue
    env SE_GUARD=1 SELL_ORDER=held "$@" DAY=$d NW=1 $COMMON TAGX=_${BT}_$t nice -n -5 $PY live/top_bench.py ckpt/$CK.pt > logs/tn_g${d}_${BT}_$t.log 2>&1
  done
  done_tonight logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json || \
    env SE_GUARD=1 SELL_ORDER=held "$@" NW=2 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_${BT}_$t nice -n -5 $PY live/bench_multi.py ckpt/$CK.pt:10:0.5 > logs/tn_g464_${BT}_$t.log 2>&1
  { echo "== $BT + $t ($envs) paired vs $BT [GPU, all rulers]"
    pair "live losses" logs/top_llall_${CK}_t10_$BT.json logs/top_llall_${CK}_t10_${BT}_$t.json
    for d in fresh fresh2 fresh3; do pair "leaders $d " logs/top_${d}_${CK}_t10_$BT.json logs/top_${d}_${CK}_t10_${BT}_$t.json; done
    pair "464" logs/ladder_${CK}_t10_r3_tau0.5_${BT}_test.json logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json; } > $B; cat $B >> $L
done
echo "[$(date -u +%H:%M) UTC] GPU chain done on $BT: $*" >> $L
