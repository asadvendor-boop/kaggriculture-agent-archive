#!/bin/bash
# Sep 28 ~16:45 PKT: today's top-7 test (fresh3, 300 worlds) for the remaining candidates on the GPU (one process, LOCKSTEP=100),
# sequential; skips results already produced tonight. Usage: srv_gpu_fresh3.sh <ckpt> <base tag> "t|VARS" ["t2|VARS" ...]
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOCKSTEP=100 ROLLOUT_DEVICE=cuda
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
T0=$(stat -c %Y /root/tn.out)
done_tonight() { [ -f "$1" ] && [ $(stat -c %Y "$1") -gt $T0 ]; }
CK=$1; BT=$2; shift 2
for spec in "$@"; do
  t=${spec%%|*}; envs=${spec#*|}
  done_tonight logs/top_fresh3_${CK}_t10_${BT}_$t.json || \
    env SE_GUARD=1 SELL_ORDER=held $envs DAY=fresh3 NW=1 N=300 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_${BT}_$t \
      nice -n -5 $PY live/top_bench.py ckpt/$CK.pt > logs/tn_gfresh3_${BT}_$t.log 2>&1
  pair "fresh3 (today's top-7) $BT+$t" logs/top_fresh3_${CK}_t10_$BT.json logs/top_fresh3_${CK}_t10_${BT}_$t.json >> $L
done
