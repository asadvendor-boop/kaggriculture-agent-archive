#!/bin/bash
# Sep 28 evening, wave 2 (operator 15:05 PKT: "do as many experiments as you want ... get me the winning result"): extra
# rule variants on one base, run NEXT TO srv_tonight_lock.sh (no eval lock; same lockstep settings, rulers, file names and
# log /root/tonight.log). One chain per call: srv_wave2.sh <ckpt> <base tag> "tag|VAR=v VAR=v" ["tag2|..."] (sequential).
# FORCE464=1 runs the 464 even when the first three rulers are worse.
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOCKSTEP=10
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
T0=$(stat -c %Y /root/tn.out)
done_tonight() { [ -f "$1" ] && [ $(stat -c %Y "$1") -gt $T0 ]; }
COMMON="N=300 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1"
declare -A NWD=( [llall]=13 [fresh]=14 [fresh2]=12 )
CK=$1; BT=$2; shift 2
for spec in "$@"; do
  t=${spec%%|*}; envs=${spec#*|}; B=/tmp/tn_blk_${BT}_$t.txt
  echo "[$(date -u +%H:%M) UTC] wave2 $BT + $t ($envs) starts" >> $L
  set -- $envs
  for d in llall fresh fresh2; do
    done_tonight logs/top_${d}_${CK}_t10_${BT}_$t.json && continue
    env SE_GUARD=1 SELL_ORDER=held "$@" DAY=$d NW=${NWD[$d]} $COMMON TAGX=_${BT}_$t $PY live/top_bench.py ckpt/$CK.pt > logs/tn_${d}_${BT}_$t.log 2>&1
  done
  while ! done_tonight logs/top_llall_${CK}_t10_$BT.json; do sleep 30; done
  read GO WB WV < <($PY - <<PY
import json
wb = wv = 0
for d in ("llall", "fresh", "fresh2"):
    B = {(r["ep"], r["seat"]): r["margin"] for r in json.load(open(f"logs/top_{d}_${CK}_t10_$BT.json"))}
    X = {(r["ep"], r["seat"]): r["margin"] for r in json.load(open(f"logs/top_{d}_${CK}_t10_${BT}_$t.json"))}
    for k in B.keys() & X.keys(): wb += B[k] > 0; wv += X[k] > 0
print(int(wv >= wb), wb, wv)
PY
)
  { echo "== $BT + $t ($envs) paired vs $BT [wave2]"
    pair "live losses" logs/top_llall_${CK}_t10_$BT.json logs/top_llall_${CK}_t10_${BT}_$t.json
    for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_${CK}_t10_$BT.json logs/top_${d}_${CK}_t10_${BT}_$t.json; done
    echo "  $BT + $t total wins (live losses + both leader tests): $WB -> $WV"; } > $B; cat $B >> $L
  if [ "$GO" = 1 ] || [ -n "$FORCE464" ]; then
    done_tonight logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json || \
      env SE_GUARD=1 SELL_ORDER=held "$@" NW=16 LOCKSTEP=10 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_${BT}_$t \
        $PY live/bench_multi.py ckpt/$CK.pt:10:0.5 > logs/tn_464_${BT}_$t.log 2>&1
    { echo "== $BT + $t 464 [wave2]"; pair "464" logs/ladder_${CK}_t10_r3_tau0.5_${BT}_test.json logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json; } > $B; cat $B >> $L
  fi
done
echo "[$(date -u +%H:%M) UTC] wave2 chain done on $BT: $*" >> $L
