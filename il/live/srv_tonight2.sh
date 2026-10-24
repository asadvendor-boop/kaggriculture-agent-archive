#!/bin/bash
# Sep 28 evening follow-up to srv_tonight.sh (same rulers, same log /root/tonight.log): extra rule variants on one base.
# Usage: srv_tonight2.sh <ckpt> <base tag> "tag|VAR=v VAR=v" ["tag2|..."]   (base env SE_GUARD=1 SELL_ORDER=held, like tonight)
# e.g. m10 = MELON_WAVE=8 MELON_DAY=10 (a second melon wave planted days 10-11: Pico / marwar22 / 吃白饭 losses, W11), or a
# tailored combo of the rules that won alone. Waits for the eval lock (runs after any srv_tonight.sh still holding it).
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
source live/srv_cpu.sh; export CPU_BUDGET=${RT_CPU:-240} BENCH_CAP=${RT_CAP:-200}; NWL=${NWL:-150}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say() { echo "[$(date -u +%H:%M) UTC] $*" >> $L; }
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
CK=$1; BT=$2; shift 2
say "follow-up starts on $BT ($CK): $*"
for spec in "$@"; do
  t=${spec%%|*}; envs=${spec#*|}
  set -- $envs
  env SE_GUARD=1 SELL_ORDER=held "$@" DAY=llall N=300 NW=$NWL TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_${BT}_$t FRESH=1 \
    $PY live/top_bench.py ckpt/$CK.pt > logs/tn_ll_${BT}_$t.log 2>&1
  ( export SE_GUARD=1 SELL_ORDER=held "$@" TAGX=_${BT}_$t; bench_gov top:10 ckpt/$CK.pt > logs/tn_top_${BT}_$t.log 2>&1 )
  ( export SE_GUARD=1 SELL_ORDER=held "$@" TAGX=_${BT}_$t; bench_gov top2:10 ckpt/$CK.pt > logs/tn_top2_${BT}_$t.log 2>&1 )
  { echo "== $BT + $t ($envs) paired vs $BT"
    pair "live losses" logs/top_llall_${CK}_t10_$BT.json logs/top_llall_${CK}_t10_${BT}_$t.json
    for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_${CK}_t10_$BT.json logs/top_${d}_${CK}_t10_${BT}_$t.json; done; } >> $L
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
  echo "  $BT + $t total wins (live losses + both leader tests): $WB -> $WV" >> $L
  if [ "$GO" = 1 ] || [ -n "$FORCE464" ]; then
    ( export SE_GUARD=1 SELL_ORDER=held "$@" TAGX=_r3_tau0.5_${BT}_$t; bench_gov ladder ckpt/$CK.pt:10:0.5 > logs/tn_464_${BT}_$t.log 2>&1 )
    { echo "== $BT + $t 464"; pair "464" logs/ladder_${CK}_t10_r3_tau0.5_${BT}_test.json logs/ladder_${CK}_t10_r3_tau0.5_${BT}_${t}_test.json; } >> $L
  fi
done
say "follow-up done on $BT"
