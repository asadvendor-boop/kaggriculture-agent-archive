#!/bin/bash
# Sep 28 ~03:15 PKT: one rule-layer setting on v13_p3 (NEURAL_20MW8_SE), paired vs the shipped rules (SE_GUARD=1), in the
# main queue (lock /tmp/srv_a100_eval.lock): the frozen 54 live losses + both leader tests first; the 464 only when the
# setting is not worse there (wins >= shipped, or the summed margin difference > 0) - a sequential test, so a setting
# that already loses on the leader tests does not spend ~1 h of the box on the 464.
# Usage: [WAITFOR=<log text>] srv_ruletest.sh <tag> <VAR=val> ...   (e.g. srv_ruletest.sh sob SELL_ORDER=base)
# Output /root/a100_bench.log ("== RULE <tag>", "RULE <tag> total wins", "== RULE <tag> 464", "RULE <tag> done")
[ -n "$WAITFOR" ] && until grep -q "$WAITFOR" /root/a100_bench.log; do sleep 30; done
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log
source live/srv_cpu.sh; export CPU_BUDGET=40 BENCH_CAP=36
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
t=$1; shift; SETS=("$@")
# SE_GUARD=1 unless a setting overrides it (e.g. SE_GUARD=0). CK=<ckpt name> (default ilsp_v13_p3); BASET=<tag> [BASECK=<ckpt name>]: compare against that earlier rule run (of BASECK,
# default CK) instead of v13_p3 with the shipped rules (e.g. BASET=soh for settings stacked on held)
CK=${CK:-ilsp_v13_p3}; BC=${BASECK:-$CK}
if [ -n "$BASET" ]; then BLL=logs/top_sevloss_${BC}_t10_$BASET.json; BFR=logs/top_fresh_${BC}_t10_$BASET.json; BFR2=logs/top_fresh2_${BC}_t10_$BASET.json
  B464=logs/ladder_${BC}_t10_r3_tau0.5_${BASET}_test.json
else BLL=logs/top_sevloss_ilsp_v13_p3_t10_sev_base.json; BFR=logs/top_fresh_ilsp_v13_p3_t10_se.json; BFR2=logs/top_fresh2_ilsp_v13_p3_t10_se.json
  B464=logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json; fi
env SE_GUARD=1 "${SETS[@]}" DAY=sevloss N=300 NW=24 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_$t FRESH=1 \
  $PY live/top_bench.py ckpt/$CK.pt > logs/rt_ll_$t.log 2>&1
( export SE_GUARD=1 "${SETS[@]}" TAGX=_$t; bench_gov top:10 ckpt/$CK.pt > logs/rt_top_$t.log 2>&1 )
( export SE_GUARD=1 "${SETS[@]}" TAGX=_$t; bench_gov top2:10 ckpt/$CK.pt > logs/rt_top2_$t.log 2>&1 )
{ echo "== RULE $t (${SETS[*]}) on $CK, paired vs ${BASET:+$BC }${BASET:-v13_p3 with the shipped rules}"
  pair "live losses" $BLL logs/top_sevloss_${CK}_t10_$t.json
  pair "leaders fresh " $BFR logs/top_fresh_${CK}_t10_$t.json
  pair "leaders fresh2 " $BFR2 logs/top_fresh2_${CK}_t10_$t.json; } >> $L
read GO WB WV DM < <($PY - <<PY
import json
b = ["$BLL", "$BFR", "$BFR2"]
v = ["logs/top_sevloss_${CK}_t10_$t.json", "logs/top_fresh_${CK}_t10_$t.json", "logs/top_fresh2_${CK}_t10_$t.json"]
wb = wv = 0; dm = 0.0
for fb, fv in zip(b, v):
    try:
        B = {(r["ep"], r["seat"]): r["margin"] for r in json.load(open(fb))}
        V = {(r["ep"], r["seat"]): r["margin"] for r in json.load(open(fv))}
    except Exception:
        continue
    for k in B.keys() & V.keys():
        wb += B[k] > 0; wv += V[k] > 0; dm += V[k] - B[k]
print(int(wv >= wb or dm > 0), wb, wv, f"{dm:+.0f}")
PY
)
echo "  RULE $t total wins (live losses + both leader tests): ${BASET:-shipped} $WB -> $WV, summed margin diff $DM -> $([ "$GO" = 1 ] && echo "464 next" || echo "no 464 (worse on the first rulers)")" >> $L
if [ "$GO" = 1 ]; then
  ( export SE_GUARD=1 "${SETS[@]}" TAGX=_r3_tau0.5_$t; bench_gov ladder ckpt/$CK.pt:10:0.5 > logs/rt_464_$t.log 2>&1 )
  { echo "== RULE $t 464 (vs ${BASET:+$BC }${BASET:-shipped})"; pair "464" $B464 logs/ladder_${CK}_t10_r3_tau0.5_${t}_test.json; } >> $L
fi
echo "[$(date -u +%H:%M) UTC] RULE $t done" >> $L
