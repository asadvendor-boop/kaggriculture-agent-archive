#!/bin/bash
# Sep 28 ~02:55 PKT (operator: "can we improve the SE rule even further?"): rule-layer variants of v13_p3
# (NEURAL_20MW8_SE) screened by env settings only (il/live_rollout.py reads SE_HOUR / SE_LAST_DAY / SE_RESERVE / WASH /
# WASH_MODE), paired vs the shipped setting (SE_GUARD=1, hour 12, last day 13, reserve 300, no wash):
#   the live losses of both scripts (frozen at the start) + both leader tests for every variant; the 464 for the best 2.
# Runs after the other tests (same lock). Output: /root/a100_bench.log ("== SEV ...").
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log
source live/srv_cpu.sh; export CPU_BUDGET=40 BENCH_CAP=36
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
VARS=("h0 SE_HOUR=0" "h6 SE_HOUR=6" "res1500 SE_RESERVE=1500" "last15 SE_LAST_DAY=15" "wash5 WASH=5 WASH_MODE=sells" "wash5all WASH=5 WASH_MODE=all")
$PY live/srv_liveloss_worlds.py > /dev/null; cp live/top_worlds_livelosses_300.json live/top_worlds_sevloss_300.json
echo "== SEV screening starts ($(python3 -c "import json; print(len(json.load(open('live/top_worlds_sevloss_300.json'))))") live losses frozen)" >> $L
run_ll() {   # run_ll <tag> [VAR=val ...]
  local tag=$1; shift
  env "$@" SE_GUARD=1 DAY=sevloss N=300 NW=24 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=$tag FRESH=1 \
    $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/sev_ll$tag.log 2>&1
}
run_ll _sev_base
declare -A W
for v in "${VARS[@]}"; do
  set -- $v; n=$1; shift
  run_ll _sev_$n "$@"
  ( export "$@" SE_GUARD=1 TAGX=_sev_$n; bench_gov top:10 ckpt/ilsp_v13_p3.pt > logs/sev_top_$n.log 2>&1 )    # bench_gov is a
  ( export "$@" SE_GUARD=1 TAGX=_sev_$n; bench_gov top2:10 ckpt/ilsp_v13_p3.pt > logs/sev_top2_$n.log 2>&1 )   # function: export
  { echo "== SEV $n ($*)"
    pair "live losses" logs/top_sevloss_ilsp_v13_p3_t10_sev_base.json logs/top_sevloss_ilsp_v13_p3_t10_sev_$n.json
    for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_ilsp_v13_p3_t10_se.json logs/top_${d}_ilsp_v13_p3_t10_sev_$n.json; done; } >> $L
  W[$n]=$($PY -c "
import json
w = 0
for f in ('logs/top_sevloss_ilsp_v13_p3_t10_sev_$n.json', 'logs/top_fresh_ilsp_v13_p3_t10_sev_$n.json', 'logs/top_fresh2_ilsp_v13_p3_t10_sev_$n.json'):
    try: w += sum(r['margin'] > 0 for r in json.load(open(f)))
    except Exception: pass
print(w)")
done
BASEW=$($PY -c "
import json
w = 0
for f in ('logs/top_sevloss_ilsp_v13_p3_t10_sev_base.json', 'logs/top_fresh_ilsp_v13_p3_t10_se.json', 'logs/top_fresh2_ilsp_v13_p3_t10_se.json'):
    w += sum(r['margin'] > 0 for r in json.load(open(f)))
print(w)")
echo "  SEV total wins (live losses + both leader tests): shipped $BASEW | $(for k in "${!W[@]}"; do printf '%s %s  ' $k ${W[$k]}; done)" >> $L
for v in $(for k in "${!W[@]}"; do echo "${W[$k]} $k"; done | sort -rn | head -2 | awk -v b=$BASEW '$1 > b {print $2}'); do
  spec=$(printf '%s\n' "${VARS[@]}" | awk -v n=$v '$1 == n {$1 = ""; print}')
  ( export $spec SE_GUARD=1 TAGX=_r3_tau0.5_sev_$v; bench_gov ladder ckpt/ilsp_v13_p3.pt:10:0.5 > logs/sev_464_$v.log 2>&1 )
  { echo "== SEV $v 464"; pair "464" logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_sev_${v}_test.json; } >> $L
done
echo "[$(date -u +%H:%M) UTC] SEV screening done" >> $L
