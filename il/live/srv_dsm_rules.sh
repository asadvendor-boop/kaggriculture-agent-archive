#!/bin/bash
# Sep 28 ~03:10 PKT: Route A of il/live/PLANNING_DESIGN.md - DSM's (and M&M&P&Q's) Sep 26 build order as switchable rule-layer
# rules on v13_p3 (NEURAL_20MW8_SE): MELON_WAVE=4 (4 of the policy's own wheat/carrot plantings on days 6-7 become melons,
# seeds bought from hour 22 of day 5), LAND2_DAY=8 (SW on day 8 instead of ~9.1), NO_LAND3=1 with SE_GUARD=0 (no third
# land), MELON_WAVE=6. Each alone (SE on) and bundled (with and without SE),
# paired vs the shipped rules on the frozen 54 live losses + both leader tests; the 464 for the best two that beat the
# shipped total. Runs NEXT TO the main queue (own lock, NW=12: the container quota is 46 cores). Output /root/a100_bench.log
exec 6>/tmp/srv_dsm.lock; flock -n 6 || { echo "already running"; exit 0; }
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log; NWD=${NWD:-12}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
VARS=("m4 SE_GUARD=1 MELON_WAVE=4" "l8 SE_GUARD=1 LAND2_DAY=8" "m4l8 SE_GUARD=1 LAND2_DAY=8 MELON_WAVE=4"
      "plan SE_GUARD=0 NO_LAND3=1 LAND2_DAY=8 MELON_WAVE=4" "m6 SE_GUARD=1 MELON_WAVE=6")
# (EARLY_COWS dropped after the smoke test, live/rule_smoke.py: the cows bought on days 4-5 sat in the shed until day 6 -
#  the policy builds its pastures on the new NE land on day 6, so early cows only cost cash; a unit-level rule is needed)
[ -n "$ONLY" ] && VARS=("${VARS[@]:0:$ONLY}")
echo "== DSM plan rules screening starts ($(date -u +%H:%M) UTC; NW=$NWD next to the main queue)" >> $L
declare -A W
for v in "${VARS[@]}"; do
  set -- $v; n=$1; shift
  env "$@" DAY=sevloss N=300 NW=$NWD TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_dsm_$n FRESH=1 \
    $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/dsm_ll_$n.log 2>&1
  for d in fresh fresh2; do
    env "$@" DAY=$d N=300 NW=$NWD TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_dsm_$n \
      $PY live/top_bench.py ckpt/ilsp_v13_p3.pt 2>&1 | grep -v Warning > logs/dsm_${d}_$n.log
  done
  { echo "== DSM $n ($*)"
    pair "live losses" logs/top_sevloss_ilsp_v13_p3_t10_sev_base.json logs/top_sevloss_ilsp_v13_p3_t10_dsm_$n.json
    for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_ilsp_v13_p3_t10_se.json logs/top_${d}_ilsp_v13_p3_t10_dsm_$n.json; done; } >> $L
  W[$n]=$($PY -c "
import json
w = 0
for f in ('logs/top_sevloss_ilsp_v13_p3_t10_dsm_$n.json', 'logs/top_fresh_ilsp_v13_p3_t10_dsm_$n.json', 'logs/top_fresh2_ilsp_v13_p3_t10_dsm_$n.json'):
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
echo "  DSM total wins (live losses + both leader tests): shipped $BASEW | $(for k in "${!W[@]}"; do printf '%s %s  ' $k ${W[$k]}; done)" >> $L
for v in $(for k in "${!W[@]}"; do echo "${W[$k]} $k"; done | sort -rn | head -2 | awk -v b=$BASEW '$1 > b {print $2}'); do
  spec=$(printf '%s\n' "${VARS[@]}" | awk -v n=$v '$1 == n {$1 = ""; print}')
  env $spec NW=$NWD LOCKSTEP=6 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_dsm_$v \
    $PY live/bench_multi.py ckpt/ilsp_v13_p3.pt:10:0.5 2>&1 | grep -v Warning > logs/dsm_464_$v.log
  { echo "== DSM $v 464"; pair "464" logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_dsm_${v}_test.json; } >> $L
done
echo "[$(date -u +%H:%M) UTC] DSM plan rules screening done" >> $L
