#!/bin/bash
# Sep 28 evening: the new leader-style test of TODAY's top-band games (top_worlds_fresh3_300.json, srv_fetch_today.py) for the
# bases and every variant tested tonight (same env as srv_tonight_lock.sh / srv_wave2.sh), 30 workers each (10 games per
# process), started only while the box has room (<= ~200 workers in total). Summary lines -> /root/tonight.log.
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/tonight.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOCKSTEP=10
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
T0=$(stat -c %Y /root/tn.out)
done_tonight() { [ -f "$1" ] && [ $(stat -c %Y "$1") -gt $T0 ]; }
room() { while [ $(ps -eo args | grep -c "[m]ultiprocessing.spawn") -gt ${MAXW:-200} ]; do sleep 20; done; }
run() {   # run <ckpt> <tagx> <env...>
  local CK=$1 TX=$2; shift 2
  done_tonight logs/top_fresh3_${CK}_t10$TX.json && return 0
  env "$@" DAY=fresh3 N=300 NW=30 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=$TX $PY live/top_bench.py ckpt/$CK.pt > logs/tn_fresh3$TX.log 2>&1
}
H="SE_GUARD=1 SELL_ORDER=held"
declare -A V=( [sy7]="SHEEP_YARN=7" [tom]="TOMATO_SHOPS=2" [cow]="COW_MILK=3" [fg]="FEED_GUARD=1" [se2]="SE_SHOPS=2"
               [str]="STRAW_UP=4" [tomD]="TOMATO_SHOPS=2 TOMATO_FROM_DAY=15 TOMATO_PER_SHOP=4" [can]="SELL_ORDER=canonical"
               [combo]="SHEEP_YARN=7 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4" [sysy]="SHEEP_YARN=7 SE_SHOPS=2"
               [combo2]="SHEEP_YARN=7 SE_SHOPS=2 TOMATO_SHOPS=2 COW_MILK=3 FEED_GUARD=1 STRAW_UP=4"
               [m10]="MELON_WAVE=8 MELON_DAY=10" [sy7g1]="SHEEP_YARN=7 LAND_GUARD=1" [sy7g2]="SHEEP_YARN=7 LAND_GUARD=2"
               [cowg1]="COW_MILK=3 LAND_GUARD=1" )
echo "[$(date -u +%H:%M) UTC] fresh3 runs start ($(python3 -c "import json; print(len(json.load(open('live/top_worlds_fresh3_300.json'))))") worlds)" >> $L
room; run ilsp_v13_p3 _soh $H & sleep 5
room; run ilsp_v13_newplan _nph $H & sleep 5
room; run ilsp_v13_p3 _se SE_GUARD=1 & sleep 5
for t in ${SOH_LIST:-sy7g1 sy7g2 cowg1 sy7 se2 can fg sysy str cow tomD combo2 m10 tom combo}; do
  room; run ilsp_v13_p3 _soh_$t $H ${V[$t]} & sleep 5
done
for t in ${NPH_LIST:-sy7g1 sy7 se2 can cow str combo2}; do
  room; run ilsp_v13_newplan _nph_$t $H ${V[$t]} & sleep 5
done
wait
{ echo "== FRESH3 (today's top-band games): bases vs the old live NEURAL_20MW8_SE (se)"
  for b in "ilsp_v13_p3 _soh" "ilsp_v13_newplan _nph"; do set -- $b; pair "fresh3 $2" logs/top_fresh3_ilsp_v13_p3_t10_se.json logs/top_fresh3_$1_t10$2.json; done
  echo "== FRESH3: every variant paired vs its base"
  for t in ${SOH_LIST:-sy7g1 sy7g2 cowg1 sy7 se2 can fg sysy str cow tomD combo2 m10 tom combo}; do pair "fresh3 soh+$t" logs/top_fresh3_ilsp_v13_p3_t10_soh.json logs/top_fresh3_ilsp_v13_p3_t10_soh_$t.json; done
  for t in ${NPH_LIST:-sy7g1 sy7 se2 can cow str combo2}; do pair "fresh3 nph+$t" logs/top_fresh3_ilsp_v13_newplan_t10_nph.json logs/top_fresh3_ilsp_v13_newplan_t10_nph_$t.json; done; } > /tmp/tn_fresh3_sum.txt
cat /tmp/tn_fresh3_sum.txt >> $L
echo "[$(date -u +%H:%M) UTC] fresh3 runs done" >> $L
