#!/bin/bash
# Sep 28 ~07:15 PKT: every live loss of both active scripts so far (incl. tonight's newest), replayed with the opponent on
# tape: the shipped v13_p3 (SE, canonical order) vs the pick NEURAL_20MW8NPS_SOH (soup3, SE off, held). Output a100_bench.log
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
$PY live/srv_liveloss_worlds.py > /dev/null 2>&1; cp live/top_worlds_livelosses_300.json live/top_worlds_llnow_300.json
n=$($PY -c "import json; print(len(json.load(open('live/top_worlds_llnow_300.json'))))")
SE_GUARD=1 DAY=llnow N=300 NW=30 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_se FRESH=1 \
  $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/llnow_base.log 2>&1
SE_GUARD=0 SELL_ORDER=held DAY=llnow N=300 NW=30 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_nps FRESH=1 \
  $PY live/top_bench.py ckpt/ilsp_v13_newplan_soup3.pt > logs/llnow_nps.log 2>&1
{ echo "== ALL $n live losses so far (both scripts, incl. tonight's): shipped v13_p3 SE vs the pick NEURAL_20MW8NPS_SOH"
  $PY live/pair_cmp.py logs/top_llnow_ilsp_v13_p3_t10_se.json logs/top_llnow_ilsp_v13_newplan_soup3_t10_nps.json 2>/dev/null | sed "s/^/  /" | cut -c1-230; } >> $L
echo "[$(date -u +%H:%M) UTC] LLNOW done" >> $L
