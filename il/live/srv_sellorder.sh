#!/bin/bash
# Sep 28 ~03:35 PKT: R1 of il/live/PLANNING_FORENSICS.md - our "canonical" SELL slot order puts FERTILIZER first and
# MELON last (DSM: the reverse); in same-turn sale races the rival sells first 21x/game vs our 5x (-$422/game); the
# existing switch SELL_ORDER="base" (sells sorted by base price, premium first) measured +$535/game (se 51, 91% of
# games better) in an exact market model. Test it on v13_p3 (NEURAL_20MW8_SE, SE on) with our closed-loop rulers:
# the frozen live losses, both leader tests and the 464, paired vs the shipped canonical order. Output /root/a100_bench.log
until grep -q "SEV screening done" /root/a100_bench.log; do sleep 30; done
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log
source live/srv_cpu.sh; export CPU_BUDGET=40 BENCH_CAP=36
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
SELL_ORDER=base SE_GUARD=1 DAY=sevloss N=300 NW=24 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_sob FRESH=1 \
  $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/sob_ll.log 2>&1
( export SELL_ORDER=base SE_GUARD=1 TAGX=_sob; bench_gov top:10 ckpt/ilsp_v13_p3.pt > logs/sob_top.log 2>&1 )
( export SELL_ORDER=base SE_GUARD=1 TAGX=_sob; bench_gov top2:10 ckpt/ilsp_v13_p3.pt > logs/sob_top2.log 2>&1 )
{ echo "== SELL_ORDER=base (R1) on v13_p3"
  pair "live losses" logs/top_sevloss_ilsp_v13_p3_t10_sev_base.json logs/top_sevloss_ilsp_v13_p3_t10_sob.json
  for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_ilsp_v13_p3_t10_se.json logs/top_${d}_ilsp_v13_p3_t10_sob.json; done; } >> $L
( export SELL_ORDER=base SE_GUARD=1 TAGX=_r3_tau0.5_sob; bench_gov ladder ckpt/ilsp_v13_p3.pt:10:0.5 > logs/sob_464.log 2>&1 )
{ echo "== SELL_ORDER=base (R1) 464"; pair "464" logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_sob_test.json; } >> $L
echo "[$(date -u +%H:%M) UTC] R1 sell-order test done" >> $L
