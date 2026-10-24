#!/bin/bash
# Sep 28 ~04:30 PKT: R2 of il/live/PLANNING_FORENSICS.md - the feed guard (il/features.py _feed_guard, FEED_GUARD=1):
# an animal unfed yesterday and not fed today escapes at midnight; we lose ~1.9 such animals/game (DSM 0.4). Tested on
# v13_p3 (SE on) with the closed-loop rulers: frozen live losses, both leader tests, the 464; paired vs the shipped rules.
until grep -q "R1 sell-order test done" /root/a100_bench.log; do sleep 30; done
exec 7>/tmp/srv_a100_eval.lock; flock 7
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/venv/main/bin/python; L=/root/a100_bench.log
source live/srv_cpu.sh; export CPU_BUDGET=40 BENCH_CAP=36
pair() { $PY live/pair_cmp.py $2 $3 2>/dev/null | sed "s/^/  $1 /" | cut -c1-230; }
FEED_GUARD=1 SE_GUARD=1 DAY=sevloss N=300 NW=24 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_fg FRESH=1 \
  $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/fg_ll.log 2>&1
( export FEED_GUARD=1 SE_GUARD=1 TAGX=_fg; bench_gov top:10 ckpt/ilsp_v13_p3.pt > logs/fg_top.log 2>&1 )
( export FEED_GUARD=1 SE_GUARD=1 TAGX=_fg; bench_gov top2:10 ckpt/ilsp_v13_p3.pt > logs/fg_top2.log 2>&1 )
{ echo "== FEED_GUARD=1 (R2) on v13_p3"
  pair "live losses" logs/top_sevloss_ilsp_v13_p3_t10_sev_base.json logs/top_sevloss_ilsp_v13_p3_t10_fg.json
  for d in fresh fresh2; do pair "leaders $d " logs/top_${d}_ilsp_v13_p3_t10_se.json logs/top_${d}_ilsp_v13_p3_t10_fg.json; done; } >> $L
( export FEED_GUARD=1 SE_GUARD=1 TAGX=_r3_tau0.5_fg; bench_gov ladder ckpt/ilsp_v13_p3.pt:10:0.5 > logs/fg_464.log 2>&1 )
{ echo "== FEED_GUARD=1 (R2) 464"; pair "464" logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_se_test.json logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_fg_test.json; } >> $L
echo "[$(date -u +%H:%M) UTC] R2 feed-guard test done" >> $L
