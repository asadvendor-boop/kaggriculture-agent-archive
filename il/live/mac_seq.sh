#!/bin/bash
# Sep 29: the M5 GPU (mps) serves ONE process at a time well (4 parallel processes ran at 8-23% CPU, waiting) ->
# run everything SEQUENTIALLY in one process chain: phase 1 = fresh3 for every spec, phase 2 = the 464 for every spec.
# Usage: mac_seq.sh "tag|override" ...   (ALL env + override, same tags/files as mac_allminus.sh)
cd /Users/asad.ali/kaggriculture/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
export OMP_NUM_THREADS=1 ROLLOUT_DEVICE=mps LOCKSTEP=100
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12"
for spec in "$@"; do t=${spec%%|*}; ov=${spec#*|}
  env $ALL $ov DAY=fresh3 N=300 NW=1 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_soh_st2_$t nice -n 10 $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/mac_f3_$t.log 2>&1
  echo "$(date +%H:%M) fresh3 done $t" >> logs/mac_seq.log; done
for spec in "$@"; do t=${spec%%|*}; ov=${spec#*|}
  env $ALL $ov NW=1 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_soh_st2_$t nice -n 10 $PY live/bench_multi.py ckpt/ilsp_v13_p3.pt:10:0.5 > logs/mac_464_$t.log 2>&1
  echo "$(date +%H:%M) 464 done $t" >> logs/mac_seq.log; done
