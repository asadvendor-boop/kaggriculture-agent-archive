#!/bin/bash
# Sep 29 prep while waiting for the operator's GO (no training): (1) top up the winner datasets with the newest games (builder skips built seats),
# (2) integrity check, (3) ALLG-rules baseline on fresh4 (today's held-out leader games) on the Mac GPU. Log il/logs/prep29.log
K=/Users/asad.ali/kaggriculture; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; LOG=$K/il/logs/prep29.log
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG; }
cd $K; say "top-up build"; bash il/live/mac_win29_build.sh 0 > il/logs/win29_topup.out 2>&1
say "top-up done: $(grep -E 'READY' il/logs/win29_build.log | tail -1)"
nice -n 10 $PY il/ilwin/check_ds.py il/data/win29 il/data/win29np il/data/win29l 2>&1 | tee -a $LOG
cd $K/il; export OMP_NUM_THREADS=1 ROLLOUT_DEVICE=mps LOCKSTEP=100
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12"
say "fresh4 baseline ALLG rules (v13_p3, team 10, tau .5) start"
env $ALL STRADDLE_PREBUY=1 DAY=fresh4 N=300 NW=1 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_soh_st2_wg nice -n 10 $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/mac_f4_wg.log 2>&1
say "fresh4 baseline done: $(tail -3 logs/mac_f4_wg.log | tr '\n' ' ' | cut -c1-260)"
