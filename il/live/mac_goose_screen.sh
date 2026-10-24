#!/bin/bash
# Sep 29 23:10 PKT: screen earlier / rival-matching goose rules on tonight's field (llB = all 85 evening games vs 2600+, llA = 61 earlier
# games today), Mac GPU, paired vs v13_p3 + ALLG rules. Summary: il/logs/goose_screen.log
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; LOG=logs/goose_screen.log
export OMP_NUM_THREADS=1 ROLLOUT_DEVICE=mps LOCKSTEP=100
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12 STRADDLE_PREBUY=1"
V=("base|" "g8|GOOSE_FROM_DAY=8" "g8r|GOOSE_FROM_DAY=8 GOOSE_RIVAL=1" "g7r2|GOOSE_FROM_DAY=7 GOOSE_RIVAL=1 GOOSE_GAP=2" "g8re3|GOOSE_FROM_DAY=8 GOOSE_RIVAL=1 GOOSE_EGG=3" "g6r|GOOSE_FROM_DAY=6 GOOSE_RIVAL=1")
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] goose screen start" >> $LOG
for day in llB llA; do
  for spec in "${V[@]}"; do t=${spec%%|*}; ov=${spec#*|}
    env $ALL $ov DAY=$day N=300 NW=1 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_gs_$t nice -n 10 $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/gs_${day}_$t.log 2>&1
    [ $t != base ] && $PY live/pair_cmp.py logs/top_${day}_ilsp_v13_p3_t10_gs_base.json logs/top_${day}_ilsp_v13_p3_t10_gs_$t.json 2>&1 | head -1 | cut -c1-200 | tee -a $LOG
  done
  env $ALL DAY=$day N=300 NW=1 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_gs_S nice -n 10 $PY live/top_bench.py ckpt/ilsp_v13_win_S.pt > logs/gs_${day}_S.log 2>&1
  $PY live/pair_cmp.py logs/top_${day}_ilsp_v13_p3_t10_gs_base.json logs/top_${day}_ilsp_v13_win_S_t10_gs_S.json 2>&1 | head -1 | cut -c1-200 | tee -a $LOG
done
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] GOOSE SCREEN DONE" >> $LOG
