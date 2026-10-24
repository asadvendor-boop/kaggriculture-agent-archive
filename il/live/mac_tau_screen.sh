#!/bin/bash
# Sep 29 23:40 PKT: order-threshold (TAU) re-tune for v13_p3 + ALLG rules: TAU 0.40/0.45/0.55/0.60 vs 0.50 on tonight's games (llB),
# fresh4 (today's leaders) and the 464; judged on the INTACT subsets (il/live/intact_cmp.py). Summary: il/logs/tau_screen.log
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; LOG=logs/tau_screen.log
export OMP_NUM_THREADS=1 ROLLOUT_DEVICE=mps LOCKSTEP=100
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12 STRADDLE_PREBUY=1"
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] tau screen start" >> $LOG
for tau in 0.45 0.55 0.40 0.60; do
  env $ALL DAY=llB N=300 NW=1 TEAM=10 TAU=$tau MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_gs_tau$tau nice -n 10 $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/tau_llB_$tau.log 2>&1
  env $ALL DAY=fresh4 N=300 NW=1 TEAM=10 TAU=$tau MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_soh_st2_wg_tau$tau nice -n 10 $PY live/top_bench.py ckpt/ilsp_v13_p3.pt > logs/tau_f4_$tau.log 2>&1
  env $ALL NW=1 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau${tau}_soh_st2_wg nice -n 10 $PY live/bench_multi.py ckpt/ilsp_v13_p3.pt:10:$tau > logs/tau_464_$tau.log 2>&1
  { echo "== TAU $tau vs 0.50 (v13_p3 + ALLG rules)"
    $PY live/intact_cmp.py logs/top_llB_ilsp_v13_p3_t10_gs_base.json logs/top_llB_ilsp_v13_p3_t10_gs_tau$tau.json
    $PY live/intact_cmp.py logs/top_fresh4_ilsp_v13_p3_t10_soh_st2_wg.json logs/top_fresh4_ilsp_v13_p3_t10_soh_st2_wg_tau$tau.json
    $PY live/intact_cmp.py logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_soh_st2_wg_test.json logs/ladder_ilsp_v13_p3_t10_r3_tau${tau}_soh_st2_wg_test.json
    $PY live/pair_cmp.py logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_soh_st2_wg_test.json logs/ladder_ilsp_v13_p3_t10_r3_tau${tau}_soh_st2_wg_test.json | head -1
  } 2>&1 | cut -c1-230 | tee -a $LOG
done
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] TAU SCREEN DONE" >> $LOG
