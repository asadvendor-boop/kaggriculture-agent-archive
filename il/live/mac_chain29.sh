#!/bin/bash
# Sep 29 chain on the Mac M5 GPU (ONE GPU job at a time): [A running] -> eval A -> train R (A + repeat opponents' last 3 days, 1 pass) -> eval R
#   -> train T (A + top-10 leaders' last day, 1 pass) -> eval T -> uniform soup of A, R, T -> eval soup.  Log il/logs/win29_chain.log
# Eval = ALLG rules (v13_p3-family rules + STRADDLE_PREBUY) on fresh4 (today's held-out leader games), fresh3, the 464; paired vs v13_p3 (baselines exist).
K=/Users/asad.ali/kaggriculture; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd $K/il
LOG=$K/il/logs/win29_chain.log
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG; }
export OMP_NUM_THREADS=1 ROLLOUT_DEVICE=mps LOCKSTEP=100
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12"
BASE4=logs/top_fresh4_ilsp_v13_p3_t10_soh_st2_wg.json; BASE3=logs/top_fresh3_ilsp_v13_p3_t10_soh_st2_wg.json; BASE464=logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_soh_st2_wg_test.json
wd() { max=$1; lf=$2; shift 2; "$@" > $lf 2>&1 & p=$!; t0=$(date +%s)          # hard time limit (the M5 GPU hung once)
  while kill -0 $p 2>/dev/null; do sleep 20; [ $(( $(date +%s) - t0 )) -gt $max ] && { kill -9 $p; return 9; }; done; wait $p; }
evalck() { c=$1; say "eval $c start"
  for day in fresh4 fresh3; do
    wd 1500 logs/mac_${day}_$c.log env $ALL STRADDLE_PREBUY=1 DAY=$day N=300 NW=1 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_soh_st2_wg nice -n 10 $PY live/top_bench.py ckpt/$c.pt || say "EVAL $day $c FAILED/TIMEOUT"
  done
  wd 1800 logs/mac_464_$c.log env $ALL STRADDLE_PREBUY=1 NW=1 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_soh_st2_wg nice -n 10 $PY live/bench_multi.py ckpt/$c.pt:10:0.5 || say "EVAL 464 $c FAILED/TIMEOUT"
  say "== $c vs v13_p3 with the same ALLG rules (fresh4 base 177 W, fresh3 144, 464 457)"
  for pr in "$BASE4 logs/top_fresh4_${c}_t10_soh_st2_wg.json" "$BASE3 logs/top_fresh3_${c}_t10_soh_st2_wg.json" "$BASE464 logs/ladder_${c}_t10_r3_tau0.5_soh_st2_wg_test.json"; do
    $PY live/pair_cmp.py $pr 2>&1 | head -6 | tee -a $LOG; done
}
train() { tag=$1; seats=$2; x6=$3; wj=$4; say "train $tag start"; bash $K/il/live/mac_ft.sh $tag "$seats" "$x6" $wj 1 5e-5 > logs/mac_ft_$tag.out 2>&1
  grep -q "DONE $tag" logs/winft_mac.log || { say "TRAIN $tag FAILED - chain stops"; exit 1; }; say "train $tag done"; }
N0=${A_LOG_FROM:-0}                                     # only look at winft_mac.log lines written after the restarted A began (line N0+1 on)
until tail -n +$((N0 + 1)) logs/winft_mac.log | grep -qE "DONE A|TRAINING FAILED A"; do sleep 30; done
tail -n +$((N0 + 1)) logs/winft_mac.log | grep -q "DONE A" || { say "A FAILED - chain stops"; exit 1; }
say "A trained (A wrote its own DONE line)"; evalck ilsp_v13_win_A
$PY -c "import json; a=json.load(open('data/win29/win_weights.json')); a.update(json.load(open('data/win29r/win_weights.json'))); json.dump(a, open('data/w_A_R.json','w'))"
$PY -c "import json; a=json.load(open('data/win29/win_weights.json')); a.update(json.load(open('data/win29t/win_weights.json'))); json.dump(a, open('data/w_A_T.json','w'))"
train R "il/data/win29/seats:il/data/win29r/seats" "il/data/win29/x6:il/data/win29r/x6" il/data/w_A_R.json; evalck ilsp_v13_win_R
train T "il/data/win29/seats:il/data/win29t/seats" "il/data/win29/x6:il/data/win29t/x6" il/data/w_A_T.json; evalck ilsp_v13_win_T
say "soup of A, R, T"; $PY live/soup.py ckpt/ilsp_v13_win_soup.pt ckpt/ilsp_v13_win_A.pt ckpt/ilsp_v13_win_R.pt ckpt/ilsp_v13_win_T.pt 2>&1 | tail -2 | tee -a $LOG
evalck ilsp_v13_win_soup
say "CHAIN DONE"
