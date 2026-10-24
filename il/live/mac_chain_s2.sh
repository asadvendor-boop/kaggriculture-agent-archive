#!/bin/bash
# Sep 29 (operator: "IL from the other 8 top-10 leaders that match our slot"): S2 = S + DECEM, UMG, seek (<=60 each) + DSM farm moves only; our wins 110.
# (Victor @ Tufa Labs, Vadim Vasilenko, akmr, Anton Tikhonov) + our own v13_p3-family wins as anchor; no mixed-style losses. Train 1 pass, test.
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; LOG=$K/il/logs/win29_chain.log
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
excl() { $PY - "$@" <<'PYEOF' 2>&1 | tee -a $LOG
import json, sys, statistics as St
EXC = {"M & M & P & Q", "Boey"}
for day in ("fresh4", "fresh3"):
    b = {(r["ep"], r["seat"]): r for r in json.load(open(f"logs/top_{day}_ilsp_v13_p3_t10_soh_st2_wg.json"))}
    for tag in sys.argv[1:]:
        try: c = {(r["ep"], r["seat"]): r for r in json.load(open(f"logs/top_{day}_{tag}_t10_soh_st2_wg.json"))}
        except Exception: continue
        k = [x for x in b if x in c and b[x]["opp"] not in EXC]; d = [c[x]["margin"] - b[x]["margin"] for x in k]
        print(f"  {day} w/o MMPQ+Boey ({len(k)}): v13_p3 {sum(b[x]['margin']>0 for x in k)} -> {tag} {sum(c[x]['margin']>0 for x in k)} W | margin {St.mean(d):+,.0f} (se {St.pstdev(d)/len(d)**0.5:,.0f})")
PYEOF
}
until grep -q "BUGCHECK DONE" logs/bugcheck29.log; do sleep 20; done
say "CTRL 464 (does ANY fine-tune in this pipeline gain on the 464?)"
wd 1800 logs/mac_464_ilsp_v13_ctrl.log env $ALL STRADDLE_PREBUY=1 NW=1 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_soh_st2_wg nice -n 10 $PY live/bench_multi.py ckpt/ilsp_v13_ctrl.pt:10:0.5 || say "CTRL 464 FAILED"
$PY live/pair_cmp.py $BASE464 logs/ladder_ilsp_v13_ctrl_t10_r3_tau0.5_soh_st2_wg_test.json 2>&1 | head -2 | tee -a $LOG
say "train S2 start (8 same-style top-10 leaders, DSM farm moves only, 110 of our wins)"
bash $K/il/live/mac_ft.sh S2 il/data/win29s2/seats il/data/win29s2/x6 il/data/win29s2/win_weights.json 1 5e-5 > logs/mac_ft_S2.out 2>&1
tail -n 5 logs/winft_mac.log | grep -q "DONE S2" || { say "TRAIN S2 FAILED"; exit 1; }
evalck ilsp_v13_win_S2
excl ilsp_v13_win_S ilsp_v13_win_S2 ilsp_v13_ctrl
say "S2 CHAIN DONE"
