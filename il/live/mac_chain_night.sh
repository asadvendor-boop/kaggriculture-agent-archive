#!/bin/bash
# Sep 29 evening, UNATTENDED while the operator travels (laptop may sleep: every timer here is sleep-aware).
# 1) control 464  2) train S2 + fresh4/fresh3/464  3) report incl. "without MMPQ+Boey"  4) REACTIVE head-to-heads vs the LIVE ALLG package
# (v13_p3 reference, S, S2)  5) slot sweep of v13_p3 (slots 5 Vadim, 3 DECEM, 1 MMPQ) on fresh4 + 464.   Summary: il/logs/night29_summary.log
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
LOG=$K/il/logs/win29_chain.log; SUM=$K/il/logs/night29_summary.log
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG $SUM; }
caffeinate -i -w $$ &                                   # no idle sleep while this runs (a closed lid still sleeps; timers below survive it)
export OMP_NUM_THREADS=1 ROLLOUT_DEVICE=mps LOCKSTEP=100
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12 STRADDLE_PREBUY=1"
BASE4=logs/top_fresh4_ilsp_v13_p3_t10_soh_st2_wg.json; BASE3=logs/top_fresh3_ilsp_v13_p3_t10_soh_st2_wg.json; BASE464=logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_soh_st2_wg_test.json
wd() { max=$1; lf=$2; shift 2; "$@" > $lf 2>&1 & p=$!; t0=$(date +%s); last=$t0        # hard limit on AWAKE time (sleep gaps > 120 s are not counted)
  while kill -0 $p 2>/dev/null; do sleep 20; now=$(date +%s); [ $(( now - last )) -gt 120 ] && t0=$(( t0 + now - last )); last=$now
    [ $(( now - t0 )) -gt $max ] && { kill -9 $p; return 9; }; done; wait $p; }
cmp() { $PY live/pair_cmp.py "$@" 2>&1 | head -6 | tee -a $LOG $SUM; }
f4() { c=$1; tx=$2; shift 2; wd 1500 logs/mac_fresh4_$c$tx.log env "$@" $ALL DAY=fresh4 N=300 NW=1 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_soh_st2_wg$tx nice -n 10 $PY live/top_bench.py ckpt/$c.pt || say "fresh4 $c$tx FAILED/TIMEOUT"; }
f3() { c=$1; tx=$2; shift 2; wd 1500 logs/mac_fresh3_$c$tx.log env "$@" $ALL DAY=fresh3 N=300 NW=1 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 FRESH=1 TAGX=_soh_st2_wg$tx nice -n 10 $PY live/top_bench.py ckpt/$c.pt || say "fresh3 $c$tx FAILED/TIMEOUT"; }
l464() { c=$1; team=$2; tx=$3; wd 1800 logs/mac_464_$c$tx.log env $ALL NW=1 MIDNIGHT=1 FINALFIX=1 HOMING=1 TAGX=_r3_tau0.5_soh_st2_wg$tx nice -n 10 $PY live/bench_multi.py ckpt/$c.pt:$team:0.5 || say "464 $c$tx FAILED/TIMEOUT"; }
excl() { $PY - "$@" <<'PYEOF' 2>&1 | tee -a $LOG $SUM
import json, sys, statistics as St
EXC = {"M & M & P & Q", "Boey"}
for day in ("fresh4", "fresh3"):
    b = {(r["ep"], r["seat"]): r for r in json.load(open(f"logs/top_{day}_ilsp_v13_p3_t10_soh_st2_wg.json"))}
    for fn in sys.argv[1:]:
        try: c = {(r["ep"], r["seat"]): r for r in json.load(open(fn.replace("DAY", day)))}
        except Exception: continue
        k = [x for x in b if x in c and b[x]["opp"] not in EXC]; d = [c[x]["margin"] - b[x]["margin"] for x in k]
        print(f"  {day} w/o MMPQ+Boey ({len(k)}): v13_p3 {sum(b[x]['margin']>0 for x in k)} -> {sum(c[x]['margin']>0 for x in k)} W | margin {St.mean(d):+,.0f} (se {St.pstdev(d)/len(d)**0.5:,.0f}) | {fn.split('/')[-1].replace('top_DAY_','')}")
PYEOF
}
say "NIGHT CHAIN start"
until grep -q "BUGCHECK DONE" logs/bugcheck29.log; do sleep 20; done
say "== CONTROL (v13_p3 re-trained on its own DSM data, same pipeline as S)"; tail -4 logs/bugcheck29.log | grep -E "wins" | tee -a $SUM
l464 ilsp_v13_ctrl 10 ""; cmp $BASE464 logs/ladder_ilsp_v13_ctrl_t10_r3_tau0.5_soh_st2_wg_test.json
say "train S2 start (8 same-style top-10 leaders, DSM farm moves only, 110 of our wins)"
bash $K/il/live/mac_ft.sh S2 il/data/win29s2/seats il/data/win29s2/x6 il/data/win29s2/win_weights.json 1 5e-5 > logs/mac_ft_S2.out 2>&1
if tail -n 5 logs/winft_mac.log | grep -q "DONE S2"; then
  say "S2 trained"; f4 ilsp_v13_win_S2 "" TEAM=10; f3 ilsp_v13_win_S2 "" TEAM=10; l464 ilsp_v13_win_S2 10 ""
  say "== S2 vs v13_p3 (ALLG rules)"
  cmp $BASE4 logs/top_fresh4_ilsp_v13_win_S2_t10_soh_st2_wg.json; cmp $BASE3 logs/top_fresh3_ilsp_v13_win_S2_t10_soh_st2_wg.json
  cmp $BASE464 logs/ladder_ilsp_v13_win_S2_t10_r3_tau0.5_soh_st2_wg_test.json
else say "TRAIN S2 FAILED (continuing with the rest)"; fi
excl logs/top_DAY_ilsp_v13_win_S_t10_soh_st2_wg.json logs/top_DAY_ilsp_v13_win_S2_t10_soh_st2_wg.json logs/top_DAY_ilsp_v13_ctrl_t10_soh_st2_wg.json
say "== REACTIVE head-to-heads vs the LIVE ALLG package (40 seeds x 2 seats, both sides react; seeds 900000..)"
for c in ilsp_v13_p3 ilsp_v13_win_S ilsp_v13_win_S2; do
  [ -f ckpt/$c.pt ] || continue
  wd 3000 logs/h2h_night_$c.log env $ALL NW=3 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 ROLLOUT_DEVICE=cpu nice -n 10 $PY live/h2h_rules.py ckpt/$c.pt $K/package/NEURAL_20MW8_ST2_ALLG/main.py 40 night_${c}_vsALLG || say "h2h $c FAILED/TIMEOUT"
  $PY - logs/h2h_night_${c}_vsALLG.json <<'PYEOF' 2>&1 | tee -a $LOG $SUM
import json, sys, statistics as St
try:
    r = json.load(open(sys.argv[1])); m = [x["margin"] for x in r]
    print(f"  h2h {sys.argv[1].split('h2h_night_')[-1][:-5]}: {sum(v > 0 for v in m)}-{sum(v <= 0 for v in m)} of {len(m)}, mean margin {St.mean(m):+,.0f} (se {St.pstdev(m)/len(m)**0.5:,.0f})")
except Exception as e: print("  h2h read failed", e)
PYEOF
done
say "== SLOT SWEEP: v13_p3 played in other style slots (ALLG rules)"
for t in 5 3 1; do
  f4 ilsp_v13_p3 _slot$t TEAM=$t; cmp $BASE4 logs/top_fresh4_ilsp_v13_p3_t${t}_soh_st2_wg_slot$t.json
  l464 ilsp_v13_p3 $t _slot$t; cmp $BASE464 logs/ladder_ilsp_v13_p3_t${t}_r3_tau0.5_soh_st2_wg_slot${t}_test.json
done
say "NIGHT CHAIN DONE"
