#!/bin/bash
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
LOG=$K/il/logs/win29_chain.log; SUM=$K/il/logs/night29_summary.log
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $LOG $SUM; }
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
# Sep 29 21:05 PKT: improve S cheaply while the CPU head-to-heads run: Sb50 = 50/50 weight blend of S and v13_p3 (no training), S_h1 = S's half pass.
for c in ilsp_v13_win_Sb50 ilsp_v13_win_S_h1; do
  say "extra: eval $c"; f4 $c "" TEAM=10; f3 $c "" TEAM=10; l464 $c 10 ""
  say "== $c vs v13_p3 (ALLG rules, Mac GPU screening)"
  cmp $BASE4 logs/top_fresh4_${c}_t10_soh_st2_wg.json; cmp $BASE3 logs/top_fresh3_${c}_t10_soh_st2_wg.json; cmp $BASE464 logs/ladder_${c}_t10_r3_tau0.5_soh_st2_wg_test.json
done
excl logs/top_DAY_ilsp_v13_win_S_t10_soh_st2_wg.json logs/top_DAY_ilsp_v13_win_Sb50_t10_soh_st2_wg.json logs/top_DAY_ilsp_v13_win_S_h1_t10_soh_st2_wg.json
say "EXTRA DONE"
