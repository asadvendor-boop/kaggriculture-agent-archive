#!/bin/bash
# Sep 30 night: MULTI-STYLE reactive head-to-heads (near-clone duels mislead): v13_p3 (= live ALLG) and S (live 56683782), both with the ALLG
# env rules, vs three different reacting styles: V94 (public 2945 Farm), M7HDB (STACK1_RACE5 scripted racer), FORK_M (copycat proxy).
# 40 seeds x 2 seats each, Mac GPU (operator: GPU for everything; one process, 80 games batched). Starts after the TAU screen (4-core rule). Summary: il/logs/night29b_summary.log
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; SUM=logs/night29b_summary.log
until grep -q "TAU SCREEN DONE" logs/tau_screen.log 2>/dev/null; do sleep 30; done
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] multi-style h2h start" >> $SUM
ALL="SE_GUARD=1 SELL_ORDER=held STRADDLE=2 SHEEP_MIRROR=1 SHEEP_MGAP=4 SHEEP_HOLD=1 SHEEP_KEEP=1 LAND_GUARD=2 SHEEP_LAST_DAY=15 SHEEP_FLOOR=150 GOOSE_EGG=2 GOOSE_YARN=2 GOOSE_GAP=3 GOOSE_BATCH=6 GOOSE_FROM_DAY=10 GOOSE_HOUR=16 GOOSE_LAST_DAY=13 COW_CARE=2 STRAW2=4 STRAW2_P=120 STRAW2_FROM=11 COW2=2 COW2_FLOOR=170 COW2_LAST_DAY=12 STRADDLE_PREBUY=1"
for opp in V94 M7HDB FORK_M; do
  for c in ilsp_v13_p3 ilsp_v13_win_S; do
    env $ALL NW=1 LOCKSTEP=80 TEAM=10 TAU=0.5 MIDNIGHT=1 FINALFIX=1 HOMING=1 ROLLOUT_DEVICE=mps OMP_NUM_THREADS=1 nice -n 10 $PY live/h2h_rules.py ckpt/$c.pt $opp 40 panel_${c}_vs$opp > logs/h2h_panel_${c}_$opp.log 2>&1
  done
  $PY - $opp <<'PYEOF' 2>&1 | tee -a $SUM
import json, sys, statistics as St
o = sys.argv[1]; r = {}
for c in ("ilsp_v13_p3", "ilsp_v13_win_S"):
    try: r[c] = {x["ep"]: x for x in json.load(open(f"logs/h2h_panel_{c}_vs{o}.json"))}
    except Exception as e: print("  missing", c, o, e); sys.exit()
k = [e for e in r["ilsp_v13_p3"] if e in r["ilsp_v13_win_S"]]; b = [r["ilsp_v13_p3"][e]["margin"] for e in k]; s = [r["ilsp_v13_win_S"][e]["margin"] for e in k]
d = [y - x for x, y in zip(b, s)]
print(f"  vs {o:7s}: ALLG-net {sum(v > 0 for v in b)}-{sum(v <= 0 for v in b)} ({St.mean(b):+,.0f}) | S {sum(v > 0 for v in s)}-{sum(v <= 0 for v in s)} ({St.mean(s):+,.0f}) | S minus ALLG {St.mean(d):+,.0f} (se {St.pstdev(d)/len(d)**0.5:,.0f})")
PYEOF
done
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] H2H PANEL DONE" >> $SUM
