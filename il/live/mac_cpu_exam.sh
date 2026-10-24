#!/bin/bash
# Sep 29 night: CPU EXAM queue = each candidate PACKAGE on the 464 worlds on CPU (exactly as Kaggle runs it), paired vs the live ALLG
# package's own CPU run. 4-core rule: pkg464 workers are PAUSED (SIGSTOP, by PID) while the night chain's CPU head-to-heads run.
K=/Users/asad.ali/kaggriculture; cd $K/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; SUM=logs/night29_summary.log
say() { echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] $*" | tee -a $SUM; }
( while true; do                                       # the gate
    P=$(pgrep -f "live/pkg464.py" | tr '\n' ' ')
    if pgrep -f "live/h2h_rules.py" > /dev/null; then [ -n "$P" ] && kill -STOP $P 2>/dev/null; else [ -n "$P" ] && kill -CONT $P 2>/dev/null; fi
    sleep 10; done ) &
GATE=$!
until [ -f logs/pkg464_ALLGpkg.json ]; do sleep 30; done
say "CPU EXAM: S package on the 464"
bash live/mac_pkg464.sh NEURAL_20MW8_ALLG_S Spkg 2 > /dev/null 2>&1
say "== CPU EXAM S package vs live ALLG package (464, CPU, exact Kaggle runtime)"; $PY live/pair_cmp.py logs/pkg464_ALLGpkg.json logs/pkg464_Spkg.json 2>&1 | head -2 | tee -a $SUM
until [ -f $K/package/NEURAL_20MW8_ALLG_S2.tar.gz ] || grep -q "PKG night DONE" logs/pkg_night.log 2>/dev/null; do sleep 30; done
if [ -f $K/package/NEURAL_20MW8_ALLG_S2/main.py ]; then
  say "CPU EXAM: S2 package on the 464"; bash live/mac_pkg464.sh NEURAL_20MW8_ALLG_S2 S2pkg 2 > /dev/null 2>&1
  say "== CPU EXAM S2 package vs live ALLG package"; $PY live/pair_cmp.py logs/pkg464_ALLGpkg.json logs/pkg464_S2pkg.json 2>&1 | head -2 | tee -a $SUM
fi
kill $GATE 2>/dev/null; P=$(pgrep -f "live/pkg464.py" | tr '\n' ' '); [ -n "$P" ] && kill -CONT $P 2>/dev/null
say "CPU EXAM DONE"
