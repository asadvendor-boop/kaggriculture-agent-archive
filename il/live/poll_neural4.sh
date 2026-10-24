#!/bin/bash
# LOCAL live-ladder monitor: both active submissions - since Sep 27 NEURAL_20MW_SE = 56589891 (label N20MW) and NEURAL_20MW5_SE =
# 56595847 (label N20MW5; RL24 and DSM_S0 retired) -
# NEURAL_10M_RL24 = 56571775 (label N10MR). Every 5 minutes: refresh episode lists (public ListEpisodes), tape new games
# here (data/tapes_n20mw, data/tapes_n20mw5), log new results to il/logs/poll_neural.log. (Server copy dropped: stopped.)
cd /Users/asad.ali/kaggriculture
while true; do
  for spec in "56589891 N20MW data/tapes_n20mw" "56595847 N20MW5 data/tapes_n20mw5"; do
    set -- $spec
    nice -n 10 .venv/bin/python tools/pull.py $2 $1 > il/logs/poll_$2_pull.out 2>&1 || echo "[$(date +%H:%M)] $2 pull ERROR: $(tail -1 il/logs/poll_$2_pull.out)"
    nice -n 10 .venv/bin/python tools/tape_sub.py $1 $3 > il/logs/poll_$2_tape.out 2>&1 || echo "[$(date +%H:%M)] $2 tape ERROR: $(tail -1 il/logs/poll_$2_tape.out)"
    nice -n 10 .venv/bin/python il/live/poll_report.py $1 $2 2>&1
    sleep 5
  done
  sleep 300
done >> il/logs/poll_neural.log 2>&1
