#!/bin/bash
# Sep 28 (A100 box to be stopped): the live poller on the Mac - every 5 minutes, for each submission in
# il/live/poll_subs.txt ("<sub id> <label> <tape dir>", re-read every round so a new upload is added without a restart):
# public episode list (tools/pull.py -> data/rows_<sub>.json), new games taped (tools/tape_sub.py), one line per new
# result + a record line (il/live/poll_report.py). Public endpoints only, no credentials. Light: nice 15, ~1% of a core.
# Log: il/logs/live_poll_mac.log
cd /Users/asad.ali/kaggriculture; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3   # arm64 (the kagsim_trace build); the default python3 here is x86_64 miniforge
while true; do
  while read -r sub lab dir; do
    [ -z "$sub" ] && continue
    mkdir -p "$dir"
    nice -n 15 $PY tools/pull.py $lab $sub > il/logs/poll_${lab}_pull.out 2>&1 || echo "[$(date -u +%H:%M) UTC] $lab pull ERROR: $(tail -1 il/logs/poll_${lab}_pull.out)"
    nice -n 15 $PY tools/tape_sub.py $sub $dir > il/logs/poll_${lab}_tape.out 2>&1 || echo "[$(date -u +%H:%M) UTC] $lab tape ERROR: $(tail -1 il/logs/poll_${lab}_tape.out)"
    nice -n 15 $PY il/live/poll_report.py $sub $lab 2>&1
    sleep 5
  done < il/live/poll_subs.txt
  sleep 300
done >> il/logs/live_poll_mac.log 2>&1
