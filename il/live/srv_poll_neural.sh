#!/bin/bash
# Live monitor (operator, Sep 25): NEURAL_10M_S4 = 56553797, NEURAL_4M_S4 = 56553911. Every 5 minutes: refresh the episode
# lists (public ListEpisodes, no credentials), tape new games (data/tapes_n10m, data/tapes_n4m), log new results to
# il/logs/poll_neural.log.
cd /Users/asad.ali/kaggriculture
PY=/venv/main/bin/python
while true; do
  for spec in "56553797 N10M data/tapes_n10m" "56553911 N4M data/tapes_n4m"; do
    set -- $spec
    nice -n 10 $PY tools/pull.py $2 $1 > il/logs/poll_$2_pull.out 2>&1 || true
    nice -n 10 $PY tools/tape_sub.py $1 $3 > il/logs/poll_$2_tape.out 2>&1 || true
    nice -n 10 $PY il/live/poll_report.py $1 $2 >> il/logs/poll_neural.log 2>&1 || true
    sleep 5
  done
  sleep 300
done
