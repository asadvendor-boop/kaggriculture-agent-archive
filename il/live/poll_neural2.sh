#!/bin/bash
# LOCAL live-ladder monitor (operator, Sep 25): NEURAL_10M_S4 = 56553797, NEURAL_4M_S4 = 56553911. Every 5 minutes:
# refresh both episode lists (public ListEpisodes), download + tape every new game HERE (data/tapes_n10m, data/tapes_n4m),
# copy the new tapes and episode lists to the A100 server too (rsync, no second Kaggle download), and log each new
# result and the record to il/logs/poll_neural.log.
cd /Users/asad.ali/kaggriculture
SRV="root@<SERVER_IP>:/Users/asad.ali/kaggriculture/data/"
while true; do
  for spec in "56553797 N10M data/tapes_n10m" "56553911 N4M data/tapes_n4m"; do
    set -- $spec
    nice -n 10 .venv/bin/python tools/pull.py $2 $1 > il/logs/poll_$2_pull.out 2>&1 || echo "[$(date +%H:%M)] $2 pull ERROR: $(tail -1 il/logs/poll_$2_pull.out)"
    nice -n 10 .venv/bin/python tools/tape_sub.py $1 $3 > il/logs/poll_$2_tape.out 2>&1 || echo "[$(date +%H:%M)] $2 tape ERROR: $(tail -1 il/logs/poll_$2_tape.out)"
    nice -n 10 .venv/bin/python il/live/poll_report.py $1 $2 2>&1
    rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -p 13692" $3 data/rows_$1.json $SRV > /dev/null 2>&1 || echo "[$(date +%H:%M)] $2 copy to server ERROR (server unreachable?)"
    sleep 5
  done
  sleep 300
done >> il/logs/poll_neural.log 2>&1
