#!/bin/bash
# LOCAL live-ladder monitor (operator, Sep 26 01:40 PKT): only NEURAL_10M_DSM_S0 = 56559691 (the operator's latest upload).
# Every 5 minutes: refresh its episode list (public ListEpisodes), tape every new game here (data/tapes_n10md) and copy
# the tapes + episode list to the A100 server (rsync, while it is up); log new results to il/logs/poll_neural.log.
cd /Users/asad.ali/kaggriculture
SRV="root@<SERVER_IP>:/Users/asad.ali/kaggriculture/data/"
while true; do
  nice -n 10 .venv/bin/python tools/pull.py N10MD 56559691 > il/logs/poll_N10MD_pull.out 2>&1 || echo "[$(date +%H:%M)] N10MD pull ERROR: $(tail -1 il/logs/poll_N10MD_pull.out)"
  nice -n 10 .venv/bin/python tools/tape_sub.py 56559691 data/tapes_n10md > il/logs/poll_N10MD_tape.out 2>&1 || echo "[$(date +%H:%M)] N10MD tape ERROR: $(tail -1 il/logs/poll_N10MD_tape.out)"
  nice -n 10 .venv/bin/python il/live/poll_report.py 56559691 N10MD 2>&1
  rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -p 13692" data/tapes_n10md data/rows_56559691.json $SRV > /dev/null 2>&1 || true
  sleep 300
done >> il/logs/poll_neural.log 2>&1
