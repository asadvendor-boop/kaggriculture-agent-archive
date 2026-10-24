#!/bin/bash
# Sep 27 ~22:25 PKT, A100 box (operator: NEURAL_20MW8_SE is live as 56614508 - "keep downloading its replays every 5
# minutes and if any losses, download them on the server too and counter those weaknesses in the upcoming build"):
# every 5 minutes for both ACTIVE submissions - 56614508 NEURAL_20MW8_SE (N20MW8) and 56595847 NEURAL_20MW5_SE (N20MW5):
# episode list (public ListEpisodes -> data/rows_<sub>.json), new games taped (tools/tape_sub.py -> data/tapes_n20mw8,
# data/tapes_n20mw5), one line per new result, and every new LOSS appended to the specialist bank data/tapes_spec
# (srv_spec_add.py) that the anti-specialist RL and the next builds train against. Log: /root/live_poll.log
exec 8>/tmp/srv_poll_live.lock; flock -n 8 || { echo "already running"; exit 0; }
cd /Users/asad.ali/kaggriculture; PY=/venv/main/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data/tapes_n20mw8 data/tapes_n20mw5 il/logs
if [ ! -f il/logs/poll_N20MW5_seen.txt ]; then         # N20MW5's ~100 earlier games: known, not re-announced
  $PY tools/pull.py N20MW5 56595847 > il/logs/poll_N20MW5_pull.out 2>&1
  $PY -c "import json; print('\n'.join(str(r['ep']) for r in json.load(open('data/rows_56595847.json'))))" > il/logs/poll_N20MW5_seen.txt
fi
while true; do
  for spec in "56614508 N20MW8 data/tapes_n20mw8" "56595847 N20MW5 data/tapes_n20mw5"; do
    set -- $spec
    $PY tools/pull.py $2 $1 > il/logs/poll_$2_pull.out 2>&1 || echo "[$(date -u +%H:%M) UTC] $2 pull ERROR: $(tail -1 il/logs/poll_$2_pull.out)"
    $PY tools/tape_sub.py $1 $3 > il/logs/poll_$2_tape.out 2>&1 || echo "[$(date -u +%H:%M) UTC] $2 tape ERROR: $(tail -1 il/logs/poll_$2_tape.out)"
    $PY il/live/poll_report.py $1 $2 2>&1
    sleep 5
  done
  $PY il/live/srv_spec_add.py data/tapes_n20mw8 data/tapes_n20mw5 2>&1
  sleep 300
done >> /root/live_poll.log 2>&1
