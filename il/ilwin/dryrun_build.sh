#!/bin/bash
# Mac dry run, step 1: build 3 winner seats (ST2 win = straddle + SE + full verify, SOH loss = opponent labels, N20MW8 win).
S=$(cd "$(dirname "$0")" && pwd); PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd /Users/asad.ali/kaggriculture
export OMP_NUM_THREADS=2 TORCH_THREADS=2
bash $S/wait_slot.sh
T0=$(date +%s)
nice -n 15 $PY $S/build_winner_ds.py --out $S/dryrun/ds --tapes data/tapes_st2:WL data/tapes_soh:WL data/tapes_n20mw8:WL \
  --eps 114807051,114808382,114545224 --min-opp-rating 2400 --min-margin 500 --verify 1 --device cpu
nice -n 15 $PY $S/build_winner_ds.py --out $S/dryrun/ds --merge
echo "BUILD WALL $(( $(date +%s) - T0 )) s"
