#!/bin/bash
# Re-check of the straddle decomposition after the grammar fix: 3 ST2 wins (incl. 114807051) with the full verify.
S=$(cd "$(dirname "$0")" && pwd); PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd /Users/asad.ali/kaggriculture
until grep -q "DRYRUN REST DONE\|Traceback" $S/dryrun/rest.log 2>/dev/null; do sleep 5; done
export OMP_NUM_THREADS=2 TORCH_THREADS=2
bash $S/wait_slot.sh; T0=$(date +%s)
nice -n 15 $PY $S/build_winner_ds.py --out $S/dryrun/ds_fix --tapes data/tapes_st2:W --min-opp-rating 2400 --min-margin 500 \
  --eps 114807051,114804416 --verify 1 --device cpu
nice -n 15 $PY $S/build_winner_ds.py --out $S/dryrun/ds_fix --merge > /dev/null
$PY -c "
import json
for g in json.load(open('$S/dryrun/ds_fix/manifest.json'))['games']:
    print('FIX', {k: g[k] for k in ('ep','src','margin','straddle_legs','se_removed','unit_masked','heads_ignored','verify_permille')})
print('FIXSTATS', json.load(open('$S/dryrun/ds_fix/manifest.json'))['summary']['stats'])"
echo "FIX WALL $(( $(date +%s) - T0 )) s"
