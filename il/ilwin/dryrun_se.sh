#!/bin/bash
# Exercise the SE-guard attribution path: 12 won games (SOH + N20MW8), policy check only at SE-eligible steps.
S=$(cd "$(dirname "$0")" && pwd); PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3; cd /Users/asad.ali/kaggriculture
export OMP_NUM_THREADS=2 TORCH_THREADS=2
bash $S/wait_slot.sh; T0=$(date +%s)
nice -n 15 $PY $S/build_winner_ds.py --out $S/dryrun/ds_se --tapes data/tapes_soh:W data/tapes_n20mw8:W --min-opp-rating 2400 \
  --min-margin 500 --max-games 12 --device cpu
nice -n 15 $PY $S/build_winner_ds.py --out $S/dryrun/ds_se --merge > /dev/null
$PY -c "
import json
m = json.load(open('$S/dryrun/ds_se/manifest.json'))
print('SESTATS', m['summary']['stats'])
print('SEGAMES', [(g['ep'], g['se_removed']) for g in m['games'] if g['se_removed']])"
echo "SE WALL $(( $(date +%s) - T0 )) s"
