#!/bin/bash
# New daily archive games -> training seats (48-team vocabulary) -> v6 features -> order-sequence labels.
cd /Users/asad.ali/kaggriculture
for w in 0 1 2 3; do nice -n 8 .venv/bin/python il/build_dataset3.py il/data/all $w 4 data/tapes_daily data/tapes_pull > il/logs/build_daily2_w$w.log 2>&1 & done; wait
echo "[$(date +%H:%M)] seats: $(ls il/data/all | wc -l) in il/data/all"
for w in 0 1 2 3; do nice -n 8 .venv/bin/python il/build_extras6.py $w 4 > il/logs/x6_daily2_w$w.log 2>&1 & done; wait
for w in 0 1 2 3; do nice -n 8 .venv/bin/python il/build_mseq.py $w 4 > il/logs/mseq_daily2_w$w.log 2>&1 & done; wait
echo "[$(date +%H:%M)] v6 features $(ls il/data/x6 | wc -l), order labels $(ls il/data/mseq | wc -l)"
