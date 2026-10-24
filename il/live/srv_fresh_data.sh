#!/bin/bash
# Fresh leader data on the server (operator, Sep 25): the top-12 leaders' CURRENT submissions (all their games) + Kaggle's
# daily top-games datasets Sep 18-24 -> tapes -> training seats (48-team vocabulary) -> v6 features.
#   data/fresh  = top-12 submissions + daily Sep 21-24 (held-out games removed)  -> "FRESH READY"
#   data/fresh7 = the same + daily Sep 18-20 (built afterwards at low priority)   -> "FRESH7 READY"
# Kaggle is treated gently: 12 replay threads (429 Retry-After honoured), listings 3 s apart, dataset zips one at a time.
cd /Users/asad.ali/kaggriculture
PY=/venv/main/bin/python
mkdir -p data/kaggle_daily data/tapes_kd il/data/fresh_all
echo "[$(date -u +%H:%M) UTC] start"
THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_top12.log 2>&1 &
FETCH=$!
for d in 2026-09-24 2026-09-23 2026-09-22 2026-09-21 2026-09-20 2026-09-19 2026-09-18; do
  z=data/kaggle_daily/kaggriculture-episodes-$d.zip
  for a in 1 2 3 4 5 6; do
    $PY -c "import zipfile, sys; sys.exit(0 if zipfile.is_zipfile('$z') else 1)" 2>/dev/null && break
    curl -s -L --retry 3 --retry-delay 30 -o $z https://www.kaggle.com/api/v1/datasets/download/kaggle/kaggriculture-episodes-$d
    sleep 15
  done
  echo "[$(date -u +%H:%M) UTC] $d zip $(du -h $z 2>/dev/null | cut -f1)"
  sleep 5
done
wait $FETCH
echo "[$(date -u +%H:%M) UTC] top-12: $(grep -c 'completed episodes' il/logs/fetch_top12.log) listings; $(tail -1 il/logs/fetch_top12.log)"
for d in 2026-09-24 2026-09-23 2026-09-22 2026-09-21 2026-09-20 2026-09-19 2026-09-18; do
  $PY il/live/srv_import_day.py data/kaggle_daily/kaggriculture-episodes-$d.zip $d data/tapes_kd/$d 8 2>&1 | tail -2
done
# games already built for the union: identical seat files, so link them instead of rebuilding
$PY - <<'PY'
import json, os, glob
R = "/Users/asad.ali/kaggriculture"; eps = set()
for d in [f"{R}/data/tapes_top12"] + sorted(glob.glob(f"{R}/data/tapes_kd/*")):
    eps |= {int(e) for e in json.load(open(f"{d}/index.json"))}
n = 0
for f in glob.glob(f"{R}/il/data/union/*.npz"):
    b = os.path.basename(f)
    if int(b.split("_")[0]) in eps and not os.path.lexists(f"{R}/il/data/fresh_all/{b}"):
        os.symlink(f, f"{R}/il/data/fresh_all/{b}"); n += 1
print(f"{len(eps)} fresh games; union seats reused: {n}", flush=True)
PY
P1="data/tapes_top12 data/tapes_kd/2026-09-24 data/tapes_kd/2026-09-23 data/tapes_kd/2026-09-22 data/tapes_kd/2026-09-21"
P2="data/tapes_kd/2026-09-20 data/tapes_kd/2026-09-19 data/tapes_kd/2026-09-18"
build() {   # $1 tape dirs  $2 workers  $3 nice  $4 log tag
  for w in $(seq 0 $(($2 - 1))); do nice -n $3 $PY il/build_dataset3.py il/data/fresh_all $w $2 $1 > il/logs/build_$4_w$w.log 2>&1 & done; wait
  echo "[$(date -u +%H:%M) UTC] $4 seats written: $(tail -q -n1 il/logs/build_$4_w*.log | awk '{s+=$8} END {print s}'), non-exact/failed: $(tail -q -n1 il/logs/build_$4_w*.log | awk '{s+=$10} END {print s}')"
  for w in $(seq 0 $(($2 - 1))); do TAPES=$(echo $1 | tr ' ' ':') SEATS=il/data/fresh_all nice -n $3 $PY il/live/srv_build_x6.py $w $2 > il/logs/x6_$4_w$w.log 2>&1 & done; wait
  echo "[$(date -u +%H:%M) UTC] $4 v6 features: $(tail -q -n1 il/logs/x6_$4_w*.log | awk '{s+=$5} END {print s}') new, missing tape $(tail -q -n1 il/logs/x6_$4_w*.log | awk '{s+=$8} END {print s}')"
}
build "$P1" 14 5 fresh
$PY il/live/srv_fresh_split.py fresh $P1
echo "[$(date -u +%H:%M) UTC] FRESH READY: $(ls il/data/fresh | wc -l) seats in il/data/fresh"
build "$P2" 8 15 fresh7
$PY il/live/srv_fresh_split.py fresh7 $P1 $P2
echo "[$(date -u +%H:%M) UTC] FRESH7 READY: $(ls il/data/fresh7 | wc -l) seats in il/data/fresh7"
