#!/bin/bash
# New server, Sep 26 (while the operator uploads the data bundle): C++ engine build + parity check on recorded games,
# the Sep 25 daily top games -> compact tapes (streamed from the zip), and the top-10 leaders' newest games (the
# operator's submission ids + newer ones found in the rows files).  The refresh chain re-runs these steps later and
# skips everything already done.
K=/Users/asad.ali/kaggriculture; PY=/venv/main/bin/python
log() { echo "[$(date -u +%H:%M) UTC] $*"; }
cd $K/engine/src_v2
EXT=$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
c++ -O3 -std=c++17 -fvisibility=hidden -DNDEBUG -fPIC $($PY -m pybind11 --includes) -I. -shared kagsim_trace.cpp -o ../v2/kagsim_trace$EXT \
  && log "engine built: kagsim_trace$EXT"
cd $K/il && $PY live/engine_parity.py $K/data/tapes_top12 30
cd $K
$PY il/live/srv_import_day.py data/kaggle_daily/kaggriculture-episodes-2026-09-25.zip 2026-09-25 data/tapes_kd/2026-09-25 8 2>&1 | tail -2
log "Sep 25 imported"
THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_top12_b.log 2>&1
log "top-10 fetch: $(grep -c 'completed episodes' il/logs/fetch_top12_b.log) listings; $(tail -1 il/logs/fetch_top12_b.log)"
$PY il/live/srv_discover_subs.py il/logs/new_subs.json
if [ "$($PY -c "import json; print(sum(len(v) for v in json.load(open('il/logs/new_subs.json')).values()))")" -gt 0 ]; then
  SUBS_JSON=il/logs/new_subs.json THREADS=12 $PY il/live/srv_fetch_top12.py > il/logs/fetch_top12_c.log 2>&1
  log "newer leader submissions fetched: $(tail -1 il/logs/fetch_top12_c.log)"
fi
log "PREFETCH DONE: tapes_top12 $(ls data/tapes_top12 | wc -l) files"
