#!/bin/bash
# Resume the daily downloads gently once Kaggle's 429 throttle clears: probe every 5 min, then 2 parts x 8 threads.
cd /Users/asad.ali/kaggriculture
while true; do
  code=$(curl -s -L -o /dev/null -w "%{http_code}" -H "Accept-Encoding: gzip" https://www.kaggleusercontent.com/episodes/112219773.json)
  echo "[$(date +%H:%M)] probe HTTP $code"
  [ "$code" = "200" ] && break
  sleep 120
done
for k in 0 1 2 3; do THREADS=8 nice -n 5 .venv/bin/python il/live/fetch_part.py data/tapes_daily/episodes.json data/tapes_daily $k 4 > il/logs/fetch_daily4_part$k.log 2>&1 & done; wait
.venv/bin/python il/live/merge_daily_index.py
echo "[$(date +%H:%M)] daily downloads done"
