#!/bin/bash
# Sep 29: fetch (public endpoints) the repeat opponents' last-3-days games (R) and the top-10 leaders' last-day games (T). No hop 2.
cd /Users/asad.ali/kaggriculture/il; PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] fetch R start"
OUTD=tapes_rep29 OUTW=llall_rep29 SINCE=2026-09-26T11:00 MINR=2400 HOP2=99999 THREADS=6 MAXF=900 N=2 nice -n 15 $PY live/srv_fetch_today.py live/rep29_subs.json > logs/fetch_rep29.log 2>&1
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] fetch R done: $(tail -1 logs/fetch_rep29.log | cut -c1-150)"
OUTD=tapes_top10d OUTW=llall_top10d SINCE=2026-09-28T11:00 MINR=2700 HOP2=99999 THREADS=6 MAXF=900 N=2 nice -n 15 $PY live/srv_fetch_today.py live/top10_newest_subs.json > logs/fetch_top10d.log 2>&1
echo "[$(TZ=Asia/Karachi date +%H:%M) PKT] fetch T done: $(tail -1 logs/fetch_top10d.log | cut -c1-150)"
