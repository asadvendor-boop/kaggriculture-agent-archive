#!/bin/bash
# Sep 28 15:25 PKT: fetch pass 1 (hop 1 + hop 2) -> pass 2 (the operator's current top-4 submission ids, il/live/today_top4.json)
# -> the fresh3 test runs (srv_fresh3_runs.sh) on the final world file.
K=/Users/asad.ali/kaggriculture; cd $K/il
while ! grep -q "FRESH3 READY\|Traceback" /root/fetch_today.log; do sleep 20; done
HOP2=9999 THREADS=12 /venv/main/bin/python live/srv_fetch_today.py live/today_top4.json > /root/fetch_today2.log 2>&1
grep -q "FRESH3 READY" /root/fetch_today2.log && bash live/srv_fresh3_runs.sh > /root/fresh3_runs.out 2>&1
