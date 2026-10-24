#!/bin/bash
# Poll STACK1_RACE5_2015 (56547100) every 5 minutes: refresh its episode list, tape new games into data/tapes_race5b,
# print one line per NEW finished game (W/L/T, opponent, ratings, margin) -> il/logs/poll_s5b.log
cd /Users/asad.ali/kaggriculture
SUB=56547100; SEEN=il/logs/poll_s5b_seen.txt; touch $SEEN
while true; do
  nice -n 10 .venv/bin/python tools/pull.py STACK1_RACE5_2015 $SUB > il/logs/poll_s5b_pull.out 2>&1 || true
  nice -n 10 .venv/bin/python tools/tape_sub.py $SUB data/tapes_race5b > il/logs/poll_s5b_tape.out 2>&1 || true
  nice -n 10 .venv/bin/python - <<'PY' || true
import json, datetime
R = "/Users/asad.ali/kaggriculture"
rows = json.load(open(f"{R}/data/rows_56547100.json"))
seen = set(open(f"{R}/il/logs/poll_s5b_seen.txt").read().split())
ok = [r for r in rows if r["res"] in ("W", "L", "T")]
new = [r for r in ok if str(r["ep"]) not in seen]
w = sum(x["res"] == "W" for x in ok); l = sum(x["res"] == "L" for x in ok)
for r in new:
    tag = "LOSS" if r["res"] == "L" else ("win " if r["res"] == "W" else "tie ")
    print(f"{tag} ep {r['ep']} {r['end']} vs {r['opp']} ({r['opp_rating']:.0f}) {r['us_reward']:,.0f}-{r['th_reward']:,.0f} margin {r['margin']:+,.0f}", flush=True)
    open(f"{R}/il/logs/poll_s5b_seen.txt", "a").write(f"{r['ep']}\n")
if ok:
    print(f"[{datetime.datetime.now():%H:%M}] record {w}-{l} of {len(ok)}, rating {ok[-1]['rating']:.0f}, errors {len(rows)-len(ok)}", flush=True)
PY
  sleep 300
done
