"""One line per NEW finished game of a live submission (W/L, opponent, ratings, margin) + a record line; the seen list
lives in il/logs/poll_<label>_seen.txt.   Usage: poll_report.py <sub> <label>"""
import sys, json, datetime
R = "/Users/asad.ali/kaggriculture"; sub, lab = int(sys.argv[1]), sys.argv[2]
rows = json.load(open(f"{R}/data/rows_{sub}.json")); fn = f"{R}/il/logs/poll_{lab}_seen.txt"
try:
    seen = set(open(fn).read().split())
except FileNotFoundError:
    seen = set()
ok = [r for r in rows if r["res"] in ("W", "L", "T")]
for r in ok:
    if str(r["ep"]) in seen:
        continue
    tag = "LOSS" if r["res"] == "L" else ("win " if r["res"] == "W" else "tie ")
    print(f"{lab} {tag} ep {r['ep']} {r['end']} vs {r['opp']} ({r['opp_rating']:.0f}) {r['us_reward']:,.0f}-{r['th_reward']:,.0f} margin {r['margin']:+,.0f}", flush=True)
    open(fn, "a").write(f"{r['ep']}\n")
if ok:
    w = sum(x["res"] == "W" for x in ok); l = sum(x["res"] == "L" for x in ok)
    lf = f"{R}/il/logs/poll_{lab}_last.txt"; key = f"{w}-{l}-{len(rows)}"
    try:
        same = open(lf).read().strip() == key
    except FileNotFoundError:
        same = False
    if same:                                   # print the record only when it changed (no repeats every 5 minutes)
        sys.exit(0)
    open(lf, "w").write(key)
    print(f"[{datetime.datetime.now(datetime.timezone.utc):%H:%M} UTC] {lab} record {w}-{l} of {len(ok)}, rating {ok[-1]['rating']:.0f}, errors {len(rows) - len(ok)}", flush=True)
