"""Sep 30: animal leaks from the encoder's blind spots, measured on our live games vs 2600+ (ALLG + S, last 2 days), both seats, exact replay.
Per animal-day at the day's first turn: stock at the cap (cow/sheep 6, goose 4 -> the next production is wasted), care bank > 0 left unpaid
(production day passed unfed), and escapes. Rules: cow every 2 days from day 8, sheep every 3 from day 6, goose daily from day 4."""
import sys, json, gzip, glob, collections, statistics as St
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
K = "/Users/asad.ali/kaggriculture"; CAP = {"GOOSE": 4, "COW": 6, "SHEEP": 6}
tot = [collections.Counter(), collections.Counter()]; n = 0
for sub, d in ((56661802, "tapes_allg"), (56651505, "tapes_all")):
    rows = {int(r["ep"]): r for r in json.load(open(f"{K}/data/rows_{sub}.json")) if r["res"] != "ERR"}
    for ep, r in rows.items():
        if (r.get("opp_rating") or 0) < 2600: continue
        try: t = json.load(gzip.open(f"{K}/data/{d}/{ep}.json.gz", "rt"))
        except Exception: continue
        us = t["seat"]; A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][s]] for s in (0, 1)]
        g = kagsim_trace.Game(int(t["seed"]), 720, False); n += 1
        for step in range(720):
            if step % 24 == 0 and step > 0:
                o = g.observe(us)
                for side, who in ((us, 0), (1 - us, 1)):
                    for row in o["farms"][side]["tiles"]:
                        for tl in row:
                            if isinstance(tl, dict) and tl.get("animal") in CAP:
                                a = tl["animal"]; c = tot[who]
                                c[f"{a} days"] += 1
                                if int(tl.get("yield_units", 0)) >= CAP[a]: c[f"{a} at cap"] += 1
                                if int(tl.get("pending_care_bonus", 0)) >= 2: c[f"{a} bank>=2"] += 1
            g.step(A[0][step], A[1][step])
            if g.done: break
print(f"{n} games vs 2600+ (ALL + ALLG). Per game, us vs them:")
for a in ("COW", "SHEEP", "GOOSE"):
    u, th = tot[0], tot[1]
    print(f"  {a:5s}: animal-days {u[a+' days']/n:5.1f} vs {th[a+' days']/n:5.1f} | days AT CAP {u[a+' at cap']/n:4.1f} ({100*u[a+' at cap']/max(1,u[a+' days']):.0f}%) vs "
          f"{th[a+' at cap']/n:4.1f} ({100*th[a+' at cap']/max(1,th[a+' days']):.0f}%) | care bank >=2 {u[a+' bank>=2']/n:4.1f} vs {th[a+' bank>=2']/n:4.1f}")
