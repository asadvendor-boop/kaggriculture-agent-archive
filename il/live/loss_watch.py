"""Live-loss watcher (operator, Sep 26: "keep analyzing losses of our current live submissions, so we can counter those in
our current builds").  For every NEW loss of the live submissions (rows files kept current by poll_neural4.sh) whose tape
is saved: exact C++ replay of both seats -> one line: opponent, rating, margin, our land days / sheep / cows vs theirs,
whether the third field was skipped with the cash there (SE-guard case), sheep >= 16 before the third field (sheep-cap
case), and the products where the opponent out-earned us most.  Then running tallies over all analysed losses.
State: il/logs/loss_watch_seen.json.   Usage: loss_watch.py <sub>:<tape_dir> [...]"""
import sys, os, json, gzip, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from dsm_likeness import profile
from land_sheep_diag import one
K = "/Users/asad.ali/kaggriculture"; ST = f"{K}/il/logs/loss_watch_seen.json"
PROD = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
seen = json.load(open(ST)) if os.path.exists(ST) else {}
for arg in sys.argv[1:]:
    sub, tdir = arg.split(":")
    rows = json.load(open(f"{K}/data/rows_{sub}.json"))
    for r in rows:
        k = str(r["ep"])
        if r.get("res") != "L" or k in seen:
            continue
        f = f"{tdir}/{r['ep']}.json.gz"
        if not os.path.exists(f):
            continue                                       # tape not saved yet: next round
        t = json.load(gzip.open(f, "rt")); s = int(t["seat"])
        a, b = profile((f, s)), profile((f, 1 - s)); ls = one((f,))
        if not a or not b or not ls:
            continue
        u, v = a["P"], b["P"]
        gap = sorted(((v[f"rev:{p}"] - u[f"rev:{p}"], p) for p in PROD), reverse=True)[:3]
        se_case = len(ls["lands"]) < 3 and ls["trig"] is not None and ls["trig"][0] <= 13
        cap_case = ls["sheep_before3"] >= 16
        seen[k] = dict(sub=sub, opp=r["opp"], orat=r["opp_rating"], margin=r["margin"], lands=ls["lands"], th_lands=b["land_days"],
                       sheep=ls["sheep_tot"], th_sheep=v["animal:SHEEP"], cows=u["animal:COW"], th_cows=v["animal:COW"],
                       se_case=se_case, cap_case=cap_case, gap=[(p, round(g)) for g, p in gap if g > 0], end=r["end"])
        x = seen[k]
        print(f"LOSS {sub} {x['end'][5:]} vs {x['opp'][:20]:20s} ({x['orat']:.0f}) {x['margin']:+8,.0f} | our land {x['lands']} sheep {x['sheep']} cows {x['cows']}"
              f" | theirs land {x['th_lands']} sheep {x['th_sheep']} cows {x['th_cows']} | {'SE-SKIPPED ' if se_case else ''}{'SHEEP16+ ' if cap_case else ''}"
              f"| they out-earned us on " + ", ".join(f"{p} +${g:,}" for p, g in x["gap"]), flush=True)
json.dump(seen, open(ST, "w"))
L = list(seen.values())
if L:
    c = collections.Counter(p for x in L for p, _ in x["gap"][:1])
    print(f"TALLY {len(L)} analysed losses: third field skipped with cash {sum(x['se_case'] for x in L)}, sheep 16+ before it "
          f"{sum(x['cap_case'] for x in L)}, vs 2400+ {sum(x['orat'] >= 2400 for x in L)}; biggest gap product: {dict(c.most_common(4))}", flush=True)
