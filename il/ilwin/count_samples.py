"""Light census (JSON headers only, no engine): winner games per tape dir after build_winner_ds.py's filters."""
import json, gzip, glob, os, collections, sys
K = "/Users/asad.ali/kaggriculture"
held = {int(w["ep"]) for w in json.load(open(f"{K}/il/live/ladder_worlds.json"))}
for f in glob.glob(f"{K}/il/live/top_worlds_*_300.json"):
    if "llall" in f: continue
    held |= {int(w["ep"]) for w in json.load(open(f))}
held |= set(int(x) for x in json.load(open(f"{K}/il/live/fresh_heldout_eps.json")))
ll = {int(w["ep"]) for w in json.load(open(f"{K}/il/live/top_worlds_llall_300.json"))}
def census(minr, minm):
    rows = []
    for d in ["tapes_st2", "tapes_soh", "tapes_n20mw8", "tapes_npse", "tapes_nps", "tapes_n20mw5", "tapes_neural"]:
        c = collections.Counter()
        for f in glob.glob(f"{K}/data/{d}/*.json.gz"):
            x = json.load(gzip.open(f, "rt")); ep = int(x["ep"]); us = x["seat"]; rw = x["rewards"]
            if rw[0] is None or rw[1] is None: continue
            if ep in held: c["held"] += 1; continue
            if abs(rw[0] - rw[1]) < minm: c["tie"] += 1; continue
            if (x.get("opp_rating") or 0) < minr: c["lowrated"] += 1; continue
            won = rw[us] > rw[1 - us]
            c["W" if won else "L"] += 1
            if not won and ep in ll: c["L_in_llall"] += 1
        rows.append((d, c))
    return rows
for minr, minm in ((0, 0), (2400, 500)):
    print(f"== filters: opp rating >= {minr}, |margin| >= {minm}")
    tw = tl = 0
    for d, c in census(minr, minm):
        print(f"  {d:13s} W {c['W']:4d}  L {c['L']:3d}  (held-out ruler games {c['held']}, near ties {c['tie']}, low-rated {c['lowrated']}, losses in llall {c['L_in_llall']})")
