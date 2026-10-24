"""Specialist opponents for the JAX league (operator-approved plan, Sep 27): our recorded live games against players who
beat us or are rated 2,400+ (elmo, Aaweg, Yannik Schiffner, SpaTaro, lingxiaojun, yarneo, Snorlax, ...) converted to the
leader-tape format kagjax/train/tapes.py reads (dir with index.json + <ep>.json.gz: ep, seed, teams, rewards, statuses,
shops, actions, day). Only the OPPONENT seat becomes a league opponent (its team name is listed in teams.txt, to be
joined into TOP_TEAMS); our own seat's name is never listed. tapes.py still replays every tape in C++ and keeps only
those that reproduce the recorded rewards exactly, and skips all benchmark episodes.
Usage: make_specialist_tapes.py [min_opp_rating=2400] [out_dir=data/tapes_spec]"""
import sys, os, json, gzip, glob
K = "/Users/asad.ali/kaggriculture"
MINR = float(sys.argv[1]) if len(sys.argv) > 1 else 2400
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{K}/data/tapes_spec"
DIRS = ["tapes_n20mw5", "tapes_n20mw", "tapes_n10md", "tapes_n10mr", "tapes_nlive", "tapes_neural", "tapes_n10m",
        "tapes_n4m", "tapes_race", "tapes_stack1"]
os.makedirs(OUT, exist_ok=True)
index, names, seen, kept = {}, {}, set(), 0
for d in DIRS:
    for f in sorted(glob.glob(f"{K}/data/{d}/*.json.gz")):
        try:
            t = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        ep = int(t.get("ep", 0))
        if not ep or ep in seen or t.get("res") not in ("W", "L"):
            continue
        seen.add(ep)
        if not (t["res"] == "L" or (t.get("opp_rating") or 0) >= MINR):
            continue
        s = int(t["seat"]); nm = list(t.get("names") or ["?", "?"])
        opp = nm[1 - s] or t.get("opp") or "?"
        if opp in ("?", "Syed Asad Ali"):
            continue
        teams = ["us:" + (nm[s] or "?"), opp] if s == 0 else [opp, "us:" + (nm[s] or "?")]
        out = dict(ep=ep, seed=int(t["seed"]), teams=teams, rewards=t.get("rewards"), statuses=["DONE", "DONE"], shops=[],
                   actions=t["actions"], day=t.get("end", "")[:10] or None)
        json.dump(out, gzip.open(f"{OUT}/{ep}.json.gz", "wt"), separators=(",", ":"))
        index[str(ep)] = dict(teams=teams, rewards=t.get("rewards"), statuses=["DONE", "DONE"], day=out["day"], src=d,
                              res_vs_us=t["res"], opp_rating=t.get("opp_rating"))
        names[opp] = names.get(opp, 0) + 1; kept += 1
json.dump(index, open(f"{OUT}/index.json", "w"))
open(f"{OUT}/teams.txt", "w").write("|".join(sorted(names)))
print(f"{kept} specialist seats from {len(seen)} live games -> {OUT}; {len(names)} distinct opponents")
for n, c in sorted(names.items(), key=lambda x: -x[1])[:25]: print(f"  {c:3d}  {n}")
