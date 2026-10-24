"""Sep 28: every recorded LIVE LOSS of our submissions, straight from the poller's tape dirs (no specialist bank needed):
data/tapes_nps (NEURAL_20MW8NPS_SOH, 56627684), data/tapes_n20mw8, data/tapes_n20mw5 -> top_bench worlds (our seat replayed
by the candidate, the opponent's recorded moves on tape): il/live/top_worlds_llall_300.json (top_bench DAY=llall).
Prints the count per source and a short hash of the episode set.   Usage: liveloss_all.py [dir ...]"""
import sys, json, gzip, glob, os, hashlib, collections
R = "/Users/asad.ali/kaggriculture"
dirs = sys.argv[1:] or ["data/tapes_st2", "data/tapes_npse", "data/tapes_soh", "data/tapes_nps", "data/tapes_n20mw8", "data/tapes_n20mw5"]
W = []; n = collections.Counter()
for d in dirs:
    for p in sorted(glob.glob(f"{R}/{d}/*.json.gz")):
        try:
            t = json.load(gzip.open(p, "rt"))
        except Exception:
            continue
        if t.get("res") != "L" or not t.get("rewards") or None in t["rewards"]:
            continue
        s = int(t["seat"]); rw = t["rewards"]
        W.append(dict(path=p, ep=int(t["ep"]), seat=s, replaced=(t.get("names") or ["us", "them"])[s], opp=t.get("opp"),
                      rec_margin=rw[s] - rw[1 - s], rec_opp=rw[1 - s], opp_rating=t.get("opp_rating"), src=os.path.basename(d)))
        n[os.path.basename(d)] += 1
json.dump(W, open(f"{R}/il/live/top_worlds_llall_300.json", "w"), ensure_ascii=False)
h = hashlib.sha1(",".join(str(w["ep"]) for w in W).encode()).hexdigest()[:10]
print(f"{len(W)} live losses {dict(n)} {h}")
