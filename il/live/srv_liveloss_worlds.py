"""Live-loss replay worlds (Sep 27 night, operator: "keep building counters to our both current scripts' weakness as they
arrive on the live ladder"): every recorded LIVE LOSS of the two active submissions - NEURAL_20MW5_SE (tapes_n20mw5) and
NEURAL_20MW8_SE (tapes_n20mw8), appended to data/tapes_spec by srv_spec_add.py - as top_bench worlds (our seat replayed by
the candidate, the opponent's recorded moves on tape): il/live/top_worlds_livelosses_300.json (top_bench DAY=livelosses).
Prints the count and a short hash of the episode set (the tests re-run the v13_p3 baseline when it changes).
Usage: srv_liveloss_worlds.py"""
import json, os, hashlib
R = "/Users/asad.ali/kaggriculture"
idx = json.load(open(f"{R}/data/tapes_spec/index.json")); W = []
for ep, v in sorted(idx.items()):
    if v.get("src") not in ("tapes_n20mw5", "tapes_n20mw8") or v.get("res_vs_us") != "L":
        continue
    t = v["teams"]; s = 0 if str(t[0]).startswith("us:") else 1; rw = v["rewards"]; p = f"{R}/data/tapes_spec/{ep}.json.gz"
    if os.path.exists(p) and rw and None not in rw:
        W.append(dict(path=p, ep=int(ep), seat=s, replaced=t[s], opp=t[1 - s], rec_margin=rw[s] - rw[1 - s], rec_opp=rw[1 - s],
                      opp_rating=v.get("opp_rating"), src=v.get("src")))
json.dump(W, open(f"{R}/il/live/top_worlds_livelosses_300.json", "w"), ensure_ascii=False)
h = hashlib.sha1(",".join(str(w["ep"]) for w in W).encode()).hexdigest()[:10]
print(f"{len(W)} {h}")
