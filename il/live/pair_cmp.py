"""Paired comparison of two runs on the same ladder worlds (il/logs/ladder_<tag>_test.json files, or top_bench JSONs
with the same row format): wins each, flips both ways, mean margin difference with its standard error and z, and the
difference by opponent-rating bucket.   Usage: pair_cmp.py <base.json> <cand.json> [<cand2.json> ...]"""
import sys, json
import numpy as np


def load(fn):
    R = json.load(open(fn))
    return {(r.get("ep"), r.get("opp")): r for r in R}


base = load(sys.argv[1])
for fn in sys.argv[2:]:
    c = load(fn); keys = [k for k in base if k in c]
    b = np.array([base[k]["margin"] for k in keys]); m = np.array([c[k]["margin"] for k in keys]); d = m - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    up = int(((b <= 0) & (m > 0)).sum()); down = int(((b > 0) & (m <= 0)).sum())
    line = (f"{fn.split('/')[-1][:60]:60s} n {len(keys)} | wins {int((b > 0).sum())} -> {int((m > 0).sum())} (flips +{up} -{down})"
            f" | margin {b.mean():+,.0f} -> {m.mean():+,.0f}, diff {d.mean():+,.0f} se {se:,.0f} z {d.mean() / se if se else 0:+.1f}")
    rat = np.array([float(base[k].get("opp_rating") or 0) for k in keys])
    parts = []
    for lo, hi in ((0, 2000), (2000, 2400), (2400, 9999)):
        s = (rat >= lo) & (rat < hi)
        if s.any():
            parts.append(f"{lo}+: {int((b[s] > 0).sum())}->{int((m[s] > 0).sum())}/{int(s.sum())} {d[s].mean():+,.0f}")
    print(line + " | " + " ".join(parts), flush=True)
