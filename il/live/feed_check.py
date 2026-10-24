# Unfed animal-days by game day (animal on the farm, not fed by the day's end) and wheat bought per game, for seats in
# exact replays: NEURAL_RL30 live games, STACK1_RACE live games, top-10 leaders in daily top games.
import sys, os, json, gzip, glob, random, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
from product_mix import LEAD, R


def one(item):
    from harness import kagsim_trace, PASS
    path, seats = item
    d = json.load(gzip.open(path, "rt")); A = d["actions"]
    g = kagsim_trace.Game(int(d["seed"]), 720, True)
    unfed = {s: collections.Counter() for s in seats}; animals = {s: collections.Counter() for s in seats}
    wheat = {s: 0 for s in seats}; wheat_cost = {s: 0.0 for s in seats}
    for t in range(720):
        a = [A[s][t] if isinstance(A[s][t], dict) else dict(PASS) for s in (0, 1)]
        pre = g.observe(0) if t % 24 == 23 else None
        g.step(a[0], a[1])
        ts = g.trace_step(g.trace_len() - 1)
        for rc in ts.get("receipts", []):
            if rc["seat"] in seats and rc.get("op") == "BUY_PRODUCT" and rc.get("item") == "WHEAT" and rc.get("executed"):
                wheat[rc["seat"]] += rc["executed"]; wheat_cost[rc["seat"]] += rc["money_before"] - rc["money_after"]
        if pre is not None:
            fed_now = collections.defaultdict(set)
            for r in ts.get("unit_receipts", []):
                if r.get("executed") and r.get("op") == "FEED":
                    fed_now[r["seat"]].add(r.get("unit"))
            for s in seats:
                for row in pre["farms"][s]["tiles"]:
                    for tl in row:
                        if isinstance(tl, dict) and tl.get("animal"):
                            animals[s][t // 24] += 1
                            if not tl.get("fed_today"):
                                unfed[s][t // 24] += 1
        if g.done:
            break
    return [(unfed[s], animals[s], wheat[s], wheat_cost[s]) for s in seats]


if __name__ == "__main__":
    G = collections.defaultdict(list)
    for fn in glob.glob(f"{R}/data/tapes_neural/*.json.gz"):
        G["NEURAL_RL30"].append((fn, [json.load(gzip.open(fn, "rt"))["seat"]]))
    for fn in sorted(glob.glob(f"{R}/data/tapes_race/*.json.gz"))[:60]:
        G["STACK1_RACE"].append((fn, [json.load(gzip.open(fn, "rt"))["seat"]]))
    idx = json.load(open(f"{R}/data/tapes_daily/index.json")); eps = sorted(idx); random.Random(7).shuffle(eps)
    for ep in eps:
        t = idx[ep].get("teams") or []; seats = [s for s in (0, 1) if s < len(t) and t[s] in LEAD]
        if seats and len(G["leaders"]) < 80:
            G["leaders"].append((f"{R}/data/tapes_daily/{ep}.json.gz", seats))
    with mp.get_context("spawn").Pool(8) as pool:
        for name, items in G.items():
            res = [x for r in pool.map(one, items) for x in r]; n = len(res)
            U = sum((r[0] for r in res), collections.Counter()); An = sum((r[1] for r in res), collections.Counter())
            bands = [(0, 9), (10, 19), (20, 25), (26, 29)]
            print(f"{name} ({n} seats): unfed share of animal-days by day band: " +
                  "  ".join(f"d{a}-{b}: {sum(U[d] for d in range(a, b + 1)) / max(1, sum(An[d] for d in range(a, b + 1))):.0%} ({sum(U[d] for d in range(a, b + 1)) / n:.1f}/game)" for a, b in bands) +
                  f" | wheat bought {np.mean([r[2] for r in res]):.0f} units ${np.mean([r[3] for r in res]):,.0f}/game")
