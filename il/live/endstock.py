# Goods left unsold at the end of the game (shed + unit inventories at the last observation, valued at market prices).
import sys, json, gzip, glob, random, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
from product_mix import LEAD, R, PRODUCTS


def one(item):
    from harness import kagsim_trace, PASS
    path, seats = item
    d = json.load(gzip.open(path, "rt")); A = d["actions"]
    g = kagsim_trace.Game(int(d["seed"]), 720, False); last = None
    for t in range(720):
        last = [g.observe(0), g.observe(1)]
        g.step(*[A[s][t] if isinstance(A[s][t], dict) else dict(PASS) for s in (0, 1)])
        if g.done:
            break
    out = []
    for s in seats:
        o = last[s]; pr = o["market"]["prices"]; stock = collections.Counter({k: int(v) for k, v in o["private"]["shed"].items() if k in PRODUCTS})
        for inv in o["private"].get("inventories") or []:
            stock.update({k: int(v) for k, v in inv.items() if k in PRODUCTS})
        out.append((sum(stock.values()), sum(n * pr.get(k, 0) for k, n in stock.items()), g.reward(s), o["farms"][s]["money"]))
    return out


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
            res = [x for r in pool.map(one, items) for x in r]
            print(f"{name}: units left at the last turn {np.mean([r[0] for r in res]):.1f} worth ${np.mean([r[1] for r in res]):,.0f}; "
                  f"final reward ${np.mean([r[2] for r in res]):,.0f} vs money before last turn ${np.mean([r[3] for r in res]):,.0f}")
