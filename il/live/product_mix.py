# Revenue by product (and animals bought) for groups of seats in exact replays: our neural bot's live games, STACK1_RACE's
# live games, the opponents who beat the neural bot, and top-10 leaders in the daily top games.
import sys, os, json, gzip, glob, random, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
R = "/Users/asad.ali/kaggriculture"
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
LEAD = ["Boey", "M & M & P & Q", "nah id win", "DECEM", "Unknown Mother-Goose", "Vadim Vasilenko", "吃白饭的大肥鱼",
        "THIRD FARM CLUB", "Kaggledew Valley 🏆", "mtmr_s1"]


def one(item):
    from harness import kagsim_trace, PASS
    path, seats = item
    d = json.load(gzip.open(path, "rt")); A = d["actions"]
    g = kagsim_trace.Game(int(d["seed"]), 720, True)
    rev = [collections.Counter(), collections.Counter()]; buy = [collections.Counter(), collections.Counter()]
    for t in range(720):
        a0 = A[0][t] if isinstance(A[0][t], dict) else dict(PASS); a1 = A[1][t] if isinstance(A[1][t], dict) else dict(PASS)
        g.step(a0, a1)
        for rc in g.trace_step(g.trace_len() - 1).get("receipts", []):
            ex = rc.get("executed", 0) or 0
            if not ex:
                continue
            s = rc["seat"]; op = rc.get("op"); item = rc.get("item")
            if op in (6, "SELL"):
                rev[s][PRODUCTS[item] if isinstance(item, int) and item < 9 else str(item)] += (rc.get("money_after") or 0) - (rc.get("money_before") or 0)
            elif op in (5, "BUY_ANIMAL"):
                buy[s][str(item)] += ex
        if g.done:
            break
    return [(s, dict(rev[s]), dict(buy[s]), g.reward(s)) for s in seats]


def groups():
    G = collections.defaultdict(list)
    for fn in glob.glob(f"{R}/data/tapes_neural/*.json.gz"):
        d = json.load(gzip.open(fn, "rt")); G["NEURAL_RL30 (live)"].append((fn, [d["seat"]]))
        if d["res"] == "L":
            G["opponents who beat NEURAL_RL30"].append((fn, [1 - d["seat"]]))
    for fn in glob.glob(f"{R}/data/tapes_race/*.json.gz"):
        d = json.load(gzip.open(fn, "rt")); G["STACK1_RACE (live)"].append((fn, [d["seat"]]))
    idx = json.load(open(f"{R}/data/tapes_daily/index.json")); rng = random.Random(7); eps = sorted(idx); rng.shuffle(eps)
    for ep in eps:
        t = idx[ep].get("teams") or []
        seats = [s for s in (0, 1) if s < len(t) and t[s] in LEAD]
        if seats and os.path.exists(f"{R}/data/tapes_daily/{ep}.json.gz") and len(G["top-10 leaders (daily top games)"]) < 160:
            G["top-10 leaders (daily top games)"].append((f"{R}/data/tapes_daily/{ep}.json.gz", seats))
    return G


if __name__ == "__main__":
    G = groups()
    with mp.get_context("spawn").Pool(8) as pool:
        for name, items in G.items():
            res = [x for r in pool.map(one, items) for x in r]
            n = len(res); tot = np.mean([sum(r[1].values()) for r in res]); bank = np.mean([r[3] for r in res])
            print(f"\n{name}: {n} seats, mean sales ${tot:,.0f}/game, mean final ${bank:,.0f}")
            print("  sales by product ($/game, share): " + ", ".join(f"{p} {np.mean([r[1].get(p, 0) for r in res]):,.0f} ({np.mean([r[1].get(p, 0) for r in res]) / tot:.0%})" for p in PRODUCTS))
            an = collections.Counter()
            for r in res:
                an.update(r[2])
            print("  animals bought per game: " + ", ".join(f"{k} {v / n:.1f}" for k, v in an.most_common()))
