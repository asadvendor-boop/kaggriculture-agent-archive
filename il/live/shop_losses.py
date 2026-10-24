"""Check an outside claim (Kimi, Sep 26): "all 19 losses live in ICE_CREAM / SMOOTHIE / PIZZA worlds". Replays every taped live
game of our neural submissions exactly and reads the shops the town unlocked; win rate with / without each shop."""
import sys, json, gzip, glob, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
R = "/Users/asad.ali/kaggriculture"
for lab, d_ in (("NEURAL_10M_S4", "tapes_n10m"), ("NEURAL_4M_S4", "tapes_n4m")):
    games = []
    for f in sorted(glob.glob(f"{R}/data/{d_}/*.json.gz")):
        t = json.load(gzip.open(f, "rt"))
        if t.get("res") not in ("W", "L"):
            continue
        A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][k]] for k in (0, 1)]
        g = kagsim_trace.Game(int(t["seed"]), 720, False)
        for s in range(720):
            g.step(A[0][s], A[1][s])
            if g.done:
                break
        shops = set(g.observe(0)["town"].get("unlocked_shops") or [])
        games.append((t["res"], shops))
    L = [s for r, s in games if r == "L"]; n = len(games)
    print(f"== {lab}: {n} games, {len(L)} losses")
    for shop in sorted({x for _, s in games for x in s}):
        w_in = [r for r, s in games if shop in s]; w_out = [r for r, s in games if shop not in s]
        print(f"  {shop:14s} present in {len(w_in):3d} games: loss rate {sum(r == 'L' for r in w_in) / max(1, len(w_in)):5.1%} | absent: {sum(r == 'L' for r in w_out) / max(1, len(w_out)):5.1%}")
    k = sum(1 for s in L if s & {"ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "PIZZA_SHOP"})
    b = sum(1 for _, s in games if s & {"ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "PIZZA_SHOP"})
    print(f"  losses with ICE_CREAM/SMOOTHIE/PIZZA: {k}/{len(L)}; games with any of them: {b}/{n}")
