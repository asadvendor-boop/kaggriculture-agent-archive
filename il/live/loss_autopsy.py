"""Where do our live neural agents lose? For every live LOSS (and, for contrast, every win) of a submission: exact replay
of both seats (dsm_likeness.profile) + the town's shops; per loss: opponent, rating, margin, our land days / herd /
fertilizing / wheat bought vs the opponent's, and the products where the opponent out-earned us most. Then the patterns:
3rd field bought, sheep, BAKERY/YARN presence, revenue gaps by product in losses vs wins.
Usage: loss_autopsy.py <tape_dir> <rows.json>"""
import sys, json, gzip, glob, collections
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from dsm_likeness import profile
from harness import kagsim_trace, PASS
PROD = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
tdir, rows_fn = sys.argv[1], sys.argv[2]
rows = {r["ep"]: r for r in json.load(open(rows_fn))}
def shops(t):
    A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][k]] for k in (0, 1)]
    g = kagsim_trace.Game(int(t["seed"]), 720, False)
    for s in range(720):
        g.step(A[0][s], A[1][s])
        if g.done: break
    return set(g.observe(0)["town"].get("unlocked_shops") or [])
res = {"W": [], "L": []}
for f in sorted(glob.glob(f"{tdir}/*.json.gz")):
    t = json.load(gzip.open(f, "rt")); r = rows.get(t["ep"])
    if not r or r["res"] not in ("W", "L"): continue
    me = int(t["seat"]); a, b = profile((f, me)), profile((f, 1 - me))
    if not a or not b: continue
    res[r["res"]].append(dict(ep=t["ep"], opp=r["opp"], orat=r["opp_rating"], margin=r["margin"], us=a, th=b, shops=shops(t)))
L, W = res["L"], res["W"]
print(f"{len(W)} wins, {len(L)} losses")
print(f"\n{'opponent':22s} {'rating':>6s} {'margin':>8s} | ours: land  cows/sheep/geese fert wheat$ | theirs: land  cows/sheep/geese fert | they out-earned us most on")
for x in sorted(L, key=lambda x: x["margin"]):
    u, v = x["us"]["P"], x["th"]["P"]
    gap = sorted(((v[f"rev:{p}"] - u[f"rev:{p}"], p) for p in PROD), reverse=True)[:3]
    print(f"{x['opp'][:22]:22s} {x['orat']:6.0f} {x['margin']:+8,.0f} | {str(x['us']['land_days']):9s} {u['animal:COW']:2d}/{u['animal:SHEEP']:2d}/{u['animal:GOOSE']:2d} {u['fertilize']:4d} {u['wheat_bought$']:6,.0f} |"
          f" {str(x['th']['land_days']):9s} {v['animal:COW']:2d}/{v['animal:SHEEP']:2d}/{v['animal:GOOSE']:2d} {v['fertilize']:4d} | " + ", ".join(f"{p} +${g:,.0f}" for g, p in gap if g > 0))
def share(G, fn): return f"{np.mean([fn(x) for x in G]):.0%}" if G else "-"
print("\nPATTERNS (losses vs wins):")
print(f"  3rd field bought:        {share(L, lambda x: len(x['us']['land_days']) >= 3)} vs {share(W, lambda x: len(x['us']['land_days']) >= 3)}")
print(f"  opponent bought 3rd:     {share(L, lambda x: len(x['th']['land_days']) >= 3)} vs {share(W, lambda x: len(x['th']['land_days']) >= 3)}")
print(f"  our sheep >= 10:         {share(L, lambda x: x['us']['P']['animal:SHEEP'] >= 10)} vs {share(W, lambda x: x['us']['P']['animal:SHEEP'] >= 10)}")
for shop in ("BAKERY", "YARN_STORE", "BRUNCH_SPOT", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"):
    print(f"  {shop:15s} in town:  {share(L, lambda x: shop in x['shops'])} vs {share(W, lambda x: shop in x['shops'])}")
for G, lab in ((L, "losses"), (W, "wins")):
    d = {p: np.mean([x["th"]["P"][f"rev:{p}"] - x["us"]["P"][f"rev:{p}"] for x in G]) for p in PROD} if G else {}
    print(f"  opponent minus us, revenue by product in {lab}: " + ", ".join(f"{p[:5]} {v:+,.0f}" for p, v in sorted(d.items(), key=lambda kv: -kv[1])))
