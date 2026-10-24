# Where does NEURAL_RL30 win / lose on the live ladder?  Reads data/rows_56510358.json + taped games in
# data/tapes_neural, replays every taped game in the C++ engine (exact check) and prints:
#   record by opponent-rating band, the losses one by one (revenue gap by product, cost gaps, cash gap by day),
#   and the average per-product revenue gap in losses vs wins.
# Usage: live_report.py [sub] [tape_dir]
import sys, json, gzip, glob, collections, os
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS, ROOT

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
OPN = {6: "SELL", 4: "BUY_PRODUCT", 3: "BUY_SEED", 5: "BUY_ANIMAL", 2: "BUY_LAND", 1: "HIRE"}
sub = int(sys.argv[1]) if len(sys.argv) > 1 else 56510358
tdir = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/data/tapes_neural"


def replay(d):
    A = d["actions"]; g = kagsim_trace.Game(d["seed"], 720, True)
    rev = [collections.Counter(), collections.Counter()]; cost = [collections.Counter(), collections.Counter()]
    cash = [[], []]; hires = [[], []]
    for t in range(720):
        a0 = A[0][t] if isinstance(A[0][t], dict) else dict(PASS); a1 = A[1][t] if isinstance(A[1][t], dict) else dict(PASS)
        g.step(a0, a1)
        for rc in g.trace_step(g.trace_len() - 1).get("receipts", []):
            s = rc["seat"]; ex = rc.get("executed", 0) or 0
            if not ex:
                continue
            dm = (rc.get("money_after") or 0) - (rc.get("money_before") or 0); op = OPN.get(rc.get("op"), rc.get("op")); item = rc.get("item")
            itn = PRODUCTS[item] if isinstance(item, int) and item < len(PRODUCTS) else item
            if op == "SELL":
                rev[s][itn] += dm
            else:
                cost[s][f"{op}:{itn}" if op in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL") else op] += -dm
                if op == "HIRE":
                    hires[s].append(t)
        if t % 24 == 23:
            o = g.observe(0); cash[0].append(o["farms"][0]["money"]); cash[1].append(o["farms"][1]["money"])
        if g.done:
            break
    exact = abs(g.reward(0) - d["rewards"][0]) < .5 and abs(g.reward(1) - d["rewards"][1]) < .5
    return rev, cost, cash, hires, exact


rows = {str(r["ep"]): r for r in json.load(open(f"{ROOT}/data/rows_{sub}.json")) if r["res"] in ("W", "L", "T")}
ok = sorted(rows.values(), key=lambda r: r["end"])
print(f"=== submission {sub}: {len(ok)} games, record {sum(r['res']=='W' for r in ok)}-{sum(r['res']=='L' for r in ok)}, "
      f"rating now {ok[-1]['rating']:.0f}" if ok else "no games")
bands = collections.defaultdict(lambda: [0, 0, []])
for r in ok:
    b = int(r["opp_rating"] // 200 * 200); bands[b][0 if r["res"] == "W" else 1] += 1; bands[b][2].append(r["margin"])
print("record by opponent rating band:")
for b in sorted(bands):
    w, l, m = bands[b]; print(f"  {b:4d}-{b+199:4d}: {w:3d}-{l:<3d} mean margin {sum(m)/len(m):+9,.0f}")
by_opp = collections.defaultdict(lambda: [0, 0])
for r in ok:
    by_opp[r["opp"]][0 if r["res"] == "W" else 1] += 1
lost_to = sorted(((v[1], k) for k, v in by_opp.items() if v[1]), reverse=True)
if lost_to:
    print("lost to:", ", ".join(f"{k} ({n})" for n, k in lost_to[:15]))

gap = {"W": collections.defaultdict(list), "L": collections.defaultdict(list)}
for fn in sorted(glob.glob(f"{tdir}/*.json.gz")):
    d = json.load(gzip.open(fn, "rt")); r = rows.get(str(d["ep"]))
    if r is None:
        continue
    us = d["seat"]; th = 1 - us
    rev, cost, cash, hires, exact = replay(d)
    k = "L" if r["res"] == "L" else "W"
    for p in PRODUCTS + ["TOTAL"]:
        gap[k][p].append((sum(rev[us].values()) - sum(rev[th].values())) if p == "TOTAL" else rev[us][p] - rev[th][p])
    for c in ("HIRE", "BUY_LAND"):
        gap[k][c].append(cost[us][c] - cost[th][c])
    if k == "L":
        rows_ = sorted(((rev[us][p] - rev[th][p]), p) for p in PRODUCTS if rev[us][p] or rev[th][p])
        print(f"\n### LOSS ep {d['ep']} {r['end']} vs {r['opp'][:22]} ({r['opp_rating']:.0f}) seat {us} | "
              f"{r['us_reward']:,.0f} vs {r['th_reward']:,.0f} margin {r['margin']:+,.0f} | exact {exact}")
        print("   revenue gap us-them: " + ", ".join(f"{p} {v:+,.0f}" for v, p in rows_))
        cg = sorted(((cost[us][c] - cost[th][c]), c) for c in set(cost[0]) | set(cost[1]))
        print("   cost gap us-them (>=$300): " + ", ".join(f"{c} {v:+,.0f}" for v, c in cg if abs(v) >= 300))
        print(f"   hires us {len(hires[us])} (first {hires[us][:1]}) them {len(hires[th])} (first {hires[th][:1]})")
        cg_ = [cash[us][i] - cash[th][i] for i in range(len(cash[us]))]
        print("   cash gap by day: " + " ".join(f"d{i}:{v:+,.0f}" for i, v in enumerate(cg_) if i % 3 == 2 or i == len(cg_) - 1))
for k in ("W", "L"):
    if gap[k]["TOTAL"]:
        n = len(gap[k]["TOTAL"])
        print(f"\nmean revenue gap us-them in {'wins' if k == 'W' else 'losses'} ({n} games): " +
              ", ".join(f"{p} {sum(v)/n:+,.0f}" for p, v in gap[k].items() if abs(sum(v) / n) >= 200))
