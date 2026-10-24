"""Wheat study (Sep 28 night): NEURAL_20MW8_SE lost twice to wheat players (Breaking1800 -6,422, SpaTaro -4,602) who
out-earned us on WHEAT by ~$16-17k. Exact C++ replay of a recorded live game, both seats: wheat seeds bought, tiles
planted, units harvested, units fed to animals, bought as a product, sold (units, $, price), per day and in total, plus
the same totals for every crop / product - so the gap can be traced to planting, harvesting, feeding or selling.
Usage: wheat_study.py <tape.json.gz> [...]      (live tapes written by tools/tape_sub.py: our seat = "seat")"""
import sys, json, gzip, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = CROPS + ["EGG", "MILK", "WOOL", "FERTILIZER"]


def study(path):
    d = json.load(gzip.open(path, "rt"))
    A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][k]] for k in (0, 1)]
    g = kagsim_trace.Game(int(d["seed"]), 720, True)
    S = [collections.defaultdict(collections.Counter) for _ in (0, 1)]           # S[seat][key][day]
    for t in range(720):
        g.step(A[0][t], A[1][t]); tr = g.trace_step(g.trace_len() - 1); day = t // 24 + 1
        for rc in tr.get("receipts", []):
            s = rc.get("seat"); n = rc.get("executed") or 0
            if s not in (0, 1) or not n:
                continue
            op, it = rc.get("op"), rc.get("item"); dm = (rc.get("money_after") or 0) - (rc.get("money_before") or 0)
            if op == "SELL":
                S[s]["sold:" + it][day] += n; S[s]["rev:" + it][day] += dm
            elif op == "BUY_SEED":
                S[s]["seed:" + it][day] += n; S[s]["seed$:" + it][day] -= dm
            elif op == "BUY_PRODUCT":
                S[s]["bought:" + it][day] += n; S[s]["bought$:" + it][day] -= dm
            elif op == "BUY_LAND":
                S[s]["land"][day] += n
            elif op == "BUY_ANIMAL":
                S[s]["animal:" + str(it)][day] += n
        for u in tr.get("unit_receipts", []):
            s = u.get("seat")
            if s not in (0, 1) or not u.get("executed"):
                continue
            op, it, n = u.get("op"), u.get("eff_item") or u.get("arg"), u.get("eff_n") or 1
            if op == "PLANT":
                S[s]["plant:" + str(it)][day] += n
            elif op == "HARVEST":
                S[s]["harvest:" + str(it)][day] += n
            elif op == "FEED":
                S[s]["fed:" + str(it)][day] += n
            elif op == "FERTILIZE":
                S[s]["fertilize"][day] += 1
    us = int(d["seat"]); th = 1 - us; names = d.get("names") or ["?", "?"]
    tot = lambda s, k: sum(S[s][k].values())
    print(f"=== ep {d['ep']}: us ({names[us]}) {d['rewards'][us]:,.0f} vs {names[th]} {d['rewards'][th]:,.0f}")
    print(f"  {'':22s} {'us':>9s} {'them':>9s}")
    for c in PRODUCTS:
        row = []
        for k in (f"plant:{c}", f"harvest:{c}", f"fed:{c}", f"bought:{c}", f"sold:{c}", f"rev:{c}"):
            a, b = tot(us, k), tot(th, k)
            if a or b:
                row.append(f"{k.split(':')[0]} {a:,.0f}/{b:,.0f}")
        if row:
            pa = tot(us, f"rev:{c}") / max(1, tot(us, f"sold:{c}")); pb = tot(th, f"rev:{c}") / max(1, tot(th, f"sold:{c}"))
            print(f"  {c:11s} " + "  ".join(row) + f"  | price {pa:.1f}/{pb:.1f}")
    print("  land days us " + str(sorted(S[us]["land"])) + " them " + str(sorted(S[th]["land"])) +
          " | animals us " + str({k.split(':')[1]: tot(us, k) for k in S[us] if k.startswith('animal:')}) +
          " them " + str({k.split(':')[1]: tot(th, k) for k in S[th] if k.startswith('animal:')}) +
          f" | fertilize us {tot(us, 'fertilize')} them {tot(th, 'fertilize')}")
    print("  WHEAT by day (us | them): planted, harvested, fed, sold, $")
    for day in range(1, 31):
        a = [S[us][k][day] for k in ("plant:WHEAT", "harvest:WHEAT", "fed:WHEAT", "sold:WHEAT", "rev:WHEAT")]
        b = [S[th][k][day] for k in ("plant:WHEAT", "harvest:WHEAT", "fed:WHEAT", "sold:WHEAT", "rev:WHEAT")]
        if any(a) or any(b):
            print(f"   day {day:2d}: " + " ".join(f"{x:5.0f}" for x in a) + "  |  " + " ".join(f"{x:5.0f}" for x in b))


for p in sys.argv[1:]:
    study(p)
