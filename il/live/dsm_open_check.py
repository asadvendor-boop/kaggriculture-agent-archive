"""Does DSM's scripted opening (agents/cand/DSMR0.py, steps 0-143) rebuild DSM's day-6 farm in OUR worlds?
DSM itself (120/120 games): 10 melons, 10 strawberries, 2 cows, 3 sheep, 8 hands, $734-914 at step 144.
Plays the opening in N ladder test worlds (opponent on tape) and prints the farm at step 144.   Usage: dsm_open_check.py [N]"""
import sys, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
from harness import mk, PASS
from live_rollout import _game
from ladder import worlds
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
W = worlds("test"); make = mk("/Users/asad.ali/kaggriculture/agents/cand/DSMR0.py"); rows = []
for w in W[::max(1, len(W) // N)][:N]:
    g, opp = _game(w["seed"], w["seat"], "TAPE:" + w["path"]); ag = make(); me = w["seat"]
    for t in range(144):
        a = ag(g.observe(me)); b = opp(g.observe(1 - me))
        now = [None, None]; now[me] = a; now[1 - me] = b; g.step(now[0], now[1])
    o = g.observe(me); f = o["farms"][me]; c = collections.Counter()
    for row in f["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("crop"): c[t["crop"]] += 1
            if isinstance(t, dict) and t.get("animal"): c[t["animal"]] += 1
    units = list(f.get("hands") or [])
    rows.append(c)
    print(f"world {w['ep']} seat {me} vs {w['opp'][:18]:18s}: money ${f['money']:,.0f}, hands {len(units)}, " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())), flush=True)
print("farm keys:", sorted(f.keys()))
