"""Why does DSM's opening tape (DSMR0) end day 5 with 5 melons instead of 10 in most of our worlds? For one world of
each kind, log every step whose action mentions MELON: money before, the action's MELON parts, melon tiles after, and
the engine's rejections for our seat that step.   Usage: dsm_melon_diag.py <ep> [<ep> ...]"""
import sys, json
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
from harness import mk
from live_rollout import _game
from ladder import worlds
W = {w["ep"]: w for w in worlds("test")}; make = mk("/Users/asad.ali/kaggriculture/agents/cand/DSMR0.py")
def melons(o, me):
    return sum(1 for row in o["farms"][me]["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "MELON")
for ep in map(int, sys.argv[1:]):
    w = W[ep]; g, opp = _game(w["seed"], w["seat"], "TAPE:" + w["path"]); ag = make(); me = w["seat"]
    print(f"=== world {ep} seat {me} vs {w['opp']}")
    for t in range(144):
        o = g.observe(me); a = ag(o); b = opp(g.observe(1 - me)); m0 = o["farms"][me]["money"]
        now = [None, None]; now[me] = a; now[1 - me] = b; g.step(now[0], now[1])
        s = json.dumps(a)
        if "MELON" in s:
            parts = [c for c in [a.get("farmer")] + list(a.get("hands") or []) if c and "MELON" in json.dumps(c)]
            mk_ = [x for x in (a.get("market") or []) if "MELON" in json.dumps(x)]
            print(f"  step {t:3d} (day {t // 24} h{t % 24:02d}) money ${m0:,.0f}: units {parts} market {mk_} -> melon tiles {melons(g.observe(me), me)}")
