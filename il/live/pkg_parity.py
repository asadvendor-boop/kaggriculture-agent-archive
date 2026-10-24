"""Package parity (Sep 26): the exported Kaggle main.py (numpy runtime, its own embedded rule layer) plays N benchmark worlds
(pinned world, opponent on tape) and must reproduce the server benchmark's margin on every one of them exactly.
Usage: pkg_parity.py <package main.py> <benchmark rows json> [N]"""
import sys, json
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from live_rollout import run_scripted
pkg, rows_fn = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
W = {w["ep"]: w for w in json.load(open("/Users/asad.ali/kaggriculture/il/live/ladder_worlds.json"))}
rows = [r for r in json.load(open(rows_fn)) if r["ep"] in W]
pick = rows[:: max(1, len(rows) // N)][:N]
res = run_scripted(dict(us=pkg, games=[(W[r["ep"]]["seed"], W[r["ep"]]["seat"], "TAPE:" + W[r["ep"]]["path"]) for r in pick]))
ok = 0
for r, x in zip(pick, res):
    same = abs(x["margin"] - r["margin"]) < 0.5; ok += same
    print(f"  ep {r['ep']}: package {x['margin']:+10,.0f} | benchmark {r['margin']:+10,.0f} | {'EXACT' if same else 'DIFFERENT'}", flush=True)
print(f"PACKAGE PARITY {ok}/{len(pick)} exact", flush=True)
