# Side-by-side trace: BC agent in the leader's seat (world pinned, opponent on tape) vs the leader's recorded actions.
import sys, json, gzip; sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import torch; torch.set_num_threads(1)
from harness import kagsim_trace, PASS
import topruler
from agent import BCAgent
path, seat, ckpt, lo, hi = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
d = json.load(gzip.open(path, "rt")); th = 1 - seat
shops, e0, exact = topruler.recorded_world(path, d)
A = d["actions"]
g = kagsim_trace.Game(d["seed"], 720, False); g.set_forced_shops(shops)
if th == 1: g.set_weed_prefix(1, e0)
bc = BCAgent(ckpt)
short = lambda c: "".join(x[:2] for x in c[:2]) if isinstance(c, list) else "?"
for t in range(hi + 1):
    o = g.observe(seat); a = bc(o)
    rec = A[seat][t] if isinstance(A[seat][t], dict) else dict(PASS)
    if t >= lo:
        f = o["farms"][seat]
        us = [a["farmer"]] + a["hands"]; ds = [rec.get("farmer")] + list(rec.get("hands") or [])
        print(f"t{t:3d} ${f['money']:>7,.0f} hands {len(f['hands']):2d} seeds {sum(o['private']['seeds'].values()):2d} | BC  {' '.join(short(c) for c in us[:8]):40s} {a['market']}")
        print(f"{'':36s}| DSM {' '.join(short(c) for c in ds[:8]):40s} {rec.get('market')}")
    now = [None, None]; now[seat] = a; now[th] = A[th][t] if isinstance(A[th][t], dict) else dict(PASS); g.step(now[0], now[1])
