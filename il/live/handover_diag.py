"""Bug check (Sep 25): why does ANY scripted opening + handover to the network collapse (DSMR0 365->119, HighWool ->68)?
Plays the same ladder worlds (a) network from step 0, (b) scripted bot for steps 0-143 then the network - with the
PACKAGE runtime (np_runtime.NumpyAgent, as on Kaggle) - and prints money by day plus what the network does after the
handover.   Usage: handover_diag.py <ckpt> <team> <bot> [n_worlds]"""
import sys, json, collections
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
import numpy as np, torch
from harness import mk, PASS
from live_rollout import _game
from ladder import worlds
import np_runtime as R, features as F, features6 as F6
ck, team, bot = sys.argv[1], int(sys.argv[2]), sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
c = torch.load(ck, map_location="cpu", weights_only=False)
W_ = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
blocks = len({k.split(".")[1] for k in W_ if k.startswith("blocks.")})
def new_agent():
    net = R.NumpyNet(W_, width=int(c["width"]), blocks=blocks, n_teams=int(c.get("n_teams", 0)))
    return R.NumpyAgent(net, F6, tau=0.5, team=team)
mkbot = mk(bot)
for w in worlds("test")[:N]:
    for mode in ("network only", f"{bot.split('/')[-1]} 0-143 then network"):
        g, opp = _game(w["seed"], w["seat"], "TAPE:" + w["path"]); me = w["seat"]; ag = new_agent(); b = mkbot() if "then" in mode else None
        money = []; after = collections.Counter()
        for t in range(720):
            o = g.observe(me)
            if b is not None and t < 144:
                a = b(o)
            else:
                a = ag(o)
                if 144 <= t < 240:
                    after["market orders"] += len(a.get("market") or []); after["unit PASS"] += sum(1 for u in [a.get("farmer")] + list(a.get("hands") or []) if not u or u[0] == "PASS")
                    after["units"] += 1 + len(a.get("hands") or [])
            try:
                bb = opp(g.observe(1 - me))
            except Exception:
                bb = dict(PASS)
            now = [None, None]; now[me] = a; now[1 - me] = bb; g.step(now[0], now[1])
            if t % 48 == 47:
                money.append(int(g.observe(me)["farms"][me]["money"]))
            if g.done:
                break
        print(f"world {w['ep']} {mode:40s} margin {g.reward(me) - g.reward(1 - me):+9,.0f} | money every 2 days {money[:8]} | days 6-9: {dict(after)}", flush=True)
