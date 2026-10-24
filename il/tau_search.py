"""Direct policy search on the market-decision thresholds (derivative-free RL on the decoding policy).
Coordinate search over head groups, then per-product SELL thresholds; every candidate is scored by greedy play on
the same fixed set of training worlds (paired), accepted if the mean margin improves by more than DELTA.
Usage: tau_search.py <ckpt> <team> <base_tau> <out.json>   env: N_WORLDS (96), DELTA (500)
"""
import sys, os, json, random, time
import numpy as np
import torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import ppo
import features as F
from rollout import run, load_net
ck, team, base, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
NW = int(os.environ.get("N_WORLDS", "96")); DELTA = float(os.environ.get("DELTA", "500"))
dev = torch.device("mps"); net, c = load_net(ck, dev)
tr, _ = ppo.world_jobs()
ev = set(tuple(x) for x in json.load(open(f"{ppo.R}/il/eval_worlds.json")))
worlds = random.Random(11).sample(sorted(j for j in tr if j not in ev), NW)

def score(tau):
    res = []
    for b in range(0, len(worlds), 48):
        res += run(net, worlds[b:b + 48], dev, tau=tau, team=team)
    m = np.array([r["bank"] - r["opp"] for r in res])
    return float(m.mean()), int((m > 0).sum())

GROUPS = {"SELL": [h for h, _ in F.MARKET_HEADS if h.startswith("SELL:")],
          "BUY_SEED": [h for h, _ in F.MARKET_HEADS if h.startswith("BUY_SEED:")],
          "BUY_PRODUCT": ["BUY_PRODUCT:WHEAT", "BUY_PRODUCT:FERTILIZER"],
          "BUY_ANIMAL": [h for h, _ in F.MARKET_HEADS if h.startswith("BUY_ANIMAL:")],
          "HIRE_LAND": ["HIRE", "BUY_LAND"]}
tau = {"_default": base}
best, wins = score(tau); t0 = time.time()
print(f"base tau {base}: margin {best:+,.0f} wins {wins}/{NW}", flush=True)
def trial(new_tau, label):
    global tau, best
    m, w = score(new_tau)
    ok = m > best + DELTA
    print(f"  {label:34s} margin {m:+,.0f} wins {w}/{NW} {'ACCEPT' if ok else ''} ({time.time()-t0:.0f}s)", flush=True)
    if ok:
        tau, best = new_tau, m
        json.dump(tau, open(out, "w"), indent=1)
for g, heads in GROUPS.items():
    for v in (0.4, 0.5, 0.7, 0.8):
        cand = dict(tau); cand.update({h: v for h in heads}); trial(cand, f"{g} -> {v}")
for h in GROUPS["SELL"] + GROUPS["BUY_SEED"]:
    cur = tau.get(h, tau["_default"])
    for v in (round(cur - 0.15, 2), round(cur + 0.15, 2)):
        if 0.2 <= v <= 0.9:
            cand = dict(tau); cand[h] = v; trial(cand, f"{h} -> {v}")
json.dump(tau, open(out, "w"), indent=1)
print("final", json.dumps(tau), f"margin {best:+,.0f}", flush=True)
