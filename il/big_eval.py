"""Large paired evaluation for configuration selection: every config plays the same N training worlds (not the frozen
eval/test worlds) greedily vs the leader tapes; reports wins, mean margin, and the paired difference vs the first
config with its standard error.  Usage: big_eval.py <N> <label:ckpt:team:tau|json> ...
Rows cached to il/logs/big_eval_<label>.json."""
import sys, os, json, random, time
import numpy as np
import torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import ppo
from rollout import run, load_net

N = int(sys.argv[1]); cfgs = [a.split(":", 3) for a in sys.argv[2:]]
dev = torch.device("mps")
tr, _ = ppo.world_jobs()
ev = set(tuple(x) for x in json.load(open(f"{ppo.R}/il/eval_worlds.json")))
worlds = random.Random(23).sample(sorted(j for j in tr if j not in ev), N)
base = None
for label, ck, team, tau in cfgs:
    fn = f"{ppo.R}/il/logs/big_eval_{label}.json"
    if os.path.exists(fn):
        m = np.array(json.load(open(fn)))
    else:
        tau_v = json.load(open(tau)) if tau.endswith(".json") else float(tau)
        if "+" in ck:
            from ensemble import load_ensemble
            net = load_ensemble(ck.split("+"), dev)
        else:
            net, c = load_net(ck, dev)
        t0 = time.time(); res = []
        for b in range(0, N, 50):
            res += run(net, worlds[b:b + 50], dev, tau=tau_v, team=int(team))
        m = np.array([r["bank"] - r["opp"] for r in res]); json.dump(m.tolist(), open(fn, "w"))
    line = f"{label:22s} wins {(m > 0).sum()}/{N}, mean margin {m.mean():+,.0f} (se {m.std() / np.sqrt(N):,.0f})"
    if base is None:
        base = m
    else:
        d = m - base
        line += f" | paired vs first {d.mean():+,.0f} (se {d.std() / np.sqrt(N):,.0f})"
    print(line, flush=True)
