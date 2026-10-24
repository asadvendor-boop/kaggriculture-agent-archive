"""Final unbiased comparison of neural candidates.
  (1) test worlds: eval_worlds.json[48:96] (never used for style/tau/RL selection), our agent vs the leader's tape;
  (2) closed loop on fresh seeds vs scripted bots (numpy-free torch agent, C++ engine), paired seeds.
Usage: final_eval.py <n_fresh> <opp1,opp2> <label:ckpt:team:tau> [<label:ckpt:team:tau> ...]
"""
import sys, json, time
import numpy as np
import torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
torch.set_num_threads(1)
from rollout import run, load_net
from harness import mk, play
from agent import factory

n_fresh, opps = int(sys.argv[1]), [o for o in sys.argv[2].split(",") if o]
cands = [x.split(":") for x in sys.argv[3:]]
test = [tuple(x) for x in json.load(open("/Users/asad.ali/kaggriculture/il/eval_worlds.json"))][48:96]
dev = torch.device("mps")
for label, ck, team, tau in cands:
    team, tau = int(team), float(tau)
    net, c = load_net(ck, dev)
    res = run(net, test, dev, tau=tau, team=team)
    m = np.array([r["bank"] - r["opp"] for r in res]); b = np.array([r["bank"] for r in res])
    line = f"{label:14s} test worlds vs leader tapes: wins {(m > 0).sum()}/{len(m)}, margin {m.mean():+,.0f}, bank {b.mean():,.0f}"
    f = factory(ck, mk_tau=tau, team=team)
    for opp in opps:
        mo = mk(opp); rows = []; t0 = time.time()
        for s in range(250000, 250000 + n_fresh):
            x, y, _, _, err = play(s, s % 2, f, mo)
            rows.append((x, y))
        a = np.array(rows)
        line += f" | vs {opp}: W {(a[:,0] > a[:,1]).sum()}/{len(a)}, margin {(a[:,0]-a[:,1]).mean():+,.0f}, bank {a[:,0].mean():,.0f} ({time.time()-t0:.0f}s)"
    print(line, flush=True)
