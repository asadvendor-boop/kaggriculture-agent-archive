"""Which leader style makes the strongest agent?  Plays the policy with each team embedding against held-out leader
tapes (our agent in the leader's opponent's seat, world pinned) and reports wins / margin / bank per style.
Usage: eval_styles.py <ckpt> <n_worlds> <tau> [teams comma list | all] [--scripted NAME]
"""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from rollout import run, load_net
from build_dataset2 import TEAMS
import ppo

ckpt, n, tau = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
which = sys.argv[4] if len(sys.argv) > 4 else "all"
dev = torch.device("mps")
net, c = load_net(ckpt, dev)
import json
ev = [tuple(x) for x in json.load(open('/Users/asad.ali/kaggriculture/il/eval_worlds.json'))][:n]
teams = range(len(TEAMS)) if which == "all" else [int(x) for x in which.split(",")]
for t in teams:
    t0 = time.time(); res = []
    for b in range(0, len(ev), 48):
        res += run(net, ev[b:b + 48], dev, tau=tau, team=t)
    m = np.array([r["bank"] - r["opp"] for r in res]); bank = np.array([r["bank"] for r in res])
    print(f"style {t:2d} {TEAMS[t][:20]:20s}: wins {(m > 0).sum():2d}/{len(m)}, mean margin {m.mean():+8,.0f}, bank {bank.mean():8,.0f} ({time.time()-t0:.0f}s)", flush=True)
if "--scripted" in sys.argv:
    from harness import mk
    import topruler
    name = sys.argv[sys.argv.index("--scripted") + 1]; f = mk(name); res = []; t0 = time.time()
    for p, seat in ev:
        cb, lb, rl, ro, dp = topruler.run_tape(p, 1 - seat, f)
        res.append((cb, lb))
    a = np.array(res)
    print(f"scripted {name}: wins {(a[:,0] > a[:,1]).sum()}/{len(a)}, mean margin {(a[:,0]-a[:,1]).mean():+,.0f}, bank {a[:,0].mean():,.0f} ({time.time()-t0:.0f}s)", flush=True)
