# Fresh closed-loop head-to-head: neural agent (torch, CPU) vs a harness/scripted opponent on seeds 251000.. (seat = seed % 2).
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import torch; torch.set_num_threads(1)
from harness import mk, play
from agent import factory
ck, team, tau, opp, n, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
f = factory(ck, mk_tau=tau, team=team); mo = mk(opp); rows = []; t0 = time.time()
for s in range(251000, 251000 + n):
    x, y, _, _, err = play(s, s % 2, f, mo); rows.append([s, x, y])
    if len(rows) % 20 == 0:
        a = np.array(rows)[:, 1:]; print(f"{len(rows)} games: W {(a[:,0] > a[:,1]).sum()}, margin {(a[:,0]-a[:,1]).mean():+,.0f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(rows, open(out, "w"))
json.dump(rows, open(out, "w"))
a = np.array(rows)[:, 1:]; d = a[:, 0] - a[:, 1]
print(f"FINAL vs {opp.split('/')[-2] if '/' in opp else opp}: W {(d > 0).sum()}/{n}, margin {d.mean():+,.0f} (se {d.std()/np.sqrt(n):,.0f}), bank {a[:,0].mean():,.0f} vs {a[:,1].mean():,.0f}", flush=True)
