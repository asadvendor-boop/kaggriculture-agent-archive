"""Closed-loop evaluation of an imitation agent in the C++ engine.
  eval_bc.py tape <ckpt> <n> [device]   : agent in the leader's seat of held-out leader games (world pinned, opponent on
                                           tape); compares its bank with the leader's recorded bank in the same world.
  eval_bc.py fresh <ckpt> <opp> <n>     : agent vs a harness opponent (e.g. V94, STACK1_O20) on fresh seeds 250000..
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import torch
torch.set_num_threads(1)
from harness import mk, play, ROOT
import topruler
from agent import factory

mode, ckpt = sys.argv[1], sys.argv[2]
_t = os.environ.get("MK_TAU", "nan")
TAU = json.load(open(_t)) if _t.endswith(".json") else (None if float(_t) != float(_t) else float(_t))
MASK = {"none": False, "indep": True, "shared": "shared"}[os.environ.get("MASK", "indep")]
if mode == "tape":
    n = int(sys.argv[3]); dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    c = torch.load(ckpt, map_location="cpu", weights_only=False); val = c.get("files_val", [])
    import glob
    files = sorted(glob.glob(f"{ROOT}/il/data/dsm/*.npz"))
    jobs = []
    for i in val[:n]:
        b = os.path.basename(files[i]).split(".")[0]; ep, seat = b.split("_")
        for d_ in ("data/tapes_0922", "data/tapes_archive"):
            p = f"{ROOT}/{d_}/{ep}.json.gz"
            if os.path.exists(p): jobs.append((p, int(seat))); break
    f = factory(ckpt, device=dev, mk_tau=TAU, mask=MASK)
    rows = []; t0 = time.time()
    for p, seat in jobs:
        us_bank, opp_bank, rec_opp, rec_leader, dp = topruler.run_tape(p, 1 - seat, f)
        rows.append((os.path.basename(p), us_bank, opp_bank, rec_leader, rec_opp))
        print(f"{os.path.basename(p)} seat {seat}: BC {us_bank:>9,.0f} vs opp {opp_bank:>9,.0f} | DSM recorded {rec_leader:>9,.0f} vs {rec_opp:>9,.0f} | {time.time()-t0:.0f}s", flush=True)
    a = np.array([r[1:] for r in rows])
    print(f"SUMMARY {len(rows)} games: BC bank {a[:,0].mean():,.0f} (DSM {a[:,2].mean():,.0f}, ratio {a[:,0].mean()/a[:,2].mean():.2f}); "
          f"BC wins vs taped opponent {int((a[:,0] > a[:,1]).sum())}/{len(rows)} (DSM won {int((a[:,2] > a[:,3]).sum())})", flush=True)
else:
    opp, n = sys.argv[3], int(sys.argv[4]); dev = sys.argv[5] if len(sys.argv) > 5 else "cpu"
    f = factory(ckpt, device=dev, mk_tau=TAU, mask=MASK); mo = mk(opp); rows = []; t0 = time.time()
    for s in range(250000, 250000 + n):
        x, y, tmax, tsum, err = play(s, s % 2, f, mo, timed=True)
        rows.append((x, y)); print(f"seed {s}: BC {x:,.0f} vs {opp} {y:,.0f} | max step {tmax*1000:.0f} ms | {err or ''}", flush=True)
    a = np.array(rows)
    print(f"SUMMARY vs {opp}: W {int((a[:,0] > a[:,1]).sum())}/{len(rows)}, BC bank {a[:,0].mean():,.0f} vs {a[:,1].mean():,.0f}, {time.time()-t0:.0f}s", flush=True)
