"""Dry run: the exported package main.py (export.py, RULES of ST2) loads the Kaggle way (harness.mk) and plays 30 turns."""
import sys, time
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import mk, kagsim_trace, PASS
t0 = time.time(); ag = mk(sys.argv[1])(); t1 = time.time()
g = kagsim_trace.Game(900001, 720, False); acts = []
for t in range(30):
    a = ag(g.observe(0)); acts.append(a); g.step(a, dict(PASS))
print(f"package loaded in {t1-t0:.1f} s, 30 turns in {time.time()-t1:.1f} s; turn 0 market {acts[0]['market'][:4]}; money after 30 turns {g.observe(0)['farms'][0]['money']:.0f}")
