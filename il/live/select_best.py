"""Pick the RL starting point: the best DSM-style imitation model on the 464 ladder worlds (most wins, then margin) among
the fresh old-menu DSM fine-tunes (fr_*_dsm, team 10) and the gap-fixed menu-8 DSM+DECEM fine-tunes (g8_*_dd, team 10 or 3).
Prints "<ckpt> <team> <wins> <margin>" for the best and a ranking on stderr."""
import sys, json, glob, os, re
import numpy as np
R = "/Users/asad.ali/kaggriculture/il"
rows = []
for f in glob.glob(f"{R}/logs/ladder_*_test.json"):
    m = re.match(r".*/ladder_((fr_[cgtx]_dsm|g8_[cgt]_dd)_s\d)_t(\d+)_r3_tau0\.5_test\.json$", f)
    if not m:
        continue
    L = json.load(open(f)); mg = np.array([r["margin"] for r in L])
    if len(mg) != 464:
        continue
    rows.append(((mg > 0).sum(), mg.mean(), f"{R}/ckpt/{m.group(1)}.pt", int(m.group(3))))
rows.sort(reverse=True)
for w, mm, c, t in rows:
    print(f"  {w:3d} wins  {mm:+8,.0f}  {os.path.basename(c)} team {t}", file=sys.stderr)
if rows:
    w, mm, c, t = rows[0]; print(c, t, w, f"{mm:.0f}")
