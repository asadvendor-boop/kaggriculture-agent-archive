"""Sep 27 IL sprint: one table of every tested model (SE guard on), from the benchmark JSONs in il/logs.
Old leaders = top:10 (fresh, 135 paired worlds), today's top-10 = top2:10 (fresh2, 119), 464 ladder worlds (+ the 2400+
part); dm = mean paired margin difference vs NEURAL_20MW5_SE (g20_w5_dsm_s0 + SE), z in brackets.
Usage (from il/): python live/il_sprint_table.py"""
import glob, json, os
import numpy as np

L = "logs"
BASE = "g20_w5_dsm_s0"


def load(fn):
    try:
        return {(r.get("ep"), r.get("opp")): r for r in json.load(open(fn))}
    except (OSError, ValueError):
        return None


def cmp(base, cand, rating_min=None):
    keys = [k for k in base if k in cand and (rating_min is None or float(base[k].get("opp_rating") or 0) >= rating_min)]
    b = np.array([base[k]["margin"] for k in keys]); m = np.array([cand[k]["margin"] for k in keys]); d = m - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
    return int((m > 0).sum()), len(keys), d.mean(), (d.mean() / se if se else 0.0)


fB, f2B, lB = (load(f"{L}/top_fresh_{BASE}_t10_se.json"), load(f"{L}/top_fresh2_{BASE}_t10_se.json"),
               load(f"{L}/ladder_{BASE}_t10_r3_tau0.5_se_test.json"))
names = sorted({os.path.basename(f)[len("top_fresh_"):-len("_t10_se.json")] for f in glob.glob(f"{L}/top_fresh_*_t10_se.json")})
names = [n for n in names if n.startswith("ilsp_") or n in (BASE, "g20_w_dsm_s0")]
print(f"{'model':24s} | {'old leaders':>20s} | {'today top-10':>20s} | {'464 ladder':>20s} | {'2400+':>8s}")
for n in sorted(names, key=lambda n: (n not in (BASE, "g20_w_dsm_s0"), n)):
    cells = []
    for base, fn in ((fB, f"{L}/top_fresh_{n}_t10_se.json"), (f2B, f"{L}/top_fresh2_{n}_t10_se.json"),
                     (lB, f"{L}/ladder_{n}_t10_r3_tau0.5_se_test.json")):
        c = load(fn)
        if c is None or base is None:
            cells.append(f"{'-':>20s}"); continue
        w, k, dm, z = cmp(base, c)
        cells.append(f"{w:3d}/{k:<3d} {dm:+6,.0f} ({z:+.1f})")
    c = load(f"{L}/ladder_{n}_t10_r3_tau0.5_se_test.json")
    hi = f"{cmp(lB, c, 2400)[0]:3d}/{cmp(lB, c, 2400)[1]:<3d}" if c is not None and lB is not None else "-"
    print(f"{n.replace('ilsp_', ''):24s} | {cells[0]} | {cells[1]} | {cells[2]} | {hi:>8s}")
