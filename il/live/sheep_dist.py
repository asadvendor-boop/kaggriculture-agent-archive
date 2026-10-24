"""Sheep (and cows/geese) bought per game: DSM's own recorded games vs our live 10M DSM model, and whether many sheep go
with losses (the -34.8k loss to Seho.Connect had 23 sheep).   Usage: sheep_dist.py [n_dsm]"""
import sys, json, gzip, glob, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from dsm_likeness import profile
R = "/Users/asad.ali/kaggriculture"
if __name__ == "__main__":
    idx = json.load(open(f"{R}/data/tapes_top12/index.json")); N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    items = [(f"{R}/data/tapes_top12/{e}.json.gz", v["teams"].index("DSM")) for e, v in sorted(idx.items(), reverse=True) if "DSM" in (v.get("teams") or [])][:N]
    with mp.get_context("spawn").Pool(8) as pool:
        res = [r for r in pool.map(profile, items) if r]
    sh = np.array([r["P"]["animal:SHEEP"] for r in res]); win = np.array([r["P"]["margin"] > 0 for r in res])
    print(f"DSM {len(res)} games: sheep bought mean {sh.mean():.1f}, max {sh.max()}, distribution {dict(sorted(collections.Counter(sh.tolist()).items()))}")
    for lo, hi in ((0, 6), (6, 10), (10, 14), (14, 99)):
        m = (sh >= lo) & (sh < hi)
        if m.any(): print(f"  sheep {lo}-{hi - 1}: {m.sum()} games, DSM win rate {win[m].mean():.0%}")
