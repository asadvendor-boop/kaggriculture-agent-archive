"""Paired replay of the two live neural submissions' real games (operator, Sep 25): every taped live game of NEURAL_10M_S4
(data/tapes_n10m) and NEURAL_4M_S4 (data/tapes_n4m) is replayed with BOTH models in our seat (world pinned, opponent's
recorded actions; team 5, tau 0.5, rules on = the packages' settings). Answers: on the same worlds, is the 10M better
than the 4M, or did the live records differ by opponent luck?   Usage: srv_live_pair.py   env NW (8)"""
import sys, os, json, glob, gzip
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
R = "/Users/asad.ali/kaggriculture"
CK = {"10M": f"{R}/il/ckpt/s2_big10_s4.pt", "4M": f"{R}/il/ckpt/s2_grown_s4.pt"}


def one(job):
    from live_rollout import run_job
    name, g = job
    return name, run_job(dict(weights=CK[name], team=5, tau=0.5, T=0.2, ks=10.0, greedy=True, seed=0, games=[g]))[0]


if __name__ == "__main__":
    games, meta = [], {}
    for src in ("n10m", "n4m"):
        for f in sorted(glob.glob(f"{R}/data/tapes_{src}/*.json.gz")):
            d = json.load(gzip.open(f, "rt"))
            if d.get("res") not in ("W", "L", "T"):
                continue
            g = (int(d["seed"]), int(d["seat"]), "TAPE:" + f); games.append(g)
            meta[f] = dict(src=src, res=d["res"], opp=d["opp"], live=(d["rewards"][d["seat"]] or 0) - (d["rewards"][1 - d["seat"]] or 0))
    jobs = [(n, g) for g in games for n in CK]
    with mp.get_context("spawn").Pool(int(os.environ.get("NW", "8"))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    by = {}
    for (n, g), (_, r) in zip(jobs, res):
        by.setdefault(g[2][5:], {})[n] = r["margin"]
    for src in ("n10m", "n4m"):
        fs = [f for f in by if meta[f]["src"] == src]
        m10 = np.array([by[f]["10M"] for f in fs]); m4 = np.array([by[f]["4M"] for f in fs]); live = np.array([meta[f]["live"] for f in fs])
        d = m10 - m4
        print(f"\n=== live games of {'NEURAL_10M_S4' if src == 'n10m' else 'NEURAL_4M_S4'}: {len(fs)} (live W {sum(meta[f]['res'] == 'W' for f in fs)})")
        print(f"  replayed wins: 10M {(m10 > 0).sum()}, 4M {(m4 > 0).sum()} | mean margin 10M {m10.mean():+,.0f}, 4M {m4.mean():+,.0f}, "
              f"paired 10M-4M {d.mean():+,.0f} (se {d.std() / max(1, np.sqrt(len(d))):,.0f})")
        for f in fs:
            if meta[f]["res"] != "W" or (by[f]["10M"] > 0) != (by[f]["4M"] > 0):
                print(f"   {meta[f]['res']} vs {meta[f]['opp'][:22]:22s} live {meta[f]['live']:+9,.0f} | replay 10M {by[f]['10M']:+9,.0f}  4M {by[f]['4M']:+9,.0f}")
