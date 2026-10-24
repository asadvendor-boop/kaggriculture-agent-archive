"""End-to-end, reference side: the C++ engine (kagsim_trace) + the REAL exported numpy packages (main.py), loaded the
way Kaggle / tools/harness.py load them (exec the source in a fresh module per game and seat, take the last callable,
exceptions -> PASS as the harness does).  Same seeds and seat assignment as e2e_jax.py (learner seat 0 in games
[0, B/2) and seat 1 in [B/2, B) when an opponent package is given).  Records every action dict and the final banks.

usage: python -m kagjax.policy.e2e_ref --learner_pkg main.py [--opp_pkg main.py] --games 64 --seed0 880000
         --shard 0 --nshards 2 --out file.json.gz
"""
import sys, json, gzip, time, argparse, gc
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, mk, PASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner_pkg", required=True); ap.add_argument("--opp_pkg")
    ap.add_argument("--games", type=int, default=64); ap.add_argument("--seed0", type=int, default=880000)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = a.games
    seeds = np.arange(a.seed0, a.seed0 + B)
    lseat = np.zeros(B, np.int32)
    if a.opp_pkg:
        lseat[B // 2:] = 1
    mkL = mk(a.learner_pkg); mkO = mk(a.opp_pkg) if a.opp_pkg else mkL
    res = {}
    t0 = time.time()
    for b in range(a.shard, B, a.nshards):
        g = kagsim_trace.Game(int(seeds[b]), 720, False)
        ag = [None, None]
        ag[lseat[b]] = mkL(); ag[1 - lseat[b]] = mkO()
        acts, errs = [], []
        for t in range(720):
            obs = [g.observe(0), g.observe(1)]
            now = []
            for s in (0, 1):
                try:
                    x = ag[s](obs[s])
                except Exception as e:                       # harness semantics
                    x = dict(PASS); errs.append(f"step {t} seat {s}: {type(e).__name__}: {e}"[:200])
                now.append(x)
            acts.append(json.dumps(now))
            g.step(now[0], now[1])
            if g.done:
                break
        res[int(b)] = dict(seed=int(seeds[b]), lseat=int(lseat[b]), banks=[g.reward(0), g.reward(1)], acts=acts, errors=errs)
        del ag; gc.collect()
        print(f"game {b} seed {seeds[b]}: banks {res[int(b)]['banks']}, {len(acts)} steps, errors {len(errs)}, "
              f"{time.time() - t0:.0f}s", flush=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(dict(args=vars(a), games=res), f)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
