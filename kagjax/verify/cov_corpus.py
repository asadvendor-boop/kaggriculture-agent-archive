"""Run A's parity / fuzz corpus through the coverage-instrumented reference engine (C++ side only; no JAX).

  tapes : every recorded game of data/tapes_top12 (the corpus parity.py replays), both seats' raw actions;
          final banks checked against the tape's Kaggle rewards (sanity of the instrumented build).
  fuzz  : A's fuzz.py generators (rand_unit / rand_order / random_agent / perturb / entangle, imported, not edited)
          driving bots + random agents exactly like fuzz.run_batch, minus the JAX side.

usage (server, cwd kagjax/verify):
  KAG_ENGINE=cov_build/cov GCOV_PREFIX=... python cov_corpus.py tapes --start 0 --n 4000
  KAG_ENGINE=cov_build/cov python cov_corpus.py fuzz --games 40 --seed 0
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import harness as H  # noqa: E402  (imports kagsim_trace from $KAG_ENGINE)
from harness import kagsim_trace, PASS  # noqa: E402

assert "verify_cov" in kagsim_trace.__version__, kagsim_trace.__version__


def run_tapes(args):
    import parity  # A's loader (imports jax too, unused here)
    files = parity.tape_files(args.dir, args.n, args.start)
    n_ok = n_bad = 0
    bad = []
    t0 = time.time()
    for i in range(0, len(files), 50):
        for t in parity.load_tapes_files(files[i:i + 50]):
            acts = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][k]] for k in (0, 1)]
            g = kagsim_trace.Game(int(t["seed"]), 720, False)
            for s in range(720):
                g.step(acts[0][s], acts[1][s])
            mine = [g.reward(0), g.reward(1)]
            if [float(x) for x in t["rewards"]] == [float(x) for x in mine]:
                n_ok += 1
            else:
                n_bad += 1
                bad.append((t["_file"], mine, t["rewards"]))
        print(f"tapes {args.start + i}..: ok {n_ok} bad {n_bad} ({time.time() - t0:.0f}s)", flush=True)
    print("RESULT tapes", n_ok, "reward-exact of", n_ok + n_bad, "bad:", bad[:10])


def run_fuzz(args):
    import fuzz as FZ
    r = random.Random(args.seed)
    bots = [b for b in FZ.DEFAULT_BOTS if os.path.exists(H.AGENTS.get(b, b))]
    print("bots available:", bots, flush=True)
    t0 = time.time()
    for gi in range(0, args.games, args.batch):
        B = min(args.batch, args.games - gi)
        seeds = [r.randrange(2 ** 31) for _ in range(B)]
        refs = [kagsim_trace.Game(s, 720, False) for s in seeds]
        agents = []
        for b in range(B):
            pair = []
            for seat in range(2):
                if r.random() < args.random:
                    pair.append(None)
                else:
                    pair.append(H.mk(r.choice(bots))())
            agents.append(pair)
        obs = [[refs[b].observe(0), refs[b].observe(1)] for b in range(B)]
        for s in range(720):
            acts = []
            for b in range(B):
                pair = []
                for seat in range(2):
                    if agents[b][seat] is None:
                        a = FZ.random_agent(r, obs[b][seat])
                    else:
                        try:
                            a = agents[b][seat](obs[b][seat])
                        except Exception:
                            a = dict(PASS)
                    pair.append(FZ.perturb(r, a, obs[b][seat], args.p))
                FZ.entangle(r, pair[0], pair[1], args.p)
                if r.random() < 0.5:
                    FZ.entangle(r, pair[1], pair[0], args.p)
                acts.append(pair)
            for b in range(B):
                try:
                    refs[b].step(acts[b][0], acts[b][1])
                except Exception as e:      # the C++ binding raises on e.g. non-list hands; fuzz.py would too
                    print("step raised", type(e).__name__, e, flush=True)
                obs[b] = [refs[b].observe(0), refs[b].observe(1)]
        print(f"fuzz games {gi}..{gi + B - 1} done ({time.time() - t0:.0f}s) banks",
              [(x.reward(0), x.reward(1)) for x in refs], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tapes", "fuzz"])
    ap.add_argument("--dir", default="/Users/asad.ali/kaggriculture/data/tapes_top12")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.25)
    ap.add_argument("--random", type=float, default=0.2)
    args = ap.parse_args()
    {"tapes": run_tapes, "fuzz": run_fuzz}[args.mode](args)


if __name__ == "__main__":
    main()
