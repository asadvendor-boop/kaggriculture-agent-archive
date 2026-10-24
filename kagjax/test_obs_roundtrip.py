"""obs <-> state converters: (1) game_to_obs(obs_to_game(ref obs)) == ref obs exactly (incl. inventory key order);
(2) start the JAX engine MID-GAME from the reference's observations (steps 100/300/500/650) and continue both engines
with the recorded actions to the end, comparing the full state every step.
usage: env1.sh test_obs_roundtrip.py [--n 6]
"""
import argparse
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402

import engine as E  # noqa: E402
import rng as R  # noqa: E402
import state as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--start", type=int, default=100)
    args = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(HERE, "..", "data", "tapes_top12", "*.json.gz")))[args.start:args.start + args.n]
    ok_rt = ok_cont = total = 0
    for f in fs:
        t = json.load(gzip.open(f))
        A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][k]] for k in (0, 1)]
        seed = int(t["seed"])
        W = R.batch_words([seed])
        for k0 in (100, 300, 500, 650):
            total += 1
            g = kagsim_trace.Game(seed, 720, False)
            for s in range(k0):
                g.step(A[0][s], A[1][s])
            o0, o1 = g.observe(0), g.observe(1)
            game, t0 = S.obs_to_game(o0, o1)
            G = S.stack_games([game])
            rt = [S.obs_diff(S.game_to_obs(G, 0, seat, t0), o) for seat, o in ((0, o0), (1, o1))]
            if not any(rt):
                ok_rt += 1
            else:
                print("ROUNDTRIP DIFF", os.path.basename(f), k0, rt[0][:3], rt[1][:3])
            st = E.from_games(G, t0)
            bad = None
            for s in range(k0, 720):
                st = E.step(st, S.encode_actions([A[0][s]]), S.encode_actions([A[1][s]]), W[min(s // 24, 29)])
                g.step(A[0][s], A[1][s])
                Gs = jax.tree_util.tree_map(np.asarray, st.g)
                for seat in (0, 1):
                    d = S.obs_diff(S.game_to_obs(Gs, 0, seat, int(st.t)), g.observe(seat))
                    if d:
                        bad = (s, seat, d[:4])
                        break
                if bad:
                    break
            if bad is None and int(st.g.money[0, 0]) == g.reward(0) and int(st.g.money[0, 1]) == g.reward(1):
                ok_cont += 1
            else:
                print("CONTINUATION MISMATCH", os.path.basename(f), "from", k0, bad)
    print(f"RESULT roundtrip {ok_rt}/{total}, mid-game restart exact to the end {ok_cont}/{total}")


if __name__ == "__main__":
    main()
