"""Parity test: replay recorded games through the C++ reference (kagsim_trace) and the JAX engine with identical
actions and identical end-of-day randomness, and compare the FULL state after every step.

Compared every step, for both seats: the complete observe(seat) dict of the reference vs the same dict rebuilt from
the JAX state (state.game_to_obs): money, all 200 tiles with every visible field, unit positions, hands, owned
quadrants, hires_today, market inventory and prices, unlocked shops, day/hour/step, and the private shed, seeds and
per-unit inventories INCLUDING their key (insertion) order. Hidden tile fields are canonical in both engines (reset
on every kind change) and never read, so the observation is the complete game state. Final rewards are also checked
against the tape's recorded Kaggle rewards.

Randomness: the JAX engine gets rng.day_words(seed, day) (the exact MT19937 outputs of the reference's
random.Random((seed*1000003)^day)), so weeds and shop unlocks are reproduced, not injected from observations.

usage: env1.sh parity.py [--n 50] [--start 0] [--batch 25] [--dir ../data/tapes_top12] [--every 1]
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time

sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402

import consts as C  # noqa: E402
import engine as E  # noqa: E402
import rng as R  # noqa: E402
import state as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def tape_files(d, n, start=0):
    return sorted(glob.glob(os.path.join(d, "*.json.gz")))[start:start + n]


def load_tapes_files(fs):
    out = []
    for f in fs:
        t = json.load(gzip.open(f))
        t["_file"] = os.path.basename(f)
        out.append(t)
    return out


def load_tapes(d, n, start=0):
    return load_tapes_files(tape_files(d, n, start))


def replay_batch(seeds, actions, every=1, rewards=None, names=None, verbose=True, max_steps=720, cfg=None):
    """seeds: list[B]; actions[b][k][s]: action dicts (720 per seat). Returns per-game results."""
    B = len(seeds)
    refs = [kagsim_trace.Game(int(s), 720, False) for s in seeds]
    st = E.init(B)
    words = R.batch_words(seeds)                      # [30,B,624]
    first_bad = [None] * B
    t0 = time.time()
    tj = tr = tc = 0.0
    for s in range(max_steps):
        ta = time.time()
        a0 = S.encode_actions([actions[b][0][s] for b in range(B)])
        a1 = S.encode_actions([actions[b][1][s] for b in range(B)])
        st = E.step(st, a0, a1, words[min(s // C.TURNS_PER_DAY, 29)], cfg=cfg)
        g = jax.tree_util.tree_map(np.asarray, st.g)
        tb = time.time()
        for b in range(B):
            refs[b].step(actions[b][0][s], actions[b][1][s])
        tcc = time.time()
        tj += tb - ta
        tr += tcc - tb
        if (s + 1) % every == 0 or s >= 717:
            t = int(st.t)
            for b in range(B):
                if first_bad[b] is not None:
                    continue
                for seat in (0, 1):
                    d = S.obs_diff(S.game_to_obs(g, b, seat, t), refs[b].observe(seat))
                    if d:
                        first_bad[b] = {"step": s, "seat": seat, "diff": [list(map(str, x)) for x in d[:12]]}
                        if verbose:
                            print(f"  MISMATCH game {names[b] if names else b} after step {s} seat {seat}:")
                            for x in d[:12]:
                                print("     ", x)
                        break
        tc += time.time() - tcc
    res = []
    g = jax.tree_util.tree_map(np.asarray, st.g)
    for b in range(B):
        mine = [int(g.money[b, 0]), int(g.money[b, 1])]
        ref = [refs[b].reward(0), refs[b].reward(1)]
        r = {"game": names[b] if names else b, "seed": int(seeds[b]), "exact_steps": (
            first_bad[b]["step"] if first_bad[b] else max_steps), "mismatch": first_bad[b], "jax_reward": mine,
             "ref_reward": ref, "reward_equal": mine == ref, "oob": bool(g.oob[b])}
        if rewards is not None:
            r["tape_reward"] = rewards[b]
            r["tape_equal"] = [float(x) for x in rewards[b]] == [float(x) for x in mine]
        res.append(r)
    if verbose:
        print(f"  batch B={B}: {time.time() - t0:.1f}s (jax step+pull {tj:.1f}s, ref {tr:.1f}s, compare {tc:.1f}s)")
    return res


def pick_cfg(args):
    name = "par" if args.par else args.cfg
    return {"seq": E.CFG_SEQ, "par": E.CFG_PAR, "gpu": E.CFG_GPU, "unrolled": E.CFG_UNROLLED,
            "default": None}[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "data", "tapes_top12"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "parity.json"))
    ap.add_argument("--par", action="store_true", help="same as --cfg par")
    ap.add_argument("--cfg", default="seq", choices=["seq", "par", "gpu", "unrolled", "default"],
                    help="loop structure: engine.CFG_SEQ / CFG_PAR / CFG_GPU (identical results)")
    args = ap.parse_args()
    files = tape_files(args.dir, args.n, args.start)          # tapes are loaded one batch at a time
    allres = []
    for i in range(0, len(files), args.batch):
        tb = load_tapes_files(files[i:i + args.batch])
        acts = [[[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][k]] for k in (0, 1)] for t in tb]
        res = replay_batch([int(t["seed"]) for t in tb], acts, every=args.every,
                           rewards=[t["rewards"] for t in tb], names=[t["_file"] for t in tb],
                           cfg=pick_cfg(args))
        allres += res
        ok = sum(r["mismatch"] is None and r["reward_equal"] for r in res)
        print(f"games {i}..{i + len(tb) - 1}: {ok}/{len(tb)} exact through all steps", flush=True)
    n_ok = sum(r["mismatch"] is None and r["reward_equal"] for r in allres)
    n_tape = sum(bool(r.get("tape_equal")) for r in allres)
    summary = {"games": len(allres), "exact_all_steps": n_ok, "tape_reward_equal": n_tape, "cfg": "par" if args.par else args.cfg,
               "checked_every": args.every, "dir": os.path.abspath(args.dir), "start": args.start,
               "failures": [r for r in allres if not (r["mismatch"] is None and r["reward_equal"])]}
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))
    for r in summary["failures"][:10]:
        print("FAIL", r["game"], "first mismatch after step", r["exact_steps"], r["mismatch"]["diff"][:3]
              if r["mismatch"] else r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": {k: v for k, v in summary.items() if k != "failures"}, "results": allres},
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
