"""Encoder on the TRAINING path vs the Kaggle runtime's encoder, on random valid states (statefuzz generator,
canonicalized by the C++ engine: up to 40 units, animals carried, duplicate shops, markets at the $1 floors and far
from I0, day-end / last steps), both seats, random previous-op lists (shorter / longer than the unit count, -1s):
   JAX:  state.obs_to_game -> train/real_backend.raw_from_state -> kagjax.policy.encode.encode (vmapped, jitted)
   ref:  il/features6.encode_obs(obs, prev_ops)
planes / pos compared exactly, glob / unit features bitwise.
usage (server, CPU): python enc_fuzz.py --n 3000 --seed 0
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
for p in (os.path.join(R, "kagjax"), os.path.join(R, "kagjax", "train"), R, os.path.join(R, "il")):
    sys.path.insert(0, p)
sys.path.insert(0, os.environ.get("KAG_ENGINE", os.path.join(HERE, "cov_build", "opt")))
os.environ.setdefault("JAX_CACHE_DIR", "off")
from functools import partial  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kagsim_trace as K  # noqa: E402
import statefuzz as SF  # noqa: E402
import state as S  # noqa: E402
import engine as E  # noqa: E402
import real_backend as RB  # noqa: E402
from kagjax.policy import encode as CE  # noqa: E402
import features6 as F6  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realistic", type=int, default=0, help="1: market inventories within the range real games reach")
    args = ap.parse_args()
    if args.realistic:                                      # max town drain over a game ~1.6k units -> prices < $4096
        _orig = SF.rand_minv
        def _rm(r, fam):
            d = _orig(r, fam)
            return {k: (v if abs(v - 10000) <= 900 else 10000 + r.randint(-900, 600)) for k, v in d.items()}
        SF.rand_minv = _rm
    expl = {"price>=4096": 0, "unexplained": 0}
    idx_hist = {}
    r = random.Random(args.seed)
    tb = E.tables()
    tabs = CE.tables_jnp()

    @partial(jax.jit, static_argnums=(2,))
    def jenc(g, t, seat, prev):
        raw = RB.raw_from_state(g, t, seat, tb)
        return jax.vmap(CE.encode, in_axes=(0, 0, None))(raw, prev, tabs)

    n_tot = n_bad = 0
    first = []
    B = 64
    steps = [0, 1, 23, 24, 100, 239, 287, 311, 431, 575, 695, 700, 717, 718]
    while n_tot < args.n:
        t = r.choice(steps)
        pairs, prevs, seats = [], [], []
        for b in range(B):
            o0, o1 = SF.rand_state(r, t, r.choice(["unit", "all", "eod", "market"]))
            g = K.Game(r.randrange(2 ** 31), 720, False)
            g.set_state(o0, o1)
            pairs.append((g.observe(0), g.observe(1)))
            L = r.choice([0, 1, 3, 16, 20])
            prevs.append([r.choice([-1, r.randrange(44)]) for _ in range(L)] if r.random() < 0.9 else None)
            seats.append(r.randint(0, 1))
        gs = jax.tree_util.tree_map(jnp.asarray, S.stack_games([S.obs_to_game(a, b_)[0] for a, b_ in pairs]))
        parr = np.full((B, 16), -1, np.int32)
        for b, pv in enumerate(prevs):
            if pv:
                m = min(16, len(pv))
                parr[b, :m] = pv[:m]
        outs = {s: jax.tree_util.tree_map(np.asarray, jenc(gs, jnp.int32(t), s, jnp.asarray(parr))) for s in (0, 1)}
        for b in range(B):
            s = seats[b]
            obs = pairs[b][s]
            pl, gl, pos, uf = F6.encode_obs(obs, prevs[b])
            o = {k: v[b] for k, v in outs[s].items()}
            n = len(pos)
            ok = (np.array_equal(o["planes"], pl) and np.array_equal(o["glob"].view(np.uint32), gl.astype(np.float32).view(np.uint32))
                  and np.array_equal(o["pos"][:n], pos) and np.array_equal(o["uf"][:n].view(np.uint32), uf.astype(np.float32).view(np.uint32))
                  and int(o["valid"].sum()) == n)
            n_tot += 1
            if not ok:
                n_bad += 1
                big = max(obs["market"]["prices"].values()) >= 4096
                expl["price>=4096" if big else "unexplained"] += 1
                for i in np.nonzero(o["glob"].view(np.uint32) != gl.astype(np.float32).view(np.uint32))[0].tolist():
                    idx_hist[i] = idx_hist.get(i, 0) + 1
                if not (np.array_equal(o["planes"], pl) and np.array_equal(o["uf"][:n].view(np.uint32), uf.astype(np.float32).view(np.uint32))):
                    idx_hist["planes_or_uf"] = idx_hist.get("planes_or_uf", 0) + 1
                if len(first) < 5:
                    what = [k for k, a, bb in (("planes", o["planes"], pl), ("glob", o["glob"], gl), ("uf", o["uf"][:n], uf))
                            if not np.array_equal(np.asarray(a, np.float64), np.asarray(bb, np.float64))]
                    gd = np.nonzero(o["glob"] != gl)[0][:10].tolist()
                    first.append((t, s, what, gd, [(float(o["glob"][i]), float(gl[i])) for i in gd[:5]]))
        print(f"t={t}: {n_tot} observations, {n_bad} mismatches", flush=True)
    for x in first:
        print("MISMATCH", x)
    print("mismatch classes:", expl, "| glob indices:", dict(sorted(idx_hist.items(), key=lambda kv: -kv[1])[:20]))
    print(f"RESULT {n_tot} observations: {n_bad} encoder mismatches")


if __name__ == "__main__":
    main()
