"""All LoopCfg variants (dynamic / static trip counts, unrolling) must give bit-identical states.
Runs the bench policy (random legal-ish actions, training randomness with a fixed key) for --steps steps under each
variant and compares every state field after every step with the default variant.
usage: env1.sh test_loopcfg.py [--B 64] [--steps 60]"""
import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp

import engine as E
from bench import random_actions

VARIANTS = [E.LoopCfg(), E.LoopCfg(unit_static=True, slot_static=True),
            E.LoopCfg(unit_static=True, slot_static=True, unit_unroll=4, slot_unroll=2),
            E.LoopCfg(unit_static=False, slot_static=True, slot_unroll=10),
            E.LoopCfg(unit_par=True), E.LoopCfg(unit_par=True, unit_static=True, slot_static=True),
            E.CFG_PAR, E.LoopCfg(unit_par=True, count_search=True, slot_static=True, slot_unroll=10),
            E.LoopCfg(count_search=True, no_cond=True),
            E.LoopCfg(unit_par=True, count_search=True, slot_static=True, slot_unroll=10, no_cond=True), E.CFG_GPU, E.CFG_UNROLLED]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    B = args.B
    tb = E.tables()
    fns = [E.make_step(c) for c in VARIANTS]
    st0 = E.init(B, seed=3)
    st0 = st0._replace(g=st0.g._replace(money=jnp.full_like(st0.g.money, 200_000)))
    sts = [st0] * len(fns)
    key = jax.random.PRNGKey(11)
    bad = 0
    for s in range(args.steps):
        key, ka, kb = jax.random.split(key, 3)
        a0 = random_actions(ka, B, sts[0].g, 0, sts[0].t)
        a1 = random_actions(kb, B, sts[0].g, 1, sts[0].t)
        sts = [f(tb, st, a0, a1, None) for f, st in zip(fns, sts)]
        ref = jax.tree_util.tree_leaves(sts[0])
        for k in range(1, len(fns)):
            for x, y in zip(ref, jax.tree_util.tree_leaves(sts[k])):
                if not np.array_equal(np.asarray(x), np.asarray(y)):
                    bad += 1
                    print("MISMATCH variant", VARIANTS[k], "step", s)
                    break
        if bad:
            break
    print("mean units", float(jnp.mean(sts[0].g.n_units)), "money", float(jnp.mean(sts[0].g.money)))
    print("RESULT", "PASS" if bad == 0 else "FAIL", f"{len(VARIANTS)} variants x {args.steps} steps x B={B}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
