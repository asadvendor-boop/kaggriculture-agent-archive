"""GPU structure benchmark: the same rollout (bench.py random policy, realistic ~19 units) under several loop
structures (engine.LoopCfg), to find the fastest formulation for the device. Results are identical across variants
(test_loopcfg.py); only speed differs.
usage: python bench_variants.py --B 4096 --steps 48 [--variants 0,1,2,3]
"""
import argparse
import time

import jax
import jax.numpy as jnp
from jax import lax

import engine as E
from bench import random_actions

VARIANTS = {
    0: E.LoopCfg(),                                                                  # dynamic bounds (while)
    1: E.LoopCfg(unit_static=True, slot_static=True),                                # static trip counts
    2: E.LoopCfg(unit_static=True, slot_static=True, unit_unroll=8, slot_unroll=10),
    3: E.LoopCfg(unit_static=True, slot_static=True, unit_unroll=40, slot_unroll=10),  # fully unrolled
    4: E.LoopCfg(unit_static=False, slot_static=True, slot_unroll=10),
    5: E.LoopCfg(unit_par=True),                                                     # round-based units
    6: E.LoopCfg(unit_par=True, slot_static=True, slot_unroll=10),
    7: E.LoopCfg(unit_par=True, count_search=True),                                  # = engine.CFG_PAR
    8: E.LoopCfg(unit_par=True, count_search=True, slot_static=True, slot_unroll=10),
    9: E.CFG_GPU,                                                                    # no syncs at all
    10: E.LoopCfg(unit_par=True, count_search=True, slot_static=True, slot_unroll=10, no_cond=True),
    11: E.LoopCfg(count_search=True, no_cond=True),
    12: E.LoopCfg(unit_static=True, slot_static=True, count_search=True, no_cond=True),   # static trip counts
    13: E.LoopCfg(unit_par=True, count_search=True, no_cond=True),
}


def make_rollout(B, nsteps, cfg):
    tb = E.tables()
    stepf = E.make_step(cfg)

    @jax.jit
    def rollout(state, key):
        def body(carry, _):
            st, key = carry
            key, ka, kb = jax.random.split(key, 3)
            st = stepf(tb, st, random_actions(ka, B, st.g, 0, st.t), random_actions(kb, B, st.g, 1, st.t), None)
            return (st, key), None
        (state, key), _ = lax.scan(body, (state, key), None, length=nsteps)
        return state, key
    return rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--variants", default="0,1,2,3")
    args = ap.parse_args()
    print("devices", jax.devices(), flush=True)
    B = args.B
    for v in [int(x) for x in args.variants.split(",")]:
        cfg = VARIANTS[v]
        roll = make_rollout(B, args.steps, cfg)
        st = E.init(B, seed=1)
        st = st._replace(g=st.g._replace(money=jnp.full_like(st.g.money, 5_000_000)))
        key = jax.random.PRNGKey(0)
        t0 = time.time()
        st2, key = roll(st, key)
        jax.block_until_ready(st2)
        tc = time.time() - t0
        best = 1e9
        for rep in range(2):
            t0 = time.time()
            st2, key = roll(st2, key)
            jax.block_until_ready(st2)
            best = min(best, time.time() - t0)
        print(f"variant {v} {cfg}: compile+run {tc:.1f}s, B={B} steps={args.steps}: {best:.3f}s -> "
              f"{B * args.steps / best:,.0f} game-steps/s ({1000 * best / args.steps:.1f} ms/step)", flush=True)


if __name__ == "__main__":
    main()
