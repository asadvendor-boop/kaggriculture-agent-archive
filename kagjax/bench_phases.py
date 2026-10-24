"""Per-phase timing of one engine step on realistic mid-day states (~19 units per farm, 2-10 orders per player).
usage: python bench_phases.py --B 4096 [--cfg 0]
"""
import argparse
import time

import jax
import jax.numpy as jnp
from jax import lax

import consts as C
import engine as E
from bench import random_actions
from bench_variants import VARIANTS


def timeit(f, *args, reps=5):
    out = f(*args)
    jax.block_until_ready(out)
    best = 1e9
    for _ in range(reps):
        t0 = time.time()
        out = f(*args)
        jax.block_until_ready(out)
        best = min(best, time.time() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4096)
    ap.add_argument("--cfg", type=int, default=0)
    args = ap.parse_args()
    B = args.B
    cfg = VARIANTS[args.cfg]
    tb = E.tables()
    print("devices", jax.devices(), "B", B, cfg, flush=True)
    # realistic state: 12 steps of the bench policy from the morning (hires 3 per step in hours 0-5)
    stepf = E.make_step(cfg)
    st = E.init(B, seed=1)
    st = st._replace(g=st.g._replace(money=jnp.full_like(st.g.money, 5_000_000)))
    key = jax.random.PRNGKey(0)
    for s in range(12):
        key, ka, kb = jax.random.split(key, 3)
        st = stepf(tb, st, random_actions(ka, B, st.g, 0, st.t), random_actions(kb, B, st.g, 1, st.t), None)
    jax.block_until_ready(st)
    g, t = st.g, st.t
    print("state: t", int(t), "mean units", float(jnp.mean(g.n_units)), flush=True)
    key, ka, kb = jax.random.split(key, 3)
    a0 = random_actions(ka, B, g, 0, t)
    a1 = random_actions(kb, B, g, 1, t)
    acts = E._stack_players(a0, a1)
    umax = C.U if cfg.unit_static else jnp.max(g.n_units)
    smax = C.NSLOT if cfg.slot_static else jnp.max(jnp.where(acts.mtype != C.M_NONE, jnp.arange(1, 11), 0))
    day = t // 24

    @jax.jit
    def units(g, acts, t, umax):
        f = jax.vmap(jax.vmap(lambda *a: E._unit_phase(*a, umax=umax, unroll=cfg.unit_unroll),
                              in_axes=(0,) * 11 + (None, None)), in_axes=(0,) * 11 + (None, None))
        return f(g.tiles, g.ux, g.uy, g.shed, g.seeds, g.inv, g.ins, acts.uop, acts.uarg, acts.un, g.n_units,
                 t // 24, t)

    @jax.jit
    def market(g, acts, smax):
        f = jax.vmap(lambda *a: E._market_phase(tb, *a, smax=smax, unroll=cfg.slot_unroll))
        return f(g.money, g.shed, g.seeds, g.minv, g.hires, g.n_units, g.ux, g.uy, g.n_quads, acts.mtype,
                 acts.mitem, acts.mn)

    @jax.jit
    def eod(g, words, day):
        f = jax.vmap(lambda g1, w: E._end_of_day(g1.tiles, g1.shed, g1.inv, g1.ins, g1.ux, g1.uy, g1.n_units,
                                                 g1.hires, g1.shops, g1.n_shops, w, day)[:10])
        return f(g, words)

    @jax.jit
    def full(st, a0, a1):
        return stepf(tb, st, a0, a1, None)

    words = jax.random.bits(key, (B, C.NW_WORDS), jnp.uint32)
    if isinstance(umax, int):
        tu = timeit(jax.jit(lambda g, a, t: units(g, a, t, umax)), g, acts, t)
    else:
        tu = timeit(units, g, acts, t, umax)
    if isinstance(smax, int):
        tm = timeit(jax.jit(lambda g, a: market(g, a, smax)), g, acts)
    else:
        tm = timeit(market, g, acts, smax)
    te = timeit(eod, g, words, day)
    tf = timeit(full, st, a0, a1)
    print(f"umax={umax if isinstance(umax, int) else int(umax)} smax={smax if isinstance(smax, int) else int(smax)}")
    print(f"units {1000 * tu:.2f} ms | market {1000 * tm:.2f} ms | end-of-day {1000 * te:.2f} ms | "
          f"full step (t={int(t)}, no day end) {1000 * tf:.2f} ms  -> {B / tf:,.0f} game-steps/s", flush=True)


if __name__ == "__main__":
    main()
