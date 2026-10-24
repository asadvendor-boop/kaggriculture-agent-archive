"""Throughput benchmark: B games stepping with random legal-ish actions generated on device, jit + scan.

usage: env1.sh bench.py [--B 1024] [--steps 96] [--reps 2]      (GPU: run with the normal python, no env1.sh)
Reports game-steps/s = B * steps / wall time (after compilation). Randomness: training mode (words drawn from keys).
"""
import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import consts as C
import engine as E
from state import Actions

# unit-op mix: moves dominate, then field work, shed ops, some junk
_UOP_P = np.array([4, 10, 10, 10, 10, 3, 3, 2, 6, 8, 6, 2, 2, 1, 1, 4, 2, 3], np.float32)
_UOP_P = _UOP_P / _UOP_P.sum()
_MOP_P = np.array([0, 1.0, 0.1, 3, 2, 1, 6], np.float32)
_MOP_P = _MOP_P / _MOP_P.sum()


def random_actions(key, B, g=None, p=0, t=0):
    """Random legal-ish actions for player p. With the state g: hire 3 hands per step in hours 0-5 (~19 units, as
    strong bots do), sell what the shed holds, buy small amounts only while the bank is above $2k."""
    ks = jax.random.split(key, 8)
    uop = jax.random.choice(ks[0], C.N_ITEMS + 6, (B, C.U), p=jnp.asarray(_UOP_P)).astype(jnp.int32)
    uarg = jax.random.randint(ks[1], (B, C.U), 0, C.N_ITEMS)
    uarg = jnp.where(uop == C.OP_PLANT, uarg % C.N_CROPS, uarg)
    un = jax.random.randint(ks[2], (B, C.U), 1, 20)
    mtype = jax.random.choice(ks[3], 7, (B, C.NSLOT), p=jnp.asarray(_MOP_P)).astype(jnp.int32)
    nord = jax.random.randint(ks[4], (B, 1), 2, C.NSLOT + 1)
    mtype = jnp.where(jnp.arange(C.NSLOT)[None, :] < nord, mtype, 0)
    mitem = jax.random.randint(ks[5], (B, C.NSLOT), 0, C.N_ITEMS)
    mitem = jnp.where(mtype == C.M_BUY_SEED, mitem % C.N_CROPS, mitem)
    mitem = jnp.where(mtype == C.M_BUY_ANIMAL, C.GOOSE + mitem % 3, mitem)
    mitem = jnp.where(mtype == C.M_BUY_PRODUCT, jnp.where(mitem % 2 == 0, C.WHEAT, C.FERTILIZER), mitem)
    mitem = jnp.where(mtype == C.M_SELL, mitem % C.N_PRODUCTS, mitem)
    mn = jax.random.randint(ks[6], (B, C.NSLOT), 1, 6)
    if g is not None:
        rich = (g.money[:, p] > 2000)[:, None]
        buying = (mtype == C.M_BUY_SEED) | (mtype == C.M_BUY_PRODUCT) | (mtype == C.M_BUY_ANIMAL) | (
            mtype == C.M_BUY_LAND)
        mtype = jnp.where(buying & ~rich, C.M_SELL, mtype)
        mn = jnp.where(mtype == C.M_SELL, 50, mn)
        hire = (t % C.TURNS_PER_DAY < 6) & (jnp.arange(C.NSLOT)[None, :] < 3)
        mtype = jnp.where(hire, C.M_HIRE, mtype)
    return Actions(uop, uarg, un, mtype, mitem, mn)


def make_rollout(B, nsteps, cfg=None):
    tb = E.tables()
    stepf = E._step_jit if cfg is None else E.make_step(cfg)

    @jax.jit
    def rollout(state, key):
        def body(carry, _):
            st, key = carry
            key, ka, kb = jax.random.split(key, 3)
            a0 = random_actions(ka, B, st.g, 0, st.t)
            a1 = random_actions(kb, B, st.g, 1, st.t)
            st = stepf(tb, st, a0, a1, None)
            return (st, key), None
        (state, key), _ = lax.scan(body, (state, key), None, length=nsteps)
        return state, key
    return rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--full", action="store_true", help="also time a full 720-step episode")
    ap.add_argument("--cfg", default="default", help="default | seq | par | gpu | unrolled (engine.LoopCfg)")
    args = ap.parse_args()
    cfg = {"seq": E.CFG_SEQ, "par": E.CFG_PAR, "gpu": E.CFG_GPU, "unrolled": E.CFG_UNROLLED,
           "default": None}[args.cfg]
    print("devices", jax.devices(), "cfg", args.cfg, cfg)
    B = args.B
    roll = make_rollout(B, args.steps, cfg)
    st = E.init(B, seed=1)
    # benchmark only: a large bank so the random policy can keep hiring ~19 units/day all game (realistic loop
    # trip counts); rules are unchanged
    st = st._replace(g=st.g._replace(money=jnp.full_like(st.g.money, 5_000_000)))
    key = jax.random.PRNGKey(0)
    t0 = time.time()
    st2, key = roll(st, key)
    jax.block_until_ready(st2)
    print(f"compile + first {args.steps} steps: {time.time() - t0:.1f}s")
    for rep in range(args.reps):
        t0 = time.time()
        st2, key = roll(st2, key)
        jax.block_until_ready(st2)
        dt = time.time() - t0
        print(f"B={B} steps={args.steps}: {dt:.3f}s -> {B * args.steps / dt:,.0f} game-steps/s "
              f"(t={int(st2.t)}, mean units {float(jnp.mean(st2.g.n_units)):.1f}, "
              f"mean money {float(jnp.mean(st2.g.money)):.0f})")
    if args.full:
        roll720 = make_rollout(B, 720, cfg)
        s0 = E.init(B, seed=2)
        s0 = s0._replace(g=s0.g._replace(money=jnp.full_like(s0.g.money, 5_000_000)))
        st3, _ = roll720(s0, key)
        jax.block_until_ready(st3)
        t0 = time.time()
        st3, _ = roll720(s0, jax.random.PRNGKey(5))
        jax.block_until_ready(st3)
        dt = time.time() - t0
        print(f"full episode B={B}: {dt:.2f}s -> {B * 720 / dt:,.0f} game-steps/s, done={E.done(st3)}")


if __name__ == "__main__":
    main()
