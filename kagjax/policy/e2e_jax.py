"""End-to-end, JAX side: B games on device, both seats our policy (optionally a different checkpoint/team for the
opponent seat), rule layer as deployed (kagjax/train/actions.apply_rules, RuleCfg below), exact day words, one jitted
step = adapter + encoder + net + greedy decide + rules + engine.  Records every post-rule action and the final banks;
e2e_ref.py plays the same seeds with the C++ engine and the real exported numpy packages, e2e_compare.py compares.

precision: "highest" (fp32, the deployed arithmetic) or "default" (TF32 matmuls/convs on Ampere).  --shadow also
computes the fp32 (highest) decisions on the SAME states every step and records where the acting decision differs
(train/deploy flip rate along the trajectory).
Opponent games: the learner plays seat 0 in games [0, B/2) and seat 1 in [B/2, B) (lax-free: two policy_step calls on
the gathered rows, like the trainer's per-block opponents).

usage: python -m kagjax.policy.e2e_jax --learner CK --lteam 10 --ltau 0.5 [--opp CK --oteam 0 --otau 0.5]
         --games 64 --seed0 880000 --precision highest [--shadow] [--engine par|seq] --out file.npz
"""
import sys, os, time, json, argparse
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/kagjax/train")
import jax
import jax.numpy as jnp
from kagjax.policy import encode as E, act as A, net as N, adapter as AD
from kagjax.policy.test_selfplay import prices_tb
from kagjax import engine as EN, rng as RNG, consts as EC
import actions as TA                      # the trainer's rule layer (Agent B), read-only
import real_backend as RB

PREC = {"highest": N.HIGHEST, "default": None}
RULES = TA.RuleCfg(midnight=1, final_fix=1, homing=1, se_guard=1, se_hour=12, se_last_day=13, se_reserve=300,
                   sheep_cap=0, seed_budget=0)     # = features.py with RULES="SE_GUARD=1" (the deployed packages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", required=True); ap.add_argument("--lteam", type=int, default=10); ap.add_argument("--ltau", type=float, default=0.5)
    ap.add_argument("--opp"); ap.add_argument("--oteam", type=int, default=0); ap.add_argument("--otau", type=float, default=0.5)
    ap.add_argument("--games", type=int, default=64); ap.add_argument("--seed0", type=int, default=880000)
    ap.add_argument("--precision", default="highest"); ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--engine", default="seq"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.opp:                                             # self-play: both seats are the learner (team, tau)
        a.oteam, a.otau = a.lteam, a.ltau
    B = a.games
    pL, _ = N.load_ckpt(a.learner)
    pO = N.load_ckpt(a.opp)[0] if a.opp else pL
    seeds = np.arange(a.seed0, a.seed0 + B)
    lseat = np.zeros(B, np.int32)
    if a.opp:
        lseat[B // 2:] = 1
    rows = np.arange(2 * B)                                   # raw_both_seats order: seat 0 rows, then seat 1 rows
    seat_of = rows // B; game_of = rows % B
    idxL = rows[seat_of == lseat[game_of]]; idxO = rows[seat_of != lseat[game_of]]
    idxL_j, idxO_j = jnp.asarray(idxL), jnp.asarray(idxO)
    words = jnp.asarray(np.moveaxis(RNG.batch_words(seeds), 0, 1))        # [B,30,624]
    tb = EN.tables(); tabs = E.tables_jnp()
    loop = {} if a.engine == "seq" else {"cfg": EN.CFG_PAR}
    cfgL = A.Cfg(greedy=True, tau=a.ltau); cfgO = A.Cfg(greedy=True, tau=a.otau)
    teamL = jnp.full(len(idxL), a.lteam, jnp.int32); teamO = jnp.full(len(idxO), a.oteam, jnp.int32)
    prec = PREC[a.precision]

    def decide_rows(params, raw, prev, key, team, cfg, idx, precision):
        r = jax.tree_util.tree_map(lambda x: x[idx], raw)
        d = A.policy_step(params, r, prev[idx], key, team, cfg, tabs, precision)
        view = jax.vmap(RB.view_from_raw)(r)
        act = jax.vmap(lambda v, x: TA.apply_rules(v, x, RULES))(view, TA.Act(*d["act"]))
        return A.Act(*act), d["next_prev"]

    def all_rows(params_l, params_o, raw, prev, key, precision):
        k1, k2 = jax.random.split(key)
        aL, nL = decide_rows(params_l, raw, prev, k1, teamL, cfgL, idxL_j, precision)
        aO, nO = decide_rows(params_o, raw, prev, k2, teamO, cfgO, idxO_j, precision)
        act = jax.tree_util.tree_map(lambda l, o: jnp.zeros((2 * B,) + l.shape[1:], l.dtype).at[idxL_j].set(l).at[idxO_j].set(o), aL, aO)
        nprev = jnp.zeros((2 * B, 16), jnp.int32).at[idxL_j].set(nL).at[idxO_j].set(nO)
        return act, nprev

    @jax.jit
    def step(params_l, params_o, st, prev, prev_hi):
        raw = AD.raw_both_seats(st.g, st.t, prices_tb(tb, st.g))
        key = jax.random.PRNGKey(0)
        act, nprev = all_rows(params_l, params_o, raw, prev, key, prec)
        flip = jnp.zeros(2 * B, bool); nprev_hi = prev_hi
        if a.shadow:
            # fp32 decisions on the same state (with the fp32 run's own prev ops = the acting prev ops, since the
            # prev feature is part of the state the acting policy produced)
            act_hi, _ = all_rows(params_l, params_o, raw, prev, key, N.HIGHEST)
            flip = jnp.zeros(2 * B, bool)
            for x, y in zip(act, act_hi):
                flip = flip | (x != y).reshape(2 * B, -1).any(1)
        a0 = jax.tree_util.tree_map(lambda x: x[:B], act); a1 = jax.tree_util.tree_map(lambda x: x[B:], act)
        st2 = EN._step_impl(tb, st, AD.actions_from_policy(a0), AD.actions_from_policy(a1), words, **loop)
        return st2, nprev, act, flip

    st = EN.init(B)
    prev = jnp.full((2 * B, 16), -1, jnp.int32)
    rec = {f: [] for f in A.Act._fields}; flips = []
    t0 = time.time()
    for t in range(EC.LAST_STEP + 1):
        st, prev, act, flip = step(pL, pO, st, prev, prev)
        act = jax.device_get(act)
        for f in A.Act._fields:
            rec[f].append(np.asarray(getattr(act, f)).astype(np.int16))
        flips.append(np.asarray(flip))
        if t == 0:
            print(f"compiled + step 0 in {time.time() - t0:.0f}s", flush=True)
        if (t + 1) % 120 == 0:
            print(f"step {t + 1}: {time.time() - t0:.0f}s", flush=True)
    banks = np.asarray(jax.device_get(st.g.money))
    out = {f: np.stack(v) for f, v in rec.items()}                          # [T, 2B, ...]
    out.update(banks=banks, seeds=seeds, lseat=lseat, flips=np.stack(flips), B=np.int32(B))
    meta = dict(vars(a), device=str(jax.devices()[0]), time_s=time.time() - t0, rules=RULES._asdict())
    np.savez_compressed(a.out, meta=np.array(json.dumps(meta)), **out)
    print(f"saved {a.out}: {B} games, {time.time() - t0:.0f}s; mean banks {banks.mean(0)}; "
          f"shadow flips {int(np.stack(flips).sum())} row-steps" if a.shadow else f"saved {a.out}: {B} games, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
