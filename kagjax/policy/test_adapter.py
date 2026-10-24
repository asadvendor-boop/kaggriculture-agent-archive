"""Adapter parity: engine GameState (built by the engine's own kagjax.state.obs_to_game from real C++ observations)
-> adapter.raw_from_state -> encoder  ==  observation dict -> encode.raw_from_obs -> encoder, bit for bit; the
engine price function == observed prices; policy Act -> engine Actions == kagjax.state.encode_action(act dict).

run: kagjax/policy/run1.sh -m kagjax.policy.test_adapter [n_games] [every]
"""
import sys, json, os
import numpy as np
import jax
import jax.numpy as jnp
from kagjax.policy import encode as E, adapter as AD, act as A, net as N, refdata
from kagjax import state as S, engine as EN


def pairs(n_games, every):
    for f in refdata.tape_files(n_games, start=11):
        obs_by_t = {}
        for t, s, o, prev in refdata.iter_tape_obs(f, every=every):
            obs_by_t.setdefault(t, [None, None, None])
            obs_by_t[t][s] = o
            obs_by_t[t][2] = prev if s == 0 else obs_by_t[t][2]
            if obs_by_t[t][0] is not None and obs_by_t[t][1] is not None:
                yield t, obs_by_t.pop(t)


def main(n_games=12, every=5, chunk=128):
    tabs = E.tables_jnp()
    enc_f = jax.jit(jax.vmap(E.encode, in_axes=(0, 0, None)))
    raw_f = jax.jit(AD.raw_both_seats)
    items = list(pairs(n_games, every))
    stats = dict(pairs=len(items), raw_bad=0, enc_bad=0, price_bad=0, act_bad=0)
    t_groups = {}
    for t, (o0, o1, _) in items:
        t_groups.setdefault(t, []).append((o0, o1))
    for t, grp in sorted(t_groups.items()):
        games = [S.obs_to_game(o0, o1)[0] for o0, o1 in grp]
        g = jax.tree_util.tree_map(jnp.asarray, S.stack_games(games))
        pr = EN.prices(g)
        obs_pr = np.array([[o0["market"]["prices"][p] for p in AD.K.PRODUCTS] for o0, _ in grp], np.int32)
        stats["price_bad"] += int((np.asarray(pr) != obs_pr).sum())
        raw_st = jax.device_get(raw_f(g, t, pr))                             # [2B]
        raw_ob = E.stack([E.raw_from_obs(o0) for o0, _ in grp] + [E.raw_from_obs(o1) for _, o1 in grp])
        nb = len(grp) * 2
        # unit rows beyond n_units are stale in the engine and zero in raw_from_obs: compare masked
        for name in E.RawObs._fields:
            a = np.asarray(getattr(raw_st, name)); b = np.asarray(getattr(raw_ob, name))
            if name == "unit_xy":
                m = np.arange(a.shape[2])[None, None, :] < raw_ob.n_units[:, :, None]
                a = np.where(m[..., None], a, 0); b = np.where(m[..., None], b, 0)
            if name == "inv":
                m = np.arange(a.shape[1])[None, :] < raw_ob.n_units[:, 0:1]
                a = np.where(m[..., None], a, 0); b = np.where(m[..., None], b, 0)
            if a.shape != b.shape or not np.array_equal(a.astype(np.float64), b.astype(np.float64)):
                stats["raw_bad"] += 1
                if stats["raw_bad"] <= 5:
                    print("   RAW MISMATCH", t, name, a.shape, b.shape, flush=True)
        prev = np.full((nb, 16), -1, np.int32)
        e1 = jax.device_get(enc_f(raw_st, jnp.asarray(prev), tabs)); e2 = jax.device_get(enc_f(raw_ob, jnp.asarray(prev), tabs))
        for k in e1:
            if not np.array_equal(np.asarray(e1[k]).view(np.uint8), np.asarray(e2[k]).view(np.uint8)):
                stats["enc_bad"] += 1
                print("   ENC MISMATCH", t, k, flush=True)
        # actions: greedy decisions on random logits -> engine Actions via the adapter vs the engine's dict parser
        rng = np.random.RandomState(t)
        outs = {"op": jnp.asarray(rng.randn(nb, 16, 44).astype(np.float32) * 3),
                "qty": jnp.asarray(rng.randn(nb, 16, 12).astype(np.float32) * 3),
                "mk": jnp.asarray(rng.randn(nb, 249).astype(np.float32) * 3), "v": jnp.zeros(nb)}
        dec = jax.vmap(lambda r, o, k: A.decide(r, o, k, A.Cfg()))(raw_st, outs, jax.random.split(jax.random.PRNGKey(t), nb))
        ea = jax.device_get(AD.actions_from_policy(dec["act"]))
        for j in range(nb):
            d = A.act_to_dict(jax.tree_util.tree_map(lambda x: x[j], dec["act"]))
            ref = S.encode_action(d)
            got = [np.asarray(x[j]) for x in ea]
            if not all(np.array_equal(a_, b_) for a_, b_ in zip(got, ref)):
                stats["act_bad"] += 1
                if stats["act_bad"] <= 3:
                    print("   ACT MISMATCH", d, flush=True)
    print(f"adapter: {stats['pairs']} observation pairs ({2 * stats['pairs']} seat views): RawObs field mismatches "
          f"{stats['raw_bad']}, encoder output mismatches {stats['enc_bad']}, engine price mismatches "
          f"{stats['price_bad']}, Act->Actions mismatches {stats['act_bad']}", flush=True)
    json.dump(stats, open(os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))), "results_adapter.json"), "w"), indent=1)
    assert stats["raw_bad"] == stats["enc_bad"] == stats["price_bad"] == stats["act_bad"] == 0
    print("ADAPTER PARITY: EXACT")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
