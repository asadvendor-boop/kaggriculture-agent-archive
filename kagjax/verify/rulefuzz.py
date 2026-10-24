"""Differential fuzz of the trainer's action path vs the Kaggle runtime's Python:
   JAX:    engine state (state.obs_to_game) -> train/real_backend.raw_from_state -> view_from_raw -> actions.unit_actions
           (qty clamp) + actions.decode_market -> actions.apply_rules  -> act_to_dict
   Python: il/np_runtime.NumpyAgent's decoding (unit_command with the same clamp, features.decode_market) ->
           features.midnight_fix (the rule layer baked into the Kaggle package)
on random valid states (statefuzz generator, canonicalized by the C++ engine) at the rule-relevant steps (hour 23,
the homing window 690..717, the final turn 718, the SE window) and random policy classes (ops, quantities, market
classes incl. >10-order overflows, price ties). Rule configs: default (MIDNIGHT/FINALFIX/HOMING) and plan rules on.
usage (server): python rulefuzz.py --n 4000 --seed 0
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, os.path.join(R, "kagjax"))
sys.path.insert(0, os.path.join(R, "kagjax", "train"))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "il"))
sys.path.insert(0, os.environ.get("KAG_ENGINE", os.path.join(HERE, "cov_build", "opt")))
os.environ.setdefault("JAX_CACHE_DIR", "off")
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kagsim_trace as K  # noqa: E402
import statefuzz as SF  # noqa: E402  (my generator)
import state as S  # noqa: E402
import engine as E  # noqa: E402
import actions as A  # noqa: E402
import real_backend as RB  # noqa: E402
import features as F  # noqa: E402

STEPS = [23, 47, 71, 95, 119, 143, 167, 191, 215, 239, 240, 250, 263, 287, 300, 311, 335, 359, 431, 575, 671, 689, 690,
         691, 695, 700, 705, 710, 713, 714, 715, 716, 717, 718]


def py_action(obs, op_cls, q_cls, classes):
    """NumpyAgent's decoding (before midnight_fix) for the first min(16, n_units) units."""
    priv = obs["private"]; shed = priv.get("shed", {}) or {}; invs = priv.get("inventories") or [{}]
    n = min(16, 1 + len(obs["farms"][obs["player"]]["hands"]))
    cmds = []
    for k in range(n):
        oc = int(op_cls[k]); name = F.OPS[oc]; qc = 0
        if name.startswith("PICKUP_") or name.startswith("PLACE_"):
            qc = int(q_cls[k]); it = name.split("_", 1)[1]
            have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
            qc = max(0, min(qc, int(have) - 1))
        cmds.append(F.unit_command(oc, qc))
    mk = F.decode_market([int(c) for c in classes], obs)
    return {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": mk}


def norm(a):
    """canonical form for comparison: unit commands with ints, market orders as lists."""
    u = [a.get("farmer") or ["PASS"]] + list(a.get("hands") or [])
    u = [[c[0]] + [x if isinstance(x, str) else int(x) for x in c[1:]] for c in u]
    m = [[o[0]] + [x if isinstance(x, str) else int(x) for x in o[1:]] for o in (a.get("market") or [])]
    return u, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", type=int, default=0, help="1: SE_GUARD + SHEEP_CAP=15 + SEED_BUDGET on both sides")
    args = ap.parse_args()
    r = random.Random(args.seed)
    rc = A.RuleCfg(se_guard=1, sheep_cap=15, seed_budget=1) if args.plan else A.RuleCfg()
    F.SE_GUARD = 1 if args.plan else 0; F.SHEEP_CAP = 15 if args.plan else 0; F.SEED_BUDGET = 1 if args.plan else 0
    F.FINAL_FIX = 1; F.HOMING = 1; F.WASH = 0
    tb = E.tables()

    from functools import partial

    @partial(jax.jit, static_argnums=(2,))
    def jax_path(g, t, seat, op, q, cls):
        raw = RB.raw_from_state(g, t, seat, tb)
        view = jax.vmap(RB.view_from_raw)(raw)
        nu = jnp.minimum(16, raw.n_units[:, 0])

        def one(v, o, qq, c, n):
            u_op, u_arg, u_n = A.unit_actions(o, qq, n, v)
            oo, oi, on, nord = A.decode_market(c)
            a = A.Act(u_op, u_arg, u_n, n.astype(jnp.int32), oo, oi, on, nord)
            return a, A.apply_rules(v, a, rc)
        return jax.vmap(one)(view, op, q, cls, nu)

    n_bad, n_tot, n_changed, bad = 0, 0, 0, []
    B = 64
    while n_tot < args.n:
        t = r.choice(STEPS)
        obs_pairs, seats, pys, ops, qs, clss = [], [], [], [], [], []
        for b in range(B):
            fam = r.choice(["unit", "all", "eod", "market"])
            o0, o1 = SF.rand_state(r, t, fam)
            if r.random() < 0.3:                      # SE window setups: 3 quadrants, cash around $4,300
                for p in (0, 1):
                    o0["farms"][p]["unlocked_quadrants"] = ["NW", "NE", "SW"]
                    for y in range(5, 10):
                        for x in range(5, 10):
                            o0["farms"][p]["tiles"][y][x] = "LOCKED"
                    o0["farms"][p]["money"] = float(r.choice([4299, 4300, 4301, 9000]))
            g = K.Game(r.randrange(2 ** 31), 720, False)
            g.set_state(o0, o1)
            c0, c1 = g.observe(0), g.observe(1)
            seat = r.randint(0, 1)
            obs = (c0, c1)[seat]
            op = np.array([r.randrange(44) if r.random() < 0.5 else r.choice([0, 5, 6, 14, 20, 21, 29, 32, 33, 41])
                           for _ in range(16)], np.int32)
            q = np.array([r.randrange(12) for _ in range(16)], np.int32)
            cls = np.zeros(21, np.int32)
            for h, (_, size) in enumerate(A.MARKET_HEADS):
                if r.random() < (0.5 if r.random() < 0.3 else 0.15):
                    cls[h] = r.randrange(1, size)
            obs_pairs.append((c0, c1)); seats.append(seat); ops.append(op); qs.append(q); clss.append(cls)
            pys.append(py_action(obs, op, q, cls))
        gs = S.stack_games([S.obs_to_game(a, b_)[0] for a, b_ in obs_pairs])
        g = jax.tree_util.tree_map(jnp.asarray, gs)
        # per-seat views: run the jax path twice (seat 0 rows, seat 1 rows) and pick
        res = {}
        for seat in (0, 1):
            res[seat] = jax_path(g, jnp.int32(t), seat, jnp.asarray(np.stack(ops)), jnp.asarray(np.stack(qs)),
                                 jnp.asarray(np.stack(clss)))
        for b in range(B):
            pre, post = res[seats[b]]
            pre_b = jax.tree_util.tree_map(lambda x: np.asarray(x)[b], pre)
            post_b = jax.tree_util.tree_map(lambda x: np.asarray(x)[b], post)
            obs = obs_pairs[b][seats[b]]
            want_pre = norm(pys[b])
            got_pre = norm(A.act_to_dict(pre_b))
            want = norm(F.midnight_fix(obs, pys[b]))
            got = norm(A.act_to_dict(post_b))
            n_tot += 1
            n_changed += want != want_pre
            if got_pre != want_pre or got != want:
                n_bad += 1
                if len(bad) < 10:
                    bad.append({"t": t, "seat": seats[b], "pre_equal": got_pre == want_pre, "jax": got, "python": want,
                                "python_pre": want_pre, "obs": obs})
        print(f"t={t}: {n_tot} cases, rule changed the action in {n_changed}, mismatches {n_bad}", flush=True)
    out = os.path.join(HERE, "out", f"rulefuzz_{args.seed}_plan{args.plan}.json")
    json.dump({"cases": n_tot, "rule_changed": n_changed, "mismatches": n_bad, "examples": bad}, open(out, "w"))
    for x in bad[:3]:
        print("MISMATCH t", x["t"], "pre_equal", x["pre_equal"], "\n jax   ", x["jax"], "\n python", x["python"])
    print(f"RESULT {n_tot} cases ({n_changed} changed by the rule layer): {n_bad} mismatches -> {out}")


if __name__ == "__main__":
    main()
