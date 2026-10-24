"""Parity of the JAX action layer (actions.py) with the Python reference (il/features.py) on real C++ engine states.

Worlds: recorded top games (data/tapes_pull) replayed in the C++ engine with the world pinned (as il/rollout.World).
At every step and seat we take the observation and build several policy-representable actions:
  (a) the recorded action mapped into the policy vocabulary (unit ops -> OPS classes, market -> 21 head classes ->
      features.decode_market),
  (b) random legal unit ops + random market classes (stress: big sells, many hires, purchases, land),
  (c) (b) with units forced to PICKUP / DROP / HARVEST where legal (shed-overflow paths of the midnight sale).
and compare, for each rule configuration (defaults; SE guard; SE guard + sheep cap 15 + seed budget):
  features.midnight_fix(obs, action)  vs  act_to_dict(apply_rules(obs_to_view(obs), dict_to_act(action)))
plus decode_market (random classes) and the quantity clamp of live_rollout (random op / qty classes).

run: kagjax/train/run1.sh kagjax/train/test_rules.py [n_games]
"""
import sys, os, json, time, random
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
import features as F
import actions as A

R = "/Users/asad.ali/kaggriculture"


def legal_mask(obs, k, pos):
    from agent import legal_mask as lm             # imports torch (only for the reference mask)
    return lm(obs, k, pos)


def policy_action(obs, rec, rng, mode):
    """A policy-representable action dict for obs: mode 0 = recorded, 1 = random, 2 = random with forced shed ops."""
    me = int(obs["player"]); farm = obs["farms"][me]
    positions = ([farm["farmer"]] + list(farm.get("hands") or []))[:F.MAX_UNITS]
    shed = obs["private"].get("shed", {}) or {}; invs = obs["private"].get("inventories") or [{}]
    cmds = []
    if mode == 0:
        rc = [rec.get("farmer") or ["PASS"]] + list(rec.get("hands") or []) if isinstance(rec, dict) else [["PASS"]]
        for k in range(len(positions)):
            oc, qc = F.unit_op_label(rc[k]) if k < len(rc) else (0, -1)
            cmds.append(F.unit_command(oc, max(qc, 0)))
        classes = F.market_labels(rec.get("market") if isinstance(rec, dict) else [])
    else:
        for k, p in enumerate(positions):
            m = legal_mask(obs, k, p)
            ok = np.flatnonzero(m)
            if mode == 2:
                pref = [i for i in ok if F.OPS[i].startswith("PICKUP_") or F.OPS[i] in ("DROP", "HARVEST", "COLLECT_FERTILIZER")]
                if pref and rng.rand() < 0.8:
                    ok = np.array(pref)
            oc = int(rng.choice(ok)); qc = 0
            name = F.OPS[oc]
            if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                qc = int(rng.randint(12)); it = name.split("_", 1)[1]
                have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                qc = max(0, min(qc, int(have) - 1))
            cmds.append(F.unit_command(oc, qc))
        classes = np.zeros(F.N_HEADS, np.int64)
        for h, (name, n) in enumerate(F.MARKET_HEADS):
            if rng.rand() < (0.35 if name.startswith("SELL") else 0.12):
                classes[h] = rng.randint(1, n)
    mk = F.decode_market(classes, obs)
    return {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": mk}


def norm(a):
    """Canonical comparable form of an action dict (commands up to 16 units, market as tuples)."""
    cm = [list(a.get("farmer") or ["PASS"])] + [list(c) for c in (a.get("hands") or [])]
    cm = [[c[0]] + ([c[1]] if len(c) > 1 else []) + ([int(c[2])] if len(c) > 2 else []) for c in cm]
    mk = [tuple([o[0]] + ([o[1], int(o[2])] if len(o) >= 3 else [])) for o in (a.get("market") or [])]
    return cm, mk


def stack(xs):
    return jax.tree_util.tree_map(lambda *a: np.stack(a), *xs)


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    from rollout import load_world
    from harness import kagsim_trace, PASS
    idx = json.load(open(f"{R}/data/tapes_pull/index.json"))
    eps = sorted(idx)[:: max(1, len(idx) // (n_games * 3))][:n_games]
    rng = np.random.RandomState(0)
    cfgs = {"default": A.RuleCfg(), "se": A.RuleCfg(se_guard=1), "se_cap_seed": A.RuleCfg(se_guard=1, sheep_cap=15, seed_budget=1),
            "no_rules": A.RuleCfg(midnight=0)}
    cases = []                                           # (obs, action) pairs
    t0 = time.time()
    for ep in eps:
        path = f"{R}/data/tapes_pull/{ep}.json.gz"
        d, shops, e0 = load_world(path)
        g = kagsim_trace.Game(int(d["seed"]), 720, False); g.set_forced_shops(shops)
        g.set_weed_prefix(1, e0)
        acts = d["actions"]
        for t in range(720):
            obs = [g.observe(0), g.observe(1)]
            for s in (0, 1):
                o = obs[s]; st = int(o["step"])
                rec = acts[s][t] if t < len(acts[s]) else None
                hot = st % 24 == 23 or st >= 690 or (238 <= st < 336)
                modes = [0, 1, 2] if hot else ([0] if rng.rand() < 0.15 else [])
                for mo in modes:
                    try:
                        a = policy_action(o, rec, rng, mo)
                    except Exception as e:                # recorded commands outside the vocabulary
                        continue
                    if len(a["market"]) <= 10 and 1 + len(a["hands"]) <= 16:
                        cases.append((o, a))
            now = [acts[p][t] if t < len(acts[p]) and isinstance(acts[p][t], dict) else dict(PASS) for p in (0, 1)]
            g.step(now[0], now[1])
            if g.done:
                break
    print(f"{len(cases)} cases from {len(eps)} games ({time.time() - t0:.0f}s)", flush=True)

    views = stack([A.obs_to_view(o) for o, _ in cases])
    acts_in = stack([A.dict_to_act(a) for _, a in cases])
    views = jax.tree_util.tree_map(jnp.asarray, views); acts_in = jax.tree_util.tree_map(jnp.asarray, acts_in)
    import copy
    ok_all = True
    for name, cfg in cfgs.items():
        F.SE_GUARD = cfg.se_guard; F.SHEEP_CAP = cfg.sheep_cap; F.SEED_BUDGET = cfg.seed_budget
        F.FINAL_FIX = cfg.final_fix; F.HOMING = cfg.homing; F.WASH = 0
        fn = jax.jit(jax.vmap(lambda v, a: A.apply_rules(v, a, cfg)))
        t1 = time.time(); out = jax.tree_util.tree_map(np.asarray, fn(views, acts_in)); t_jax = time.time() - t1
        bad = 0; changed = 0; first = None; kinds = {}
        for i, (o, a) in enumerate(cases):
            ref = F.midnight_fix(o, copy.deepcopy(a)) if cfg.midnight else a
            got = A.act_to_dict(jax.tree_util.tree_map(lambda x: x[i], out))
            if norm(ref) != norm(got):
                bad += 1
                if first is None:
                    first = (int(o["step"]), norm(a), norm(ref), norm(got))
            if norm(ref) != norm(a):
                changed += 1
                st = int(o["step"])
                k = ("final" if st >= 718 else "homing+midnight" if st == 695 else "homing" if st >= 690 else
                     "midnight" if st % 24 == 23 else "plan")
                if k == "midnight" and norm(ref)[0] != norm(a)[0]:
                    k = "midnight(units?)"
                kinds[k] = kinds.get(k, 0) + 1
        print(f"[rules {name}] {len(cases)} cases, rule changed the action in {changed} {kinds}, mismatches {bad} (jax {t_jax:.1f}s)", flush=True)
        if first:
            print("  first mismatch: step", first[0], "\n  in :", first[1], "\n  ref:", first[2], "\n  jax:", first[3])
        ok_all &= bad == 0
    F.SE_GUARD = 0; F.SHEEP_CAP = 0; F.SEED_BUDGET = 0

    # decode_market on random classes (incl. class 17 and > 10 orders)
    cl = np.stack([[rng.randint(0, n) if rng.rand() < 0.5 else 0 for _, n in F.MARKET_HEADS] for _ in range(4000)])
    cl[:500, 19] = 10; cl[:500, :9] = 17
    dec = jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(A.decode_market))(jnp.asarray(cl, jnp.int32)))
    bad = 0
    for i in range(len(cl)):
        ref = F.decode_market(cl[i])
        a = A.Act(np.zeros(16, np.int32), np.zeros(16, np.int32), np.ones(16, np.int32), np.int32(1), dec[0][i], dec[1][i], dec[2][i], dec[3][i])
        bad += norm({"market": ref})[1] != norm(A.act_to_dict(a))[1]
    print(f"[decode_market] 4000 random class vectors, mismatches {bad}", flush=True)
    ok_all &= bad == 0

    # quantity clamp / class -> command (live_rollout semantics)
    sub = cases[::7]
    V = jax.tree_util.tree_map(jnp.asarray, stack([A.obs_to_view(o) for o, _ in sub]))
    opc = rng.randint(0, F.N_OPS, (len(sub), 16)); qtc = rng.randint(0, 12, (len(sub), 16))
    nu = np.array([min(16, 1 + len(o["farms"][int(o["player"])].get("hands") or [])) for o, _ in sub])
    ua = jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(A.unit_actions))(jnp.asarray(opc, jnp.int32), jnp.asarray(qtc, jnp.int32),
                                                                                jnp.asarray(nu, jnp.int32), V))
    bad = 0
    for i, (o, _) in enumerate(sub):
        shed = o["private"].get("shed", {}) or {}; invs = o["private"].get("inventories") or [{}]
        for k in range(nu[i]):
            oc, qc = int(opc[i, k]), int(qtc[i, k]); name = F.OPS[oc]
            if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                it = name.split("_", 1)[1]
                have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                qc = max(0, min(qc, int(have) - 1))
            ref = F.unit_command(oc, qc)
            got = A.act_to_dict(A.Act(ua[0][i], ua[1][i], ua[2][i], np.int32(nu[i]), np.zeros(10, np.int32), np.zeros(10, np.int32),
                                      np.zeros(10, np.int32), np.int32(0)))
            gk = ([got["farmer"]] + got["hands"])[k]
            bad += norm({"farmer": ref})[0][0] != norm({"farmer": gk})[0][0]
    print(f"[unit_actions] {int(nu.sum())} unit commands, mismatches {bad}", flush=True)
    ok_all &= bad == 0
    print("ALL OK" if ok_all else "MISMATCHES FOUND", flush=True)


if __name__ == "__main__":
    main()
