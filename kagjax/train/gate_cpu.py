"""Pre-training gate (operator, Sep 26), CPU part (server cores, no GPU needed).

  rules      item 2: C++ engine games played by the numpy Kaggle runtime (il/np_runtime.NumpyAgent) with
             g20_u_dsm_s0 and rl2_it24 (team 10, tau 0.5); every call of features.midnight_fix is intercepted, and
             the recorded (observation, pre-rule action) is replayed through actions.apply_rules (JAX) and
             features.midnight_fix (numpy/Python, incl. _plan_fix) for the default rules, SE_GUARD=1, and
             SE_GUARD=1 + SHEEP_CAP=15 + SEED_BUDGET=1.  Also real states with the market padded to 10 orders
             (the 10-order cap paths).  Counts by case; any mismatch is printed.
  roundtrip  item 4: <pt> (a JAX-trained snapshot written by ckpt_io) -> il/export.py (B64=1, RULES="SE_GUARD=1") ->
  <pt>       numpy package -> C++ games (package in both seats) -> the package's actions at every decision vs the
             JAX greedy pipeline (C's raw_from_obs + encoder + net + sample.decide + rules with SE_GUARD=1) on the
             same observations with the package's own previous ops; plus: the .pt loads in il/live_rollout.py and
             il/agent.BCAgent (torch) and acts.
Writes $GATE_DIR/gate_cpu_<mode>.json.
"""
import os, sys, json, time, copy, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, f"{R}/il"); sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import actions as A
from test_rules import norm

GD = os.environ.get("GATE_DIR", f"{R}/kagjax/logs_server/D/gate")
CFGS = {"default": A.RuleCfg(), "se_guard": A.RuleCfg(se_guard=1), "se_cap15_seed": A.RuleCfg(se_guard=1, sheep_cap=15, seed_budget=1)}


def set_py_rules(F, cfg):
    F.SE_GUARD, F.SHEEP_CAP, F.SEED_BUDGET, F.FINAL_FIX, F.HOMING, F.WASH = cfg.se_guard, cfg.sheep_cap, cfg.seed_budget, cfg.final_fix, cfg.homing, 0


def numpy_agent(ckpt, team, tau=0.5):
    import torch, features6 as F6, np_runtime as NR
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    W = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
    return NR.NumpyAgent(NR.NumpyNet(W, width=c["width"], blocks=c["blocks"], n_teams=c["n_teams"]), F6, tau=tau, team=team)


def play_and_record(agents, seed, F, play_cfg, hold_se=False):
    """One C++ game; returns [(seat, obs, pre-rule action)] for every decision (intercepting features.midnight_fix).
    hold_se: the models' own BUY_LAND is withheld while 3 quadrants are owned (days <= 13), so days 10-13 are full of
    SE-guard-eligible states (the recorded action is the one the rule layer sees)."""
    from harness import kagsim_trace
    import features6 as F6
    rec, orig = [], F.midnight_fix

    def spy(obs, act):
        farm = obs["farms"][int(obs["player"])]
        if hold_se and len(farm.get("unlocked_quadrants") or []) == 3 and int(obs["step"]) // 24 <= 13:
            act = dict(act, market=[o for o in (act.get("market") or []) if o[0] != "BUY_LAND"])
        rec.append((int(obs["player"]), copy.deepcopy(obs), copy.deepcopy(act)))
        return orig(obs, act)

    F6.midnight_fix = spy
    set_py_rules(F, play_cfg)
    try:
        g = kagsim_trace.Game(int(seed), 720, False)
        for t in range(720):
            o = [g.observe(0), g.observe(1)]
            a = [agents[p](copy.deepcopy(o[p])) for p in (0, 1)]
            g.step(a[0], a[1])
            if g.done:
                break
    finally:
        F6.midnight_fix = orig
    return rec, (g.reward(0), g.reward(1))


def case_of(obs, act):
    st = int(obs["step"]); day, hour = divmod(st, 24)
    farm = obs["farms"][int(obs["player"])]
    tags = []
    if st >= 718:
        tags.append("final_turn")
    elif st >= 690:
        tags.append("last_day_homing")
    if st % 24 == 23 and st < 718:
        tags.append("midnight_turn")
    if 10 <= day <= 13 and len(farm.get("unlocked_quadrants") or []) == 3 and float(farm["money"]) >= 4300:
        tags.append("se_window_3quads_4300")
    if len(act.get("market") or []) >= 10:
        tags.append("ten_orders")
    return tags


def compare(cases, F):
    """cases: [(obs, action)] -> per rule config: n, changed, mismatches, by-tag counts."""
    views = jax.tree_util.tree_map(lambda *x: jnp.asarray(np.stack(x)), *[A.obs_to_view(o) for o, _ in cases])
    acts = jax.tree_util.tree_map(lambda *x: jnp.asarray(np.stack(x)), *[A.dict_to_act(a) for _, a in cases])
    res = {}
    for name, cfg in CFGS.items():
        set_py_rules(F, cfg)
        out = jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(lambda v, a: A.apply_rules(v, a, cfg)))(views, acts))
        bad, changed, tags, first = 0, 0, {}, None
        for i, (o, a) in enumerate(cases):
            ref = F.midnight_fix(copy.deepcopy(o), copy.deepcopy(a))
            got = A.act_to_dict(jax.tree_util.tree_map(lambda x: x[i], out))
            ch = norm(ref) != norm(a)
            changed += ch
            for tg in case_of(o, a):
                t_ = tags.setdefault(tg, [0, 0, 0]); t_[0] += 1; t_[1] += ch; t_[2] += norm(ref) != norm(got)
            if norm(ref) != norm(got):
                bad += 1
                if first is None:
                    first = dict(step=int(o["step"]), action=norm(a), ref=norm(ref), jax=norm(got))
        res[name] = dict(states=len(cases), rule_changed_action=changed, mismatches=bad, by_case_total_changed_mismatch=tags, first_mismatch=first)
    set_py_rules(F, CFGS["default"])
    return res


def pad_to_10(obs, act):
    """Same state, market filled up to 10 orders with HIREs placed after the sells (a decode_market-shaped list)."""
    mk = list(act.get("market") or [])
    if len(mk) >= 10:
        return None
    sells = [o for o in mk if o[0] == "SELL"]; rest = [o for o in mk if o[0] != "SELL"]
    return dict(act, market=sells + [["HIRE"]] * (10 - len(mk)) + rest)


def mode_rules():
    import features as F
    ck = {"g20_u_dsm_s0": f"{R}/il/ckpt/g20_u_dsm_s0.pt", "rl2_it24": f"{R}/il/ckpt/rl2_it24.pt"}
    games = [(1234567, ("g20_u_dsm_s0", "rl2_it24"), "se_guard"), (7654321, ("rl2_it24", "g20_u_dsm_s0"), "default")]
    out = {"games": []}
    cases = []
    t0 = time.time()
    for seed, (m0, m1), play in games:
        ag = [numpy_agent(ck[m0], 10), numpy_agent(ck[m1], 10)]
        rec, rw = play_and_record(ag, seed, F, CFGS[play])
        cases += [(o, a) for _, o, a in rec]
        out["games"].append(dict(seed=seed, seat0=m0, seat1=m1, played_with=play, decisions=len(rec), final=rw, seconds=round(time.time() - t0)))
        print("[rules] game", out["games"][-1], flush=True)
    out["real_states"] = compare(cases, F)
    aug = [(o, p) for o, a in cases if (int(o["step"]) % 24 == 23 or 238 <= int(o["step"]) < 336 or int(o["step"]) >= 690)
           for p in [pad_to_10(o, a)] if p is not None]
    out["real_states_market_padded_to_10"] = compare(aug, F)
    ok = all(v["mismatches"] == 0 for k in ("real_states", "real_states_market_padded_to_10") for v in out[k].values())
    out["ALL_IDENTICAL"] = ok
    json.dump(out, open(os.path.join(GD, "gate_cpu_rules.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "games"}, default=str)[:3000], flush=True)
    print("RULES IDENTICAL" if ok else "RULES MISMATCH", flush=True)


def mode_se():
    """SE-guard states from model games with the third land withheld (default rules during play, so no guard fires)."""
    import features as F
    ck = {"g20_u_dsm_s0": f"{R}/il/ckpt/g20_u_dsm_s0.pt", "rl2_it24": f"{R}/il/ckpt/rl2_it24.pt"}
    out = {"games": []}
    cases = []
    t0 = time.time()
    for seed, (m0, m1) in ((24680, ("g20_u_dsm_s0", "rl2_it24")), (13579, ("rl2_it24", "g20_u_dsm_s0"))):
        ag = [numpy_agent(ck[m0], 10), numpy_agent(ck[m1], 10)]
        rec, rw = play_and_record(ag, seed, F, CFGS["default"], hold_se=True)
        win = [(o, a) for _, o, a in rec if 9 <= int(o["step"]) // 24 <= 14]
        cases += win
        out["games"].append(dict(seed=seed, seat0=m0, seat1=m1, decisions=len(rec), days_9_to_14=len(win), final=rw, seconds=round(time.time() - t0)))
        print("[se] game", out["games"][-1], flush=True)
    out["states_days_9_to_14"] = compare(cases, F)
    out["ALL_IDENTICAL"] = all(v["mismatches"] == 0 for v in out["states_days_9_to_14"].values())
    json.dump(out, open(os.path.join(GD, "gate_cpu_se.json"), "w"), indent=1, default=str)
    print(json.dumps(out["states_days_9_to_14"], default=str)[:2500], flush=True)
    print("SE RULES IDENTICAL" if out["ALL_IDENTICAL"] else "SE RULES MISMATCH", flush=True)


def mode_roundtrip(pt):
    import torch
    out = {"pt": pt}
    pkg = os.path.join(GD, "pkg_roundtrip", "main.py")
    os.makedirs(os.path.dirname(pkg), exist_ok=True)
    env = dict(os.environ, B64="1", RULES="SE_GUARD=1", OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, f"{R}/il/export.py", pt, pkg, "0.5", "10"], env=env, capture_output=True, text=True)
    out["export_stdout"] = (r.stdout + r.stderr)[-400:]
    assert r.returncode == 0, out["export_stdout"]
    from harness import mk, kagsim_trace
    make = mk(pkg)
    # 1) package games in the C++ engine; record observations, the package's previous ops and actions
    rec = []
    t0 = time.time()
    for seed in (2468, 1357):
        ag = [make(), make()]
        g = kagsim_trace.Game(seed, 720, False)
        for t in range(720):
            o = [g.observe(0), g.observe(1)]
            a = []
            for p in (0, 1):
                st = ag[p].__globals__["_AGENT"].state.get(p)
                prev = None if (st is None or int(o[p]["step"]) == 0 or int(o[p]["step"]) <= st["step"]) else st["prev"]
                a.append(ag[p](copy.deepcopy(o[p])))
                rec.append((o[p], prev, a[-1]))
            g.step(a[0], a[1])
            if g.done:
                break
        out.setdefault("games", []).append(dict(seed=seed, final=(g.reward(0), g.reward(1)), seconds=round(time.time() - t0)))
        print("[roundtrip] game", out["games"][-1], flush=True)
    # 2) JAX greedy pipeline on the same observations (SE guard on, tau 0.5, team 10)
    from kagjax.policy import encode as CE, act as CA, net as N
    import sample as S
    params, _ = N.load_ckpt(pt)
    tabs = CE.tables_jnp()
    import real_backend as RB
    rules = A.RuleCfg(se_guard=1)

    @jax.jit
    def act_batch(raw, prev):
        enc = jax.vmap(CE.encode, in_axes=(0, 0, None))(raw, prev, tabs)
        nu = enc["valid"].sum(-1).astype(jnp.int32)
        o = N.forward(params, enc["planes"].astype(jnp.float32) / 255.0, enc["glob"], enc["pos"], enc["uf"], jnp.full((nu.shape[0],), 10, jnp.int32),
                      precision=jax.lax.Precision.HIGHEST)
        d = S.decide(o, jax.vmap(CA.legal_mask)(raw), nu, jax.random.PRNGKey(0), 0.2, 10.0, jnp.full(21, 0.5), True)
        return jax.vmap(lambda d_, n_, v_: S.build_act(d_, n_, v_, rules))(d, nu, jax.vmap(RB.view_from_raw)(raw))

    same, first, n = 0, None, len(rec)
    CH = 64
    for i in range(0, n, CH):
        chunk = rec[i:i + CH]
        raws = CE.stack([CE.raw_from_obs(o, nu=40) for o, _, _ in chunk])
        prevs = np.stack([CE.prev_array(p) for _, p, _ in chunk])
        pad = CH - len(chunk)
        if pad:
            raws = jax.tree_util.tree_map(lambda x: np.concatenate([x, np.repeat(x[-1:], pad, 0)]), raws)
            prevs = np.concatenate([prevs, np.repeat(prevs[-1:], pad, 0)])
        acts = jax.tree_util.tree_map(np.asarray, act_batch(jax.tree_util.tree_map(jnp.asarray, raws), jnp.asarray(prevs)))
        for j, (o, _, a_pkg) in enumerate(chunk):
            got = A.act_to_dict(jax.tree_util.tree_map(lambda x: x[j], acts))
            if norm(got) == norm(a_pkg):
                same += 1
            elif first is None:
                first = dict(step=int(o["step"]), seat=int(o["player"]), package=norm(a_pkg), jax=norm(got))
    out["decisions"] = n; out["identical"] = same; out["first_difference"] = first
    print(f"[roundtrip] package vs JAX greedy decisions: {same}/{n} identical ({time.time() - t0:.0f}s)", flush=True)
    # 3) the .pt in il/live_rollout.py and il/agent.BCAgent
    import live_rollout, agent as AG
    tnet = live_rollout._net(pt)
    bc = AG.BCAgent(pt, mk_tau=0.5, team=10)
    a0 = bc(copy.deepcopy(rec[40][0]))
    out["live_rollout_net"] = f"{type(tnet).__name__} width {tnet.width} n_teams {tnet.n_teams}"
    out["bcagent_action_ok"] = isinstance(a0, dict) and "market" in a0
    out["PASS"] = same == n and out["bcagent_action_ok"]
    json.dump(out, open(os.path.join(GD, "gate_cpu_roundtrip.json"), "w"), indent=1, default=str)
    try:
        os.remove(pkg)                                        # ~90 MB
    except OSError:
        pass
    print("ROUNDTRIP PASS" if out["PASS"] else "ROUNDTRIP FAIL", json.dumps({k: out[k] for k in ("decisions", "identical", "first_difference", "live_rollout_net", "bcagent_action_ok")}, default=str), flush=True)


if __name__ == "__main__":
    os.makedirs(GD, exist_ok=True)
    if sys.argv[1] == "rules":
        mode_rules()
    elif sys.argv[1] == "se":
        mode_se()
    else:
        mode_roundtrip(sys.argv[2])
