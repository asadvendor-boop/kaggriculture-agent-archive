"""The real backend's adapters on real game states (C++ engine replays of recorded top games):
  1. real_backend.raw_from_state(A's GameState from the two observations)  ==  C's encode.raw_from_obs(obs)   (every field)
  2. real_backend.view_from_raw(raw)                                        ==  actions.obs_to_view(obs)        (every field)
  3. to_engine(dict_to_act(action)) == A's state.encode_action(action) for policy-style actions (units padded to 40)
  4. (--decide N) greedy decisions of the whole JAX pipeline (encode -> net -> sample.decide -> build_act with the rule
     layer) == the reference numpy Kaggle runtime (il/np_runtime.NumpyAgent incl. features.midnight_fix) on N
     observations, with the real checkpoint (default rf_w_dsm_s0, team 10, tau 0.5).
  5. (--rollout) a short lockstep rollout of the trainer on the real engine (tiny net) to check jit / shapes.
run: kagjax/train/run1.sh kagjax/train/test_real_backend.py [n_games] [--decide N] [--rollout]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax
import jax.numpy as jnp
import actions as A
import ppo as P                    # import the trainer's ppo before any il/ module puts il/ first on sys.path
import real_backend as RB
import state as AS
from kagjax.policy import encode as CE

R = "/Users/asad.ali/kaggriculture"


def collect(n_games, every=7):
    """[(obs0, obs1, rec_action_seat0, rec_action_seat1)] from recorded games replayed in the C++ engine."""
    from rollout import load_world
    from harness import kagsim_trace, PASS
    idx = json.load(open(f"{R}/data/tapes_pull/index.json"))
    eps = sorted(idx)[5:: max(1, len(idx) // (n_games * 2))][:n_games]
    out = []
    for ep in eps:
        d, shops, e0 = load_world(f"{R}/data/tapes_pull/{ep}.json.gz")
        g = kagsim_trace.Game(int(d["seed"]), 720, False); g.set_forced_shops(shops); g.set_weed_prefix(1, e0)
        acts = d["actions"]
        for t in range(720):
            now = [acts[p][t] if t < len(acts[p]) and isinstance(acts[p][t], dict) else dict(PASS) for p in (0, 1)]
            if t % every == 0 or t % 24 == 23 or t >= 700:
                out.append((g.observe(0), g.observe(1), now[0], now[1]))
            g.step(now[0], now[1])
            if g.done:
                break
    return out


def eq_tree(a, b):
    bad = []
    for f in a._fields:
        x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        if x.shape != y.shape or not np.array_equal(x.astype(np.float64), y.astype(np.float64)):
            bad.append(f)
    return bad


def main():
    args = sys.argv[1:]
    n_games = int(args[0]) if args and args[0].isdigit() else 2
    samples = collect(n_games)
    print(f"{len(samples)} observation pairs from {n_games} replayed games", flush=True)
    games = AS.stack_games([AS.obs_to_game(o0, o1)[0] for o0, o1, _, _ in samples])
    ok = True
    # 1 + 2: per seat, group by step (the adapter takes the shared t of a batch)
    steps = np.array([int(o0["step"]) for o0, _, _, _ in samples])
    tb = RB.AE.tables()
    raw_fn = jax.jit(RB.raw_from_state, static_argnums=(2,))
    view_fn = jax.jit(jax.vmap(RB.view_from_raw))
    bad_raw, bad_view, n = {}, {}, 0
    for t in np.unique(steps):
        sel = np.flatnonzero(steps == t)
        g = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)[sel]), games)
        for seat in (0, 1):
            raw = raw_fn(g, int(t), seat, tb)
            vw = view_fn(raw)
            for j, i in enumerate(sel):
                obs = samples[i][seat]
                ref = CE.raw_from_obs(obs, nu=40)
                got = jax.tree_util.tree_map(lambda x: np.asarray(x)[j], raw)
                for f in eq_tree(got, ref):
                    bad_raw[f] = bad_raw.get(f, 0) + 1
                vref = A.obs_to_view(obs)
                vgot = jax.tree_util.tree_map(lambda x: np.asarray(x)[j], vw)
                for f in eq_tree(vgot, vref):
                    bad_view[f] = bad_view.get(f, 0) + 1
                n += 1
    print(f"[raw_from_state] {n} seat-observations, fields with mismatches: {bad_raw or 'none'}")
    print(f"[view_from_raw]  {n} seat-observations, fields with mismatches: {bad_view or 'none'}")
    ok &= not bad_raw and not bad_view

    # 3: Act -> engine arrays == A's dict parser
    import features as F
    rng = np.random.RandomState(0)
    bad = 0
    for o0, o1, r0, r1 in samples[:300]:
        for obs, rec in ((o0, r0), (o1, r1)):
            farm = obs["farms"][int(obs["player"])]
            nu = min(16, 1 + len(farm.get("hands") or []))
            rc = [rec.get("farmer") or ["PASS"]] + list(rec.get("hands") or [])
            cmds = []
            for k in range(nu):
                try:
                    oc, qc = F.unit_op_label(rc[k]) if k < len(rc) else (0, -1)
                except Exception:
                    oc, qc = 0, -1
                cmds.append(F.unit_command(oc, max(qc, 0)))
            classes = rng.randint(0, 3, F.N_HEADS) * (rng.rand(F.N_HEADS) < 0.3)
            d = {"farmer": cmds[0], "hands": cmds[1:], "market": F.decode_market(classes)}
            a = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], A.dict_to_act(d))
            got = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], RB.to_engine(a))
            ref = AS.encode_action(d)
            bad += any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(got, ref))
    print(f"[to_engine] 600 policy actions vs state.encode_action: mismatches {bad}"); ok &= bad == 0

    if "--decide" in args:
        nd = int(args[args.index("--decide") + 1])
        ok &= test_decide(samples, nd)
    if "--rollout" in args:
        ok &= test_rollout()
    print("ALL OK" if ok else "MISMATCH", flush=True)


def test_decide(samples, nd, ckpt=f"{R}/il/ckpt/rf_w_dsm_s0.pt", team=10, tau=0.5):
    """Greedy JAX pipeline vs the numpy Kaggle runtime (np_runtime.NumpyAgent + features6 + midnight_fix)."""
    import torch
    import features as F, features6 as F6, np_runtime as NR, copy
    import sample as S
    from kagjax.policy import net as N
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    W = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
    agent = NR.NumpyAgent(NR.NumpyNet(W, width=c["width"], blocks=c["blocks"], n_teams=c["n_teams"]), F6, tau=tau, team=team)
    params, _ = N.load_ckpt(ckpt)
    rng = np.random.RandomState(1)
    pick = rng.choice(len(samples), nd, replace=False)
    be = RB.RealBackend()
    rules = A.RuleCfg()

    cs = be.consts()

    @jax.jit
    def jax_act(g, t, prev, seat_is1, cs):
        raw = RB.raw_from_state(g, t, 0, cs["a"])
        raw1 = RB.raw_from_state(g, t, 1, cs["a"])
        raw = jax.tree_util.tree_map(lambda a, b: jnp.where(seat_is1.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), raw, raw1)
        ob = be.encode(raw, prev, cs)
        out = N.forward(params, ob.planes.astype(jnp.float32) / 255.0, ob.glob, ob.pos, ob.uin, jnp.full((1,), team, jnp.int32),
                        precision=jax.lax.Precision.HIGHEST)
        d = S.decide(out, be.legal(raw), ob.nu, jax.random.PRNGKey(0), 0.2, 10.0, jnp.full(21, tau), True)
        return jax.vmap(lambda d_, nu, v: S.build_act(d_, nu, v, rules))(d, ob.nu, be.view(raw)), out

    same, first = 0, None
    t0 = time.time()
    for i in pick:
        o0, o1, _, _ = samples[i]
        seat = int(rng.randint(2))
        obs = (o0, o1)[seat]
        prev = [int(x) for x in rng.randint(0, 44, 1 + len(obs["farms"][seat].get("hands") or []))][:16]
        agent.state = {seat: {"step": int(obs["step"]) - 1, "prev": prev}}
        ref = agent(copy.deepcopy(obs))
        g = AS.stack_games([AS.obs_to_game(o0, o1)[0]])
        g = jax.tree_util.tree_map(jnp.asarray, g)
        pa = jnp.asarray(CE.prev_array(prev))[None]
        act, _ = jax_act(g, jnp.int32(obs["step"]), pa, jnp.asarray([seat == 1]), cs)
        got = A.act_to_dict(jax.tree_util.tree_map(lambda x: np.asarray(x)[0], act))
        from test_rules import norm
        if norm(got) == norm(ref):
            same += 1
        elif first is None:
            first = (int(obs["step"]), norm(ref), norm(got))
    print(f"[decide] greedy JAX pipeline vs numpy Kaggle runtime: {same}/{nd} identical actions ({time.time() - t0:.0f}s)")
    if first:
        print("  first difference at step", first[0], "\n  ref:", first[1], "\n  jax:", first[2])
    return same == nd


def test_rollout():
    """A short trainer rollout + update on the real engine with a tiny network (jit, shapes, rewards)."""
    cfg = P.Cfg(backend="real", init="tiny:8x1", games=8, blocks=2, seg=24, mb=64, epochs=1, eval_every=0)
    tr = P.Trainer(cfg)
    params = tr.be.init_params(jax.random.PRNGKey(0), width=8, blocks=1)
    bank = jax.tree_util.tree_map(lambda p: jnp.stack([p, p]), params)
    carry = tr.fresh_games(jax.random.PRNGKey(1), 8)
    t0 = time.time()
    cs = tr.be.consts()
    carry, traj, (fin, mar), boot = tr._rollout(params, bank, cs, None, jnp.asarray([0, 1]), jnp.asarray([10, 5]), carry, jax.random.PRNGKey(2),
                                               L=24, learner_greedy=False, record=True)
    jax.block_until_ready(traj["rew"])
    t1 = time.time()
    carry, traj2, _, boot2 = tr._rollout(params, bank, cs, None, jnp.asarray([0, 1]), jnp.asarray([10, 5]), carry, jax.random.PRNGKey(3),
                                         L=24, learner_greedy=False, record=True)
    jax.block_until_ready(traj2["rew"])
    t2 = time.time()
    st = tr._update(jax.tree_util.tree_map(jnp.array, params), jax.tree_util.tree_map(jnp.array, params), P.adam_init(params),
                    P.adam_init(params), params, traj2, boot2, jax.random.PRNGKey(4), train_policy=True)[4]
    mon = np.asarray(tr.be.money(carry[0]))
    print(f"[rollout] real engine, 8 games x 48 steps: compile+run {t1 - t0:.0f}s, run {t2 - t1:.1f}s; t = {int(carry[0].t)}; "
          f"money after 48 steps {mon.min():.0f}..{mon.max():.0f}; update loss {float(st['loss']):+.4f} v {float(st['l_v']):.4f}")
    return int(carry[0].t) == 48 and np.isfinite(float(st["loss"]))


if __name__ == "__main__":
    main()
