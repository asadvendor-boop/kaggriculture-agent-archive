"""Closed loop: the whole self-play step on device (JAX engine state -> adapter -> encoder -> net -> greedy decide ->
engine Actions -> JAX engine step, one jitted function) against the Kaggle stack (C++ engine + il/np_runtime.NumpyAgent,
rule layer stripped), same seeds, exact end-of-day randomness (kagjax/rng.py).  Both seats are our policy (team 10,
tau 0.5).  Per game: every decision of both seats compared (action dicts); the full observation compared every 24
steps; final banks compared.  A game stops being compared at its first divergence (float near-tie).

RULES=1: the full production agent instead: our Act goes through the trainer's rule layer (kagjax/train/actions.py
apply_rules, View from train/real_backend.view_from_raw, default RuleCfg = final fix + homing + midnight sale), against
NumpyAgent WITH features.midnight_fix (module defaults) -- i.e. the Kaggle submission reproduced on device.

run: kagjax/policy/run1.sh -m kagjax.policy.test_selfplay [ckpt] [n_games] [seed0]
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import jax
import jax.numpy as jnp
from kagjax.policy import encode as E, act as A, net as N, adapter as AD
from kagjax.policy.test_act import Recorder, TEAM, TAU
from kagjax import engine as EN, state as S, rng as RNG, consts as EC
from harness import kagsim_trace

RULES = int(os.environ.get("RULES", "0"))
if RULES:
    sys.path.insert(0, "/Users/asad.ali/kaggriculture/kagjax/train")
    import actions as TA                      # the trainer's rule layer (Agent B)
    import real_backend as RB


def prices_tb(tb, g):
    return tb.P[jnp.arange(EC.N_PRODUCTS)[None, :] * EC.NTAB + jnp.clip(g.minv - EC.INV_LO, 0, EC.NTAB - 1)]


def make_step(B, cfg=A.Cfg(greedy=True, tau=TAU), loop=None):
    """loop: engine LoopCfg (None = the engine default, sequential units; EN.CFG_PAR = the GPU-oriented, exactly
    equivalent structure)."""
    tb = EN.tables(); tabs = E.tables_jnp()
    eng_kw = {} if loop is None else {"cfg": loop}
    team = jnp.full(2 * B, TEAM, jnp.int32)

    @jax.jit
    def step(params, st, prev, words, key):
        raw = AD.raw_both_seats(st.g, st.t, prices_tb(tb, st.g))
        d = A.policy_step(params, raw, prev, key, team, cfg, tabs)
        act = d["act"]
        if RULES:
            view = jax.vmap(RB.view_from_raw)(raw)
            act = A.Act(*jax.vmap(lambda v, a: TA.apply_rules(v, a, TA.RuleCfg()))(view, TA.Act(*act)))
            d["act"] = act
        ea = AD.actions_from_policy(act)
        a0 = jax.tree_util.tree_map(lambda x: x[:B], ea); a1 = jax.tree_util.tree_map(lambda x: x[B:], ea)
        st2 = EN._step_impl(tb, st, a0, a1, words, **eng_kw)
        return st2, d["next_prev"], d["act"], d["value"]
    return step


def main(ckpt, n_games=4, seed0=777001):
    import torch
    torch.set_num_threads(1)
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    P = {k: v.numpy() for k, v in c["state"].items()}
    meta = {"width": c["width"], "blocks": c.get("blocks", 5), "n_teams": c.get("n_teams", 0)}
    del c
    params, _ = N.load_ckpt(ckpt)
    seeds = [seed0 + i for i in range(n_games)]
    B = n_games
    refs = [kagsim_trace.Game(s, 720, False) for s in seeds]
    recs = [Recorder(P, meta) for _ in range(B)]
    if RULES:                                  # NumpyAgent with the production rule layer
        import features6 as F6
        for r in recs:
            r.agent.F = F6
    words = jnp.asarray(np.moveaxis(RNG.batch_words(seeds), 0, 1))           # [B,30,624]
    st = EN.init(B)
    prev = jnp.full((2 * B, 16), -1, jnp.int32)
    step = make_step(B)
    key = jax.random.PRNGKey(0)
    diverged = [None] * B; obs_bad = [None] * B
    n_dec = 0; t0 = time.time(); t_jax = 0.0; t_ref = 0.0
    for t in range(EC.LAST_STEP + 1):
        ta = time.time()
        ref_acts = [[recs[b](refs[b].observe(s))[0] for s in (0, 1)] for b in range(B)]
        t_ref += time.time() - ta
        ta = time.time()
        st, prev, act, val = step(params, st, prev, words, key)
        act = jax.device_get(act)
        t_jax += time.time() - ta
        for b in range(B):
            if diverged[b] is not None:
                continue
            for s in (0, 1):
                jd = A.act_to_dict(jax.tree_util.tree_map(lambda x: x[s * B + b], act))
                n_dec += 1
                if jd != ref_acts[b][s]:
                    diverged[b] = dict(step=t, seat=s)
                    print(f"   game {b} diverged at step {t} seat {s}", flush=True)
                    break
        for b in range(B):
            refs[b].step(ref_acts[b][0], ref_acts[b][1])
        if (t + 1) % 24 == 0 or t == EC.LAST_STEP:
            g = jax.tree_util.tree_map(np.asarray, st.g)
            for b in range(B):
                if diverged[b] is None and obs_bad[b] is None:
                    for s in (0, 1):
                        dd = S.obs_diff(S.game_to_obs(g, b, s, int(st.t)), refs[b].observe(s))
                        if dd:
                            obs_bad[b] = dict(step=t, seat=s, diff=[list(map(str, x)) for x in dd[:5]])
                            print(f"   game {b}: observation mismatch after step {t}: {dd[:3]}", flush=True)
                            break
        if (t + 1) % 120 == 0:
            print(f"   step {t + 1}: {time.time() - t0:.0f}s (jax {t_jax:.0f}s, numpy agents {t_ref:.0f}s)", flush=True)
    g = jax.tree_util.tree_map(np.asarray, st.g)
    res = []
    for b in range(B):
        jr = [int(g.money[b, 0]), int(g.money[b, 1])]; rr = [refs[b].reward(0), refs[b].reward(1)]
        res.append(dict(seed=seeds[b], diverged=diverged[b], obs_mismatch=obs_bad[b], jax_banks=jr, ref_banks=rr,
                        banks_equal=[float(x) for x in jr] == [float(x) for x in rr]))
    ok = sum(r["diverged"] is None and r["obs_mismatch"] is None and r["banks_equal"] for r in res)
    print(f"closed loop{' WITH RULE LAYER' if RULES else ''} {os.path.basename(ckpt)}: {ok}/{B} games identical through all {EC.LAST_STEP + 1} steps "
          f"({n_dec} decisions compared); banks " + ", ".join(f"{r['jax_banks']}" for r in res), flush=True)
    out = os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))),
                       f"results_selfplay{'_rules' if RULES else ''}_{os.path.basename(ckpt).replace('.pt', '')}.json")
    json.dump(dict(ckpt=ckpt, games=res, identical=ok, decisions=n_dec, time_s=time.time() - t0, jax_s=t_jax,
                   numpy_agent_s=t_ref), open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/asad.ali/kaggriculture/il/ckpt/fr_t_dsm_s0.pt",
         int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 777001)
