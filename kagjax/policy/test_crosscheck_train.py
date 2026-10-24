"""Cross-check of the trainer's own sampler (kagjax/train/sample.py, read-only here) against the Kaggle numpy runtime
and act.py, on the NumpyAgent's recorded logits (same records as test_act.py):
  greedy: sample.decide(greedy=True) + actions.unit_actions / decode_market vs NumpyAgent's action dict
  PPO   : sample.logp_terms vs act.logp_terms-equivalent terms on identical logits and sampled actions
run: kagjax/policy/run1.sh -m kagjax.policy.test_crosscheck_train [ckpt] [n_games] [n_tape_games]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/kagjax/train")
import jax
import jax.numpy as jnp
from kagjax.policy import consts as K, encode as E, act as A, net as N
from kagjax.policy.test_act import Recorder, collect, pad_logits, TEAM, TAU
import sample as TS          # trainer (Agent B), read-only
import actions as TA


def main(ckpt, n_games=2, n_tape=4):
    import torch
    torch.set_num_threads(1)
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    P = {k: v.numpy() for k, v in c["state"].items()}
    meta = {"width": c["width"], "blocks": c.get("blocks", 5), "n_teams": c.get("n_teams", 0)}
    del c
    R = collect(Recorder(P, meta), n_games, n_tape)
    n = len(R)
    raw = E.stack([E.raw_from_obs(r["obs"]) for r in R])
    lg = [pad_logits(r) for r in R]
    out = {"op": jnp.asarray(np.stack([x[0] for x in lg])), "qty": jnp.asarray(np.stack([x[1] for x in lg])),
           "mk": jnp.asarray(np.stack([x[2] for x in lg]))}
    umask = jax.jit(jax.vmap(A.legal_mask))(raw)
    nu = jnp.asarray(np.minimum(16, raw.n_units[:, 0]))
    tau = jnp.full(K.N_HEADS, TAU, jnp.float32)
    dec = jax.device_get(TS.decide(out, umask, nu, jax.random.PRNGKey(0), 0.5, 4.0, tau, True))
    # engine-native act via the trainer's own builders (no rules)
    bad_greedy = 0
    for j, r in enumerate(R):
        v = TA.obs_to_view(r["obs"])
        uo, ua, un = TA.unit_actions(jnp.asarray(dec.op[j]), jnp.asarray(dec.qty[j]), int(nu[j]), v)
        oo, oi, on, nn = TA.decode_market(jnp.asarray(dec.classes[j]))
        a = TA.Act(uo, ua, un, jnp.int32(int(nu[j])), oo, oi, on, nn)
        d = TA.act_to_dict(a)
        if d != r["ref_act"]:
            bad_greedy += 1
    # PPO terms: trainer vs act.logp_terms on identical logits (feed the logits through a stub forward)
    rng = np.random.RandomState(0)
    um = np.asarray(umask); valid = np.arange(16)[None, :] < np.asarray(nu)[:, None]
    uact = np.where(valid, np.array([[rng.choice(np.nonzero(um[i, k])[0]) for k in range(16)] for i in range(n)]), -1)
    qact = np.where(valid & (uact >= K.OP_PICKUP0), rng.randint(0, 12, (n, 16)), -1)
    mfire = rng.rand(n, K.N_HEADS) < 0.3
    tt = jax.device_get(TS.logp_terms(out, umask, jnp.asarray(uact), jnp.asarray(qact), jnp.asarray(mfire), 0.5, 4.0, tau))
    cfg = A.Cfg(greedy=False, T=0.5, tau=TAU, ks=4.0)
    zu = jnp.where(umask, out["op"] / cfg.T, A.NEG); lpu_all = jax.nn.log_softmax(zu, -1)
    lpu = np.take_along_axis(np.asarray(lpu_all), np.maximum(uact, 0)[..., None], -1)[..., 0]
    a_ = np.asarray(jnp.clip(jax.vmap(lambda m: A.market_fire_logits(m, TAU, 4.0))(out["mk"]), -A.LOGIT_CLAMP, A.LOGIT_CLAMP))
    lpm = np.where(mfire, np.asarray(jax.nn.log_sigmoid(a_)), np.asarray(jax.nn.log_sigmoid(-a_)))
    # float64 reference of the market term (ppo.py formula without float32 cancellation)
    mk64 = np.asarray(out["mk"], np.float64); lpm64 = np.zeros_like(lpm)
    for h in range(K.N_HEADS):
        z = mk64[:, K.HEAD_OFF[h]:K.HEAD_OFF[h + 1]]
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        pnz = np.clip(1 - p[:, 0], 1e-6, 1 - 1e-6)
        pf = np.clip(1 / (1 + np.exp(-4.0 * (np.log(pnz / (1 - pnz))))), 1e-6, 1 - 1e-6)
        lpm64[:, h] = np.where(mfire[:, h], np.log(pf), np.log(1 - pf))
    res = dict(decisions=n, greedy_mismatch_vs_numpy_agent=bad_greedy,
               lpu_train_vs_act=float(np.abs(np.where(valid, tt["lpu"] - lpu, 0)).max()),
               lpm_train_vs_f64=float(np.abs(tt["lpm"] - lpm64).max()), lpm_act_vs_f64=float(np.abs(lpm - lpm64).max()))
    print(f"trainer sample.py cross-check on {n} NumpyAgent decisions: greedy action mismatches {bad_greedy}; "
          f"lpu vs act.py {res['lpu_train_vs_act']:.1e}; market log-prob vs float64: trainer {res['lpm_train_vs_f64']:.1e}, "
          f"act.py {res['lpm_act_vs_f64']:.1e}", flush=True)
    json.dump(res, open(os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))),
                                     "results_crosscheck_train.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/asad.ali/kaggriculture/il/ckpt/fr_t_dsm_s0.pt",
         int(sys.argv[2]) if len(sys.argv) > 2 else 2, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
