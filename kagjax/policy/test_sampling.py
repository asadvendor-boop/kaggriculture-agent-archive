"""Sampling mode (the PPO rollout policy of il/live_rollout.py / il/ppo.py) of act.decide:
 1. distributions: per state, 4000 draws with different keys vs the analytic reference computed in float64 numpy
    from the same logits exactly as live_rollout does (ops ~ softmax(masked / T), qty ~ softmax(qty / T), market fire
    ~ Bernoulli(rollout.sharpen(1 - p0, tau, ks)), class = 1 + argmax p[1:]); reported as the max total-variation
    distance per unit / the max |fire rate - sharpen| in standard errors
 2. consistency: decide's log-probs of its own sampled actions == logp_terms on those actions (max abs diff)

run: kagjax/policy/run1.sh -m kagjax.policy.test_sampling [ckpt]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import jax
import jax.numpy as jnp
from kagjax.policy import consts as K, encode as E, act as A, net as N, refdata
from rollout import sharpen

T, TAU, KS, TEAM = 0.5, 0.5, 4.0, 10


def main(ckpt, n_states=48, draws=4000):
    params, _ = N.load_ckpt(ckpt)
    tabs = E.tables_jnp()
    raws, prevs = [], []
    for f in refdata.tape_files(4, start=23):
        for t, s, o, prev in refdata.iter_tape_obs(f, every=29):
            raws.append(E.raw_from_obs(o)); prevs.append(E.prev_array(prev))
    idx = np.linspace(0, len(raws) - 1, n_states).astype(int)
    raw = jax.tree_util.tree_map(jnp.asarray, E.stack([raws[i] for i in idx])); prev = jnp.asarray(E.stack([prevs[i] for i in idx]))
    enc = jax.jit(jax.vmap(E.encode, in_axes=(0, 0, None)))(raw, prev, tabs)
    pl, gl, pos, uin = E.net_inputs(enc, tabs)
    out = jax.device_get(N.forward_jit(params, pl, gl, pos, uin, jnp.full(n_states, TEAM, jnp.int32)))
    masks = np.asarray(jax.jit(jax.vmap(A.legal_mask))(raw))
    valid = np.asarray(enc["valid"])
    cfg = A.Cfg(greedy=False, T=T, tau=TAU, ks=KS)
    per = {k: jnp.asarray(out[k]) for k in ("op", "qty", "mk", "v")}

    def draw(key):
        d = jax.vmap(lambda r, o, k: A.decide(r, o, k, cfg))(raw, per, jax.random.split(key, n_states))
        return {k: d[k] for k in ("op", "qact", "fire", "mclass", "lp_u", "lp_q", "lp_m", "valid")}
    D = jax.device_get(jax.jit(jax.vmap(draw))(jax.random.split(jax.random.PRNGKey(1), draws)))   # [draws, n, ...]
    tv_op = tv_q = 0.0; z_fire = 0.0; cls_bad = 0
    for i in range(n_states):
        for k in range(16):
            if not valid[i, k]:
                continue
            z = np.where(masks[i, k], out["op"][i, k].astype(np.float64), -1e9)
            p = np.exp((z - z.max()) / T); p /= p.sum()                       # live_rollout's sampling distribution
            emp = np.bincount(D["op"][:, i, k], minlength=K.N_OPS) / draws
            tv_op = max(tv_op, 0.5 * np.abs(emp - p).sum())
            zq = out["qty"][i, k].astype(np.float64)
            pq = np.exp((zq - zq.max()) / T); pq /= pq.sum()
            qa = D["qact"][:, i, k]; qa = qa[qa >= 0]
            if len(qa) > 500:
                tv_q = max(tv_q, 0.5 * np.abs(np.bincount(qa, minlength=12) / len(qa) - pq).sum())
        for h in range(K.N_HEADS):
            zz = out["mk"][i, K.HEAD_OFF[h]:K.HEAD_OFF[h + 1]].astype(np.float64)
            p = np.exp(zz - zz.max()); p /= p.sum()
            pf = float(sharpen(1.0 - p[0], TAU, KS))
            rate = D["fire"][:, i, h].mean()
            se = max(np.sqrt(pf * (1 - pf) / draws), 1.0 / draws)
            z_fire = max(z_fire, abs(rate - pf) / se)
            fired = D["fire"][:, i, h]
            if fired.any():
                cls_bad += int((D["mclass"][fired, i, h] != 1 + np.argmax(p[1:])).sum())
    # consistency of the rollout log-probs with logp_terms on the first draw
    d0 = jax.tree_util.tree_map(lambda x: x[0], D)
    lt = jax.device_get(A.logp_terms(params, enc["planes"], enc["glob"], enc["pos"], enc["uf"], jnp.full(n_states, TEAM, jnp.int32),
                                     jnp.asarray(masks), jnp.asarray(d0["op"]), jnp.asarray(d0["qact"]), jnp.asarray(d0["fire"]), cfg, tabs))
    c_u = float(np.abs(np.where(d0["valid"], lt["lpu"] - d0["lp_u"], 0)).max())
    c_q = float(np.abs(np.where(d0["qact"] >= 0, lt["lpq"] - d0["lp_q"], 0)).max())
    c_m = float(np.abs(lt["lpm"] - d0["lp_m"]).max())
    res = dict(states=n_states, draws=draws, max_tv_op=tv_op, max_tv_qty=tv_q, max_fire_z=z_fire, class_mismatch=cls_bad,
               consistency_lpu=c_u, consistency_lpq=c_q, consistency_lpm=c_m)
    print(f"sampling: {n_states} states x {draws} draws: max TV(op) {tv_op:.4f}, max TV(qty) {tv_q:.4f} "
          f"(sampling noise ~{np.sqrt(44 / draws) / 2:.3f}), market fire rate vs rollout.sharpen max {z_fire:.2f} s.e., "
          f"fired class != argmax: {cls_bad}; decide vs logp_terms: lpu {c_u:.1e} lpq {c_q:.1e} lpm {c_m:.1e}", flush=True)
    json.dump(res, open(os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))), "results_sampling.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/asad.ali/kaggriculture/il/ckpt/fr_t_dsm_s0.pt")
