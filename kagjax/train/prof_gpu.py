"""Per-component timing of one lockstep rollout step on the GPU (real engine, real checkpoint), to find the bottleneck:
engine step, observation / encoding / legal masks / views (both seats), learner forward, opponent forward (vmap over
blocks vs lax.map vs one plain batched forward), decisions + rule layer, for each precision.
bash kagjax/train/srv_launch.sh <log> [PROF_B=1024] [INIT=...] --gpu python kagjax/train/prof_gpu.py
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import ppo as P
import sample as S
import actions as A


def timeit(f, *a, n=10):
    r = f(*a); jax.block_until_ready(r)
    t0 = time.time()
    for _ in range(n):
        r = f(*a)
    jax.block_until_ready(r)
    return (time.time() - t0) / n * 1e3


def main():
    B = int(os.environ.get("PROF_B", "1024")); K = int(os.environ.get("PROF_K", "4"))
    init = os.environ.get("INIT", "/Users/asad.ali/kaggriculture/il/ckpt/g20_u_dsm_s0.pt")
    cfg = P.Cfg(backend="real", init=init, games=B, blocks=K, seg=720, mb=1024, eval_every=0)
    tr = P.Trainer(cfg); be = tr.be
    params, _ = be.load_params(init, "state")
    bank = jax.tree_util.tree_map(lambda p: jnp.stack([p, p]), params)
    cs = be.consts()
    st, prev, lseat, tid = tr.fresh_games(jax.random.PRNGKey(0), B)
    # advance 30 steps so the state is not trivial
    st_, _, _, _ = tr._rollout(params, bank, cs, None, jnp.asarray([1] * K), jnp.asarray([10] * K), (st, prev, lseat, tid), jax.random.PRNGKey(1),
                               L=30, learner_greedy=False, record=False)
    st = st_[0]
    res = {"B": B, "K": K}
    obs_fn = jax.jit(lambda s, cs: (be.encode(be.observe(s, 0, cs), prev[:, 0], cs), be.legal(be.observe(s, 0, cs)), be.view(be.observe(s, 0, cs))))
    res["observe+encode+legal+view (one seat) ms"] = timeit(obs_fn, st, cs)
    o, m, v = obs_fn(st, cs)
    tm = jnp.full((B,), 10, jnp.int32)
    for prec in ("default", "highest", "bf16"):
        f = jax.jit(lambda p, o: tr._fwd(p, o, tm, prec))
        res[f"forward B={B} {prec} ms"] = timeit(f, params, o)
    out = tr._fwd(params, o, tm, "default")
    Bk = B // K
    ob = jax.tree_util.tree_map(lambda x: x.reshape((K, Bk) + x.shape[1:]), o)
    idx = jnp.asarray([1] * K); tms = jnp.asarray([10, 5, 3, 0] * (K // 4 + 1))[:K]

    def blk(xs):
        i, t_, o_ = xs
        return tr._fwd(jax.tree_util.tree_map(lambda b: b[i], bank), o_, jnp.full((Bk,), t_, jnp.int32), "default")

    res[f"opponent vmap over {K} blocks ms"] = timeit(jax.jit(lambda ob: jax.vmap(blk)((idx, tms, ob))), ob)
    res[f"opponent lax.map over {K} blocks ms"] = timeit(jax.jit(lambda ob: lax.map(blk, (idx, tms, ob))), ob)
    tau = jnp.full(21, 0.5)
    dec = jax.jit(lambda out, m, nu, k: S.decide(out, m, nu, k, 0.2, 10.0, tau, False))
    res["decide ms"] = timeit(dec, out, m, o.nu, jax.random.PRNGKey(3))
    d = dec(out, m, o.nu, jax.random.PRNGKey(3))
    rules = A.RuleCfg()
    bld = jax.jit(jax.vmap(lambda d_, nu, v_: S.build_act(d_, nu, v_, rules)))
    res["build_act + rules ms"] = timeit(bld, d, o.nu, v)
    a = bld(d, o.nu, v)
    res["engine step ms"] = timeit(jax.jit(lambda s, a, cs: be.step(s, a, a, cs)), st, a, cs)
    full = jax.jit(lambda p, s, c, k: tr.rollout_local(p, bank, cs, None, idx, tms, c, k, L=24, learner_greedy=False, record=True, axis=None)[1]["rew"])
    res["full rollout step (record) ms"] = timeit(full, params, st, (st, prev, lseat, tid), jax.random.PRNGKey(4), n=2) / 24
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(os.environ.get("PROF_OUT", "/Users/asad.ali/kaggriculture/kagjax/logs_server/D/prof_gpu.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
