"""Throughput of the trainer options on the GPU (shared A100: run back-to-back, repeat once).

BENCH=rollout : lockstep step time of the rollout at B = BENCH_B (default 1024, no recording) for the opponent-forward
                modes: map over K=4 blocks (one param set per block), OPP_GROUPS=2 and OPP_GROUPS=1 (one forward over all
                opponent rows with per-row style ids); plus the learner forward alone.
BENCH=update  : one recorded rollout (B=128 games x 720 steps = 92,160 samples; value-head features recorded), then the
                PPO update of every variant in BENCH_VARIANTS on that same rollout: compile time, one timed update,
                samples/s (per iteration) and sample-passes/s (x EPOCHS), compiled memory (XLA memory_analysis).
Results: $BENCH_OUT/bench_<BENCH>_<tag>.json.   bash kagjax/train/srv_launch.sh <log> BENCH=update --gpu python kagjax/train/bench_update.py
"""
import os, sys, json, time, dataclasses
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import ppo as P

R = "/Users/asad.ali/kaggriculture"
OUT = os.environ.get("BENCH_OUT", f"{R}/kagjax/logs_server/D/speed")
INIT = os.environ.get("INIT", f"{R}/il/ckpt/g20_w_dsm_s0.pt")
IL = os.environ.get("IL", f"{R}/il/ckpt/g20_w_s0.pt")
TAG = os.environ.get("BENCH_TAG", time.strftime("%H%M"))
BASE = dict(backend="real", init=INIT, il=IL, games=128, blocks=4, seg=720, eval_every=0, mb=256, micro=0, epochs=2,
            lr=5e-6, vlr=1e-4)
VARIANTS = {   # name -> Cfg overrides (LR by the square-root rule from the gate's MB=256: x sqrt(MB / 256))
    "gate_E2_MB256": dict(),
    "E1_MB256": dict(epochs=1),
    "E1_MB1024": dict(epochs=1, mb=1024, micro=512, lr=1e-5, vlr=2e-4),
    "E1_MB4096": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4),
    "E1_MB4096_ref25": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4, ref_frac=0.25),
    "E1_MB4096_ref25_small": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="small:128x4"),
    "E1_MB4096_ref25_head": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="head", value_target="final"),
    "E2_MB4096_ref25_head": dict(epochs=2, mb=4096, micro=512, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="head", value_target="final"),
    "E1_MB4096_head": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4, value_init="head", value_target="final"),
    "E1_MB4096_ref25_head_bf16": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="head",
                                      value_target="final", update_precision="bf16"),
    "E1_MB4096_ref25_head_bf16_recompute": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="head",
                                                value_target="final", update_precision="bf16", old_logp="recompute"),
    "E1_MB4096_ref25_head_bf16grad": dict(epochs=1, mb=4096, micro=512, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="head",
                                          value_target="final", update_precision="bf16grad"),
    "E1_MB4096_ref25_head_m1024": dict(epochs=1, mb=4096, micro=1024, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="head", value_target="final"),
    "E1_MB4096_ref25_head_m2048": dict(epochs=1, mb=4096, micro=2048, lr=2e-5, vlr=4e-4, ref_frac=0.25, value_init="head", value_target="final"),
}


if os.environ.get("BENCH_MOCK"):                              # code-path check on the Mac (tiny net, mock engine)
    BASE.update(backend="mock", init="tiny:8x1", il="", episode_slots=48, games=8, seg=48, mb=16)
    for _v in VARIANTS.values():
        for _k, _d in (("mb", 16), ("micro", 8)):
            if _k in _v:
                _v[_k] = max(8, _v[_k] // _d)


def log(rec, fn):
    try:                                                      # process high-water mark so far (cumulative over variants)
        st = jax.devices()[0].memory_stats() or {}
        pk = dict(proc_peak_gib=round(st.get("peak_bytes_in_use", 0) / 2 ** 30, 2), in_use_gib=round(st.get("bytes_in_use", 0) / 2 ** 30, 2))
    except Exception:
        pk = {}
    rec = dict(rec, model=os.path.basename(INIT).replace(".pt", ""), **pk)
    print(json.dumps(rec), flush=True)
    with open(fn, "a") as f:
        f.write(json.dumps(rec) + "\n")


def value_params(tr, params, cfg, key):
    if cfg.value_init == "head":
        return {"w": jnp.array(params["v"]["w"]), "b": jnp.array(params["v"]["b"])}
    if cfg.value_init.startswith("small:"):
        w, b = [int(x) for x in cfg.value_init.split(":")[1].split("x")]
        return tr.be.init_params(key, width=w, blocks=b, n_teams=int(params["team"].shape[0]))
    return jax.tree_util.tree_map(jnp.array, params)


def mem(compiled):
    try:
        ma = compiled.memory_analysis()
        return dict(temp_gib=round(ma.temp_size_in_bytes / 2 ** 30, 2), arg_gib=round(ma.argument_size_in_bytes / 2 ** 30, 2),
                    out_gib=round(ma.output_size_in_bytes / 2 ** 30, 2))
    except Exception as e:
        return dict(err=repr(e)[:160])


def bench_rollout(fn):
    """All modes compiled first, then timed round-robin (BENCH_ROUNDS rounds of L steps each): the shared GPU's load
    changes by the minute, so modes are compared inside the same few minutes; the median round is reported."""
    B0 = int(os.environ.get("BENCH_B", "1024")); L = int(os.environ.get("BENCH_L", "24"))
    rounds = int(os.environ.get("BENCH_ROUNDS", "3"))
    be = P.Trainer(P.Cfg(**dict(BASE, games=B0, blocks=4))).be
    if BASE["init"].startswith("tiny:"):
        params = il = be.init_params(jax.random.PRNGKey(1), width=8, blocks=1)
    else:
        params, _ = be.load_params(INIT, "state"); il, _ = be.load_params(IL, "state")
    bank = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), params, il)
    cs = be.consts()
    modes = (("map_K4", B0, dict(opp_groups=0)), ("groups_G2", B0, dict(opp_groups=2)), ("groups_G1", B0, dict(opp_groups=1)),
             ("map_K4_oppbf16", B0, dict(opp_groups=0, opp_precision="bf16")),         # opponents in bf16 (learner TF32)
             ("map_K4", 2 * B0, dict(opp_groups=0)))                                    # bigger batch (may not fit the cap)
    live = []
    for name, B, over in modes:
        try:
            cfg = P.Cfg(**dict(BASE, games=B, blocks=4, **over))
            tr = P.Trainer(cfg)
            G = cfg.opp_groups or 4
            oi = jnp.asarray([1] * G, jnp.int32); ot = jnp.asarray([10, 5, 3, 0], jnp.int32)
            carry = tr.fresh_games(jax.random.PRNGKey(0), B)
            t0 = time.time()
            carry, _, _, _ = tr._rollout(params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(1), L=L, learner_greedy=False, record=False)
            jax.block_until_ready(carry[0].t)
            live.append(dict(name=name, B=B, tr=tr, oi=oi, ot=ot, carry=carry, first=time.time() - t0, dts=[]))
        except Exception as e:                                                          # (e.g. out of memory at 2B)
            log(dict(bench="rollout_error", mode=name, B=B, err=repr(e)[:300]), fn)
    for r in range(rounds):
        for m in live:
            try:
                t1 = time.time()
                m["carry"], _, _, _ = m["tr"]._rollout(params, bank, cs, None, m["oi"], m["ot"], m["carry"], jax.random.PRNGKey(2 + r), L=L,
                                                       learner_greedy=False, record=False)
                jax.block_until_ready(m["carry"][0].t)
                m["dts"].append((time.time() - t1) / L)
            except Exception as e:
                log(dict(bench="rollout_error", mode=m["name"], B=m["B"], err=repr(e)[:300]), fn)
    for m in live:
        if m["dts"]:
            dt = float(np.median(m["dts"]))
            log(dict(bench="rollout", mode=m["name"], B=m["B"], L=L, rounds=len(m["dts"]), first_call_s=round(m["first"], 1),
                     ms_per_step=round(1e3 * dt, 1), ms_rounds=[round(1e3 * x, 1) for x in m["dts"]],
                     game_steps_per_s=round(m["B"] / dt)), fn)


def bench_update(fn):
    names = [n for n in os.environ.get("BENCH_VARIANTS", ",".join(VARIANTS)).split(",") if n]
    cfg0 = P.Cfg(**dict(BASE, value_init="head", value_target="final"))
    tr0 = P.Trainer(cfg0)
    be = tr0.be
    if cfg0.init.startswith("tiny:"):
        params = be.init_params(jax.random.PRNGKey(1), width=8, blocks=1); il = params
    else:
        params, _ = be.load_params(INIT, "state"); il, _ = be.load_params(IL, "state")
    ref = jax.tree_util.tree_map(jnp.array, params)
    bank = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), params, il)
    cs = be.consts()
    carry = tr0.fresh_games(jax.random.PRNGKey(0), cfg0.games)
    oi = jnp.asarray([1, 1, 0, 1], jnp.int32); ot = jnp.asarray([10, 5, 10, 0], jnp.int32)
    t0 = time.time()
    carry, traj, _, boot = tr0._rollout(params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(1), L=tr0.E, learner_greedy=False, record=True)
    jax.block_until_ready(traj["rew"])
    N = tr0.N
    log(dict(bench="collect", games=cfg0.games, samples=N, seconds=round(time.time() - t0, 1)), fn)
    del bank, carry
    for name in names:
        try:
            bench_variant(fn, name, params, ref, traj, boot, N)
        except Exception as e:
            log(dict(bench="update_error", variant=name, err=repr(e)[:300]), fn)


def bench_variant(fn, name, params, ref, traj, boot, N):
    if True:
        cfg = P.Cfg(**dict(BASE, **VARIANTS[name]))
        tr = P.Trainer(cfg)
        vp = value_params(tr, params, cfg, jax.random.PRNGKey(3))
        mk = lambda: (jax.tree_util.tree_map(jnp.array, params), jax.tree_util.tree_map(jnp.array, vp))
        p1, v1 = mk()
        args = (p1, v1, P.adam_init(p1), P.adam_init(v1), ref, traj, boot, jax.random.PRNGKey(4))
        t0 = time.time()
        compiled = tr._update.lower(*args, train_policy=True, diag=False).compile()
        t_comp = time.time() - t0
        m = mem(compiled)
        del compiled
        pb = probe_tf32()
        t0 = time.time()
        out = tr._update(*args, train_policy=True, diag=False)
        st = {k: float(v) for k, v in out[4].items()}
        t_run = time.time() - t0
        pa = probe_tf32()
        del out, p1, v1, args
        log(dict(bench="update", variant=name, samples=N, epochs=cfg.epochs, mb=cfg.mb, micro=tr.micro, ref_frac=cfg.ref_frac,
                 value=cfg.value_init, target=cfg.value_target, compile_s=round(t_comp, 1), update_s=round(t_run, 1),
                 samples_per_s=round(N / t_run), sample_passes_per_s=round(N * cfg.epochs / t_run), **m,
                 klu=st["klu"], klm=st["klm"], clipfrac=st["clipfrac"], ev_gae=st["ev_gae"], ev_mc=st["ev_mc"],
                 finite=st["finite"], probe_before=pb, probe_after=pa, **{k: st[k] for k in st if k.startswith("r0_")}), fn)


_PROBE = None


def probe_tf32():
    """TF32 matmul TFLOP/s (8192^2, 3 reps): the contention level around one timed run; 0 on the CPU mock."""
    global _PROBE
    if os.environ.get("BENCH_MOCK"):
        return 0.0
    if _PROBE is None:
        _PROBE = (jnp.ones((8192, 8192), jnp.float32), jax.jit(lambda a: jnp.dot(a, a)))
        _PROBE[1](_PROBE[0]).block_until_ready()
    x, f = _PROBE
    t0 = time.time()
    for _ in range(3):
        y = f(x)
    y.block_until_ready()
    return round(3 * 2 * 8192 ** 3 / (time.time() - t0) / 1e12, 1)


def probe():
    """TF32 / bf16 matmul throughput right now (the GPU is shared: the contention level for projections)."""
    x = jnp.ones((8192, 8192), jnp.float32)
    res = {}
    for name, f in (("tf32", jax.jit(lambda a: jnp.dot(a, a))),
                    ("bf16", jax.jit(lambda a: (a.astype(jnp.bfloat16) @ a.astype(jnp.bfloat16)).astype(jnp.float32)))):
        f(x).block_until_ready()
        t0 = time.time()
        for _ in range(5):
            y = f(x)
        y.block_until_ready()
        res[name + "_tflops"] = round(5 * 2 * 8192 ** 3 / (time.time() - t0) / 1e12, 1)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    which = os.environ.get("BENCH", "update")
    fn = os.path.join(OUT, f"bench_{which}_{TAG}.jsonl")
    log(dict(bench="start", which=which, devices=str(jax.devices()), init=INIT, time=time.strftime("%H:%M:%S"),
             **(probe() if not os.environ.get("BENCH_MOCK") else {})), fn)
    (bench_rollout if which == "rollout" else bench_update)(fn)
    log(dict(bench="done", time=time.strftime("%H:%M:%S"), **(probe() if not os.environ.get("BENCH_MOCK") else {})), fn)


if __name__ == "__main__":
    main()
