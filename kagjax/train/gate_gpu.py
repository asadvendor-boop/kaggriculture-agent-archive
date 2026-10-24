"""Pre-training gate (operator, Sep 26), GPU part.  Real engine, warm start g20_w_dsm_s0, IL clones g20_w_s0 (operator: rating-weighted models).

  item 1  LR = VLR = 0 for 2 iterations (twice: OLD_LOGP=behavior, the default, and OLD_LOGP=recompute): ratio
          statistics of every minibatch (max |ratio - 1| for unit ops / quantities / market fires), clip fractions,
          KL to the reference, max |log-prob stored at rollout - log-prob recomputed in the update|, and the saved
          params compared bit for bit with the start checkpoint.
  item 3  one full episode (720 slots) of 64 games with WINB = 0.3: sum of the learner's per-step rewards vs
          final margin / 2e4 + WINB sign(final margin) on every game; GAE with gamma = lambda = 1 must return the
          reward-to-go at every valid step; the dummy slot (t = 719) is masked (valid = False, advantage 0).
  item 4  (preparation) 2 real updates at LR 5e-5 -> pt/it2.pt, the JAX-trained snapshot for the round trip.
Writes $GATE_DIR/gate_gpu.json.   bash kagjax/train/srv_launch.sh <log> GATE_DIR=<dir> --gpu python kagjax/train/gate_gpu.py
"""
import os, sys, json, time, dataclasses
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import ppo as P
import ckpt_io

R = "/Users/asad.ali/kaggriculture"
GD = os.environ.get("GATE_DIR", f"{R}/kagjax/logs_server/D/gate")
INIT = os.environ.get("INIT", f"{R}/il/ckpt/g20_w_dsm_s0.pt")
IL = os.environ.get("IL", f"{R}/il/ckpt/g20_w_s0.pt")
BASE = dict(backend="real", init=INIT, il=IL, games=int(os.environ.get("GATE_B", "64")), blocks=4, mb=256, eval_every=0,
            snap_every=0, pt_every=0, self_max=2, ckpt_every=2, log_every=1)


def stats(run_dir):
    return [json.loads(l) for l in open(os.path.join(run_dir, "stats.jsonl"))]


def item1(out, old_logp):
    rd = os.path.join(GD, f"lr0_{old_logp}")
    cfg = P.Cfg(**BASE, seg=240, epochs=2, iters=2, lr=0.0, vlr=0.0, diag_old=1, old_logp=old_logp, run_dir=rd)
    P.main(cfg)
    st = stats(rd)
    p, v, _, _, _ = ckpt_io.load_train_state(os.path.join(rd, "state_2.npz"))
    import torch
    ref = torch.load(INIT, map_location="cpu", weights_only=False)["state"]
    sp, sv = ckpt_io.to_sd(p), ckpt_io.to_sd(v)
    changed_p = [k for k in ref if not np.array_equal(ref[k].numpy().astype(np.float32), sp[k])]
    changed_v = [k for k in ref if not np.array_equal(ref[k].numpy().astype(np.float32), sv[k])]
    keys = ["r0_ru_maxdev", "r0_rq_maxdev", "r0_rm_maxdev", "ru_maxdev", "rq_maxdev", "rm_maxdev", "ru_mean", "clipfrac", "r0_clipfrac",
            "klu", "klm", "akl", "old_diff_lpu", "old_diff_lpq", "old_diff_lpm", "finite"]
    out[f"item1_{old_logp}"] = dict(per_iteration=[{k: s.get(k) for k in keys} for s in st],
                                    policy_params_changed=changed_p, value_params_changed=changed_v,
                                    bit_unchanged=not changed_p and not changed_v)
    print("[item1]", old_logp, json.dumps(out[f"item1_{old_logp}"]), flush=True)


def item3(out):
    cfg = P.Cfg(**BASE, seg=720, winb=0.3, iters=1)
    tr = P.Trainer(cfg)
    be = tr.be
    params, _ = be.load_params(INIT, "state"); il, _ = be.load_params(IL, "state")
    bank = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), params, il)
    cs = be.consts()
    B = cfg.games
    carry = tr.fresh_games(jax.random.PRNGKey(7), B)
    oi = jnp.asarray([1, 1, 0, 1], jnp.int32); ot = jnp.asarray([10, 5, 10, 0], jnp.int32)       # IL@10, IL@5, LATEST, IL@0
    t0 = time.time()
    carry, traj, (fin, mar), boot = tr._rollout(params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(8), L=720, learner_greedy=False, record=True)
    rew = np.asarray(traj["rew"], np.float64); valid = np.asarray(traj["valid"]); done = np.asarray(traj["done"])
    fin = np.asarray(mar, np.float64)
    target = fin / cfg.rscale + cfg.winb * np.sign(fin)
    err = np.abs(rew.sum(0) - target)
    # GAE with gamma = lambda = 1 -> advantage + value = reward-to-go (any values; bootstrap irrelevant: done at t = 718)
    v = np.random.RandomState(0).randn(*rew.shape).astype(np.float32)
    adv = np.asarray(P.gae(jnp.asarray(rew, jnp.float32), jnp.asarray(v), jnp.asarray(np.random.randn(B).astype(np.float32)),
                           jnp.asarray(done), jnp.asarray(valid), 1.0), np.float64)
    rtg = np.cumsum(rew[::-1], 0)[::-1]
    gerr = np.abs((adv + v) - rtg)[valid].max()
    out["item3"] = dict(games=B, episode_s=round(time.time() - t0, 1), t_end=int(carry[0].t),
                        final_margin_mean=float(fin.mean()), max_abs_err_reward_units=float(err.max()),
                        max_abs_err_dollars=float(err.max() * cfg.rscale), gae_lambda1_max_abs_err=float(gerr),
                        dummy_slot_masked=bool((~valid[719]).all() and valid[:719].all() and np.all(adv[719] == 0)),
                        done_only_at_718=bool(done[718].all() and not done[:718].any() and not done[719].any()))
    print("[item3]", json.dumps(out["item3"]), flush=True)


def item4_prep(out):
    rd = os.path.join(GD, "trained")
    cfg = P.Cfg(**dict(BASE, pt_every=2), seg=240, epochs=2, iters=2, lr=5e-5, run_dir=rd)
    P.main(cfg)
    out["item4_snapshot"] = dict(pt=os.path.join(rd, "pt", "it2.pt"), stats=[{k: s[k] for k in ("klu", "klm", "clipfrac", "l_v", "ent", "finite")} for s in stats(rd)])
    print("[item4 prep]", json.dumps(out["item4_snapshot"]), flush=True)


def precision_probe():
    """f32 matmul throughput with the XLA default precision vs HIGHEST (tells whether TF32 tensor cores are used)."""
    x = jnp.ones((8192, 8192), jnp.float32)
    res = {}
    for name, prec in (("default", None), ("highest", jax.lax.Precision.HIGHEST), ("bf16", "bf16")):
        f = jax.jit(lambda a: (a.astype(jnp.bfloat16) @ a.astype(jnp.bfloat16)).astype(jnp.float32)) if prec == "bf16" else \
            jax.jit(lambda a, p=prec: jnp.dot(a, a, precision=p))
        f(x).block_until_ready()
        t0 = time.time()
        for _ in range(5):
            y = f(x)
        y.block_until_ready()
        res[name + "_tflops"] = round(5 * 2 * 8192 ** 3 / (time.time() - t0) / 1e12, 1)
    return res


def main():
    os.makedirs(GD, exist_ok=True)
    fn = os.path.join(GD, "gate_gpu.json")
    out = json.load(open(fn)) if os.environ.get("GATE_SKIP") and os.path.exists(fn) else {}
    out.update({"devices": str(jax.devices()), "init": INIT, "il": IL})
    try:
        out["matmul_precision_probe"] = precision_probe()
        print("[probe]", out["matmul_precision_probe"], flush=True)
    except Exception as e:
        out["matmul_precision_probe"] = repr(e)
    for name, fn in (("item1_behavior", lambda: item1(out, "behavior")), ("item1_recompute", lambda: item1(out, "recompute")),
                     ("item3", lambda: item3(out)), ("item4_prep", lambda: item4_prep(out))):
        if name in os.environ.get("GATE_SKIP", "").split(","):
            continue
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            out[name + "_error"] = repr(e)
        out[name + "_seconds"] = round(time.time() - t0, 1)
        json.dump(out, open(os.path.join(GD, "gate_gpu.json"), "w"), indent=1)
    print("[gate_gpu] done", flush=True)


if __name__ == "__main__":
    main()
