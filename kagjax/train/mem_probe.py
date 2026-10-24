"""GPU memory of one training iteration for several games-per-GPU sizes, WITHOUT training (compile only, ~1 min per size).

For each B: the rollout (720 recorded steps, TF32 learner, bf16 opponents) and the PPO update (fast preset, value head,
bf16grad, MB 4096 / MICRO 1024) of the 10M are compiled for ONE device and XLA's buffer assignment is read:
  rollout  = arguments + outputs (the trajectory) + temporaries;  update = arguments (incl. the trajectory) + temporaries.
Peak estimate = the larger of the two + the device tape bank (TAPE seats x ~0.31 MB) + 1 GiB (evaluation programs,
allocator slack).  Choose the largest B whose estimate stays below 85% of the device memory (>= 15% headroom).
Calibration: A100-40GB, fast preset, B=2048, OPP fp32, no tapes: measured peak_bytes_in_use 20.4 GiB (Sep 27, ded_10m_b2048).
--run B: also run ONE real iteration at that size and print the measured peak (a few minutes).
usage: CUDA_VISIBLE_DEVICES=0 python kagjax/train/mem_probe.py [--b 2048,3072,4096,5120] [--tapes 2745] [--run 4096]
"""
import os, sys, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp

R = "/Users/asad.ali/kaggriculture"
GiB = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", default="2048,3072,4096,5120")
    ap.add_argument("--tapes", type=int, default=2745)        # Sep 27: 2,423 leader + 322 specialist training seats
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--init", default=f"{R}/il/ckpt/rf_w_dsm_s0.pt")
    ap.add_argument("--il", default=f"{R}/il/ckpt/rf_w_s0.pt")
    a = ap.parse_args()
    os.environ.setdefault("KAGJAX_BACKEND", "real")
    import ppo as P
    dev = jax.devices()[0]
    limit = (dev.memory_stats() or {}).get("bytes_limit", 0)
    print(f"device {dev} memory limit {limit / GiB:.1f} GiB (XLA_PYTHON_CLIENT_MEM_FRACTION {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default')})",
          flush=True)
    tape_gib = a.tapes * (30 * 624 * 4 + 720 * 40 * 2 * 3 + 720 * 10 * (2 * 2 + 4) + 8 * 4 + 30 * 4 + 4) / GiB
    base = dict(backend="real", init=a.init, il=a.il, blocks=8, seg=720, mb=4096, micro=1024, epochs=1, lr=2e-5, vlr=4e-4, ref_frac=0.25,
                value_init="head", value_target="final", update_precision="bf16grad", opp_precision="bf16", eval_every=0, devices=1)
    rows = []
    params = il = None
    for B in [int(x) for x in a.b.split(",") if x]:
        cfg = P.Cfg(**dict(base, games=B))
        tr = P.Trainer(cfg)
        be = tr.be
        if params is None:
            params, _ = be.load_params(a.init, "state"); il, _ = be.load_params(a.il, "state")
        bank = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *([params, il, params] + [params] * cfg.self_max))
        cs = be.consts()
        vp = {"w": params["v"]["w"], "b": params["v"]["b"]}
        carry = tr.fresh_games(jax.random.PRNGKey(0), B)
        oi = jnp.ones((cfg.blocks,), jnp.int32); ot = jnp.full((cfg.blocks,), 10, jnp.int32); om = jnp.zeros((cfg.blocks,), jnp.int32)
        t0 = time.time()
        ro_args = (params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(1), om, jnp.float32(0.5))
        ro = tr._rollout.lower(*ro_args, L=cfg.seg, learner_greedy=False, record=True).compile()
        out_shape = jax.eval_shape(lambda *x: tr._rollout(*x, L=cfg.seg, learner_greedy=False, record=True), *ro_args)
        traj_s, boot_s = out_shape[1], out_shape[3]
        up = tr._update.lower(params, vp, P.adam_init(params), P.adam_init(vp), params, traj_s, boot_s, jax.random.PRNGKey(2),
                              jnp.float32(0.05), None, None, train_policy=True, diag=False, anchor=True).compile()
        m_ro, m_up = ro.memory_analysis(), up.memory_analysis()
        ro_tot = m_ro.argument_size_in_bytes + m_ro.output_size_in_bytes + m_ro.temp_size_in_bytes
        up_tot = m_up.argument_size_in_bytes + m_up.temp_size_in_bytes
        est = max(ro_tot, up_tot) / GiB + tape_gib + 1.0
        r = dict(B=B, rollout_arg=m_ro.argument_size_in_bytes / GiB, rollout_out=m_ro.output_size_in_bytes / GiB,
                 rollout_temp=m_ro.temp_size_in_bytes / GiB, update_arg=m_up.argument_size_in_bytes / GiB, update_temp=m_up.temp_size_in_bytes / GiB,
                 tapes=tape_gib, estimate_gib=est, share=est * GiB / limit if limit else None, compile_s=round(time.time() - t0))
        rows.append(r)
        print(f"B={B:5d} per GPU: rollout {r['rollout_arg']:.1f} args + {r['rollout_out']:.1f} trajectory + {r['rollout_temp']:.1f} temp | "
              f"update {r['update_arg']:.1f} args + {r['update_temp']:.1f} temp | tapes {tape_gib:.1f} | ESTIMATE {est:.1f} GiB"
              + (f" = {100 * r['share']:.0f}% of the device" if limit else "") + f" ({r['compile_s']}s)", flush=True)
        del ro, up, tr, carry
    ok = [r["B"] for r in rows if r["share"] is not None and r["share"] <= 0.85]
    if ok:
        print(f"-> largest B with >= 15% headroom: {max(ok)} games per GPU", flush=True)
    if a.run:
        print(f"real iteration at B={a.run} (takes a few minutes) ...", flush=True)
        cfg = P.Cfg(**dict(base, games=a.run))
        tr = P.Trainer(cfg)
        bank = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *([params, il, params] + [params] * cfg.self_max))
        cs = tr.be.consts(); vp = {"w": jnp.array(params["v"]["w"]), "b": jnp.array(params["v"]["b"])}
        carry = tr.fresh_games(jax.random.PRNGKey(0), a.run)
        oi = jnp.ones((cfg.blocks,), jnp.int32); ot = jnp.full((cfg.blocks,), 10, jnp.int32); om = jnp.zeros((cfg.blocks,), jnp.int32)
        t0 = time.time()
        carry, traj, _, boot = tr._rollout(params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(1), om, jnp.float32(0.5), L=cfg.seg,
                                           learner_greedy=False, record=True)
        jax.block_until_ready(traj["rew"]); t1 = time.time()
        p2 = jax.tree_util.tree_map(jnp.array, params)
        out = tr._update(p2, vp, P.adam_init(p2), P.adam_init(vp), params, traj, boot, jax.random.PRNGKey(2), jnp.float32(0.05), None, None,
                         train_policy=True, diag=False, anchor=True)
        jax.block_until_ready(out[4]["loss"]); t2 = time.time()
        st = dev.memory_stats() or {}
        n = cfg.seg * a.run
        print(f"B={a.run}: measured peak {st.get('peak_bytes_in_use', 0) / GiB:.1f} GiB of {limit / GiB:.1f} (+ tapes {tape_gib:.1f} in a run) | "
              f"rollout {t1 - t0:.0f}s ({n / (t1 - t0):,.0f} samples/s incl. compile) update {t2 - t1:.0f}s", flush=True)
    json.dump(rows, open(os.environ.get("MEM_PROBE_OUT", "/tmp/kagjax_mem_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
