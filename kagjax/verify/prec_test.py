"""Integration risk: TF32 / bf16 rollouts vs the fp32 policy the Kaggle numpy runtime executes.

phase inputs (CPU):  real observations of recorded top games (C++ engine), both seats, every 3rd step, encoded with
                     the REFERENCE encoder il/features6.encode_obs (prev ops = the recorded previous action's classes),
                     legal masks from il/np_runtime.legal_mask_np; numpy runtime (NumpyNet) logits for a subset.
phase gpu (GPU):     the same inputs through kagjax.policy.net.forward at precision HIGHEST (fp32), None (XLA default =
                     TF32 on Ampere, what ppo.py uses by default for rollouts AND updates) and bf16 (ROLLOUT_PRECISION
                     =bf16, via train/backend.Common.net_apply); batch-size dependence of TF32 (rollout batch vs update
                     minibatch); decision-level agreement with fp32 (greedy ops / qty / fires, as NumpyAgent decides)
                     and the log-prob noise PPO sees (T=0.2 unit ops, KSHARP=10 fires at tau 0.5).
usage: python prec_test.py inputs <ckpt> <n_games> <out.npz>     (CPU)
       python prec_test.py gpu <ckpt> <in.npz> <out.json>         (GPU, via srv_jax.sh --gpu)
"""
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, R)
sys.path.insert(0, f"{R}/il")
sys.path.insert(0, f"{R}/kagjax/train")


def phase_inputs(ckpt, n_games, out):
    sys.path.insert(0, os.path.join(HERE, "cov_build", "opt"))
    import kagsim_trace as K
    import features as F
    import features6 as F6
    import np_runtime as NR
    import torch
    files = sorted(glob.glob(f"{R}/data/tapes_top12/*.json.gz"))[500:500 + n_games]
    P, G, POS, U, M, NU, SEAT, STEP = [], [], [], [], [], [], [], []
    for f in files:
        t = json.load(gzip.open(f))
        acts = [[x if isinstance(x, dict) else {"farmer": ["PASS"], "hands": [], "market": []} for x in t["actions"][k]]
                for k in (0, 1)]
        g = K.Game(int(t["seed"]), 720, False)
        prev = [None, None]
        for s in range(719):
            if s % 3 == 1:
                for seat in (0, 1):
                    o = g.observe(seat)
                    planes, gl, pos, uf = F6.encode_obs(o, prev[seat])
                    n = len(pos)
                    mask = np.zeros((16, 44), bool)
                    mask[:, 0] = True
                    for k in range(n):
                        mask[k] = NR.legal_mask_np(o, k, pos[k], F6)
                    pp = np.zeros((16, 2), np.int32); pp[:n] = pos
                    uu = np.zeros((16, uf.shape[1]), np.float32); uu[:n] = uf
                    P.append(planes); G.append(gl); POS.append(pp); U.append(uu); M.append(mask); NU.append(n)
                    SEAT.append(seat); STEP.append(s)
            for seat in (0, 1):
                a = acts[seat][s]
                prev[seat] = [F.unit_op_label(u)[0] for u in [a.get("farmer")] + list(a.get("hands") or [])]
            g.step(acts[0][s], acts[1][s])
        print("inputs from", os.path.basename(f), len(P), flush=True)
    planes = np.stack(P); globs = np.stack(G).astype(np.float32); pos = np.stack(POS); uin = np.stack(U)
    # numpy runtime reference on a subset (what the Kaggle package computes)
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
    blocks = len({k.split(".")[1] for k in st if k.startswith("blocks.")})
    net = NR.NumpyNet(st, width=c["width"], blocks=blocks, n_teams=c.get("n_teams", 0))
    sub = np.arange(0, len(P), max(1, len(P) // 150))
    ref_op, ref_q, ref_mk = [], [], []
    t0 = time.time()
    for i in sub:
        n = NU[i]
        a, b, m = net.forward(planes[i].astype(np.float32) / 255.0, globs[i], pos[i, :n].astype(np.int64), uin[i, :n], 10)
        oo = np.zeros((16, 44), np.float32); oo[:n] = a
        qq = np.zeros((16, 12), np.float32); qq[:n] = b
        ref_op.append(oo); ref_q.append(qq); ref_mk.append(m)
    print(f"numpy runtime logits for {len(sub)} obs in {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, planes=planes, glob=globs, pos=pos, uin=uin, mask=np.stack(M), nu=np.array(NU),
                        seat=np.array(SEAT), step=np.array(STEP), sub=sub, ref_op=np.stack(ref_op),
                        ref_q=np.stack(ref_q), ref_mk=np.stack(ref_mk))
    print("saved", out, planes.shape)


def phase_gpu(ckpt, inp, out):
    import jax
    import jax.numpy as jnp
    from kagjax.policy import net as N
    import backend as BE
    import actions as A
    import sample as SM
    print(jax.devices(), flush=True)
    z = np.load(inp)
    params, meta = N.load_ckpt(ckpt, "state")
    common = BE.Common()
    Ntot = z["planes"].shape[0]
    planes = jnp.asarray(z["planes"].astype(np.float32) / 255.0)       # exact (numpy division is correctly rounded)
    glob_ = jnp.asarray(z["glob"]); pos = jnp.asarray(z["pos"]); uin = jnp.asarray(z["uin"])
    team = jnp.full((Ntot,), 10, jnp.int32)
    mask = z["mask"]; nu = z["nu"]
    live = np.arange(16)[None, :] < nu[:, None]

    def run(prec, bs):
        f = jax.jit(lambda p, a, b, c, d, e: common.net_apply(p, a, b, c, d, e, prec))
        outs = []
        for i in range(0, Ntot, bs):
            sl = slice(i, min(Ntot, i + bs))
            n = sl.stop - sl.start
            pad = bs - n
            padf = lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, 0)]) if pad else x
            o = f(params, padf(planes[sl]), padf(glob_[sl]), padf(pos[sl]), padf(uin[sl]), padf(team[sl]))
            outs.append({k: np.asarray(v)[:n] for k, v in o.items() if k in ("op", "qty", "mk", "v")})
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    t0 = time.time()
    res = {"n_obs": int(Ntot), "n_units": int(live.sum())}
    O = {"hi": run("highest", 256), "tf32": run("default", 256), "tf32_b64": run("default", 64),
         "tf32_b1024": run("default", 1024), "bf16": run("bf16", 256)}
    res["forward_s"] = time.time() - t0
    sub = z["sub"]
    # (1) fp32 JAX vs the numpy runtime (sanity anchor)
    lv = live[sub]
    res["hi_vs_numpy_maxabs"] = {"op": float(np.abs(O["hi"]["op"][sub] - z["ref_op"])[lv].max()),
                                 "qty": float(np.abs(O["hi"]["qty"][sub] - z["ref_q"])[lv].max()),
                                 "mk": float(np.abs(O["hi"]["mk"][sub] - z["ref_mk"]).max())}

    def greedy(o):
        zo = np.where(mask, o["op"], -1e9)
        op = zo.argmax(-1)
        q = o["qty"].argmax(-1)
        h = SM.heads(jnp.asarray(o["mk"]))
        lo = np.asarray(jax.nn.logsumexp(h[..., 1:], -1) - h[..., 0])
        cls = np.asarray(1 + jnp.argmax(h[..., 1:], -1))
        return op, q, lo, cls

    def logp_units(o, T=0.2):
        zo = np.where(mask, o["op"].astype(np.float64) / T, -1e9)
        zo = zo - zo.max(-1, keepdims=True)
        return zo - np.log(np.exp(zo).sum(-1, keepdims=True))

    def fire_logs(o, ks=10.0, tau=0.5):
        h = SM.heads(jnp.asarray(o["mk"], jnp.float32))
        lo = np.asarray(jax.nn.logsumexp(h[..., 1:], -1) - h[..., 0]).astype(np.float64)
        LC = SM.LOGIT_CLAMP
        a = np.clip(ks * (np.clip(lo, -LC, LC) - np.log(tau / (1 - tau))), -LC, LC)
        return -np.log1p(np.exp(-a)), -np.log1p(np.exp(a))        # log pf, log(1-pf)

    g_hi = greedy(O["hi"])
    lp_hi = logp_units(O["hi"])
    f1_hi, f0_hi = fire_logs(O["hi"])
    rng = np.random.RandomState(0)
    for name in ("tf32", "tf32_b64", "tf32_b1024", "bf16"):
        o = O[name]
        g = greedy(o)
        pp = (g_hi[0] >= A.OP_PICKUP0) & live
        fire_hi, fire_o = g_hi[2] > 0, g[2] > 0
        r = {"op_maxabs": float(np.abs(o["op"] - O["hi"]["op"])[live].max()),
             "op_meanabs": float(np.abs(o["op"] - O["hi"]["op"])[live].mean()),
             "mk_maxabs": float(np.abs(o["mk"] - O["hi"]["mk"]).max()),
             "v_maxabs": float(np.abs(o["v"] - O["hi"]["v"]).max()),
             "greedy_op_disagree": float((g[0] != g_hi[0])[live].mean()),
             "greedy_op_disagree_n": int((g[0] != g_hi[0])[live].sum()),
             "obs_with_any_op_flip": float(((g[0] != g_hi[0]) & live).any(1).mean()),
             "greedy_qty_disagree": float((g[1] != g_hi[1])[pp].mean()) if pp.any() else 0.0,
             "fire_disagree": float((fire_o != fire_hi).mean()),
             "fire_disagree_n": int((fire_o != fire_hi).sum()),
             "class_disagree_when_fired": float((g[3] != g_hi[3])[fire_hi & fire_o].mean())}
        # PPO log-prob noise at T=0.2: sample a ~ pi_fp32, compare log pi_x(a) - log pi_fp32(a)
        lp = logp_units(o)
        p_hi = np.exp(lp_hi)
        cum = p_hi.cumsum(-1)
        u = rng.random_sample(cum.shape[:2])[..., None]
        a = np.minimum((cum < u).sum(-1), 43)
        d = np.take_along_axis(lp, a[..., None], -1)[..., 0] - np.take_along_axis(lp_hi, a[..., None], -1)[..., 0]
        kl = (p_hi * (lp_hi - lp)).sum(-1)
        r.update(unit_logp_absdiff_mean=float(np.abs(d)[live].mean()), unit_logp_absdiff_p99=float(np.quantile(np.abs(d)[live], 0.99)),
                 unit_logp_absdiff_max=float(np.abs(d)[live].max()), unit_ratio_outside_clip=float((np.abs(np.exp(d)) > 1.2)[live].mean() + (np.exp(d) < 0.8)[live].mean()),
                 unit_kl_mean=float(kl[live].mean()), unit_kl_max=float(kl[live].max()))
        f1, f0 = fire_logs(o)
        pf = np.exp(f1_hi)
        fired = rng.random_sample(pf.shape) < pf
        dm = np.where(fired, f1 - f1_hi, f0 - f0_hi)
        r.update(fire_logp_absdiff_mean=float(np.abs(dm).mean()), fire_logp_absdiff_p99=float(np.quantile(np.abs(dm), 0.99)),
                 fire_logp_absdiff_max=float(np.abs(dm).max()), fire_ratio_outside_clip=float(((np.exp(dm) > 1.2) | (np.exp(dm) < 0.8)).mean()))
        res[name + "_vs_fp32"] = r
        print(name, json.dumps(r), flush=True)
    # batch-size dependence of TF32 (behavior log-probs from the rollout batch vs recomputed in update minibatches)
    for a_, b_ in (("tf32", "tf32_b64"), ("tf32", "tf32_b1024")):
        res[f"{a_}_vs_{b_}_op_maxabs"] = float(np.abs(O[a_]["op"] - O[b_]["op"])[live].max())
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    print("hi_vs_numpy", res["hi_vs_numpy_maxabs"])


if __name__ == "__main__":
    if sys.argv[1] == "inputs":
        phase_inputs(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        phase_gpu(sys.argv[2], sys.argv[3], sys.argv[4])
