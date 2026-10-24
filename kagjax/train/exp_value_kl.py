"""Value-function options and the KL-anchor subsample, measured on the SAME rollouts (real engine, GPU).

EXP=collect : R = 4 rollouts of B = 128 games x 720 steps, the frozen g20_w_dsm_s0 (team 10, T-sampled as in training)
              vs the imitation clones (g20_w_s0 as DSM 10 / Vadim 5 / DECEM 3 / Boey 0, one block each); saved to
              $EXP_DIR/roll_<r>.npz (the value-head features included) - identical for every option below.
EXP=kl      : REF_FRAC unbiasedness.  ref = g20_w_s0 (so the KL is not zero), 4,096 samples of rollout 0: per-sample KL
              terms exactly as ppo.policy_loss sums them; the subsampled estimator (M = f x 4,096 without replacement,
              weight 1/f) over 2,000 draws vs the full value (bias / noise), and the gradient of beta * KL: 8 draws vs
              the full gradient (cosine similarity, relative error of their mean).
EXP=value   : the trainer's own update, policy frozen (value-only path), for each option in EXP_OPTIONS on rollouts
              0..3 in order: the statistics before each update are a held-out fit (the value function has not seen
              that rollout): explained variance of the Monte-Carlo return (ev_mc) and of the GAE(0.95) return (ev_gae),
              the advantage standard deviation; plus the time of the value-only update.
Results: $EXP_DIR/exp_<EXP>.jsonl.   bash kagjax/train/srv_launch.sh <log> EXP=collect --gpu python kagjax/train/exp_value_kl.py
"""
import os, sys, json, time, dataclasses
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import ppo as P
import sample as S
from backend import Obs

R = "/Users/asad.ali/kaggriculture"
EXP_DIR = os.environ.get("EXP_DIR", f"{R}/kagjax/logs_server/D/speed/exp")
INIT = f"{R}/il/ckpt/g20_w_dsm_s0.pt"
IL = f"{R}/il/ckpt/g20_w_s0.pt"
NR, B = 4, 128
CFG0 = dict(backend="real", init=INIT, il=IL, games=B, blocks=4, seg=720, eval_every=0, value_init="head", value_target="final",
            lr=0.0)
OPTIONS = {   # value-network options; LR = 0 (policy frozen); VLR by the square-root rule from the gate's MB=256
    "full_togo_gateopt": dict(value_init="policy", value_target="togo", epochs=2, mb=256, micro=256, vlr=1e-4),
    "full_togo": dict(value_init="policy", value_target="togo", epochs=1, mb=4096, micro=512, vlr=4e-4),
    "full_final": dict(value_init="policy", value_target="final", epochs=1, mb=4096, micro=512, vlr=4e-4),
    "small_togo": dict(value_init="small:128x4", value_target="togo", epochs=1, mb=4096, micro=1024, vlr=4e-4),
    "small_final": dict(value_init="small:128x4", value_target="final", epochs=1, mb=4096, micro=1024, vlr=4e-4),
    "head_final": dict(value_init="head", value_target="final", epochs=1, mb=4096, micro=1024, vlr=4e-4),
    "head_togo": dict(value_init="head", value_target="togo", epochs=1, mb=4096, micro=1024, vlr=4e-4),
    "head_final_vlr1e3": dict(value_init="head", value_target="final", epochs=1, mb=4096, micro=1024, vlr=1e-3),
}


MOCK = bool(os.environ.get("EXP_MOCK"))
if MOCK:                                                     # code-path check on the Mac (tiny nets, mock engine)
    B = 8
    CFG0.update(backend="mock", init="tiny:8x1", il="", episode_slots=48, seg=48)
    for _o in OPTIONS.values():
        _o.update(mb=min(_o["mb"], 64), micro=min(_o["micro"], 32))
        if _o["value_init"].startswith("small:"):
            _o["value_init"] = "small:8x1"


def log(rec, fn):
    print(json.dumps(rec), flush=True)
    with open(fn, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load(tr, path, seed):
    if MOCK:
        return tr.be.init_params(jax.random.PRNGKey(seed), width=8, blocks=1)
    return tr.be.load_params(path, "state")[0]


def setup():
    tr = P.Trainer(P.Cfg(**CFG0))
    return tr, load(tr, INIT, 1)


def collect(fn):
    tr, params = setup()
    il = load(tr, IL, 2)
    bank = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), params, il)
    cs = tr.be.consts()
    oi = jnp.asarray([1, 1, 1, 1], jnp.int32); ot = jnp.asarray([10, 5, 3, 0], jnp.int32)
    for r in range(NR):
        t0 = time.time()
        carry = tr.fresh_games(jax.random.PRNGKey(100 + r), B)
        _, traj, (fin, mar), boot = tr._rollout(params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(200 + r), L=tr.E,
                                               learner_greedy=False, record=True)
        traj, boot = jax.device_get((traj, boot))
        np.savez(os.path.join(EXP_DIR, f"roll_{r}.npz"), **{"t_" + k: v for k, v in traj.items()},
                 b_planes=boot[0].planes, b_glob=boot[0].glob, b_pos=boot[0].pos, b_uin=boot[0].uin, b_nu=boot[0].nu, b_marg=boot[1])
        m = np.asarray(mar)
        log(dict(exp="collect", r=r, seconds=round(time.time() - t0, 1), margin_mean=float(m.mean()), win=float((m > 0).mean())), fn)


def load_roll(r):
    z = np.load(os.path.join(EXP_DIR, f"roll_{r}.npz"))
    traj = {k[2:]: jnp.asarray(z[k]) for k in z.files if k.startswith("t_")}
    boot = (Obs(jnp.asarray(z["b_planes"]), jnp.asarray(z["b_glob"]), jnp.asarray(z["b_pos"]), jnp.asarray(z["b_uin"]),
                jnp.asarray(z["b_nu"])), jnp.asarray(z["b_marg"]))
    return traj, boot


def kl(fn):
    tr, params = setup()
    ref = load(tr, IL, 2)
    traj, _ = load_roll(0)
    N = traj["rew"].size
    keys = ("planes", "glob", "pos", "uin", "umask", "uact", "qact", "mfire", "valid")
    flat = {k: traj[k].reshape((N,) + traj[k].shape[2:]) for k in keys}
    rng = np.random.RandomState(0)
    valid = np.flatnonzero(np.asarray(flat["valid"]))
    NP_ = 4096 if not MOCK else 256
    idx = jnp.asarray(np.sort(rng.choice(valid, NP_, replace=False)))
    d = {k: v[idx] for k, v in flat.items()}
    del traj, flat                                   # (GPU memory: only the 4,096 samples stay)
    tau = jnp.asarray(tr.tau)
    beta = tr.cfg.beta

    def terms(p, dd):
        out = tr._fwd(p, Obs(dd["planes"], dd["glob"], dd["pos"].astype(jnp.int32), dd["uin"].astype(jnp.float32), None),
                      jnp.full((dd["glob"].shape[0],), 10, jnp.int32), "default")
        return S.logp_terms(out, dd["umask"], dd["uact"], dd["qact"], dd["mfire"], tr.cfg.T, tr.cfg.ks, tau)

    CH = min(1024, NP_)
    chunk = lambda f, dd: [f({k: v[i:i + CH] for k, v in dd.items()}) for i in range(0, NP_, CH)]
    cat = lambda xs, k: jnp.concatenate([x[k] for x in xs])
    rt = chunk(jax.jit(lambda dd: terms(ref, dd)), d)
    rl, rp = cat(rt, "lpu_all"), cat(rt, "pf")
    ct = chunk(jax.jit(lambda dd: terms(params, dd)), d)
    vm = cat(ct, "valid") & d["valid"][:, None]
    pu = jnp.exp(cat(ct, "lpu_all"))
    klu_i = jnp.where(vm, jnp.where(d["umask"], pu * (cat(ct, "lpu_all") - rl), 0.0).sum(-1), 0.0).sum(-1)       # per sample
    pf = cat(ct, "pf")
    klm_i = jnp.where(d["valid"][:, None], pf * jnp.log(pf / rp) + (1 - pf) * jnp.log((1 - pf) / (1 - rp)), 0.0).sum(-1)
    nvm, nm = float(vm.sum()), float(d["valid"].sum() * 21)
    ku, km = np.asarray(klu_i, np.float64), np.asarray(klm_i, np.float64)
    full = (ku.sum() / nvm, km.sum() / nm)
    res = dict(exp="kl", samples=NP_, klu_full=full[0], klm_full=full[1])
    for f in (0.5, 0.25):
        M = int(f * NP_)
        est = np.array([[ku[s].sum() * NP_ / M / nvm, km[s].sum() * NP_ / M / nm]
                        for s in (rng.choice(NP_, M, replace=False) for _ in range(2000))])
        res[f"f{f}"] = dict(klu_mean=est[:, 0].mean(), klu_bias_in_se=(est[:, 0].mean() - full[0]) / (est[:, 0].std() / np.sqrt(len(est))),
                            klu_rel_sd_per_draw=est[:, 0].std() / full[0], klm_mean=est[:, 1].mean(),
                            klm_bias_in_se=(est[:, 1].mean() - full[1]) / (est[:, 1].std() / np.sqrt(len(est))),
                            klm_rel_sd_per_draw=est[:, 1].std() / full[1])
    # gradient of beta * (klu + klm) w.r.t. the policy params: full vs subsample draws (f = 0.25, weight 4)

    def kl_loss(p, dd, rl_, rp_, w):
        c = terms(p, dd)
        vm_ = c["valid"] & dd["valid"][:, None]
        pu_ = jnp.exp(c["lpu_all"])
        u = jnp.where(vm_, jnp.where(dd["umask"], pu_ * (c["lpu_all"] - rl_), 0.0).sum(-1) * w[:, None], 0.0).sum() / nvm
        pf_ = c["pf"]
        m_ = jnp.where(dd["valid"][:, None], (pf_ * jnp.log(pf_ / rp_) + (1 - pf_) * jnp.log((1 - pf_) / (1 - rp_))) * w[:, None], 0.0).sum() / nm
        return beta * (u + m_)

    grad = jax.jit(jax.grad(kl_loss))
    add = lambda a, b: jax.tree_util.tree_map(jnp.add, a, b)

    CHG = min(256, NP_)                              # gradient chunks: 1,024 samples need a 4.2 GiB buffer (8 GB cap)

    def grad_on(rows, w):
        g = None
        for i in range(0, len(rows), CHG):
            r_ = rows[i:i + CHG]
            gi = grad(params, {k: v[r_] for k, v in d.items()}, rl[r_], rp[r_], jnp.full((len(r_),), w, jnp.float32))
            g = gi if g is None else add(g, gi)
        return g

    flatg = lambda g: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g)])
    gf = flatg(grad_on(np.arange(NP_), 1.0))
    draws = [flatg(grad_on(np.sort(rng.choice(NP_, NP_ // 4, replace=False)), 4.0)) for _ in range(8)]
    cos = lambda a, b: float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    rel = lambda a, b: float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    gm = sum(draws) / len(draws)
    res["grad"] = dict(full_norm=float(jnp.linalg.norm(gf)), per_draw_cos=[round(cos(x, gf), 4) for x in draws],
                       per_draw_rel_err=[round(rel(x, gf), 3) for x in draws], mean8_cos=cos(gm, gf), mean8_rel_err=rel(gm, gf))
    log(res, fn)


def value(fn):
    tr0, params = setup()
    names = [n for n in os.environ.get("EXP_OPTIONS", ",".join(OPTIONS)).split(",") if n]
    rolls = [load_roll(r) for r in range(NR)]
    copy = lambda t: jax.tree_util.tree_map(jnp.array, t)
    for name in names:
        cfg = P.Cfg(**dict(CFG0, **OPTIONS[name]))
        tr = P.Trainer(cfg)
        if cfg.value_init == "head":
            vp = {"w": jnp.array(params["v"]["w"]), "b": jnp.array(params["v"]["b"])}
        elif cfg.value_init.startswith("small:"):
            w, b = [int(x) for x in cfg.value_init.split(":")[1].split("x")]
            vp = tr.be.init_params(jax.random.PRNGKey(2), width=w, blocks=b, n_teams=int(params["team"].shape[0]))
        else:
            vp = copy(params)
        ov = P.adam_init(vp)
        for r in range(NR):
            traj, boot = rolls[r]
            p1 = copy(params)
            t0 = time.time()
            out = tr._update(p1, vp, P.adam_init(p1), ov, params, traj, boot, jax.random.PRNGKey(300 + r), train_policy=False, diag=False)
            st = {k: float(v) for k, v in out[4].items()}
            dt = time.time() - t0
            vp, ov = out[1], out[3]
            log(dict(exp="value", option=name, rollout=r, heldout=r > 0, ev_mc=st["ev_mc"], ev_gae=st["ev_gae"], adv_std=st["adv_std"],
                     g_std=st["g_std"], l_v_after=st["l_v"], seconds=round(dt, 1), finite=st["finite"]), fn)
            del out, p1


def main():
    os.makedirs(EXP_DIR, exist_ok=True)
    which = os.environ.get("EXP", "collect")
    fn = os.path.join(EXP_DIR, f"exp_{which}.jsonl")
    log(dict(exp="start", which=which, devices=str(jax.devices()), time=time.strftime("%H:%M:%S")), fn)
    {"collect": collect, "kl": kl, "value": value}[which](fn)
    log(dict(exp="done", time=time.strftime("%H:%M:%S")), fn)


if __name__ == "__main__":
    main()
