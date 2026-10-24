"""Parity of the JAX PPO math with the PyTorch trainer on identical inputs (fp32, CPU):
  1. sample.logp_terms           vs il/ppo.logp_terms                (all terms)
  2. ppo.policy_loss + jax.grad  vs the il/ppo_live.py loss lines + autograd (loss value and every gradient)
  3. ppo.gae                     vs the il/ppo_live.py GAE loop       (episodes of different lengths, done at the end)
  4. ppo.adam_step + clip        vs torch.optim.Adam + clip_grad_norm_ (3 steps)
  5. sample.decide               distribution check: greedy == argmax path of live_rollout; sampled fire rates match
                                 sharpen(); sampled op frequencies match softmax(z / T)
The torch network is a small il/model.Net (width 8, 1 block, 48 teams, v6 inputs); JAX runs kagjax.policy.net on the
converted weights.  run: kagjax/train/run1.sh kagjax/train/test_loss.py
"""
import os, sys
os.environ.setdefault("DEVICE", "cpu"); os.environ["T"] = "0.2"; os.environ["KSHARP"] = "10"; os.environ["TAU"] = "0.5"
os.environ["TEAM"] = "10"
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as Fn
import jax
import jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


TP = _load("il_ppo", "/Users/asad.ali/kaggriculture/il/ppo.py")                                   # the torch trainer
JP = _load("jppo", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo.py"))          # this trainer
import sample as S
from model import Net
from kagjax.policy import net as N
import actions as A

rng = np.random.RandomState(0)
T, KS, TAU, CLIP, BETA = 0.2, 10.0, 0.5, 0.2, 0.05


def tnet(seed, scale=1.0):
    torch.manual_seed(seed)
    n = Net(width=8, blocks=1, n_teams=48, g_in=179, n_aux=0)
    with torch.no_grad():
        for p in n.parameters():
            p.mul_(scale)
    return n.eval()


def jparams(n):
    return jax.tree_util.tree_map(jnp.asarray, N.convert_state_dict(n.state_dict()))


def batch(n):
    planes = (rng.randint(0, 256, (n, 47, 10, 10)) * (rng.rand(n, 47, 10, 10) < 0.2)).astype(np.uint8)
    glob = rng.rand(n, 179).astype(np.float32)
    pos = rng.randint(0, 10, (n, 16, 2)).astype(np.int8)
    nu = rng.randint(1, 17, n)
    uin = ((rng.rand(n, 16, 76) < 0.2) * rng.rand(n, 16, 76)).astype(np.float16)
    umask = rng.rand(n, 16, 44) < 0.3
    umask[:, :, 0] = True
    live = np.arange(16)[None] < nu[:, None]
    umask &= live[:, :, None]; uin *= live[:, :, None]; pos *= live[:, :, None]
    uact = np.full((n, 16), -1, np.int16); qact = np.full((n, 16), -1, np.int16)
    for i in range(n):
        for k in range(nu[i]):
            uact[i, k] = rng.choice(np.flatnonzero(umask[i, k]))
            if rng.rand() < 0.3:                       # a PICKUP / PLACE decision
                uact[i, k] = rng.randint(20, 44); umask[i, k, uact[i, k]] = True; qact[i, k] = rng.randint(12)
    mfire = (rng.rand(n, 21) < 0.3).astype(np.int8)
    return dict(planes=planes, glob=glob, pos=pos, uin=uin, umask=umask, uact=uact, qact=qact, mfire=mfire)


def to_torch(D):
    return {k: torch.from_numpy(v) for k, v in D.items()}


def jax_terms(p, D):
    out = N.forward(p, jnp.asarray(D["planes"], jnp.float32) / 255.0, jnp.asarray(D["glob"]), jnp.asarray(D["pos"], jnp.int32),
                    jnp.asarray(D["uin"], jnp.float32), jnp.full((D["planes"].shape[0],), 10, jnp.int32), precision=jax.lax.Precision.HIGHEST)
    return out, S.logp_terms(out, jnp.asarray(D["umask"]), jnp.asarray(D["uact"]), jnp.asarray(D["qact"]), jnp.asarray(D["mfire"]), T, KS, TAU)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def market_f64_check(out, D):
    """lpm under the stable and the torch-style float32 formulas vs a float64 evaluation of the same logits."""
    mk = np.asarray(out["mk"], np.float64); f = D["mfire"].astype(np.float64)
    lt = np.log(TAU / (1 - TAU)); res = {}
    lpm64 = np.zeros((mk.shape[0], 21))
    for h in range(21):
        z = mk[:, A.HEAD_OFF[h]:A.HEAD_OFF[h + 1]]
        m = z.max(1, keepdims=True); lse1 = m[:, 0] + np.log(np.exp(z[:, 1:] - m).sum(1)); d = lse1 - z[:, 0]
        cl = np.log((1 - 1e-6) / 1e-6)
        a = np.clip(KS * (np.clip(d, -cl, cl) - lt), -cl, cl)
        lpm64[:, h] = f[:, h] * -np.log1p(np.exp(-a)) + (1 - f[:, h]) * -np.log1p(np.exp(a))
    for form in ("stable", "torch"):
        S.MARKET_FORMULA = form
        t = S.logp_terms(out, jnp.asarray(D["umask"]), jnp.asarray(D["uact"]), jnp.asarray(D["qact"]), jnp.asarray(D["mfire"]), T, KS, TAU)
        res[form] = float(np.abs(np.asarray(t["lpm"], np.float64) - lpm64).max())
    S.MARKET_FORMULA = "torch"
    return res


def main():
    ok = True
    nb = 96
    S.MARKET_FORMULA = "torch"             # parity with il/ppo.py uses its exact formula; the trainer default is "stable"
    D = batch(nb)
    Dt = to_torch(D); idx = torch.arange(nb)
    net, old_net, ref_net = tnet(1), tnet(1, 1.02), tnet(2)
    # 1. logp terms
    with torch.no_grad():
        tt = TP.logp_terms(net, Dt, idx)
    _, jt = jax_terms(jparams(net), D)
    for k in ("lpu", "lpq", "lpm", "pf", "lpu_all"):
        a, b = np.asarray(jt[k]), tt[k].numpy()
        m = np.isfinite(b) & (b > -1e8)
        e = rel(a[m], b[m])
        print(f"[logp_terms] {k:8s} max rel err {e:.2e}"); ok &= e < 1e-4
    ok &= bool((np.asarray(jt["valid"]) == tt["valid"].numpy()).all() and (np.asarray(jt["qvalid"]) == tt["qvalid"].numpy()).all())

    # 2. loss and gradients (ppo_live lines)
    a = torch.from_numpy(rng.randn(nb).astype(np.float32))
    with torch.no_grad():
        old = TP.logp_terms(old_net, Dt, idx); refp = TP.logp_terms(ref_net, Dt, idx)
    cur = TP.logp_terms(net, Dt, idx)
    ru = torch.exp(cur["lpu"] - old["lpu"]); vm = cur["valid"]
    l_u = -(torch.min(ru * a[:, None], torch.clamp(ru, 1 - CLIP, 1 + CLIP) * a[:, None])[vm]).mean()
    rq = torch.exp(cur["lpq"] - old["lpq"]); qm = cur["qvalid"]
    l_q = -(torch.min(rq * a[:, None], torch.clamp(rq, 1 - CLIP, 1 + CLIP) * a[:, None])[qm]).mean()
    rm = torch.exp(cur["lpm"] - old["lpm"])
    l_m = -torch.min(rm * a[:, None], torch.clamp(rm, 1 - CLIP, 1 + CLIP) * a[:, None]).mean()
    pu = torch.exp(cur["lpu_all"])
    klu = (pu * (cur["lpu_all"] - refp["lpu_all"])).masked_fill(~cur["umask"], 0).sum(-1)[vm].mean()
    pf, pr = cur["pf"], refp["pf"]
    klm = (pf * torch.log(pf / pr) + (1 - pf) * torch.log((1 - pf) / (1 - pr))).mean()
    loss_t = l_u + 0.5 * l_q + l_m + BETA * (klu + klm)
    net.zero_grad(); loss_t.backward()
    g_t = {k: (p.grad.numpy() if p.grad is not None else np.zeros(tuple(p.shape), np.float32)) for k, p in net.named_parameters()}

    _, jold = jax_terms(jparams(old_net), D); _, jref = jax_terms(jparams(ref_net), D)
    oldj = {k: jold[k] for k in ("lpu", "lpq", "lpm")}; refj = dict(rlpu_all=jref["lpu_all"], rpf=jref["pf"])
    dj = {k: jnp.asarray(v) for k, v in D.items()}; dj["valid"] = jnp.ones(nb, bool)

    def lossj(p):
        _, c = jax_terms(p, D)
        return JP.policy_loss(c, dj, jnp.asarray(a.numpy()), oldj, refj, CLIP, BETA)

    (lj, st), gj = jax.value_and_grad(lossj, has_aux=True)(jparams(net))
    print(f"[loss] torch {loss_t.item():+.7f} jax {float(lj):+.7f} | terms u {l_u.item():+.5f}/{float(st['l_u']):+.5f} q {l_q.item():+.5f}/{float(st['l_q']):+.5f} "
          f"m {l_m.item():+.5f}/{float(st['l_m']):+.5f} klu {klu.item():.5f}/{float(st['klu']):.5f} klm {klm.item():.5f}/{float(st['klm']):.5f}")
    ok &= abs(loss_t.item() - float(lj)) < 1e-5 * max(1.0, abs(loss_t.item()))
    gsd = N.to_state_dict(gj)
    worst = max((rel(gsd[k], g_t[k]), k) for k in g_t if np.abs(g_t[k]).max() > 0)
    zero_ok = all(np.abs(gsd[k]).max() == 0 for k in g_t if np.abs(g_t[k]).max() == 0)
    print(f"[grad] {len(g_t)} tensors, worst max rel err {worst[0]:.2e} ({worst[1]}); zero-grad tensors match: {zero_ok}")
    ok &= worst[0] < 1e-3 and zero_ok

    # 3. GAE
    lens = [5, 9, 3]; Lmax = 9; lam = 0.95
    rew = [rng.randn(n).astype(np.float32) for n in lens]; val = [rng.randn(n).astype(np.float32) for n in lens]
    ref_adv = []
    for r, v in zip(rew, val):
        L_ = len(r); adv = np.zeros(L_, np.float32); last = 0.0
        for t in range(L_ - 1, -1, -1):
            nv = v[t + 1] if t + 1 < L_ else 0.0
            last = r[t] + nv - v[t] + lam * last; adv[t] = last
        ref_adv.append(adv)
    R_ = np.zeros((Lmax, 3), np.float32); V_ = np.zeros((Lmax, 3), np.float32); Dn = np.zeros((Lmax, 3), bool); Vd = np.zeros((Lmax, 3), bool)
    for j, (r, v) in enumerate(zip(rew, val)):
        R_[:len(r), j] = r; V_[:len(r), j] = v; Dn[len(r) - 1, j] = True; Vd[:len(r), j] = True
    adv_j = np.asarray(JP.gae(jnp.asarray(R_), jnp.asarray(V_), jnp.asarray(rng.randn(3).astype(np.float32)), jnp.asarray(Dn), jnp.asarray(Vd), lam))
    e = max(np.abs(adv_j[:len(r), j] - ref_adv[j]).max() for j, r in enumerate(rew))
    print(f"[gae] max abs err {e:.2e}"); ok &= e < 1e-5
    # truncated segment: bootstrap
    r = rng.randn(6).astype(np.float32); v = rng.randn(6).astype(np.float32); vb = np.float32(0.7); last = 0.0; ref = np.zeros(6)
    for t in range(5, -1, -1):
        nv = v[t + 1] if t < 5 else vb
        last = r[t] + nv - v[t] + lam * last; ref[t] = last
    adv_b = np.asarray(JP.gae(jnp.asarray(r[:, None]), jnp.asarray(v[:, None]), jnp.asarray([vb]), jnp.zeros((6, 1), bool), jnp.ones((6, 1), bool), lam))[:, 0]
    e = np.abs(adv_b - ref).max(); print(f"[gae bootstrap] max abs err {e:.2e}"); ok &= e < 1e-5

    # 4. Adam + clip (3 steps on the gradients above, then on scaled copies)
    tn = tnet(1); opt = torch.optim.Adam(tn.parameters(), lr=1e-3)
    pj = jparams(tn); st_j = JP.adam_init(pj)
    for step in range(3):
        opt.zero_grad()
        for (k, p) in tn.named_parameters():
            p.grad = torch.from_numpy((g_t[k] * (step + 1) * 7.0).astype(np.float32))
        torch.nn.utils.clip_grad_norm_(tn.parameters(), 1.0); opt.step()
        gsd_t = dict({k: g_t[k] * (step + 1) * 7.0 for k in g_t}, coords=np.zeros((2, 10, 10), np.float32))
        gj_ = jax.tree_util.tree_map(jnp.asarray, N.convert_state_dict(gsd_t))        # the identical gradients
        gj_, _ = JP.clip_by_global_norm(JP.freeze_grads(gj_), 1.0)
        pj, st_j = JP.adam_step(pj, gj_, st_j, 1e-3)
    sd_t = {k: v.detach().numpy() for k, v in tn.state_dict().items()}; sd_j = N.to_state_dict(pj)
    e = max(np.abs(sd_j[k] - sd_t[k]).max() for k in sd_t)
    print(f"[adam+clip] 3 steps, max abs param diff {e:.2e}"); ok &= e < 1e-6

    # market log-prob accuracy (float32 formulas vs float64), on logits scaled up so that P(order) -> 0 / 1 occurs
    out_big, _ = jax_terms(jparams(tnet(3, 6.0)), D)
    acc = market_f64_check(out_big, D)
    print(f"[market lpm vs float64] stable formula (trainer default) max abs err {acc['stable']:.2e}; torch arithmetic {acc['torch']:.2e}")
    ok &= acc["stable"] <= acc["torch"] + 1e-6 and acc["stable"] < 1e-4

    # 5. decide: greedy path and sampling statistics
    out, _ = jax_terms(jparams(net), D)
    nu = jnp.asarray((D["uact"] >= 0).sum(1), jnp.int32)
    um = jnp.asarray(D["umask"])
    dg = S.decide(out, um, nu, jax.random.PRNGKey(0), T, KS, jnp.full(21, TAU), True)
    z = np.where(D["umask"], np.asarray(out["op"], np.float64), -1e9)
    ok_g = (np.asarray(dg.op)[D["uact"] >= 0] == z.argmax(-1)[D["uact"] >= 0]).all()
    mk = np.asarray(out["mk"], np.float64); fire_ref = np.zeros((nb, 21), bool); cls_ref = np.zeros((nb, 21), int)
    for h in range(21):
        zz = mk[:, A.HEAD_OFF[h]:A.HEAD_OFF[h + 1]]; p = np.exp(zz - zz.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        fire_ref[:, h] = 1 - p[:, 0] > TAU; cls_ref[:, h] = np.where(fire_ref[:, h], 1 + p[:, 1:].argmax(1), 0)
    ok_g &= (np.asarray(dg.fire) == fire_ref).all() and (np.asarray(dg.classes) == cls_ref).all()
    print(f"[decide greedy] unit argmax / market fire + class identical to live_rollout greedy: {bool(ok_g)}"); ok &= bool(ok_g)
    # sampled: 400 draws of one sample's decisions
    i0 = 0
    keys = jax.random.split(jax.random.PRNGKey(1), 400)
    one = jax.tree_util.tree_map(lambda x: jnp.repeat(x[i0:i0 + 1], 400, 0), out)
    ds = jax.vmap(lambda k, o: S.decide(jax.tree_util.tree_map(lambda x: x[None], o), um[i0:i0 + 1], nu[i0:i0 + 1], k, T, KS, jnp.full(21, TAU), False))(
        keys, jax.tree_util.tree_map(lambda x: x, one))
    f_emp = np.asarray(ds.fire)[:, 0].mean(0)
    pnz = np.clip(1 - np.array([np.exp(mk[i0, A.HEAD_OFF[h]] - np.log(np.exp(mk[i0, A.HEAD_OFF[h]:A.HEAD_OFF[h + 1]]).sum())) for h in range(21)]), 1e-6, 1 - 1e-6)
    from rollout import sharpen
    f_ref = sharpen(pnz, TAU, KS)
    e_f = np.abs(f_emp - f_ref).max()
    pz = np.exp((z[i0, 0] - z[i0, 0].max()) / T); pz /= pz.sum()
    op_emp = np.bincount(np.asarray(ds.op)[:, 0, 0], minlength=44) / 400
    e_o = np.abs(op_emp - pz).max()
    print(f"[decide sampled] fire rate vs sharpen(): max abs diff {e_f:.3f}; unit-op frequency vs softmax(z/T): max abs diff {e_o:.3f} (400 draws)")
    ok &= e_f < 0.1 and e_o < 0.1
    print("ALL OK" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
