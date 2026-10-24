"""Parity of act.py with the Kaggle numpy runtime (il/np_runtime.NumpyAgent, greedy, tau 0.5, team 10, rule layer
midnight_fix stripped) and with ppo.logp_terms.

A. market decode: decode_market on 200k random class vectors == features.decode_market (incl. > 10 orders)
B. games: NumpyAgent plays both seats of fresh engine games (on-policy states); plus leader-tape states (both seats,
   prev = the tape's previous ops).  At every recorded decision:
     B1 legal masks == agent.legal_mask for every unit
     B2 decode logic: JAX decide() on the NumpyAgent's own logits == its action dict, exactly
     B3 end to end: JAX encode + JAX forward + decide() == its action dict; mismatches are listed with the logit
        margin of the flipped decision (float noise if < 1e-4)
C. PPO: logp_terms (JAX) vs ppo.logp_terms (torch, same recorded batch, sampled actions): max abs diff

run: kagjax/policy/run1.sh -m kagjax.policy.test_act [ckpt] [n_games] [n_tape_games]
"""
import sys, os, time, json, types, gzip
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import jax
import jax.numpy as jnp
from kagjax.policy import consts as K, encode as E, net as N, act as A, refdata
import features as F
import features6 as F6
import np_runtime
from agent import legal_mask as ref_legal_mask
from harness import kagsim_trace, PASS

TEAM, TAU = 10, 0.5
CHUNK = 64


def features6_without_rules():
    m = types.ModuleType("features6_norules")
    m.__dict__.update({k: v for k, v in F6.__dict__.items() if k != "midnight_fix"})
    return m


class Recorder:
    """NumpyAgent whose net.forward outputs are kept (the logits of the decision just made)."""
    def __init__(self, P, meta):
        self.net = np_runtime.NumpyNet(P, meta["width"], meta["blocks"], meta["n_teams"])
        self.agent = np_runtime.NumpyAgent(self.net, features6_without_rules(), tau=TAU, team=TEAM)
        f = self.net.forward
        def fwd(*a, **k):
            self.last = f(*a, **k)
            return self.last
        self.net.forward = fwd

    def __call__(self, obs, prev=None, force_prev=False):
        me, step = int(obs["player"]), int(obs["step"])
        if force_prev:
            self.agent.state[me] = {"step": step - 1, "prev": prev} if step > 0 else None
            if self.agent.state[me] is None:
                del self.agent.state[me]
        st = self.agent.state.get(me)
        used = None if (st is None or step == 0 or step <= st["step"]) else st["prev"]
        a = self.agent(obs)
        ops = list(self.agent.state[me]["prev"])
        lop, lq, lm = self.last
        return a, used, ops, (lop, lq, lm)


def test_decode_market(n=200000):
    rng = np.random.RandomState(0)
    sizes = np.array(K.HEAD_SIZES)
    cls = (rng.rand(n, K.N_HEADS) * sizes).astype(np.int32)
    cls *= (rng.rand(n, K.N_HEADS) < rng.choice([0.1, 0.3, 0.7, 1.0], (n, 1)))      # sparse and dense
    f = jax.jit(jax.vmap(A.decode_market))
    o_op, o_item, o_n, n_ord = jax.device_get(f(jnp.asarray(cls)))
    bad = 0; over10 = 0
    for i in range(n):
        ref = F.decode_market([int(c) for c in cls[i]])
        a = A.Act(np.zeros(16, np.int32), np.zeros(16, np.int32), np.ones(16, np.int32), np.int32(0),
                  o_op[i], o_item[i], o_n[i], n_ord[i])
        got = A.act_to_dict(a)["market"]
        bad += got != ref
        over10 += int((cls[i, :19] > 0).sum() + cls[i, 19] + (cls[i, 20] > 0) > 10)   # candidates before truncation
    print(f"A. decode_market: {n - bad}/{n} random class vectors identical to features.decode_market "
          f"({over10} of them with > 10 candidate orders)", flush=True)
    assert bad == 0
    return dict(n=n, bad=int(bad), over10=int(over10))


def collect(rec, n_games, n_tape_games, seed0=4242):
    """-> list of records: dict(obs, prev, ref_act, ref_ops, lop, lq, lm)."""
    R = []
    t0 = time.time()
    for gi in range(n_games):                         # on-policy: the numpy agent plays both seats
        g = kagsim_trace.Game(seed0 + gi, 720, False)
        rec.agent.state = {}
        for t in range(720):
            acts = [None, None]
            for s in (0, 1):
                o = g.observe(s)
                a, used, ops, lg = rec(o)
                R.append(dict(obs=o, prev=used, ref_act=a, ref_ops=ops, lop=lg[0], lq=lg[1], lm=lg[2], src="self"))
                acts[s] = a
            g.step(acts[0], acts[1])
            if g.done:
                break
        print(f"   self-play game {gi + 1}/{n_games}: {len(R)} decisions so far, {time.time() - t0:.0f}s", flush=True)
    for f in refdata.tape_files(n_tape_games, start=5):    # leader tapes, prev = the tape's own previous ops
        for t, s, o, prev in refdata.iter_tape_obs(f, every=7):
            a, used, ops, lg = rec(o, prev, force_prev=True)
            R.append(dict(obs=o, prev=used, ref_act=a, ref_ops=ops, lop=lg[0], lq=lg[1], lm=lg[2], src="tape"))
    print(f"   collected {len(R)} decisions in {time.time() - t0:.0f}s", flush=True)
    return R


def pad_logits(r):
    n = len(r["ref_ops"])
    lop = np.zeros((16, K.N_OPS), np.float32); lq = np.zeros((16, K.N_QTY), np.float32)
    lop[:n] = r["lop"]; lq[:n] = r["lq"]
    return lop, lq, np.asarray(r["lm"], np.float32)


def margins(r, got_dict):
    """For a mismatch: smallest logit gap among the decisions that differ (units: top-2 legal op gap; market: |logit
    P(order) - logit tau| or the class top-2 gap)."""
    out = []
    lop, lq, lm = r["lop"], r["lq"], r["lm"]
    o = r["obs"]
    for k in range(len(r["ref_ops"])):
        allowed = ref_legal_mask(o, k, (int(([o["farms"][o["player"]]["farmer"]] + o["farms"][o["player"]]["hands"])[k][0]),
                                        int(([o["farms"][o["player"]]["farmer"]] + o["farms"][o["player"]]["hands"])[k][1])))
        z = np.sort(np.where(allowed, lop[k], -1e9))[::-1]
        out.append(("unit", k, float(z[0] - z[1])))
        zq = np.sort(lq[k])[::-1]
        out.append(("qty", k, float(zq[0] - zq[1])))
    for h in range(K.N_HEADS):
        z = lm[K.HEAD_OFF[h]:K.HEAD_OFF[h + 1]].astype(np.float64)
        p = np.exp(z - z.max()); p /= p.sum()
        out.append(("fire", h, float(abs((1 - p[0]) - TAU))))
        zz = np.sort(z[1:])[::-1]
        out.append(("class", h, float(zz[0] - zz[1]) if len(zz) > 1 else 1e9))
    return min(out, key=lambda x: x[2])


def run_parity(params, R, precision=N.HIGHEST):
    tabs = E.tables_jnp()
    cfg = A.Cfg(greedy=True, T=0.5, tau=TAU, ks=4.0)
    mask_f = jax.jit(jax.vmap(A.legal_mask))
    dec_f = jax.jit(jax.vmap(lambda r, o, k: A.decide(r, o, k, cfg)))
    step_f = jax.jit(lambda p, r, pr, k, t: A.policy_step(p, r, pr, k, t, cfg, tabs, precision))
    st = dict(n=len(R), mask_rows=0, mask_bad=0, decode_bad=0, e2e_bad=0, e2e_ops_bad=0, e2e_mismatch_margins=[],
              units=0, qty_decisions=0, market_orders=0, fires=0)
    key = jax.random.PRNGKey(0)
    for c0 in range(0, len(R), CHUNK):
        rr = R[c0:c0 + CHUNK]; n = len(rr)
        rr_p = rr + [rr[-1]] * (CHUNK - n)
        raw = E.stack([E.raw_from_obs(r["obs"]) for r in rr_p])
        prev = E.stack([E.prev_array(r["prev"]) for r in rr_p])
        masks = np.asarray(mask_f(raw))
        lg = [pad_logits(r) for r in rr_p]
        outs = {"op": jnp.asarray(np.stack([x[0] for x in lg])), "qty": jnp.asarray(np.stack([x[1] for x in lg])),
                "mk": jnp.asarray(np.stack([x[2] for x in lg])), "v": jnp.zeros(CHUNK)}
        dec = jax.device_get(dec_f(raw, outs, jax.random.split(key, CHUNK)))
        e2e = jax.device_get(step_f(params, raw, prev, key, jnp.full(CHUNK, TEAM, jnp.int32)))
        for j in range(n):
            r = rr[j]; o = r["obs"]; me = int(o["player"])
            units = [o["farms"][me]["farmer"]] + list(o["farms"][me]["hands"])
            nu = min(16, len(units))
            for k in range(nu):
                ref_m = ref_legal_mask(o, k, (int(units[k][0]), int(units[k][1])))
                st["mask_rows"] += 1
                st["mask_bad"] += int((ref_m != masks[j, k]).any())
            ref = r["ref_act"]
            a_dec = A.act_to_dict(jax.tree_util.tree_map(lambda x: x[j], dec["act"]))
            a_e2e = A.act_to_dict(jax.tree_util.tree_map(lambda x: x[j], e2e["act"]))
            st["units"] += nu
            st["qty_decisions"] += int((dec["qact"][j] >= 0).sum())
            st["market_orders"] += len(ref["market"]); st["fires"] += int(dec["fire"][j].sum())
            if a_dec != ref:
                st["decode_bad"] += 1
                if st["decode_bad"] <= 3:
                    print("   DECODE MISMATCH", o["step"], ref, a_dec, flush=True)
            if list(e2e["op"][j][:nu]) != list(r["ref_ops"]):
                st["e2e_ops_bad"] += 1
            if a_e2e != ref:
                st["e2e_bad"] += 1
                m = margins(r, a_e2e)
                st["e2e_mismatch_margins"].append(m)
                if st["e2e_bad"] <= 5:
                    print(f"   e2e mismatch step {o['step']} seat {me}: smallest margin {m}", flush=True)
    return st


def test_ppo_terms(ckpt, params, R, n=128):
    """ppo.logp_terms (torch) vs act.logp_terms (JAX) on a batch of recorded states with random sampled actions."""
    os.environ.update(DEVICE="cpu", TEAM=str(TEAM), T="0.5", KSHARP="4", TAU=str(TAU))
    import torch
    torch.set_num_threads(1)
    import ppo
    net, c = ppo.load(ckpt); net.eval()
    rng = np.random.RandomState(3)
    sel = [R[i] for i in rng.choice(len(R), n, replace=False)]
    raw = E.stack([E.raw_from_obs(r["obs"]) for r in sel]); prev = E.stack([E.prev_array(r["prev"]) for r in sel])
    tabs = E.tables_jnp()
    enc = jax.device_get(jax.jit(jax.vmap(E.encode, in_axes=(0, 0, None)))(raw, prev, tabs))
    umask = np.asarray(jax.jit(jax.vmap(A.legal_mask))(raw))
    valid = enc["valid"]
    uact = np.full((n, 16), -1, np.int64); qact = np.full((n, 16), -1, np.int64)
    for i in range(n):
        for k in range(16):
            if valid[i, k]:
                uact[i, k] = rng.choice(np.nonzero(umask[i, k])[0])
                if uact[i, k] >= K.OP_PICKUP0:
                    qact[i, k] = rng.randint(0, 12)
    mfire = (rng.rand(n, K.N_HEADS) < 0.3).astype(np.int8)
    D = {"planes": torch.from_numpy(enc["planes"]), "glob": torch.from_numpy(enc["glob"]),
         "pos": torch.from_numpy(enc["pos"].astype(np.int8)), "uin": torch.from_numpy(enc["uf"].astype(np.float16)),
         "umask": torch.from_numpy(umask), "uact": torch.from_numpy(uact.astype(np.int16)),
         "qact": torch.from_numpy(qact.astype(np.int16)), "mfire": torch.from_numpy(mfire)}
    torch.backends.nnpack.set_flags(False)
    with torch.no_grad():
        ref = ppo.logp_terms(net, D, torch.arange(n))
    ref = {k: v.numpy() for k, v in ref.items()}
    cfg = A.Cfg(greedy=False, T=0.5, tau=TAU, ks=4.0)
    got = jax.device_get(jax.jit(lambda p: A.logp_terms(p, jnp.asarray(enc["planes"]), jnp.asarray(enc["glob"]),
                                                         jnp.asarray(enc["pos"]), jnp.asarray(enc["uf"].astype(np.float16).astype(np.float32)),
                                                         jnp.full(n, TEAM, jnp.int32), jnp.asarray(umask), jnp.asarray(uact.astype(np.int32)),
                                                         jnp.asarray(qact.astype(np.int32)), jnp.asarray(mfire), cfg))(params))
    # float64 reference of the market term from the torch logits (ppo.py's formula without float32 cancellation)
    with torch.no_grad():
        lm64 = forward_mk = ppo.forward(net, D["planes"].float() / 255.0, D["glob"], D["pos"].long(), D["uin"].float())[2].double().numpy()
    lpm64 = np.zeros((n, K.N_HEADS))
    for h in range(K.N_HEADS):
        z = lm64[:, K.HEAD_OFF[h]:K.HEAD_OFF[h + 1]]
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        pnz = np.clip(1 - p[:, 0], 1e-6, 1 - 1e-6)
        pf = np.clip(1 / (1 + np.exp(-4.0 * (np.log(pnz / (1 - pnz)) - np.log(TAU / (1 - TAU))))), 1e-6, 1 - 1e-6)
        lpm64[:, h] = mfire[:, h] * np.log(pf) + (1 - mfire[:, h]) * np.log(1 - pf)
    d = {}
    d["lpm_jax_vs_f64"] = float(np.abs(got["lpm"] - lpm64).max())
    d["lpm_torch_vs_f64"] = float(np.abs(ref["lpm"] - lpm64).max())
    vm = ref["valid"]; qm = ref["qvalid"]
    d["lpu"] = float(np.abs(ref["lpu"][vm] - got["lpu"][vm]).max())
    d["lpq"] = float(np.abs(ref["lpq"][qm] - got["lpq"][qm]).max()) if qm.any() else 0.0
    d["lpm"] = float(np.abs(ref["lpm"] - got["lpm"]).max())
    d["pf"] = float(np.abs(ref["pf"] - got["pf"]).max())
    d["lpu_all(legal)"] = float(np.abs(np.where(ref["umask"], ref["lpu_all"] - got["lpu_all"], 0)).max())
    d["v"] = float(np.abs(ref["v"] - got["v"]).max())
    print("C. PPO logp_terms JAX vs torch (max abs diff): " + ", ".join(f"{k} {v:.2e}" for k, v in d.items()), flush=True)
    return d


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/Users/asad.ali/kaggriculture/il/ckpt/fr_t_dsm_s0.pt"
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_tape = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    res = {"ckpt": ckpt}
    res["A_decode_market"] = test_decode_market()
    import torch
    torch.set_num_threads(1)
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    P = {k: v.numpy() for k, v in c["state"].items()}
    meta = {"width": c["width"], "blocks": c.get("blocks", 5), "n_teams": c.get("n_teams", 0)}
    del c
    params, _ = N.load_ckpt(ckpt)
    rec = Recorder(P, meta)
    print(f"B. {os.path.basename(ckpt)}: collecting NumpyAgent decisions ({n_games} self-play games, {n_tape} tape games)", flush=True)
    R = collect(rec, n_games, n_tape)
    st = run_parity(params, R)
    mm = st.pop("e2e_mismatch_margins")
    st["e2e_mismatch_max_margin"] = max((m[2] for m in mm), default=0.0)
    st["e2e_mismatch_kinds"] = sorted({m[0] for m in mm})
    print(f"B1 legal masks: {st['mask_rows'] - st['mask_bad']}/{st['mask_rows']} unit rows identical", flush=True)
    print(f"B2 decode on identical logits: {st['n'] - st['decode_bad']}/{st['n']} decisions identical "
          f"({st['units']} unit commands, {st['qty_decisions']} quantities, {st['market_orders']} market orders)", flush=True)
    print(f"B3 end to end (JAX encode+net+decide): {st['n'] - st['e2e_bad']}/{st['n']} action dicts identical, "
          f"op lists differ in {st['e2e_ops_bad']}; mismatches all near-ties: max margin {st['e2e_mismatch_max_margin']:.2e} "
          f"{st['e2e_mismatch_kinds']}", flush=True)
    res["B"] = st
    res["C_ppo"] = test_ppo_terms(ckpt, params, R)
    out = os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))), f"results_act_{os.path.basename(ckpt).replace('.pt', '')}.json")
    json.dump(res, open(out, "w"), indent=1, default=str)
    assert st["mask_bad"] == 0 and st["decode_bad"] == 0
    print("ACT PARITY DONE ->", out)
