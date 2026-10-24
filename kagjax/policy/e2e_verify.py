"""For every game where the JAX run and the reference diverged (e2e_compare.py output): replay the reference game in the
C++ engine up to the divergence step with a FRESH copy of the real package for the diverging seat (fed every
observation, so its internal previous-op state is rebuilt; its actions are asserted equal to the recorded ones),
capture that package's logits at the divergence, and report the margin of the decision that flipped:
  unit op   : gap between the two best legal op logits of that unit (numpy runtime), and the JAX logits' order
  quantity  : gap between the two best quantity logits
  market    : smallest |P(order) - tau| and smallest class top-2 gap over the heads
A divergence is float noise when that margin is below the logit agreement of the two backends (~1e-4).

usage: python -m kagjax.policy.e2e_verify --cmp cmp.json --ref ref.json.gz --lpkg main.py --lckpt ck [--opkg main.py
         --ockpt ck --oteam 0] [--lteam 10] --out verify.json
"""
import sys, json, gzip, argparse
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from harness import kagsim_trace, mk
import jax
import jax.numpy as jnp
from kagjax.policy import encode as E, net as N, act as A, consts as K
from agent import legal_mask as ref_legal_mask


def capture_agent(pkg):
    ag = mk(pkg)()
    G = ag.__globals__
    net, core = G["_NET"], G["_AGENT"]
    cap = {}
    f0 = net.forward

    def fwd(*a, **k):
        cap["out"] = f0(*a, **k)
        return cap["out"]
    net.forward = fwd
    return ag, core, cap


def margins(obs, lop, lq, lm, jd, rd, tau):
    me = int(obs["player"]); units = [obs["farms"][me]["farmer"]] + list(obs["farms"][me]["hands"])
    ju = [jd.get("farmer")] + list(jd.get("hands") or []); ru = [rd.get("farmer")] + list(rd.get("hands") or [])
    out = []
    for k in range(min(len(ju), len(ru), len(lop))):
        if ju[k] == ru[k]:
            continue
        allowed = ref_legal_mask(obs, k, (int(units[k][0]), int(units[k][1])))
        z = np.where(allowed, lop[k].astype(np.float64), -1e9)
        o = np.argsort(-z)
        if ju[k][0] != ru[k][0] or (len(ju[k]) > 1 and len(ru[k]) > 1 and ju[k][1] != ru[k][1]):
            out.append(dict(kind="unit_op", unit=k, jax=ju[k], ref=ru[k], top2=[K.OPS[o[0]], K.OPS[o[1]]],
                            gap=float(z[o[0]] - z[o[1]])))
        else:
            zq = np.sort(lq[k].astype(np.float64))[::-1]
            out.append(dict(kind="qty", unit=k, jax=ju[k], ref=ru[k], gap=float(zq[0] - zq[1])))
    if jd.get("market") != rd.get("market"):
        fm, cm = 1e9, 1e9
        for h in range(K.N_HEADS):
            z = lm[K.HEAD_OFF[h]:K.HEAD_OFF[h + 1]].astype(np.float64)
            p = np.exp(z - z.max()); p /= p.sum()
            fm = min(fm, abs((1 - p[0]) - tau))
            zz = np.sort(z[1:])[::-1]
            if len(zz) > 1 and 1 - p[0] > tau:
                cm = min(cm, float(zz[0] - zz[1]))
        out.append(dict(kind="market", jax=jd.get("market"), ref=rd.get("market"), min_fire_margin=fm, min_class_gap=cm,
                        gap=min(fm, cm)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--jax", required=True)
    ap.add_argument("--lpkg", required=True); ap.add_argument("--lckpt", required=True); ap.add_argument("--lteam", type=int, default=10)
    ap.add_argument("--opkg"); ap.add_argument("--ockpt"); ap.add_argument("--oteam", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cmp = json.load(open(a.cmp)); ref = json.load(gzip.open(a.ref, "rt"))["games"]
    z = np.load(a.jax); B = int(z["B"])
    params = {"L": N.load_ckpt(a.lckpt)[0]}
    if a.ockpt:
        params["O"] = N.load_ckpt(a.ockpt)[0]
    tabs = E.tables_jnp()
    res = []
    for d in cmp["vs_reference"]["diverged"]:
        b, T, S = d["game"], d["first_divergence"]["step"] if d["first_divergence"] else None, None
        if T is None:
            res.append(dict(game=b, note="actions identical, banks differ", d=d)); continue
        S = d["first_divergence"]["seat"]
        g = ref[str(b)]
        role = "L" if S == g["lseat"] else "O"
        pkg = a.lpkg if role == "L" else a.opkg
        ag, core, cap = capture_agent(pkg)
        game = kagsim_trace.Game(int(g["seed"]), 720, False)
        prev_used = None
        for t in range(T + 1):
            obs = game.observe(S)
            st = core.state.get(S)
            prev_used = None if (st is None or t == 0 or t <= st["step"]) else list(st["prev"])
            act = ag(obs)
            rec = json.loads(g["acts"][t])
            assert act == rec[S], (b, t)
            if t < T:
                game.step(rec[0], rec[1])
        lop, lq, lm = cap["out"]
        jd = A.act_to_dict(A.Act(*(z[f][T, S * B + b] for f in A.Act._fields)))
        m = margins(obs, lop, lq, lm, jd, rec[S], a.tau)
        # JAX logits on the same state (fp32, CPU) for the flipped units
        raw = E.stack([E.raw_from_obs(obs)]); prev = E.stack([E.prev_array(prev_used)])
        enc = jax.vmap(E.encode, in_axes=(0, 0, None))(jax.tree_util.tree_map(jnp.asarray, raw), jnp.asarray(prev), tabs)
        team = jnp.asarray([a.lteam if role == "L" else a.oteam], jnp.int32)
        out = jax.device_get(N.forward_jit(params[role], *E.net_inputs(enc, tabs), team))
        n = len(lop)
        for x in m:
            if x["kind"] == "unit_op":
                k = x["unit"]; i0, i1 = K.OP_INDEX[x["top2"][0]], K.OP_INDEX[x["top2"][1]]
                x["numpy_logits"] = [float(lop[k][i0]), float(lop[k][i1])]
                x["jax_logits"] = [float(out["op"][0, k, i0]), float(out["op"][0, k, i1])]
        res.append(dict(game=b, seed=g["seed"], step=T, seat=S, role=role, margins=m,
                        max_abs_logit_diff=float(max(np.abs(out["op"][0, :n] - lop).max(), np.abs(out["mk"][0] - lm).max())),
                        smallest_gap=min((x["gap"] for x in m), default=None)))
        print(f"game {b} step {T} seat {S} ({role}): " + "; ".join(
            f"{x['kind']} gap {x['gap']:.2e}" + (f" {x.get('top2')} numpy {x.get('numpy_logits')} jax {x.get('jax_logits')}" if x['kind'] == 'unit_op' else "")
            for x in m) + f" | backend logit diff {res[-1]['max_abs_logit_diff']:.1e}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    gaps = [r["smallest_gap"] for r in res if r.get("smallest_gap") is not None]
    print(f"{len(res)} divergences verified; largest flipped-decision margin {max(gaps) if gaps else None}", flush=True)


if __name__ == "__main__":
    main()
