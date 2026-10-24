"""Compare e2e_jax.py (npz) with e2e_ref.py (json.gz shards): per game the first step where an action differs
(seat, which unit / the market list) and whether the final banks are equal.  With --base (the fp32 JAX run on the same
seeds) a TF32 run is also compared with fp32: first divergence, shadow flips (acting TF32 decision != fp32 decision on
the same state), bank and margin differences.

usage: python -m kagjax.policy.e2e_compare --jax run.npz [--ref s0.json.gz s1.json.gz ...] [--base fp32.npz] --out res.json
"""
import sys, json, gzip, argparse
import numpy as np
from kagjax.policy import act as A


def load_jax(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}


def jdict(z, t, row):
    return A.act_to_dict(A.Act(*(z[f][t, row] for f in A.Act._fields)))


def diff_desc(x, y):
    """Short description of how two action dicts differ."""
    ux = [x.get("farmer")] + list(x.get("hands") or []); uy = [y.get("farmer")] + list(y.get("hands") or [])
    out = []
    for k in range(max(len(ux), len(uy))):
        a = ux[k] if k < len(ux) else None; b = uy[k] if k < len(uy) else None
        if a != b:
            out.append(f"unit {k}: {a} vs {b}")
    if x.get("market") != y.get("market"):
        out.append(f"market: {x.get('market')} vs {y.get('market')}")
    return out[:4]


def compare_ref(z, refs):
    B = int(z["B"]); games = {}
    for r in refs:
        games.update({int(k): v for k, v in r["games"].items()})
    res = []
    for b in sorted(games):
        g = games[b]
        first = None
        for t, js in enumerate(g["acts"]):
            ref = json.loads(js)
            for s in (0, 1):
                mine = jdict(z, t, s * B + b)
                if mine != ref[s]:
                    first = dict(step=t, seat=s, lseat=int(z["lseat"][b]), diff=diff_desc(mine, ref[s]))
                    break
            if first:
                break
        jb = [float(x) for x in z["banks"][b]]; rb = [float(x) for x in g["banks"]]
        res.append(dict(game=b, seed=g["seed"], identical=first is None and jb == rb, first_divergence=first,
                        jax_banks=jb, ref_banks=rb, ref_errors=len(g.get("errors", []))))
    return res


def compare_base(z, base):
    B = int(z["B"]); T = z["u_op"].shape[0]; res = []
    for b in range(B):
        first = None
        for t in range(T):
            for s in (0, 1):
                row = s * B + b
                if any((z[f][t, row] != base[f][t, row]).any() for f in A.Act._fields):
                    first = dict(step=t, seat=s)
                    break
            if first:
                break
        fl = z["flips"][:, [b, B + b]]
        dbank = (z["banks"][b] - base["banks"][b]).astype(float)
        m_t = float(z["banks"][b, 0] - z["banks"][b, 1]); m_b = float(base["banks"][b, 0] - base["banks"][b, 1])
        res.append(dict(game=b, first_divergence_vs_fp32=first, flipped_decisions=int(fl.sum()),
                        steps_with_flip=int(fl.any(1).sum()), dbank=dbank.tolist(), dmargin=m_t - m_b))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jax", required=True); ap.add_argument("--ref", nargs="*"); ap.add_argument("--base")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = load_jax(a.jax)
    meta = json.loads(str(z["meta"]))
    out = dict(jax=a.jax, meta=meta)
    if a.ref:
        refs = [json.load(gzip.open(p, "rt")) for p in a.ref]
        r = compare_ref(z, refs)
        n_id = sum(x["identical"] for x in r)
        out["vs_reference"] = dict(games=len(r), identical=n_id, diverged=[x for x in r if not x["identical"]], all=r)
        print(f"{a.jax}: {n_id}/{len(r)} games identical to the C++ engine + exported numpy packages "
              f"(every action of both seats, final banks)", flush=True)
        for x in r:
            if not x["identical"]:
                print("   DIVERGED", x["game"], x["seed"], x["first_divergence"], x["jax_banks"], x["ref_banks"], flush=True)
    if a.base:
        base = load_jax(a.base)
        r = compare_base(z, base)
        div = [x for x in r if x["first_divergence_vs_fp32"] is not None]
        dm = np.array([x["dmargin"] for x in r]); db = np.array([x["dbank"] for x in r])
        fl = np.array([x["flipped_decisions"] for x in r]); T = z["u_op"].shape[0]
        out["vs_fp32"] = dict(games=len(r), diverged=len(div), mean_first_divergence_step=float(np.mean([x["first_divergence_vs_fp32"]["step"] for x in div])) if div else None,
                              flipped_decisions_total=int(fl.sum()), decisions_total=int(2 * len(r) * T),
                              flip_rate=float(fl.sum() / (2 * len(r) * T)), mean_dmargin=float(dm.mean()), sd_dmargin=float(dm.std()),
                              mean_abs_dbank=float(np.abs(db).mean()), mean_dbank=float(db.mean()), all=r)
        v = out["vs_fp32"]
        print(f"{a.jax} vs fp32 {a.base}: {v['diverged']}/{v['games']} games diverge (mean first step "
              f"{v['mean_first_divergence_step']}); flipped decisions {v['flipped_decisions_total']}/{v['decisions_total']} "
              f"({100 * v['flip_rate']:.3f}%); margin diff {v['mean_dmargin']:+.0f} +- {v['sd_dmargin']:.0f}; "
              f"mean |bank diff| {v['mean_abs_dbank']:.0f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
