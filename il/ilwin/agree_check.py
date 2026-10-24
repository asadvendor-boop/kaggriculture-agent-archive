"""Drift gauge: action agreement between a fine-tuned checkpoint and its base (v13_p3) on held-out states, per head, with the
decisions the live agent actually takes (greedy op argmax, quantity argmax, market head fires when P(order) > TAU, class =
most likely non-zero), both nets conditioned on team 10 (what we play with).

State sets (--set name=dir[:dir]): e.g. win=<winner val seats>, reh=<rehearsal val seats (leaders)>. Only ep % 37 == 0 seat
files are used by default (the trainers' validation split; --all-files to lift that). Slices from the global features:
  yarn (>=1 YARN_STORE), milk (>=2 of SMOOTHIE/ICE_CREAM/PIZZA), eggs (>=2 of BAKERY/BRUNCH), melon (days 8-15),
  straw (>=3 strawberry buyers: BRUNCH/FARMERS/ICE_CREAM/SMOOTHIE), crop (>=2 PIZZA/FARMERS)
and the heads that matter there (sheep, cows, geese, melon seeds, strawberry seeds; PLANT_* ops).
Target: overall agreement high (op >= ~97%, all-market-heads >= ~90%), changes concentrated in the weakness slices.
Usage: agree_check.py <new.pt> [--base ckpt] --set win=DIR --set reh=DIR1:DIR2 [--x6 DIR:DIR] [--n 20000] [--out f.json]
"""
import sys, os, glob, json, argparse, collections
import numpy as np
import torch
import torch.nn.functional as Fn
K = os.environ.get("KROOT", "/Users/asad.ali/kaggriculture")
sys.path.insert(0, f"{K}/il")
import features as F
import features6 as F6
from model import Net, split_heads

HN = [h for h, _ in F.MARKET_HEADS]; HI = {h: i for i, h in enumerate(HN)}
SH = {s: 66 + i for i, s in enumerate(F.SHOPS)}                 # glob index of each shop count (/2)


def load_net(p, dev):
    c = torch.load(p, map_location="cpu", weights_only=False)
    n = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"), n_aux=c.get("n_aux", 0),
            ar=c.get("ar", 0), menu=c.get("menu", 0))
    n.load_state_dict(c["state"]); n.eval()
    return n.to(dev)


def slices(glob98):
    g = glob98; sc = lambda s: np.rint(g[:, SH[s]] * 2)
    day = np.rint(g[:, 1] * 29)
    return {"all": np.ones(len(g), bool),
            "yarn": sc("YARN_STORE") >= 1,
            "milk": (sc("SMOOTHIE_SHOP") + sc("ICE_CREAM_SHOP") + sc("PIZZA_SHOP")) >= 2,
            "eggs": (sc("BAKERY") + sc("BRUNCH_SPOT")) >= 2,
            "melon": (day >= 8) & (day <= 15),
            "straw": (sc("BRUNCH_SPOT") + sc("FARMERS_MARKET") + sc("ICE_CREAM_SHOP") + sc("SMOOTHIE_SHOP")) >= 3,
            "crop": (sc("PIZZA_SHOP") + sc("FARMERS_MARKET")) >= 2}


FOCUS = {"yarn": ["BUY_ANIMAL:SHEEP", "SELL:WOOL"], "milk": ["BUY_ANIMAL:COW", "SELL:MILK"], "eggs": ["BUY_ANIMAL:GOOSE", "SELL:EGG"],
         "melon": ["BUY_SEED:MELON", "SELL:MELON"], "straw": ["BUY_SEED:STRAWBERRY", "SELL:STRAWBERRY"],
         "crop": ["BUY_SEED:TOMATO", "BUY_LAND"], "all": ["BUY_LAND", "HIRE", "BUY_PRODUCT:WHEAT"]}
PLANTS = [F.OP_INDEX["PLANT_" + c] for c in F.CROPS]


def sample_states(files, x6idx, n, rng):
    """Uniformly sampled (file, step) states -> input tensors + labels + glob (numpy)."""
    per = max(1, int(np.ceil(n / max(1, len(files)))))
    rows = []
    for f in files:
        try:
            z = np.load(f); x6 = np.load(x6idx[os.path.basename(f)[:-4]])
        except Exception:
            continue
        T = z["op"].shape[0]
        if x6.shape[0] != T:
            continue
        ts = rng.choice(T, min(T, per), replace=False)
        rows.append(dict(planes=z["planes"][ts], glob=np.concatenate([z["glob"][ts].astype(np.float32), x6[ts].astype(np.float32)], 1),
                         pos=z["pos"][ts], ufeat=z["ufeat"][ts], prev_op=z["prev_op"][ts], op=z["op"][ts], qty=z["qty"][ts],
                         market=z["market"][ts]))
    D = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    if len(D["op"]) > n:
        sel = rng.choice(len(D["op"]), n, replace=False); D = {k: v[sel] for k, v in D.items()}
    return D


@torch.no_grad()
def run(net, D, dev, team, tau, bs=512):
    outs = collections.defaultdict(list)
    for b in range(0, len(D["op"]), bs):
        sl = slice(b, b + bs)
        planes = torch.from_numpy(D["planes"][sl]).to(dev).float() / 255.0
        prev = torch.from_numpy(D["prev_op"][sl].astype(np.int64)).to(dev)
        oh = Fn.one_hot(prev.clamp(min=0), F.N_OPS).float() * (prev >= 0).float()[..., None]
        uin = torch.cat([torch.from_numpy(D["ufeat"][sl].astype(np.float32)).to(dev), oh], 2)
        tt = torch.full((planes.shape[0],), team, dtype=torch.long, device=dev)
        lop, lq, lm, v = net(planes, torch.from_numpy(D["glob"][sl]).to(dev), torch.from_numpy(D["pos"][sl].astype(np.int64)).to(dev), uin, tt)
        outs["op"].append(lop.argmax(-1).cpu().numpy()); outs["qty"].append(lq.argmax(-1).cpu().numpy())
        cls = []
        for h in split_heads(lm):
            p = torch.softmax(h.float(), -1)
            fire = (1.0 - p[:, 0]) > tau
            cls.append(torch.where(fire, 1 + p[:, 1:].argmax(-1), torch.zeros_like(fire, dtype=torch.long)))
        outs["mk"].append(torch.stack(cls, 1).cpu().numpy()); outs["v"].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("--base", default=f"{K}/il/ckpt/ilsp_v13_p3.pt")
    ap.add_argument("--set", action="append", default=[]); ap.add_argument("--x6", default="")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--team", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--all-files", action="store_true")
    ap.add_argument("--max-files", type=int, default=400); ap.add_argument("--out", default="")
    ap.add_argument("--device", default=os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    dev = torch.device(a.device)
    x6dirs = [d for d in a.x6.split(":") if d] + [f"{K}/il/data/x6"]
    base_net = load_net(a.base, dev); new_net = load_net(a.new, dev)
    rep = {"new": a.new, "base": a.base, "tau": a.tau, "team": a.team, "sets": {}}
    for spec in a.set:
        name, dirs = spec.split("=", 1)
        dirs = [d for d in dirs.split(":") if d]
        x6idx = {}
        for d in reversed(x6dirs + [os.path.join(os.path.dirname(os.path.abspath(x.rstrip("/"))), "x6") for x in dirs]):
            for f in glob.glob(f"{d}/*.npy"):
                x6idx[os.path.basename(f)[:-4]] = f
        files = sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz") if os.path.basename(f)[:-4] in x6idx
                       and (a.all_files or int(os.path.basename(f).split("_")[0]) % 37 == 0))
        rng = np.random.RandomState(0)
        if len(files) > a.max_files:
            files = [files[i] for i in sorted(rng.choice(len(files), a.max_files, replace=False))]
        if not files:
            print(f"[{name}] no files"); continue
        D = sample_states(files, x6idx, a.n, rng)
        B = run(base_net, D, dev, a.team, a.tau); N = run(new_net, D, dev, a.team, a.tau)
        valid = D["ufeat"][:, :, :F.MAX_UNITS].astype(np.float32).sum(2) > 0.5        # index one-hot = a real unit
        SL = slices(D["glob"][:, :F.G])
        res = {"files": len(files), "states": int(len(D["op"]))}
        for sn, m in SL.items():
            if m.sum() == 0:
                continue
            um = valid & m[:, None]
            r = {"n": int(m.sum()), "op_agree": float((B["op"][um] == N["op"][um]).mean()) if um.any() else None}
            pq = um & np.isin(B["op"], [F.OP_INDEX[o] for o in F.OPS if o.startswith(("PICKUP_", "PLACE_"))]) & (B["op"] == N["op"])
            r["qty_agree"] = float((B["qty"][pq] == N["qty"][pq]).mean()) if pq.any() else None
            r["market_all_agree"] = float((B["mk"][m] == N["mk"][m]).all(1).mean())
            r["value_mad"] = float(np.abs(B["v"][m] - N["v"][m]).mean())
            lab_ok = um & (D["op"] >= 0)
            if lab_ok.any():
                r["op_acc_vs_labels"] = [float((B["op"][lab_ok] == D["op"][lab_ok]).mean()), float((N["op"][lab_ok] == D["op"][lab_ok]).mean())]
            nz = (D["market"][m] > 0)
            if nz.any():
                r["mk_nz_recall_vs_labels"] = [float((B["mk"][m][nz] == D["market"][m][nz]).mean()), float((N["mk"][m][nz] == D["market"][m][nz]).mean())]
            r["heads"] = {}
            for h in FOCUS.get(sn, []):
                j = HI[h]
                r["heads"][h] = dict(agree=float((B["mk"][m, j] == N["mk"][m, j]).mean()),
                                     fire_base=float((B["mk"][m, j] > 0).mean()), fire_new=float((N["mk"][m, j] > 0).mean()),
                                     fire_label=float((D["market"][m, j] > 0).mean()))
            r["plant_share"] = {F.OPS[p]: [float((B["op"][um] == p).mean()), float((N["op"][um] == p).mean())] for p in PLANTS} if um.any() else {}
            res[sn] = r
        worst = sorted(((float((B["mk"][:, j] == N["mk"][:, j]).mean()), HN[j]) for j in range(len(HN))))[:6]
        res["least_agreeing_heads"] = worst
        rep["sets"][name] = res
        a_ = res["all"]
        print(f"[{name}] {res['states']} states from {res['files']} files | op agree {a_['op_agree']:.4f} | qty {a_['qty_agree'] if a_['qty_agree'] is None else round(a_['qty_agree'], 4)}"
              f" | all market heads {a_['market_all_agree']:.4f} | value MAD {a_['value_mad']:.3f}"
              + (f" | op acc vs labels {a_['op_acc_vs_labels'][0]:.4f} -> {a_['op_acc_vs_labels'][1]:.4f}" if 'op_acc_vs_labels' in a_ else "")
              + (f" | market nz-recall {a_['mk_nz_recall_vs_labels'][0]:.3f} -> {a_['mk_nz_recall_vs_labels'][1]:.3f}" if 'mk_nz_recall_vs_labels' in a_ else ""))
        for sn in ("yarn", "milk", "eggs", "melon", "straw", "crop"):
            if sn not in res:
                continue
            r = res[sn]
            hs = "; ".join(f"{h} agree {v['agree']:.3f} fire {v['fire_base']:.3f}->{v['fire_new']:.3f} (label {v['fire_label']:.3f})" for h, v in r["heads"].items())
            print(f"   {sn:5s} n {r['n']:6d} op {r['op_agree'] if r['op_agree'] is None else round(r['op_agree'], 4)} mk-all {r['market_all_agree']:.4f} | {hs}")
        print("   least-agreeing heads:", ", ".join(f"{h} {v:.3f}" for v, h in worst))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
