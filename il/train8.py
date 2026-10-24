"""v8 = train6 with the extended fixed market menu (features8: 23 heads, 651 classes; labels il/data/m8/<ep>_<seat>.npy
built by build_m8.py).  INIT from an old-menu v6 checkpoint: the market head is carried over class by class (exact
quantities keep their rows, new quantities start from the nearest old row a little less likely, the two padding heads
start at "no padding"), so training starts from the v6 policy's behaviour.

Usage: train8.py <ckpt_out> <epochs> <width> <data_dir> [<data_dir> ...]   (env as train6.py; FREEZE=1 trains only m1, m2, mk)
"""
import sys, os, glob, time
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import features as F
import features6 as F6
import features8 as F8
from model import Net, split_heads

SEED = int(os.environ.get("SEED", "0"))          # training-run seed (default 0 = the original runs)
torch.manual_seed(SEED)
import contextlib
dev = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "mps"))   # Mac: mps (unchanged)
AMP = int(os.environ.get("AMP", "0")) and dev.type == "cuda"          # bf16 autocast on CUDA only
amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if AMP else contextlib.nullcontext
LOAD_THREADS = int(os.environ.get("LOAD_THREADS", "1")); PREFETCH = int(os.environ.get("PREFETCH", "0"))
out, EPOCHS, WIDTH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dirs = sys.argv[4:]
MK_NZ_W = float(os.environ.get("MK_NZ_W", "3")); LOSS_W = float(os.environ.get("LOSS_W", "0.7"))
CHUNK = int(os.environ.get("CHUNK", "200")); INIT = os.environ.get("INIT"); MAX_LR = float(os.environ.get("MAX_LR", "2e-3"))
N_TEAMS = int(os.environ.get("N_TEAMS", "11"))
AUX_W = float(os.environ.get("AUX_W", "0.1")); AUX_H = (24, 72, 168); N_AUX = 2 * len(AUX_H)
X6 = "/Users/asad.ali/kaggriculture/il/data/x6"
M8 = "/Users/asad.ali/kaggriculture/il/data/m8"
FREEZE = int(os.environ.get("FREEZE", "0"))
BS = 512

files = sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz") if os.path.exists(f"{X6}/{os.path.basename(f)[:-4]}.npy")
               and os.path.exists(f"{M8}/{os.path.basename(f)[:-4]}.npy"))
ep_of = lambda f: int(os.path.basename(f).split("_")[0])
val_files = [f for f in files if ep_of(f) % 37 == 0]
tr_files = [f for f in files if ep_of(f) % 37 != 0]
VAL_MAX = int(os.environ.get("VAL_MAX", "0"))          # cap the (GPU-resident) validation set: memory pressure
if VAL_MAX and len(val_files) > VAL_MAX:
    val_files = val_files[::-(-len(val_files) // VAL_MAX)]
print(f"files: train {len(tr_files)}, val {len(val_files)}", flush=True)


def _one(f):
    """One seat file -> its arrays (None if unusable), menu-8 market labels; files load independently (threads)."""
    out = {}
    try:
        z = np.load(f)
        T = z["op"].shape[0]; m = z["meta"]
        x6 = np.load(f"{X6}/{os.path.basename(f)[:-4]}.npy"); m8 = np.load(f"{M8}/{os.path.basename(f)[:-4]}.npy")
        if m8.shape[0] != T or x6.shape[0] != T:
            return None
        for k in ("planes", "pos", "ufeat", "prev_op", "op", "qty"):
            out[k] = z[k]
        out["market"] = m8                                                   # menu-8 market labels
        out["glob"] = np.concatenate([z["glob"].astype(np.float32), x6.astype(np.float32)], 1)
        mo = z["money"].astype(np.float64); diff = (mo[:, 0] - mo[:, 1]) / 2e4; own = mo[:, 0] / 1e4
        fd, fo = (float(m[2]) - float(m[3])) / 2e4, float(m[2]) / 1e4        # final banks (after the last turn)
        ax = np.zeros((T, N_AUX), np.float32)
        for j, h in enumerate(AUX_H):
            idx = np.arange(T) + h
            ok = idx < T
            ax[:, j] = np.where(ok, diff[np.minimum(idx, T - 1)], fd) - diff
            ax[:, len(AUX_H) + j] = np.where(ok, own[np.minimum(idx, T - 1)], fo) - own
        out["aux"] = np.clip(ax, -5, 5)
    except Exception:
        return None
    team = int(m[6]) if len(m) > 6 else 10
    win = int(m[7]) if len(m) > 7 else int(m[2] > m[3])
    out["team"] = np.full(T, team, np.int64); out["w"] = np.full(T, 1.0 if win else LOSS_W, np.float32)
    out["val"] = np.full(T, np.clip((m[2] - m[3]) / 2e4, -3, 3), np.float32)
    return out


def load(fs):
    if LOAD_THREADS > 1:                                                  # zlib decompression releases the GIL
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(LOAD_THREADS) as ex:
            res = list(ex.map(_one, fs))
    else:
        res = [_one(f) for f in fs]
    res = [r for r in res if r is not None]
    D = {k: np.concatenate([r[k] for r in res]) for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market", "team", "w", "val", "aux")}
    return {"planes": torch.from_numpy(D["planes"]).to(dev), "glob": torch.from_numpy(D["glob"].astype(np.float32)).to(dev),
            "pos": torch.from_numpy(D["pos"].astype(np.int64)).to(dev), "ufeat": torch.from_numpy(D["ufeat"].astype(np.float32)).to(dev),
            "prev_op": torch.from_numpy(D["prev_op"].astype(np.int64)).to(dev), "op": torch.from_numpy(D["op"].astype(np.int64)).to(dev),
            "qty": torch.from_numpy(D["qty"].astype(np.int64)).to(dev), "market": torch.from_numpy(D["market"].astype(np.int64)).to(dev),
            "team": torch.from_numpy(D["team"]).to(dev), "w": torch.from_numpy(D["w"]).to(dev), "val": torch.from_numpy(D["val"]).to(dev),
            "aux": torch.from_numpy(D["aux"]).to(dev)}


def losses(net, D, idx, stats=None):
    planes = D["planes"][idx].float() / 255.0
    prev = D["prev_op"][idx]
    oh = Fn.one_hot(prev.clamp(min=0), F.N_OPS).float() * (prev >= 0).float()[..., None]
    uin = torch.cat([D["ufeat"][idx], oh], 2)
    lop, lq, lm, v, ap = net(planes, D["glob"][idx], D["pos"][idx], uin, D["team"][idx], aux=True)
    op, qty, mk, w = D["op"][idx], D["qty"][idx], D["market"][idx], D["w"][idx]
    mask = op >= 0
    wu = w[:, None].expand_as(op)[mask]
    l_op = (Fn.cross_entropy(lop[mask], op[mask], reduction="none") * wu).sum() / wu.sum()
    qm = qty >= 0
    wq = w[:, None].expand_as(qty)[qm]
    l_q = (Fn.cross_entropy(lq[qm], qty[qm], reduction="none") * wq).sum() / wq.sum() if qm.any() else lop.sum() * 0
    heads = split_heads(lm, F8.HEAD_SIZES)
    wz = torch.where(mk > 0, MK_NZ_W, 1.0) * w[:, None]
    l_m = sum((Fn.cross_entropy(h, mk[:, j], reduction="none") * wz[:, j]).sum() / wz[:, j].sum() for j, h in enumerate(heads)) / len(heads)
    l_v = Fn.mse_loss(v, D["val"][idx])
    l_a = Fn.mse_loss(ap, D["aux"][idx])
    if stats is not None:
        pred = lop.argmax(-1)
        stats["op_ok"] += (pred[mask] == op[mask]).sum().item(); stats["op_n"] += mask.sum().item()
        mp = torch.stack([h.argmax(-1) for h in heads], 1); nz = mk > 0
        stats["m_nz_ok"] += (mp[nz] == mk[nz]).sum().item(); stats["m_nz_n"] += nz.sum().item()
        stats["m_fp"] += ((mp > 0) & (mk == 0)).sum().item(); stats["m_z_n"] += (~nz).sum().item()
    return l_op + 0.5 * l_q + l_m + 0.1 * l_v + AUX_W * l_a, (l_op.item(), l_q.item(), l_m.item(), l_a.item())


BLOCKS = int(os.environ.get("BLOCKS", "5"))
net = Net(width=WIDTH, blocks=BLOCKS, n_teams=N_TEAMS, g_in=F6.G, n_aux=N_AUX, menu=8).to(dev)
if INIT:
    c0 = torch.load(INIT, map_location="cpu", weights_only=False)
    st0 = dict(c0["state"])
    tw = st0.get("team.weight")
    if tw is not None and tw.shape[0] != N_TEAMS:
        w = net.team.weight.detach().clone()
        k = min(tw.shape[0], N_TEAMS)
        w[:k] = tw[:k]
        if N_TEAMS > k:
            w[k:] = tw.mean(0, keepdim=True) + 0.1 * torch.randn(N_TEAMS - k, tw.shape[1])
        st0["team.weight"] = w
        print(f"team embedding expanded {tw.shape[0]} -> {N_TEAMS}", flush=True)
    w1 = st0.get("g1.weight")
    if w1 is not None and w1.shape[1] != net.g1.weight.shape[1]:         # net surgery: v5 -> v6 global input
        nw = torch.zeros_like(net.g1.weight.detach().cpu())
        nw[:, :F.G] = w1[:, :F.G]                                          # the 98 v5 features
        nw[:, F6.G:] = w1[:, F.G:]                                         # team-style embedding columns
        st0["g1.weight"] = nw                                              # the 81 new inputs start at zero
        print(f"net surgery: g1 {tuple(w1.shape)} -> {tuple(nw.shape)} (new inputs zero-initialised)", flush=True)
    if c0.get("menu", 0) != 8 and "mk.weight" in st0:                  # old menu -> menu 8, class by class
        ow, ob = st0["mk.weight"], st0["mk.bias"]
        nw_ = torch.zeros(F8.N_CLASSES, ow.shape[1]); nb_ = torch.zeros(F8.N_CLASSES)
        old_off = np.cumsum([0] + [n for _, n in F.MARKET_HEADS])
        old_vals = {"SELL": list(range(17)) + [1000], "BUY_PRODUCT:WHEAT": list(range(17)) + [35]}
        for h, (name, vals) in enumerate(F8.MARKET_HEADS):
            o = F8.MARKET_OFF[h]
            if name not in F.HEAD_INDEX:                                   # padding heads: start at "no padding"
                nb_[o] = 3.0
                continue
            j = F.HEAD_INDEX[name]; n_old = F.MARKET_HEADS[j][1]
            ov = old_vals.get("SELL" if name.startswith("SELL:") else name, list(range(n_old)))
            for k, v in enumerate(vals):
                if v in ov:
                    src_ = old_off[j] + ov.index(v); pen = 0.0
                else:                                                      # nearest smaller old quantity, less likely
                    src_ = old_off[j] + max(i for i, x in enumerate(ov) if x <= v and x != 1000); pen = 2.0
                nw_[o + k] = ow[src_]; nb_[o + k] = ob[src_] - pen
        st0["mk.weight"], st0["mk.bias"] = nw_, nb_
        print(f"market head surgery: {tuple(ow.shape)} -> {tuple(nw_.shape)}", flush=True)
    missing = net.load_state_dict(st0, strict=False)
    print("init from", INIT, "missing/unexpected:", missing, flush=True)
if FREEZE:
    for n_, p_ in net.named_parameters():
        p_.requires_grad = n_.split(".")[0] in ("m1", "m2", "mk")
print(f"params {sum(p.numel() for p in net.parameters()):,} (trainable {sum(p.numel() for p in net.parameters() if p.requires_grad):,})", flush=True)
VA = load(val_files)
steps_total = sum(1 for _ in tr_files) * 719 * EPOCHS // BS
opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=MAX_LR / 10, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=max(10, steps_total), pct_start=0.05)
best = 1e9; it = 0; t0 = time.time()
rng = np.random.RandomState(SEED)


def validate():
    net.eval(); st = {k: 0 for k in ("op_ok", "op_n", "m_nz_ok", "m_nz_n", "m_fp", "m_z_n")}; vl = 0.0; nb = 0
    with torch.no_grad():
        NV = VA["op"].shape[0]
        for b in range(0, NV, 2048):
            with amp_ctx():
                l, _ = losses(net, VA, torch.arange(b, min(NV, b + 2048), device=dev), st)
            vl += l.item(); nb += 1
    net.train()
    return vl / nb, st


for ep in range(EPOCHS):
    order = rng.permutation(len(tr_files))
    if PREFETCH:                                                          # load the next chunk while training
        from concurrent.futures import ThreadPoolExecutor
        _pf = ThreadPoolExecutor(1); _fut = _pf.submit(load, [tr_files[i] for i in order[0:CHUNK]])
    for c in range(0, len(order), CHUNK):
        if PREFETCH:
            D = _fut.result()
            if c + CHUNK < len(order):
                _fut = _pf.submit(load, [tr_files[i] for i in order[c + CHUNK:c + 2 * CHUNK]])
        else:
            D = load([tr_files[i] for i in order[c:c + CHUNK]])
        N = D["op"].shape[0]; perm = torch.randperm(N, device=dev); agg = np.zeros(4); nb = 0
        for b in range(0, N - BS + 1, BS):
            if it >= steps_total - 1:
                break
            with amp_ctx():
                loss, parts = losses(net, D, perm[b:b + BS])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1; agg += parts; nb += 1
        del D
        if (c // CHUNK) % 8 == 7 or c + CHUNK >= len(order):
            vl, st = validate()
            print(f"ep {ep+1} chunk {c//CHUNK+1}/{(len(order)+CHUNK-1)//CHUNK} it {it}/{steps_total} {time.time()-t0:.0f}s | train op/q/m/aux {agg[0]/max(1,nb):.3f}/{agg[1]/max(1,nb):.3f}/{agg[2]/max(1,nb):.3f}/{agg[3]/max(1,nb):.3f} "
                  f"| val loss {vl:.3f} op acc {st['op_ok']/st['op_n']:.4f} market nz-recall {st['m_nz_ok']/max(1,st['m_nz_n']):.3f} fp {st['m_fp']/max(1,st['m_z_n']):.4f}", flush=True)
            if vl < best:
                best = vl
                torch.save({"state": net.state_dict(), "width": WIDTH, "blocks": BLOCKS, "n_teams": N_TEAMS, "g_in": F6.G, "n_aux": N_AUX, "feat": 6, "menu": 8, "val_loss": vl, "it": it,
                            "val_files": val_files}, out)
    torch.save({"state": net.state_dict(), "width": WIDTH, "blocks": BLOCKS, "n_teams": N_TEAMS, "g_in": F6.G, "n_aux": N_AUX, "feat": 6, "menu": 8, "it": it, "val_files": val_files}, out.replace(".pt", "_last.pt"))
print("best val loss", best, flush=True)
