"""v7 = train6 + the order-by-order market head (features_ar; labels il/data/mseq/<ep>_<seat>.npy), trained with teacher
forcing (AR_W, non-STOP slots weighted AR_NZ_W).  INIT from any v6 checkpoint (the ar_* weights start fresh).
v6 = train2 + "if nobody acts" future features (features6: 81 extra global inputs read from il/data/x6/<ep>_<seat>.npy)
+ auxiliary future-prediction heads (money margin / own money change 24, 72, 168 turns ahead; AUX_W) + net surgery
from a v5 checkpoint (INIT): the new input columns start at zero, so training starts from v5's exact behaviour.

Multi-leader behaviour cloning with a team-style embedding, streamed from disk in chunks (Apple GPU / MPS).

Usage: train2.py <ckpt_out> <epochs> <width> <data_dir> [<data_dir> ...]
Env: MK_NZ_W (weight of non-zero market targets, default 3), LOSS_W (weight of steps from games the leader lost,
default 0.7), CHUNK (files per chunk, default 200), INIT (checkpoint to start from), MAX_LR (default 2e-3).
Validation: games with ep % 37 == 0 (fixed ~3%), loaded once.
"""
import sys, os, glob, time
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import features as F
import features6 as F6
from model import Net, split_heads

torch.manual_seed(0)
dev = torch.device("mps")
out, EPOCHS, WIDTH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dirs = sys.argv[4:]
MK_NZ_W = float(os.environ.get("MK_NZ_W", "3")); LOSS_W = float(os.environ.get("LOSS_W", "0.7"))
CHUNK = int(os.environ.get("CHUNK", "200")); INIT = os.environ.get("INIT"); MAX_LR = float(os.environ.get("MAX_LR", "2e-3"))
N_TEAMS = int(os.environ.get("N_TEAMS", "11"))
AUX_W = float(os.environ.get("AUX_W", "0.1")); AUX_H = (24, 72, 168); N_AUX = 2 * len(AUX_H)
X6 = "/Users/asad.ali/kaggriculture/il/data/x6"
MS = "/Users/asad.ali/kaggriculture/il/data/mseq"
AR_W = float(os.environ.get("AR_W", "1.0")); AR_NZ_W = float(os.environ.get("AR_NZ_W", "3.0"))
FREEZE = int(os.environ.get("FREEZE", "0"))       # 1: train ONLY the order-by-order head (ar_*); the INIT network stays exact
BS = 512

files = sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz") if os.path.exists(f"{X6}/{os.path.basename(f)[:-4]}.npy")
               and os.path.exists(f"{MS}/{os.path.basename(f)[:-4]}.npy"))
ep_of = lambda f: int(os.path.basename(f).split("_")[0])
val_files = [f for f in files if ep_of(f) % 37 == 0]
tr_files = [f for f in files if ep_of(f) % 37 != 0]
VAL_MAX = int(os.environ.get("VAL_MAX", "0"))          # cap the (GPU-resident) validation set: memory pressure
if VAL_MAX and len(val_files) > VAL_MAX:
    val_files = val_files[::-(-len(val_files) // VAL_MAX)]
print(f"files: train {len(tr_files)}, val {len(val_files)}", flush=True)


def load(fs):
    parts = {k: [] for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market", "team", "w", "val", "aux", "mseq")}
    for f in fs:
        try:
            z = np.load(f)
            T = z["op"].shape[0]; m = z["meta"]
            x6 = np.load(f"{X6}/{os.path.basename(f)[:-4]}.npy"); ms = np.load(f"{MS}/{os.path.basename(f)[:-4]}.npy")
            if x6.shape[0] != T or ms.shape[0] != T:
                continue
            parts["mseq"].append(ms)
            for k in ("planes", "pos", "ufeat", "prev_op", "op", "qty", "market"):
                parts[k].append(z[k])
            parts["glob"].append(np.concatenate([z["glob"].astype(np.float32), x6.astype(np.float32)], 1))
            mo = z["money"].astype(np.float64); diff = (mo[:, 0] - mo[:, 1]) / 2e4; own = mo[:, 0] / 1e4
            fd, fo = (float(m[2]) - float(m[3])) / 2e4, float(m[2]) / 1e4        # final banks (after the last turn)
            ax = np.zeros((T, N_AUX), np.float32)
            for j, h in enumerate(AUX_H):
                idx = np.arange(T) + h
                ok = idx < T
                ax[:, j] = np.where(ok, diff[np.minimum(idx, T - 1)], fd) - diff
                ax[:, len(AUX_H) + j] = np.where(ok, own[np.minimum(idx, T - 1)], fo) - own
            parts["aux"].append(np.clip(ax, -5, 5))
        except Exception:
            continue
        team = int(m[6]) if len(m) > 6 else 10
        win = int(m[7]) if len(m) > 7 else int(m[2] > m[3])
        parts["team"].append(np.full(T, team, np.int64)); parts["w"].append(np.full(T, 1.0 if win else LOSS_W, np.float32))
        parts["val"].append(np.full(T, np.clip((m[2] - m[3]) / 2e4, -3, 3), np.float32))
    D = {k: np.concatenate(v) for k, v in parts.items()}
    return {"planes": torch.from_numpy(D["planes"]).to(dev), "glob": torch.from_numpy(D["glob"].astype(np.float32)).to(dev),
            "pos": torch.from_numpy(D["pos"].astype(np.int64)).to(dev), "ufeat": torch.from_numpy(D["ufeat"].astype(np.float32)).to(dev),
            "prev_op": torch.from_numpy(D["prev_op"].astype(np.int64)).to(dev), "op": torch.from_numpy(D["op"].astype(np.int64)).to(dev),
            "qty": torch.from_numpy(D["qty"].astype(np.int64)).to(dev), "market": torch.from_numpy(D["market"].astype(np.int64)).to(dev),
            "team": torch.from_numpy(D["team"]).to(dev), "w": torch.from_numpy(D["w"]).to(dev), "val": torch.from_numpy(D["val"]).to(dev),
            "aux": torch.from_numpy(D["aux"]).to(dev), "mseq": torch.from_numpy(D["mseq"].astype(np.int64)).to(dev)}


def losses(net, D, idx, stats=None):
    planes = D["planes"][idx].float() / 255.0
    prev = D["prev_op"][idx]
    oh = Fn.one_hot(prev.clamp(min=0), F.N_OPS).float() * (prev >= 0).float()[..., None]
    uin = torch.cat([D["ufeat"][idx], oh], 2)
    lop, lq, lm, v, ap, hm = net(planes, D["glob"][idx], D["pos"][idx], uin, D["team"][idx], ret_hm=True)
    op, qty, mk, w = D["op"][idx], D["qty"][idx], D["market"][idx], D["w"][idx]
    mask = op >= 0
    wu = w[:, None].expand_as(op)[mask]
    l_op = (Fn.cross_entropy(lop[mask], op[mask], reduction="none") * wu).sum() / wu.sum()
    qm = qty >= 0
    wq = w[:, None].expand_as(qty)[qm]
    l_q = (Fn.cross_entropy(lq[qm], qty[qm], reduction="none") * wq).sum() / wq.sum() if qm.any() else lop.sum() * 0
    heads = split_heads(lm)
    wz = torch.where(mk > 0, MK_NZ_W, 1.0) * w[:, None]
    l_m = sum((Fn.cross_entropy(h, mk[:, j], reduction="none") * wz[:, j]).sum() / wz[:, j].sum() for j, h in enumerate(heads)) / len(heads)
    l_v = Fn.mse_loss(v, D["val"][idx])
    l_a = Fn.mse_loss(ap, D["aux"][idx])
    lab = D["mseq"][idx]; aop, ait, aq = net.ar_tf(hm, lab)
    first_stop = torch.where((lab[:, :, 0] == 0).any(1), (lab[:, :, 0] == 0).float().argmax(1), torch.full_like(lab[:, 0, 0], lab.shape[1]))
    valid = torch.arange(lab.shape[1], device=lab.device)[None, :] <= first_stop[:, None]
    wo = torch.where(lab[:, :, 0] > 0, AR_NZ_W, 1.0) * w[:, None] * valid
    l_aro = (Fn.cross_entropy(aop.flatten(0, 1), lab[:, :, 0].flatten(), reduction="none") * wo.flatten()).sum() / wo.sum()
    has = ((lab[:, :, 0] >= 2) & (lab[:, :, 0] <= 5)).float() * w[:, None]
    l_ari = (Fn.cross_entropy(ait.flatten(0, 1), lab[:, :, 1].flatten(), reduction="none") * has.flatten()).sum() / has.sum().clamp(min=1)
    l_arq = (Fn.cross_entropy(aq.flatten(0, 1), lab[:, :, 2].flatten(), reduction="none") * has.flatten()).sum() / has.sum().clamp(min=1)
    l_ar = l_aro + 0.5 * l_ari + 0.5 * l_arq
    if stats is not None:
        nz = (lab[:, :, 0] > 0) & valid
        stats["ar_op_ok"] += ((aop.argmax(-1) == lab[:, :, 0]) & nz).sum().item(); stats["ar_op_n"] += nz.sum().item()
    if stats is not None:
        pred = lop.argmax(-1)
        stats["op_ok"] += (pred[mask] == op[mask]).sum().item(); stats["op_n"] += mask.sum().item()
        mp = torch.stack([h.argmax(-1) for h in heads], 1); nz = mk > 0
        stats["m_nz_ok"] += (mp[nz] == mk[nz]).sum().item(); stats["m_nz_n"] += nz.sum().item()
        stats["m_fp"] += ((mp > 0) & (mk == 0)).sum().item(); stats["m_z_n"] += (~nz).sum().item()
    return l_op + 0.5 * l_q + l_m + 0.1 * l_v + AUX_W * l_a + AR_W * l_ar, (l_op.item(), l_q.item(), l_m.item(), l_ar.item())


BLOCKS = int(os.environ.get("BLOCKS", "5"))
net = Net(width=WIDTH, blocks=BLOCKS, n_teams=N_TEAMS, g_in=F6.G, n_aux=N_AUX, ar=1).to(dev)
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
    missing = net.load_state_dict(st0, strict=False)
    print("init from", INIT, "missing/unexpected:", missing, flush=True)
if FREEZE:
    for n_, p_ in net.named_parameters():
        p_.requires_grad = n_.startswith("ar_")
print(f"params {sum(p.numel() for p in net.parameters()):,} (trainable {sum(p.numel() for p in net.parameters() if p.requires_grad):,})", flush=True)
VA = load(val_files)
steps_total = sum(1 for _ in tr_files) * 719 * EPOCHS // BS
opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=MAX_LR / 10, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=max(10, steps_total), pct_start=0.05)
best = 1e9; it = 0; t0 = time.time()
rng = np.random.RandomState(0)


def validate():
    net.eval(); st = {k: 0 for k in ("op_ok", "op_n", "m_nz_ok", "m_nz_n", "m_fp", "m_z_n", "ar_op_ok", "ar_op_n")}; vl = 0.0; nb = 0; va = 0.0
    with torch.no_grad():
        NV = VA["op"].shape[0]
        for b in range(0, NV, 2048):
            l, pr = losses(net, VA, torch.arange(b, min(NV, b + 2048), device=dev), st); vl += l.item(); va += pr[3]; nb += 1
    net.train()
    st["ar_loss"] = va / nb
    return (va / nb if FREEZE else vl / nb), st         # FREEZE: model selection on the order-head loss alone


for ep in range(EPOCHS):
    order = rng.permutation(len(tr_files))
    for c in range(0, len(order), CHUNK):
        D = load([tr_files[i] for i in order[c:c + CHUNK]])
        N = D["op"].shape[0]; perm = torch.randperm(N, device=dev); agg = np.zeros(4); nb = 0
        for b in range(0, N - BS + 1, BS):
            if it >= steps_total - 1:
                break
            loss, parts = losses(net, D, perm[b:b + BS])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1; agg += parts; nb += 1
        del D
        if (c // CHUNK) % 8 == 7 or c + CHUNK >= len(order):
            vl, st = validate()
            print(f"ep {ep+1} chunk {c//CHUNK+1}/{(len(order)+CHUNK-1)//CHUNK} it {it}/{steps_total} {time.time()-t0:.0f}s | train op/q/m/ar {agg[0]/max(1,nb):.3f}/{agg[1]/max(1,nb):.3f}/{agg[2]/max(1,nb):.3f}/{agg[3]/max(1,nb):.3f} "
                  f"| val loss {vl:.3f} op acc {st['op_ok']/st['op_n']:.4f} market nz-recall {st['m_nz_ok']/max(1,st['m_nz_n']):.3f} ar order-op acc {st['ar_op_ok']/max(1,st['ar_op_n']):.3f} ar loss {st['ar_loss']:.4f}", flush=True)
            if vl < best:
                best = vl
                torch.save({"state": net.state_dict(), "width": WIDTH, "blocks": BLOCKS, "n_teams": N_TEAMS, "g_in": F6.G, "n_aux": N_AUX, "feat": 6, "ar": 1, "val_loss": vl, "it": it,
                            "val_files": val_files}, out)
    torch.save({"state": net.state_dict(), "width": WIDTH, "blocks": BLOCKS, "n_teams": N_TEAMS, "g_in": F6.G, "n_aux": N_AUX, "feat": 6, "ar": 1, "it": it, "val_files": val_files}, out.replace(".pt", "_last.pt"))
print("best val loss", best, flush=True)
