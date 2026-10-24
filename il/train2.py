"""Multi-leader behaviour cloning with a team-style embedding, streamed from disk in chunks (Apple GPU / MPS).

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
from model import Net, split_heads

torch.manual_seed(0)
dev = torch.device("mps")
out, EPOCHS, WIDTH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dirs = sys.argv[4:]
MK_NZ_W = float(os.environ.get("MK_NZ_W", "3")); LOSS_W = float(os.environ.get("LOSS_W", "0.7"))
CHUNK = int(os.environ.get("CHUNK", "200")); INIT = os.environ.get("INIT"); MAX_LR = float(os.environ.get("MAX_LR", "2e-3"))
N_TEAMS = int(os.environ.get("N_TEAMS", "11"))
BS = 512

files = sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz"))
ep_of = lambda f: int(os.path.basename(f).split("_")[0])
val_files = [f for f in files if ep_of(f) % 37 == 0]
tr_files = [f for f in files if ep_of(f) % 37 != 0]
VAL_MAX = int(os.environ.get("VAL_MAX", "0"))          # cap the (GPU-resident) validation set: memory pressure
if VAL_MAX and len(val_files) > VAL_MAX:
    val_files = val_files[::-(-len(val_files) // VAL_MAX)]
print(f"files: train {len(tr_files)}, val {len(val_files)}", flush=True)


def load(fs):
    parts = {k: [] for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market", "team", "w", "val")}
    for f in fs:
        try:
            z = np.load(f)
            T = z["op"].shape[0]; m = z["meta"]
            for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market"):
                parts[k].append(z[k])
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
            "team": torch.from_numpy(D["team"]).to(dev), "w": torch.from_numpy(D["w"]).to(dev), "val": torch.from_numpy(D["val"]).to(dev)}


def losses(net, D, idx, stats=None):
    planes = D["planes"][idx].float() / 255.0
    prev = D["prev_op"][idx]
    oh = Fn.one_hot(prev.clamp(min=0), F.N_OPS).float() * (prev >= 0).float()[..., None]
    uin = torch.cat([D["ufeat"][idx], oh], 2)
    lop, lq, lm, v = net(planes, D["glob"][idx], D["pos"][idx], uin, D["team"][idx])
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
    if stats is not None:
        pred = lop.argmax(-1)
        stats["op_ok"] += (pred[mask] == op[mask]).sum().item(); stats["op_n"] += mask.sum().item()
        mp = torch.stack([h.argmax(-1) for h in heads], 1); nz = mk > 0
        stats["m_nz_ok"] += (mp[nz] == mk[nz]).sum().item(); stats["m_nz_n"] += nz.sum().item()
        stats["m_fp"] += ((mp > 0) & (mk == 0)).sum().item(); stats["m_z_n"] += (~nz).sum().item()
    return l_op + 0.5 * l_q + l_m + 0.1 * l_v, (l_op.item(), l_q.item(), l_m.item(), l_v.item())


BLOCKS = int(os.environ.get("BLOCKS", "5"))
net = Net(width=WIDTH, blocks=BLOCKS, n_teams=N_TEAMS).to(dev)
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
    missing = net.load_state_dict(st0, strict=False)
    print("init from", INIT, "missing/unexpected:", missing, flush=True)
print(f"params {sum(p.numel() for p in net.parameters()):,}", flush=True)
VA = load(val_files)
steps_total = sum(1 for _ in tr_files) * 719 * EPOCHS // BS
opt = torch.optim.AdamW(net.parameters(), lr=MAX_LR / 10, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=max(10, steps_total), pct_start=0.05)
best = 1e9; it = 0; t0 = time.time()
rng = np.random.RandomState(0)


def validate():
    net.eval(); st = {k: 0 for k in ("op_ok", "op_n", "m_nz_ok", "m_nz_n", "m_fp", "m_z_n")}; vl = 0.0; nb = 0
    with torch.no_grad():
        NV = VA["op"].shape[0]
        for b in range(0, NV, 2048):
            l, _ = losses(net, VA, torch.arange(b, min(NV, b + 2048), device=dev), st); vl += l.item(); nb += 1
    net.train()
    return vl / nb, st


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
            print(f"ep {ep+1} chunk {c//CHUNK+1}/{(len(order)+CHUNK-1)//CHUNK} it {it}/{steps_total} {time.time()-t0:.0f}s | train op/q/m {agg[0]/max(1,nb):.3f}/{agg[1]/max(1,nb):.3f}/{agg[2]/max(1,nb):.3f} "
                  f"| val loss {vl:.3f} op acc {st['op_ok']/st['op_n']:.4f} market nz-recall {st['m_nz_ok']/max(1,st['m_nz_n']):.3f} fp {st['m_fp']/max(1,st['m_z_n']):.4f}", flush=True)
            if vl < best:
                best = vl
                torch.save({"state": net.state_dict(), "width": WIDTH, "blocks": BLOCKS, "n_teams": N_TEAMS, "val_loss": vl, "it": it,
                            "val_files": val_files}, out)
    torch.save({"state": net.state_dict(), "width": WIDTH, "blocks": BLOCKS, "n_teams": N_TEAMS, "it": it, "val_files": val_files}, out.replace(".pt", "_last.pt"))
print("best val loss", best, flush=True)
