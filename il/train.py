"""Behaviour cloning on leader games (Apple GPU via MPS).  Usage: train.py <data_dir> <ckpt_out> [epochs] [width]
Whole dataset lives on the GPU (unified memory); split by game (10% validation)."""
import sys, os, glob, time, json, math
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import features as F
from model import Net, split_heads, HEAD_SIZES, UB

torch.manual_seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data_dir, out = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
WIDTH = int(sys.argv[4]) if len(sys.argv) > 4 else 96
BS = 512
MK_NZ_W = float(os.environ.get("MK_NZ_W", "1"))    # loss weight of non-zero market targets

files = sorted(glob.glob(f"{data_dir}/*.npz"))
rng = np.random.RandomState(0); perm = rng.permutation(len(files))
nval = max(1, len(files) // 10)
val_set = set(perm[:nval].tolist())


def load(idx):
    parts = {k: [] for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market", "val")}
    for i in idx:
        z = np.load(files[i])
        T = z["op"].shape[0]
        for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market"):
            parts[k].append(z[k])
        m = z["meta"]; parts["val"].append(np.full(T, np.clip((m[2] - m[3]) / 2e4, -3, 3), np.float32))
    return {k: np.concatenate(v) for k, v in parts.items()}


t0 = time.time()
tr = load([i for i in range(len(files)) if i not in val_set]); va = load(sorted(val_set))
print(f"train steps {len(tr['op'])}, val steps {len(va['op'])}, load {time.time()-t0:.0f}s", flush=True)


def to_dev(d):
    return {"planes": torch.from_numpy(d["planes"]).to(dev), "glob": torch.from_numpy(d["glob"].astype(np.float32)).to(dev),
            "pos": torch.from_numpy(d["pos"].astype(np.int64)).to(dev), "ufeat": torch.from_numpy(d["ufeat"].astype(np.float32)).to(dev),
            "prev_op": torch.from_numpy(d["prev_op"].astype(np.int64)).to(dev), "op": torch.from_numpy(d["op"].astype(np.int64)).to(dev),
            "qty": torch.from_numpy(d["qty"].astype(np.int64)).to(dev), "market": torch.from_numpy(d["market"].astype(np.int64)).to(dev),
            "val": torch.from_numpy(d["val"]).to(dev)}


TR, VA = to_dev(tr), to_dev(va)
del tr, va


def batch_inputs(D, idx):
    planes = D["planes"][idx].float() / 255.0
    glob_ = D["glob"][idx]
    pos = D["pos"][idx]
    prev = D["prev_op"][idx]
    oh = Fn.one_hot(prev.clamp(min=0), F.N_OPS).float() * (prev >= 0).float()[..., None]
    uin = torch.cat([D["ufeat"][idx], oh], 2)
    return planes, glob_, pos, uin


def losses(net, D, idx, stats=None):
    planes, glob_, pos, uin = batch_inputs(D, idx)
    lop, lq, lm, v = net(planes, glob_, pos, uin)
    op, qty, mk = D["op"][idx], D["qty"][idx], D["market"][idx]
    mask = op >= 0
    l_op = Fn.cross_entropy(lop[mask], op[mask])
    qm = qty >= 0
    l_q = Fn.cross_entropy(lq[qm], qty[qm]) if qm.any() else lop.sum() * 0
    heads = split_heads(lm)
    wz = torch.where(mk > 0, MK_NZ_W, 1.0)
    l_m = sum((Fn.cross_entropy(h, mk[:, j], reduction='none') * wz[:, j]).sum() / wz[:, j].sum() for j, h in enumerate(heads)) / len(heads)
    l_v = Fn.mse_loss(v, D["val"][idx])
    if stats is not None:
        pred = lop.argmax(-1)
        stats["op_ok"] += (pred[mask] == op[mask]).sum().item(); stats["op_n"] += mask.sum().item()
        if qm.any():
            stats["q_ok"] += (lq.argmax(-1)[qm] == qty[qm]).sum().item(); stats["q_n"] += qm.sum().item()
        mp = torch.stack([h.argmax(-1) for h in heads], 1)
        nz = mk > 0
        stats["m_nz_ok"] += (mp[nz] == mk[nz]).sum().item(); stats["m_nz_n"] += nz.sum().item()
        stats["m_fp"] += ((mp > 0) & (mk == 0)).sum().item(); stats["m_z_n"] += (~nz).sum().item()
        stats["m_step_ok"] += (mp == mk).all(1).sum().item(); stats["steps"] += len(idx)
        # per-class recall for a few important ops
        for name in ("WATER", "HARVEST", "NORTH", "FEED", "CARE", "COLLECT_FERTILIZER", "PLANT_WHEAT", "PICKUP_WHEAT", "DROP", "PASS"):
            c = F.OP_INDEX[name]; sel = mask & (op == c)
            stats.setdefault("cls", {}).setdefault(name, [0, 0])
            stats["cls"][name][0] += (pred[sel] == c).sum().item(); stats["cls"][name][1] += sel.sum().item()
    return l_op + 0.5 * l_q + l_m + 0.1 * l_v, (l_op.item(), l_q.item(), l_m.item(), l_v.item())


net = Net(width=WIDTH).to(dev)
nparam = sum(p.numel() for p in net.parameters())
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
N = TR["op"].shape[0]
steps_per_epoch = N // BS
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * steps_per_epoch, pct_start=0.1)
print(f"params {nparam:,}, device {dev}, steps/epoch {steps_per_epoch}", flush=True)
best = 1e9
for ep in range(EPOCHS):
    net.train(); t1 = time.time(); agg = np.zeros(4); perm_ = torch.randperm(N, device=dev)
    for b in range(steps_per_epoch):
        idx = perm_[b * BS:(b + 1) * BS]
        loss, parts = losses(net, TR, idx)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        agg += parts
    net.eval(); st = {k: 0 for k in ("op_ok", "op_n", "q_ok", "q_n", "m_nz_ok", "m_nz_n", "m_fp", "m_z_n", "m_step_ok", "steps")}
    vl = 0.0; nb = 0
    with torch.no_grad():
        NV = VA["op"].shape[0]
        for b in range(0, NV, 2048):
            idx = torch.arange(b, min(NV, b + 2048), device=dev)
            l, _ = losses(net, VA, idx, st); vl += l.item(); nb += 1
    vl /= nb
    cls = " ".join(f"{k[:9]} {v[0]/max(1,v[1]):.2f}" for k, v in st["cls"].items())
    print(f"ep {ep+1}/{EPOCHS} {time.time()-t1:.0f}s train op/q/m/v {agg[0]/steps_per_epoch:.3f}/{agg[1]/steps_per_epoch:.3f}/{agg[2]/steps_per_epoch:.3f}/{agg[3]/steps_per_epoch:.3f} "
          f"| val loss {vl:.3f} op acc {st['op_ok']/st['op_n']:.4f} qty acc {st['q_ok']/max(1,st['q_n']):.3f} "
          f"market nz-recall {st['m_nz_ok']/max(1,st['m_nz_n']):.3f} fp-rate {st['m_fp']/max(1,st['m_z_n']):.4f} exact-steps {st['m_step_ok']/st['steps']:.3f}\n    {cls}", flush=True)
    if vl < best:
        best = vl
        torch.save({"state": net.state_dict(), "width": WIDTH, "epoch": ep + 1, "val_loss": vl, "files_val": sorted(val_set)}, out)
print("best val loss", best, flush=True)
