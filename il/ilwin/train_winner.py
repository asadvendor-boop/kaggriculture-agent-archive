"""Winner-imitation fine-tune (Sep 29) = il/train6.py (the trainer of v13_p3: same Net, losses, AdamW, OneCycleLR, batch 512,
v6 features, aux heads) with the minimum changes for a short, low-LR fine-tune with REHEARSAL:

  * two data groups: WIN_DIRS (winner seats from build_winner_ds.py) and REH ("dir:ratio,dir:ratio": per pass, ratio x
    #winner seats are drawn WITHOUT replacement from each rehearsal dir, fresh draw every pass, shuffled together with the
    winner seats into the chunks). PASSES (float) counts passes over the winner seats.
  * checkpoints every SAVE_EVERY of a pass (default 0.5): <out>_h1.pt, _h2.pt, ... (same dict format as train6: loads with
    the same Net() arguments everywhere - live_rollout, export.py, srv_mkpkg.sh). <out> itself = the final weights.
  * validation reported separately on winner val games and rehearsal val games (ep % 37 == 0 as in train6): rehearsal val
    loss = the forgetting gauge. No checkpoint selection by val loss (it did not rank playing strength, Sep 27).
  * market labels of -100 (ignored heads, see build_winner_ds.py) are excluded from the loss AND its normaliser; op -1 masked.
  * WIN_TEAM_MODE: ours (team 10 for every winner seat) | true (meta[8]: opponents' vocab id or the generic id) | meta
    (meta[6] as built). Rehearsal seats always keep their stored team.
  * X6_DIRS: colon list of v6-feature dirs searched in order (default: <each data dir>/../x6 then il/data/x6).
Env (train6 names kept): INIT, MAX_LR, SEED, BLOCKS, N_TEAMS, MK_NZ_W, LOSS_W, CHUNK, VAL_MAX, AUX_W, LOAD_THREADS, PREFETCH,
DEVICE, AMP, RATING_W (rehearsal per-seat weights, e.g. il/data/rating_w_fresh2.json), WIN_W (winner weights json),
BS (default 512), MAX_STEPS / VAL_STATES / TORCH_THREADS (dry runs), PCT_START (0.05).
Usage: train_winner.py <ckpt_out> <passes> <width>
"""
import sys, os, glob, time, json, contextlib
import numpy as np
import torch
import torch.nn.functional as Fn
K = os.environ.get("KROOT", "/Users/asad.ali/kaggriculture")
sys.path.insert(0, f"{K}/il")
import features as F
import features6 as F6
from model import Net, split_heads

SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
dev = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
AMP = int(os.environ.get("AMP", "0")) and dev.type == "cuda"
amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if AMP else contextlib.nullcontext
LOAD_THREADS = int(os.environ.get("LOAD_THREADS", "1")); PREFETCH = int(os.environ.get("PREFETCH", "0"))
out, PASSES, WIDTH = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
MK_NZ_W = float(os.environ.get("MK_NZ_W", "3")); LOSS_W = float(os.environ.get("LOSS_W", "0.7"))
CHUNK = int(os.environ.get("CHUNK", "100")); INIT = os.environ.get("INIT"); MAX_LR = float(os.environ.get("MAX_LR", "5e-5"))
N_TEAMS = int(os.environ.get("N_TEAMS", "48"))
AUX_W = float(os.environ.get("AUX_W", "0.1")); AUX_H = (24, 72, 168); N_AUX = 2 * len(AUX_H)
BS = int(os.environ.get("BS", "512")); MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
SAVE_EVERY = float(os.environ.get("SAVE_EVERY", "0.5")); PCT_START = float(os.environ.get("PCT_START", "0.05"))
WIN_TEAM_MODE = os.environ.get("WIN_TEAM_MODE", "ours"); OUR_TEAM = int(os.environ.get("OUR_TEAM", "10"))
RW = json.load(open(os.environ["RATING_W"])) if os.environ.get("RATING_W") else {}
WW = json.load(open(os.environ["WIN_W"])) if os.environ.get("WIN_W") else {}
RW_DEF = float(os.environ.get("RW_DEF", "1.0"))
WIN_DIRS = [d for d in os.environ.get("WIN_DIRS", "").split(":") if d]
REH = [(x.rsplit(":", 1)[0], float(x.rsplit(":", 1)[1])) for x in os.environ.get("REH", "").split(",") if x]
X6_DIRS = [d for d in os.environ.get("X6_DIRS", "").split(":") if d]
if not X6_DIRS:
    X6_DIRS = [os.path.join(os.path.dirname(os.path.abspath(d.rstrip("/"))), "x6") for d in WIN_DIRS] + [f"{K}/il/data/x6"]
X6 = {}
for d in reversed(X6_DIRS):                                   # earlier dirs win
    for f in glob.glob(f"{d}/*.npy"):
        X6[os.path.basename(f)[:-4]] = f
base = lambda f: os.path.basename(f)[:-4]
ep_of = lambda f: int(os.path.basename(f).split("_")[0])
win_files = sorted(f for d in WIN_DIRS for f in glob.glob(f"{d}/*.npz") if base(f) in X6)
win_eps = {ep_of(f) for f in win_files}
WIN_SET = set(win_files)
win_val = [f for f in win_files if ep_of(f) % 37 == 0]; win_tr = [f for f in win_files if ep_of(f) % 37 != 0]
reh_tr, reh_val = [], []
for d, r in REH:
    fs = sorted(f for f in glob.glob(f"{d}/*.npz") if base(f) in X6 and ep_of(f) not in win_eps)
    reh_tr.append((d, r, [f for f in fs if ep_of(f) % 37 != 0])); reh_val += [f for f in fs if ep_of(f) % 37 == 0]
reh_val = sorted(set(reh_val))
VAL_MAX = int(os.environ.get("VAL_MAX", "180"))
if VAL_MAX and len(reh_val) > VAL_MAX:
    reh_val = reh_val[::-(-len(reh_val) // VAL_MAX)]
if VAL_MAX and len(win_val) > VAL_MAX:
    win_val = win_val[::-(-len(win_val) // VAL_MAX)]
print(f"winner seats: train {len(win_tr)}, val {len(win_val)} | rehearsal: " + ", ".join(f"{d} x{r} ({len(fs)} train)" for d, r, fs in reh_tr)
      + f", val {len(reh_val)} | x6 index {len(X6)} files | team mode {WIN_TEAM_MODE}", flush=True)
assert win_tr, "no winner seats found (WIN_DIRS / X6_DIRS)"


def _one(f):
    out = {}
    try:
        z = np.load(f)
        T = z["op"].shape[0]; m = z["meta"]
        x6 = np.load(X6[base(f)])
        if x6.shape[0] != T:
            return None
        for k in ("planes", "pos", "ufeat", "prev_op", "op", "qty", "market"):
            out[k] = z[k]
        out["glob"] = np.concatenate([z["glob"].astype(np.float32), x6.astype(np.float32)], 1)
        mo = z["money"].astype(np.float64); diff = (mo[:, 0] - mo[:, 1]) / 2e4; own = mo[:, 0] / 1e4
        fd, fo = (float(m[2]) - float(m[3])) / 2e4, float(m[2]) / 1e4
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
    if f in WIN_SET:
        if WIN_TEAM_MODE == "ours":
            team = OUR_TEAM
        elif WIN_TEAM_MODE == "true" and len(m) > 8:
            team = int(m[8])
        w = WW.get(base(f), 1.0)
    else:
        w = RW.get(base(f), RW_DEF) if RW else 1.0
    win = int(m[7]) if len(m) > 7 else int(m[2] > m[3])
    out["team"] = np.full(T, team, np.int64); out["w"] = np.full(T, (1.0 if win else LOSS_W) * w, np.float32)
    out["val"] = np.full(T, np.clip((m[2] - m[3]) / 2e4, -3, 3), np.float32)
    return out


def load_cpu(fs):
    if LOAD_THREADS > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(LOAD_THREADS) as ex:
            res = list(ex.map(_one, fs))
    else:
        res = [_one(f) for f in fs]
    res = [r for r in res if r is not None]
    D = {k: np.concatenate([r[k] for r in res]) for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market", "team", "w", "val", "aux")}
    return {"planes": torch.from_numpy(D["planes"]), "glob": torch.from_numpy(D["glob"].astype(np.float32)),
            "pos": torch.from_numpy(D["pos"].astype(np.int64)), "ufeat": torch.from_numpy(D["ufeat"].astype(np.float32)),
            "prev_op": torch.from_numpy(D["prev_op"].astype(np.int64)), "op": torch.from_numpy(D["op"].astype(np.int64)),
            "qty": torch.from_numpy(D["qty"].astype(np.int64)), "market": torch.from_numpy(D["market"].astype(np.int64)),
            "team": torch.from_numpy(D["team"]), "w": torch.from_numpy(D["w"]), "val": torch.from_numpy(D["val"]),
            "aux": torch.from_numpy(D["aux"])}


def to_dev(D):
    return {k: v.to(dev) for k, v in D.items()}


def load(fs):                                       # synchronous callers (validation sets): CPU build + device move in THIS thread
    return to_dev(load_cpu(fs))


def losses(net, D, idx, stats=None):
    planes = D["planes"][idx].float() / 255.0
    prev = D["prev_op"][idx]
    oh = Fn.one_hot(prev.clamp(min=0), F.N_OPS).float() * (prev >= 0).float()[..., None]
    uin = torch.cat([D["ufeat"][idx], oh], 2)
    lop, lq, lm, v, ap = net(planes, D["glob"][idx], D["pos"][idx], uin, D["team"][idx], aux=True)
    op, qty, mk, w = D["op"][idx], D["qty"][idx], D["market"][idx], D["w"][idx]
    mask = op >= 0
    wu = w[:, None].expand_as(op)[mask]
    l_op = (Fn.cross_entropy(lop[mask], op[mask], reduction="none") * wu).sum() / wu.sum().clamp(min=1e-6)
    qm = qty >= 0
    wq = w[:, None].expand_as(qty)[qm]
    l_q = (Fn.cross_entropy(lq[qm], qty[qm], reduction="none") * wq).sum() / wq.sum() if qm.any() else lop.sum() * 0
    heads = split_heads(lm)
    valid = (mk >= 0).float()                                   # -100 = ignored head (winner data only)
    wz = torch.where(mk > 0, MK_NZ_W, 1.0) * w[:, None] * valid
    mk_t = mk.clamp(min=0)
    l_m = sum((Fn.cross_entropy(h, mk_t[:, j], reduction="none") * wz[:, j]).sum() / wz[:, j].sum().clamp(min=1e-6) for j, h in enumerate(heads)) / len(heads)
    l_v = Fn.mse_loss(v, D["val"][idx])
    l_a = Fn.mse_loss(ap, D["aux"][idx])
    if stats is not None:
        pred = lop.argmax(-1)
        stats["op_ok"] += (pred[mask] == op[mask]).sum().item(); stats["op_n"] += mask.sum().item()
        mp = torch.stack([h.argmax(-1) for h in heads], 1); nz = mk > 0
        stats["m_nz_ok"] += (mp[nz] == mk[nz]).sum().item(); stats["m_nz_n"] += nz.sum().item()
        stats["m_fp"] += ((mp > 0) & (mk == 0)).sum().item(); stats["m_z_n"] += (mk == 0).sum().item()
    return l_op + 0.5 * l_q + l_m + 0.1 * l_v + AUX_W * l_a, (l_op.item(), l_q.item(), l_m.item(), l_a.item())


BLOCKS = int(os.environ.get("BLOCKS", "10"))
net = Net(width=WIDTH, blocks=BLOCKS, n_teams=N_TEAMS, g_in=F6.G, n_aux=N_AUX).to(dev)
if INIT:
    c0 = torch.load(INIT, map_location="cpu", weights_only=False)
    assert int(c0["width"]) == WIDTH and int(c0.get("blocks", 5)) == BLOCKS and int(c0.get("g_in") or 0) == F6.G, "INIT shape mismatch"
    st0 = dict(c0["state"])
    tw = st0.get("team.weight")
    if tw is not None and tw.shape[0] != N_TEAMS:              # train6's expansion: new rows = mean + 0.1 noise
        w = net.team.weight.detach().clone().cpu()
        k = min(tw.shape[0], N_TEAMS)
        w[:k] = tw[:k]
        if N_TEAMS > k:
            w[k:] = tw.mean(0, keepdim=True) + 0.1 * torch.randn(N_TEAMS - k, tw.shape[1])
        st0["team.weight"] = w
        print(f"team embedding expanded {tw.shape[0]} -> {N_TEAMS}", flush=True)
    missing = net.load_state_dict(st0, strict=False)
    print("init from", INIT, "missing/unexpected:", missing, flush=True)
print(f"params {sum(p.numel() for p in net.parameters()):,} | device {dev} | MAX_LR {MAX_LR} passes {PASSES} BS {BS}", flush=True)
VW = load_cpu(win_val) if win_val else None
VR = load_cpu(reh_val) if reh_val else None
VAL_STATES = int(os.environ.get("VAL_STATES", "0"))           # validate on a random subset of states (Mac: keeps the GPU footprint small)
if VAL_STATES:
    _g = torch.Generator().manual_seed(0)
    for _V in (VW, VR):
        if _V is not None and _V["op"].shape[0] > VAL_STATES:
            _sel = torch.randperm(_V["op"].shape[0], generator=_g)[:VAL_STATES]
            for _k in list(_V):
                _V[_k] = _V[_k][_sel]
VW = to_dev(VW) if VW is not None else None
VR = to_dev(VR) if VR is not None else None
PROG = int(os.environ.get("PROGRESS_EVERY", "25"))           # Sep 29: a progress line every PROG steps (the stall watchdog reads it)
T_SEAT = 719
n_reh_per_pass = sum(min(len(fs), int(round(r * len(win_tr)))) for _, r, fs in reh_tr)
seats_per_pass = len(win_tr) + n_reh_per_pass
steps_per_pass = max(1, seats_per_pass * T_SEAT // BS)
steps_total = max(2, int(round(steps_per_pass * PASSES)))
if MAX_STEPS:
    steps_total = min(steps_total, MAX_STEPS)
save_at = sorted({int(round(steps_per_pass * SAVE_EVERY * k)) for k in range(1, int(np.ceil(PASSES / SAVE_EVERY)) + 1)
                  if int(round(steps_per_pass * SAVE_EVERY * k)) <= steps_total} | {steps_total})
print(f"per pass: {len(win_tr)} winner + {n_reh_per_pass} rehearsal seats = {steps_per_pass} steps; total {steps_total}; saves at {save_at}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=MAX_LR / 10, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=max(10, steps_total), pct_start=PCT_START)
it = 0; t0 = time.time(); rng = np.random.RandomState(SEED)


def validate(V):
    if V is None:
        return float("nan"), None
    net.eval(); st = {k: 0 for k in ("op_ok", "op_n", "m_nz_ok", "m_nz_n", "m_fp", "m_z_n")}; vl = 0.0; nb = 0
    with torch.no_grad():
        NV = V["op"].shape[0]
        for b in range(0, NV, 2048):
            with amp_ctx():
                l, _ = losses(net, V, torch.arange(b, min(NV, b + 2048), device=dev), st)
            vl += l.item(); nb += 1
    net.train()
    return vl / nb, st


def report(tag):
    s = []
    for name, V in (("win", VW), ("reh", VR)):
        vl, st = validate(V)
        if st:
            s.append(f"{name} val {vl:.4f} op acc {st['op_ok']/max(1,st['op_n']):.4f} nz-recall {st['m_nz_ok']/max(1,st['m_nz_n']):.3f} fp {st['m_fp']/max(1,st['m_z_n']):.4f}")
    print(f"[{tag}] it {it}/{steps_total} {time.time()-t0:.0f}s | " + " | ".join(s), flush=True)


def save(path):
    torch.save({"state": net.state_dict(), "width": WIDTH, "blocks": BLOCKS, "n_teams": N_TEAMS, "g_in": F6.G, "n_aux": N_AUX,
                "feat": 6, "it": it, "val_files": win_val + reh_val,
                "ft": dict(init=INIT, max_lr=MAX_LR, passes=PASSES, reh=REH, win_dirs=WIN_DIRS, team_mode=WIN_TEAM_MODE,
                           seed=SEED, steps_total=steps_total, bs=BS)}, path)
    print(f"saved {path} (it {it})", flush=True)


report("start")
agg = np.zeros(4); nb = 0; p = 0
while it < steps_total:
    it_pass0 = it
    files = list(win_tr)
    for d, r, fs in reh_tr:
        n = min(len(fs), int(round(r * len(win_tr))))
        files += [fs[i] for i in rng.choice(len(fs), n, replace=False)] if n else []
    order = rng.permutation(len(files)); p += 1
    chunks = [[files[i] for i in order[c:c + CHUNK]] for c in range(0, len(order), CHUNK)]
    if PREFETCH:
        from concurrent.futures import ThreadPoolExecutor
        _pf = ThreadPoolExecutor(1); _fut = _pf.submit(load_cpu, chunks[0])
    for ci in range(len(chunks)):
        if PREFETCH:
            D = to_dev(_fut.result())
            if ci + 1 < len(chunks):
                _fut = _pf.submit(load_cpu, chunks[ci + 1])
        else:
            D = load(chunks[ci])
        N = D["op"].shape[0]; perm = torch.randperm(N, device=dev)
        for b in range(0, N - BS + 1, BS):
            if it >= steps_total:
                break
            with amp_ctx():
                loss, parts = losses(net, D, perm[b:b + BS])
            opt.zero_grad(); loss.backward(); opt.step()
            if it < steps_total - 1:
                sched.step()
            it += 1; agg += parts; nb += 1
            if PROG and it % PROG == 0 and it not in save_at:
                print(f"  it {it}/{steps_total} {time.time()-t0:.0f}s ({(time.time()-t0)/max(1,it):.2f} s/it incl. start)", flush=True)
            if it in save_at:
                print(f"pass {p} chunk {ci+1}/{len(chunks)} it {it} | train op/q/m/aux " + "/".join(f"{x/max(1,nb):.3f}" for x in agg)
                      + f" lr {sched.get_last_lr()[0]:.2e}", flush=True)
                agg[:] = 0; nb = 0
                report(f"save {save_at.index(it) + 1}")
                save(out.replace(".pt", f"_h{save_at.index(it) + 1}.pt"))
        del D
        if dev.type == "mps":
            torch.mps.empty_cache()
        if it >= steps_total:
            break
    if PREFETCH:
        _pf.shutdown(wait=False)
    if it == it_pass0:
        print("no optimizer step in a whole pass (BS larger than a chunk?) - stopping", flush=True); break
save(out)
print("done", flush=True)
