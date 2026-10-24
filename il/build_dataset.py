"""Build the behaviour-cloning dataset: replay leader games in the C++ engine and record (observation features,
leader action labels) for every step of the leader's seat.

Usage: build_dataset.py <team> <out_dir> [<worker> <n_workers>]
Writes one compressed .npz per game: planes [T,C,10,10] u8, glob [T,G] f16, pos [T,16,2] i8, ufeat [T,16,32] f16,
prev_op [T,16] i8, op [T,16] i8 (-1 = no unit), qty [T,16] i8 (-1 = n/a), market [T,21] i8, plus meta.
"""
import sys, os, json, gzip, glob, time
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from harness import kagsim_trace, PASS, ROOT
import features as F

UB = F.MAX_UNITS + 2 + len(F.ITEMS) + 2          # unit features without the previous-op one-hot (32)


def games_for(team):
    out = []
    for d_ in ("data/tapes_0922", "data/tapes_archive"):
        idx = json.load(open(f"{ROOT}/{d_}/index.json"))
        for ep, m in idx.items():
            for s in (0, 1):
                if (m.get("teams") or [None, None])[s] == team:
                    out.append((f"{ROOT}/{d_}/{ep}.json.gz", s))
    return sorted(out)


def build_one(path, seat, out_dir):
    d = json.load(gzip.open(path, "rt"))
    A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][s]] for s in (0, 1)]
    g = kagsim_trace.Game(d["seed"], 720, False)
    T = 720
    planes = np.zeros((T, F.C, F.B, F.B), np.uint8); glob_ = np.zeros((T, F.G), np.float16)
    pos = np.zeros((T, F.MAX_UNITS, 2), np.int8); uf = np.zeros((T, F.MAX_UNITS, UB), np.float16)
    prev = np.full((T, F.MAX_UNITS), -1, np.int8); op = np.full((T, F.MAX_UNITS), -1, np.int8)
    qty = np.full((T, F.MAX_UNITS), -1, np.int8); mk = np.zeros((T, F.N_HEADS), np.int8)
    money = np.zeros((T, 2), np.float32)
    prev_ops = None
    n_steps = 0
    for t in range(T):
        o = g.observe(seat)
        p, gl, ps, u = F.encode_obs(o, None)
        n = len(ps)
        planes[t] = p; glob_[t] = gl; pos[t, :n] = ps; uf[t, :n] = u[:, :UB]
        if prev_ops is not None:
            prev[t, :n] = [prev_ops[k] if k < len(prev_ops) else -1 for k in range(n)]
        a = A[seat][t]
        cmds = [a.get("farmer")] + list(a.get("hands") or [])
        cur = []
        for k in range(n):
            oc, qc = F.unit_op_label(cmds[k] if k < len(cmds) else ["PASS"])
            op[t, k] = oc; qty[t, k] = qc; cur.append(oc)
        prev_ops = cur
        mk[t] = F.market_labels(a.get("market"))
        money[t] = (o["farms"][seat]["money"], o["farms"][1 - seat]["money"])
        g.step(A[0][t], A[1][t])
        n_steps = t + 1
        if g.done:
            break
    rec = d["rewards"]
    exact = abs(g.reward(0) - rec[0]) < 0.5 and abs(g.reward(1) - rec[1]) < 0.5
    ep = d["ep"]
    np.savez_compressed(f"{out_dir}/{ep}_{seat}.npz", planes=planes[:n_steps], glob=glob_[:n_steps], pos=pos[:n_steps],
                        ufeat=uf[:n_steps], prev_op=prev[:n_steps], op=op[:n_steps], qty=qty[:n_steps],
                        market=mk[:n_steps], money=money[:n_steps],
                        meta=np.array([ep, seat, rec[seat], rec[1 - seat], int(exact), int(d.get("seed", 0)) % (2**31)], np.int64))
    return exact, rec[seat] - rec[1 - seat]


if __name__ == "__main__":
    team, out_dir = sys.argv[1], sys.argv[2]
    w, nw = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    os.makedirs(out_dir, exist_ok=True)
    jobs = games_for(team)[w::nw]
    t0 = time.time(); ok = 0
    for i, (path, seat) in enumerate(jobs):
        ep = os.path.basename(path).split(".")[0]
        if os.path.exists(f"{out_dir}/{ep}_{seat}.npz"):
            continue
        exact, m = build_one(path, seat, out_dir)
        ok += exact
        if i % 20 == 0:
            print(f"worker {w}: {i+1}/{len(jobs)} games, exact {ok}, {time.time()-t0:.0f}s", flush=True)
    print(f"worker {w} done: {len(jobs)} games, exact {ok}, {time.time()-t0:.0f}s", flush=True)
