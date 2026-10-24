"""Vocabulary behaviour-cloning dataset (team ids from il/data/team_vocab.json; seats already built in
il/data/leaders or the output dir are skipped).  Based on build_dataset2: every seat played by a top-10 team (or DSM) in the given tape dirs becomes
one .npz (same arrays as build_dataset.py) plus meta [ep, seat, bank, opp_bank, exact, seed, team_id, win].
Only games that replay exactly in the C++ engine are kept.

Usage: build_dataset2.py <out_dir> <worker> <n_workers> <tape_dir> [<tape_dir> ...]
"""
import sys, os, json, gzip, time
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from harness import kagsim_trace, PASS
import features as F

TEAMS = json.load(open("/Users/asad.ali/kaggriculture/il/data/team_vocab.json"))
TEAM_ID = {t: i for i, t in enumerate(TEAMS)}
UB = F.MAX_UNITS + 2 + len(F.ITEMS) + 2


def build_game(path, seats, out_dir):
    d = json.load(gzip.open(path, "rt"))
    A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][s]] for s in (0, 1)]
    g = kagsim_trace.Game(d["seed"], 720, False)
    T = 720
    buf = {}
    for s in seats:
        buf[s] = dict(planes=np.zeros((T, F.C, F.B, F.B), np.uint8), glob=np.zeros((T, F.G), np.float16),
                      pos=np.zeros((T, F.MAX_UNITS, 2), np.int8), ufeat=np.zeros((T, F.MAX_UNITS, UB), np.float16),
                      prev_op=np.full((T, F.MAX_UNITS), -1, np.int8), op=np.full((T, F.MAX_UNITS), -1, np.int8),
                      qty=np.full((T, F.MAX_UNITS), -1, np.int8), market=np.zeros((T, F.N_HEADS), np.int8),
                      money=np.zeros((T, 2), np.float32), prev=None)
    n_steps = 0
    for t in range(T):
        for s in seats:
            b = buf[s]; o = g.observe(s)
            p, gl, ps, u = F.encode_obs(o, None)
            n = len(ps)
            b["planes"][t] = p; b["glob"][t] = gl; b["pos"][t, :n] = ps; b["ufeat"][t, :n] = u[:, :UB]
            if b["prev"] is not None:
                b["prev_op"][t, :n] = [b["prev"][k] if k < len(b["prev"]) else -1 for k in range(n)]
            a = A[s][t]
            cmds = [a.get("farmer")] + list(a.get("hands") or [])
            cur = []
            for k in range(n):
                oc, qc = F.unit_op_label(cmds[k] if k < len(cmds) else ["PASS"])
                b["op"][t, k] = oc; b["qty"][t, k] = qc; cur.append(oc)
            b["prev"] = cur
            b["market"][t] = F.market_labels(a.get("market"))
            b["money"][t] = (o["farms"][s]["money"], o["farms"][1 - s]["money"])
        g.step(A[0][t], A[1][t])
        n_steps = t + 1
        if g.done:
            break
    rec = d["rewards"]
    if rec[0] is None or rec[1] is None:
        return 0
    exact = abs(g.reward(0) - rec[0]) < 0.5 and abs(g.reward(1) - rec[1]) < 0.5
    if not exact:
        return -1
    for s in seats:
        b = buf[s]; team = d["teams"][s]
        np.savez_compressed(f"{out_dir}/{d['ep']}_{s}.npz", **{k: v[:n_steps] for k, v in b.items() if k != "prev"},
                            meta=np.array([d["ep"], s, rec[s], rec[1 - s], 1, int(d["seed"]) % (2**31), TEAM_ID[team],
                                           int(rec[s] > rec[1 - s])], np.int64))
    return len(seats)


if __name__ == "__main__":
    out_dir, w, nw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    dirs = sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for dd in dirs:
        idx = json.load(open(f"{dd}/index.json"))
        for ep, m in sorted(idx.items()):
            seats = [s for s in (0, 1) if (m.get("teams") or [None, None])[s] in TEAM_ID]
            if seats and os.path.exists(f"{dd}/{ep}.json.gz"):
                jobs.append((f"{dd}/{ep}.json.gz", seats, ep))
    jobs = jobs[w::nw]
    t0 = time.time(); ok = bad = skipped = 0
    for i, (path, seats, ep) in enumerate(jobs):
        seats = [s for s in seats if not (os.path.exists(f"{out_dir}/{ep}_{s}.npz") or os.path.exists(f"/Users/asad.ali/kaggriculture/il/data/leaders/{ep}_{s}.npz"))]
        if not seats:
            skipped += 1; continue
        try:
            r = build_game(path, seats, out_dir)
        except Exception as ex:
            print("fail", ep, str(ex)[:80], flush=True); bad += 1; continue
        if r > 0: ok += r
        else: bad += 1
        if (i + 1) % 100 == 0:
            print(f"worker {w}: {i+1}/{len(jobs)} games, seats written {ok}, non-exact/failed {bad}, {time.time()-t0:.0f}s", flush=True)
    print(f"worker {w} done: {len(jobs)} games, seats written {ok}, non-exact/failed {bad}, skipped {skipped}, {time.time()-t0:.0f}s", flush=True)
