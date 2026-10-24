"""A Kaggle package (scripted main.py) on N of the 464 ladder worlds vs a stored benchmark of a model: identical margins?
Usage: pkg_compare.py <package main.py or tar.gz> <benchmark rows json> [N]      env NW (workers)"""
import sys, os, json, tarfile, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):   # packages run numpy:
    os.environ[_v] = "1"                                                   # 1 math thread per worker (Sep 26 fix)
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")


def _run(job):
    from live_rollout import run_scripted
    return run_scripted(job)


if __name__ == "__main__":
    pkg, rows_fn = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    W = {w["ep"]: w for w in json.load(open("/Users/asad.ali/kaggriculture/il/live/ladder_worlds.json"))}
    rows = [r for r in json.load(open(rows_fn)) if r["ep"] in W]
    pick = rows[:: max(1, len(rows) // N)][:N]
    games = [(W[r["ep"]]["seed"], W[r["ep"]]["seat"], "TAPE:" + W[r["ep"]]["path"]) for r in pick]
    nw = int(os.environ.get("NW", "12"))
    with mp.get_context("spawn").Pool(nw) as pool:
        res = sum(pool.map(_run, [dict(us=pkg, games=games[k::nw]) for k in range(nw)]), [])
    by = {x["opp"][5:]: x for x in res}
    same = diff = 0
    for r, gm in zip(pick, games):
        m = by[gm[2][5:]]["margin"]
        if abs(m - r["margin"]) < 0.5: same += 1
        else:
            diff += 1; print(f"  DIFFERENT ep {r['ep']}: package {m:+,.0f} vs benchmark {r['margin']:+,.0f}", flush=True)
    print(f"{os.path.basename(pkg)} vs {os.path.basename(rows_fn)}: {same}/{len(pick)} worlds identical, {diff} different", flush=True)
