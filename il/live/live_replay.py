"""Replay our live ladder games with a model (world pinned, the opponent's recorded actions): the "live games" ruler for
runtime rules and new models (Sep 26).  Rows in the ladder-benchmark format (ep, opp, opp_rating, margin, bank, opp_bank,
rec_margin) -> il/logs/live_<tag>.json, so live/pair_cmp.py compares two runs game by game (wins AND losses, not only
the losses).  Rules via env (SE_GUARD, SHEEP_CAP, ...) as for bench_multi.py.
Usage: live_replay.py <ckpt> <team> <tag> <tape_dir> [<tape_dir> ...]      env NW (8), LOCKSTEP (6)"""
import sys, os, json, gzip, glob
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
R = "/Users/asad.ali/kaggriculture"


def _run(job):
    from live_rollout import run_job
    return run_job(job)


if __name__ == "__main__":
    ck, team, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    W = []
    for d_ in sys.argv[4:]:
        for f in sorted(glob.glob(f"{d_}/*.json.gz")):
            t = json.load(gzip.open(f, "rt"))
            if t.get("res") not in ("W", "L"):
                continue
            s = int(t["seat"])
            W.append(dict(ep=int(t["ep"]), opp=t.get("opp"), opp_rating=float(t.get("opp_rating") or 0), path=f, seat=s, seed=int(t["seed"]),
                          rec_margin=t["rewards"][s] - t["rewards"][1 - s]))
    L = int(os.environ.get("LOCKSTEP", "6"))
    games = [(w["seed"], w["seat"], "TAPE:" + w["path"]) for w in W]
    jobs = [dict(weights=ck, team=team, tau=0.5, T=0.2, ks=10.0, greedy=True, seed=k, games=games[k:k + L]) for k in range(0, len(games), L)]
    with mp.get_context("spawn").Pool(int(os.environ.get("NW", "8")), maxtasksperchild=40) as pool:
        res = sum(pool.map(_run, jobs, chunksize=1), [])
    by = {r["opp"][5:]: r for r in res}
    rows = [dict(ep=w["ep"], opp=w["opp"], opp_rating=w["opp_rating"], margin=by[w["path"]]["margin"], bank=by[w["path"]]["bank"],
                 opp_bank=by[w["path"]]["opp_bank"], rec_margin=w["rec_margin"]) for w in W]
    fn = f"{R}/il/logs/live_{tag}.json"; json.dump(rows, open(fn, "w"), ensure_ascii=False)
    import numpy as np
    m = np.array([r["margin"] for r in rows]); rec = np.array([r["rec_margin"] for r in rows])
    print(f"{tag}: {len(rows)} live games replayed | W {(m > 0).sum()} (live {(rec > 0).sum()}) | margin {m.mean():+,.0f} (live {rec.mean():+,.0f})"
          f" | exact replays of the live result: {(np.abs(m - rec) < 1).sum()} -> {fn}", flush=True)
