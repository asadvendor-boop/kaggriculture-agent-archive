"""Benchmark several checkpoints on the 464 held-out ladder worlds through ONE shared pool (dynamic scheduling: small
lockstep jobs of LOCKSTEP games, pool.imap_unordered, so every core stays busy until the last game).  Writes the same
il/logs/ladder_<tag>_test.json files and summary lines as ladder.py, so results are directly comparable.

Usage: bench_multi.py <ckpt>[:team[:tau]] ...      env NW (default 30), LOCKSTEP (6), TAGX (default _r3_tau<tau>),
       PREFIX_BOT (scripted opening in our seat, e.g. agents/cand/DSMR0.py) + PREFIX_STEPS (default 144),
       MIDNIGHT / FINALFIX / HOMING as ladder.py (default 1)
"""
import sys, os, json, collections
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
R = "/Users/asad.ali/kaggriculture"


def main():
    from ladder import worlds, summary
    from live_rollout import run_job
    W = worlds("test"); L = int(os.environ.get("LOCKSTEP", "6"))
    cands = []
    for spec in sys.argv[1:]:
        parts = spec.split(":"); ck = parts[0]
        team = int(parts[1]) if len(parts) > 1 else 5; tau = float(parts[2]) if len(parts) > 2 else 0.5
        tag = os.path.basename(ck).replace(".pt", "") + f"_t{team}" + os.environ.get("TAGX", f"_r3_tau{tau}")
        cands.append((ck, team, tau, tag))
    PRE = (os.environ["PREFIX_BOT"], int(os.environ.get("PREFIX_STEPS", "144"))) if os.environ.get("PREFIX_BOT") else None
    games = [(w["seed"], w["seat"], "TAPE:" + w["path"]) + ((PRE,) if PRE else ()) for w in W]   # PREFIX_BOT plays our first steps
    jobs = []
    for ci, (ck, team, tau, tag) in enumerate(cands):
        for k in range(0, len(games), L):
            jobs.append(dict(weights=ck, team=team, tau=tau, T=0.2, ks=10.0, greedy=True, seed=k, games=games[k:k + L], cand=ci))
    got = collections.defaultdict(list)
    with mp.get_context("spawn").Pool(int(os.environ.get("NW", "30")), maxtasksperchild=40) as pool:
        for ci, res in pool.imap_unordered(_run, jobs):
            got[ci] += res
            if len(got[ci]) == len(games):
                ck, team, tau, tag = cands[ci]
                by = {r["opp"][5:]: r for r in got[ci]}
                rows = [dict(ep=w["ep"], opp=w["opp"], opp_rating=w["opp_rating"], margin=by[w["path"]]["margin"], bank=by[w["path"]]["bank"],
                             opp_bank=by[w["path"]]["opp_bank"], rec_margin=w["rec_margin"], opp_kept=by[w["path"]]["opp_bank"] >= 0.95 * w["rec_opp"],
                             src=w["src"]) for w in W]
                json.dump(rows, open(f"{R}/il/logs/ladder_{tag}_test.json", "w"), ensure_ascii=False)
                print(summary(tag, rows), flush=True)


def _run(job):
    from live_rollout import run_job
    return job["cand"], run_job({k: v for k, v in job.items() if k != "cand"})


if __name__ == "__main__":
    main()
