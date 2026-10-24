"""Ladder worlds = live ladder games of our submissions (tape_sub format: seat, seed, opp, opp_rating, res), replayed with
the world pinned: our bot plays the seat our submission played, the opponent repeats its recorded moves.
A local benchmark whose opponents are the real ladder population (not only our own bot family).

  ladder.py list                         -> il/live/ladder_worlds.json (dedup by ep; ep % 2 == 1 -> held-out)
  ladder.py eval <split> <cand> [...]    -> W/L, mean margin, by opponent-rating band; split = test|train|all
     cand = checkpoint path (neural, env TEAM/TAU) or a harness bot name/path (scripted)
Results cached in il/logs/ladder_<cand>_<split>.json.
"""
import sys, os, json, gzip, glob, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
R = "/Users/asad.ali/kaggriculture"
DIRS = ["data/tapes_neural", "data/tapes_race", "data/tapes_stack1", "data/tapes_ladder"]
WF = f"{R}/il/live/ladder_worlds.json"


def do_list():
    seen, out = set(), []
    for d_ in DIRS:
        for fn in sorted(glob.glob(f"{R}/{d_}/*.json.gz")):
            try:
                d = json.load(gzip.open(fn, "rt"))
            except Exception:
                continue
            if d["ep"] in seen or "seed" not in d or d.get("res") not in ("W", "L", "T"):
                continue
            seen.add(d["ep"]); us = d["seat"]
            out.append(dict(path=fn, ep=d["ep"], seat=us, seed=int(d["seed"]), opp=d.get("opp"), opp_rating=d.get("opp_rating") or 0,
                            rec_margin=d["rewards"][us] - d["rewards"][1 - us], rec_opp=d["rewards"][1 - us], src=d_.split("/")[-1]))
    json.dump(out, open(WF, "w"), ensure_ascii=False)
    te = [w for w in out if w["ep"] % 2 == 1]
    print(f"{len(out)} ladder worlds ({len(te)} held-out); by source:", dict(collections.Counter(w["src"] for w in out)))


def worlds(split):
    W = json.load(open(WF))
    return W if split == "all" else [w for w in W if (w["ep"] % 2 == 1) == (split == "test")]


def evaluate(cand, split, nw=None):
    from live_rollout import run_job, run_scripted
    W = worlds(split); nw = nw or int(os.environ.get("NW", "8"))
    games = [(w["seed"], w["seat"], "TAPE:" + w["path"]) for w in W]
    neural = cand.endswith(".pt")
    jobs = []
    for k in range(nw):
        g = games[k::nw]
        if not g:
            continue
        if neural:
            tau_ = json.loads(os.environ["TAU_HEADS"]) if os.environ.get("TAU_HEADS") else float(os.environ.get("TAU", "0.6"))
            jobs.append(dict(weights=cand, team=int(os.environ.get("TEAM", "5")), tau=tau_,
                             T=0.2, ks=10.0, greedy=True, seed=k, games=g))
        else:
            jobs.append(dict(us=cand, games=g))
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = sum(pool.map(run_job if neural else run_scripted, jobs), [])
    by = {r["opp"][5:]: r for r in res}
    rows = []
    for w in W:
        r = by[w["path"]]
        rows.append(dict(ep=w["ep"], opp=w["opp"], opp_rating=w["opp_rating"], margin=r["margin"], bank=r["bank"], opp_bank=r["opp_bank"],
                         rec_margin=w["rec_margin"], opp_kept=r["opp_bank"] >= 0.95 * w["rec_opp"], src=w["src"]))
    return rows


def summary(label, rows):
    m = np.array([r["margin"] for r in rows])
    kept = np.mean([r["opp_kept"] for r in rows])
    line = f"{label:28s} n {len(m)}  W {(m > 0).sum()}-{(m < 0).sum()}  mean margin {m.mean():+8,.0f} (se {m.std() / np.sqrt(len(m)):,.0f})  opp kept 95% of its score {kept:.0%}"
    bands = collections.defaultdict(list)
    for r in rows:
        bands[min(2400, int(r["opp_rating"] // 400 * 400))].append(r["margin"])
    line += "  | " + " ".join(f"{b}+: {sum(x > 0 for x in v)}/{len(v)}" for b, v in sorted(bands.items()))
    return line


if __name__ == "__main__":
    if sys.argv[1] == "list":
        do_list()
    else:
        split = sys.argv[2]
        for cand in sys.argv[3:]:
            tag = os.path.basename(cand).replace(".pt", "").replace("/", "_") + (f"_t{os.environ['TEAM']}" if "TEAM" in os.environ and cand.endswith(".pt") else "") + os.environ.get("TAGX", "")
            fn = f"{R}/il/logs/ladder_{tag}_{split}.json"
            rows = json.load(open(fn)) if os.path.exists(fn) and not os.environ.get("FRESH") else evaluate(cand, split)
            json.dump(rows, open(fn, "w"), ensure_ascii=False)
            print(summary(tag, rows), flush=True)
            if len(sys.argv) > 4 or True:
                rec = np.array([r["rec_margin"] for r in rows])
                print(f"{'  (recorded live result)':28s} n {len(rec)}  W {(rec > 0).sum()}-{(rec < 0).sum()}  mean margin {rec.mean():+8,.0f}", flush=True) if cand == sys.argv[3] else None
