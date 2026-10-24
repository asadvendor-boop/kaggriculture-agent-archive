"""Leader benchmark: our agent takes one leader's seat in a recent leader-vs-leader game (world pinned to the recorded
game) and plays against the OTHER leader's recorded actions.  Harder than the ladder benchmark (whose opponents are
our own ladder opponents): it asks whether we beat the top teams' recorded play, and whether we do better than the
leader we replaced did in the same world.

Usage: top_bench.py <cand.pt | scripted spec> [...]      env: DAY (default 2026-09-24), N (300), NW (4), TEAM, TAU,
MIDNIGHT/FINALFIX/HOMING as ladder.py, TAGX, FRESH=1 to ignore cached results."""
import sys, os, json, gzip, random, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
R = "/Users/asad.ali/kaggriculture"
LEAD = {"DSM", "M & M & P & Q", "DECEM", "Vadim Vasilenko", "Majkel1337", "Fourth Quadrant", "Unknown Mother-Goose",
        "tetsuya & yuanzhe & guoqi", "Kaggledew Valley 🏆", "mtmr_s1", "吃白饭的大肥鱼", "SpaTaro", "Artem The Farmer 🍅",
        "Boey", "THIRD FARM CLUB", "nah id win", "Azat Akhtyamov", "Snorlax"}


def worlds():
    day = os.environ.get("DAY", "2026-09-24"); n = int(os.environ.get("N", "300"))
    fn = f"{R}/il/live/top_worlds_{day}_{n}.json"
    if os.path.exists(fn):
        return json.load(open(fn))
    out = []
    for d in ("data/tapes_daily", "data/tapes_pull"):
        for ep, v in json.load(open(f"{R}/{d}/index.json")).items():
            t = v.get("teams") or []; rw = v.get("rewards") or []
            if v.get("day") != day or len(t) != 2 or len(rw) != 2 or not (t[0] in LEAD and t[1] in LEAD):
                continue
            p = f"{R}/{d}/{ep}.json.gz"
            if os.path.exists(p):
                for s in (0, 1):
                    out.append(dict(path=p, ep=int(ep), seat=s, replaced=t[s], opp=t[1 - s], rec_margin=rw[s] - rw[1 - s], rec_opp=rw[1 - s]))
    out.sort(key=lambda w: (w["ep"], w["seat"])); random.Random(17).shuffle(out); out = out[:n]
    json.dump(out, open(fn, "w"), ensure_ascii=False)
    return out


def evaluate(cand):
    from live_rollout import run_job, run_scripted
    W = worlds(); nw = int(os.environ.get("NW", "4"))
    games = [(0, w["seat"], "TAPE:" + w["path"]) for w in W]
    neural = cand.endswith(".pt"); jobs = []
    for k in range(nw):
        g = games[k::nw]
        if neural:
            tau_ = json.loads(os.environ["TAU_HEADS"]) if os.environ.get("TAU_HEADS") else float(os.environ.get("TAU", "0.6"))
            jobs.append(dict(weights=cand, team=int(os.environ.get("TEAM", "5")), tau=tau_, T=0.2, ks=10.0, greedy=True, seed=k, games=g))
        else:
            jobs.append(dict(us=cand, games=g))
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = sum(pool.map(run_job if neural else run_scripted, jobs), [])
    by = {(r["opp"][5:], r["seat"]): r for r in res}
    return [dict(w, margin=by[(w["path"], w["seat"])]["margin"], bank=by[(w["path"], w["seat"])]["bank"],
                 opp_bank=by[(w["path"], w["seat"])]["opp_bank"]) for w in W]


def summary(tag, rows):
    m = np.array([r["margin"] for r in rows]); rec = np.array([r["rec_margin"] for r in rows])
    kept = np.array([r["opp_bank"] >= 0.95 * r["rec_opp"] for r in rows]); mk = m[kept]
    print(f"{tag:34s} n {len(m)}  W {(m > 0).sum()}-{(m < 0).sum()}  mean margin {m.mean():+8,.0f} (se {m.std() / np.sqrt(len(m)):,.0f})"
          f"  | beat the replaced leader's own result in {(m > rec).mean():.0%} (leaders' own W {(rec > 0).mean():.0%}, margin {rec.mean():+,.0f})"
          f"  | opp kept 95% of its score in {kept.mean():.0%}: there W {(mk > 0).mean():.0%} margin {mk.mean():+,.0f}", flush=True)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["opp"]].append(r["margin"])
    print("   vs " + ", ".join(f"{o[:12]} {np.mean(np.array(v) > 0):.0%}/{len(v)}" for o, v in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]), flush=True)


if __name__ == "__main__":
    for cand in sys.argv[1:]:
        tag = os.path.basename(cand).replace(".pt", "").replace("/", "_") + (f"_t{os.environ['TEAM']}" if "TEAM" in os.environ and cand.endswith(".pt") else "") + os.environ.get("TAGX", "")
        fn = f"{R}/il/logs/top_{os.environ.get('DAY', '2026-09-24')}_{tag}.json"
        rows = json.load(open(fn)) if os.path.exists(fn) and not os.environ.get("FRESH") else evaluate(cand)
        json.dump(rows, open(fn, "w"), ensure_ascii=False)
        summary(tag, rows)
