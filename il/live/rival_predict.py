"""How predictable are the rival's sale times, and what does a forecast-based sell planner earn?  (Sep 25)

Replays live games exactly (market_prize.replay) and, for the 7 sale-only products:
 1. regularity: a rival sale of product p at turn t is "foreseen" if the rival also sold p within +-1 turn of t-24
    (same hour yesterday).  recall = foreseen units / all rival units; precision = share of yesterday's sale hours
    that were followed by a sale today within +-1 turn.
 2. money: the within-day re-timing of OUR sells (as in market_prize "sameday": each unit sold between its arrival
    and the end of its day, or as late as the agent really held it) chosen three ways and all scored on what really
    happened:  hindsight (true rival path), forecast (knowable at the unit's arrival: market inventory then, the
    town's fixed consumption, and the rival repeating yesterday's sales at the same hours), and town-only.
Usage: rival_predict.py <label>:<tape_dir>[:<rows_json>] ...    env ORDER (rival_first | ours_first), NP
"""
import sys, os, json, glob
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import market_prize as MP
PURE, ITEMS, LAST = MP.PURE, MP.ITEMS, MP.LAST
ORDER = os.environ.get("ORDER", "rival_first")


def analyse(path):
    import features6 as F6
    try:
        R = MP.replay(path)
    except Exception:
        return None
    T = R["T"]; out = {}
    for p in PURE:
        j = ITEMS.index(p)
        x_act = R["ours"][:T, j]; r = R["rival"][:T, j]; dr = R["drain"][:T, j]
        # ---- 1. regularity of the rival's sale hours
        fore = sum(int(r[t]) for t in range(24, T) if r[t] and r[max(0, t - 25):t - 22].sum() > 0)
        prev_hours = [t for t in range(24, T) if r[t - 24] > 0]
        prec_hit = sum(1 for t in prev_hours if r[max(0, t - 1):t + 2].sum() > 0)
        stats = dict(units=int(r[24:].sum()), fore=fore, prev=len(prev_hours), prec_hit=prec_hit)
        # ---- 2. money
        C_prev = np.concatenate([[0], np.cumsum(x_act)[:-1]])
        B = R["inv"][:T, j] - C_prev
        Bp = B + r if ORDER == "rival_first" else B.copy()
        avail = np.maximum.accumulate(C_prev + R["shed"][:T, j]); N = int(avail[-1])
        if N == 0:
            out[p] = dict(stats=stats); continue
        lo = int(B.min()) - int(r.max()) - 60; hi = int(B.max()) + N + int(r.max()) + 60
        TAB = np.array([F6.price(p, L) for L in range(lo, hi + 1)], float)
        Pv = lambda L: TAB[np.clip(np.asarray(L, int) - lo, 0, len(TAB) - 1)]
        tau = np.searchsorted(avail, np.arange(N) + 1)
        t_act = np.repeat(np.arange(T), x_act)
        last = min(T - 1, LAST)
        if len(t_act) < N:
            t_act = np.concatenate([t_act, np.full(N - len(t_act), last)])
        t_act = t_act[:N]
        lim = np.maximum(np.minimum((tau // 24) * 24 + 23, last), t_act)
        r_fc = np.concatenate([np.zeros(24, int), r[:-24]])[:T]                  # rival repeats yesterday's hours
        def revenue(tn):
            tn = np.sort(tn); return float(Pv(Bp[tn] + np.arange(len(tn))).sum())
        def rival_rev(tn):
            tn = np.sort(tn)
            C = np.searchsorted(tn, np.arange(T), side="left" if ORDER == "rival_first" else "right")
            return float(sum(Pv(B[t] + C[t] + np.arange(int(r[t]))).sum() for t in range(T) if r[t]))
        pick = {"hindsight": np.empty(N, int), "forecast": np.empty(N, int), "town": np.empty(N, int)}
        for n in range(N):
            a, b = int(min(tau[n], last)), int(lim[n])
            seg = slice(a, b + 1)
            pick["hindsight"][n] = a + int(np.argmin(Bp[seg]))
            steps = np.arange(a, b + 1)
            cum_dr = np.concatenate([[0], np.cumsum(dr[a:b])])                 # town consumption from a to t (known)
            cum_rf = np.concatenate([[0], np.cumsum(r_fc[a:b])])               # forecast rival sales from a to t
            F = B[a] + cum_rf - cum_dr + (r_fc[steps] if ORDER == "rival_first" else 0)
            pick["forecast"][n] = a + int(np.argmin(F))
            pick["town"][n] = a + int(np.argmin(B[a] - cum_dr))
        base_own, base_riv = revenue(t_act), rival_rev(t_act)
        res = dict(stats=stats)
        for k, tn in pick.items():
            res[k] = (revenue(tn) - base_own, rival_rev(tn) - base_riv)
        out[p] = res
    return out


def report(label, rows):
    rows = [x for x in rows if x]; n = len(rows)
    print(f"\n=== {label}: {n} games (same-turn order {ORDER})", flush=True)
    S = {p: np.zeros(4) for p in PURE}; M = {k: np.zeros(2) for k in ("hindsight", "forecast", "town")}
    Mp = {p: {k: np.zeros(2) for k in M} for p in PURE}
    for x in rows:
        for p, v in x.items():
            st = v["stats"]; S[p] += [st["units"], st["fore"], st["prev"], st["prec_hit"]]
            for k in M:
                if k in v:
                    M[k] += v[k]; Mp[p][k] += v[k]
    tot = sum(S.values())
    print(f"  rival sale regularity (same hour yesterday +-1): recall {tot[1]/max(1,tot[0]):.0%} of rival units, precision {tot[3]/max(1,tot[2]):.0%}")
    print("   by product: " + ", ".join(f"{p} {S[p][1]/max(1,S[p][0]):.0%}/{S[p][3]/max(1,S[p][2]):.0%}" for p in PURE))
    for k, lab in (("hindsight", "hindsight (perfect foresight)"), ("forecast", "FORECAST (yesterday's hours + town)"), ("town", "town schedule only")):
        own, riv = M[k] / n
        print(f"  {lab:38s} own {own:+8,.0f}  rival {riv:+8,.0f}  MARGIN {own - riv:+8,.0f} $/game")
    print("  forecast margin by product: " + ", ".join(f"{p} {(Mp[p]['forecast'][0]-Mp[p]['forecast'][1])/n:+,.0f}" for p in PURE))


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "9"))) as pool:
        for spec in sys.argv[1:]:
            parts = spec.split(":"); label, tdir = parts[0], parts[1]
            paths = sorted(glob.glob(f"{tdir}/*.json.gz"))
            if len(parts) > 2:
                keep = {str(x["ep"]) for x in json.load(open(parts[2])) if x.get("res") in ("W", "L", "T")}
                paths = [q for q in paths if os.path.basename(q).split(".")[0] in keep]
            report(label, pool.map(analyse, paths, chunksize=2))
