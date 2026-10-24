"""Market prize: how much sales revenue (and margin) an agent left on the table by WHEN it sold, in real games.

For each recorded game (both seats' actions replayed exactly with the C++ engine, trace on) and each of the 7 products
that are only ever sold (never consumed on the farm), our production and the rival's orders are held fixed and only
the timing of OUR sells changes.  Engine facts used (verified on receipts): a sell of k units at market inventory L
earns price(L), price(L+1), ... price(L+k-1) (features6.price); the town's consumption is a fixed schedule, so the
inventory we face at turn t is B(t) + (our cumulative sells), B = the inventory path without our sells.  The rival's
same-turn units are priced first (a conservative choice, used for the actual schedule too, so the comparison is fair).

Schedules (per product; the n-th unit we sell may be sold at any turn >= the turn it was available):
  actual   what the agent did (model revenue, checked against the engine's receipts)
  sameday  best turn from arrival to the end of that day (or as late as the agent really held it)  -> realistic
  hold     best turn from arrival to the end of the game (unbounded holding; shed overflow reported) -> upper bound
  margin   like hold, but also counting what our unit does to the rival's later sale prices          -> margin view
Output per agent: $/game gains in own revenue, rival revenue and margin, plus the model check.

Usage: market_prize.py <label>:<tape_dir>[:<rows_json>] ...   (rows_json: keep only games listed there)   env NP
"""
import sys, os, json, gzip, glob
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
PURE = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
LAST = 718                                              # the engine's last acting turn
ORDER = os.environ.get("ORDER", "rival_first")          # same-turn pricing: rival_first (conservative) | ours_first


def replay(path):
    from harness import kagsim_trace, PASS
    d = json.load(gzip.open(path, "rt")); A = d["actions"]; me = int(d["seat"])
    g = kagsim_trace.Game(int(d["seed"]), 720, True)
    T = 0; pre_inv, pre_shed, ours, rival, rev_eng, drain = [], [], [], [], np.zeros(9), []
    for t in range(720):
        a0 = A[0][t] if isinstance(A[0][t], dict) else dict(PASS); a1 = A[1][t] if isinstance(A[1][t], dict) else dict(PASS)
        g.step(a0, a1); tr = g.trace_step(g.trace_len() - 1)
        pre_inv.append(tr["after_actor"]["minv"][:9]); pre_shed.append(tr["after_actor"]["shed"][me][:9])
        drain.append([max(0, a - b) for a, b in zip(tr["after_market"]["minv"][:9], tr["after_town"]["minv"][:9])])
        o, r = np.zeros(9, int), np.zeros(9, int)
        for rc in tr.get("receipts", []):
            if rc.get("op") == "SELL" and rc.get("executed"):
                j = ITEMS.index(rc["item"])
                if rc["seat"] == me:
                    o[j] += rc["executed"]; rev_eng[j] += (rc.get("money_after") or 0) - (rc.get("money_before") or 0)
                else:
                    r[j] += rc["executed"]
        ours.append(o); rival.append(r); T = t + 1
        if g.done:
            break
    return dict(inv=np.array(pre_inv, np.int64), shed=np.array(pre_shed, np.int64), ours=np.array(ours), rival=np.array(rival), drain=np.array(drain, np.int64),
                rev_eng=rev_eng, T=T, me=me, bank=(g.reward(me), g.reward(1 - me)))


def analyse(item):
    import features6 as F6
    path = item
    try:
        R = replay(path)
    except Exception as ex:
        return None
    T = R["T"]; out = {}
    for p in PURE:
        j = ITEMS.index(p)
        x_act = R["ours"][:T, j]; r = R["rival"][:T, j]
        C_prev = np.concatenate([[0], np.cumsum(x_act)[:-1]])          # our units sold before turn t
        B = R["inv"][:T, j] - C_prev                                    # inventory path without any of our sells
        Bp = B + r if ORDER == "rival_first" else B.copy()             # rival's same-turn units priced first (or ours)
        avail = np.maximum.accumulate(C_prev + R["shed"][:T, j])        # units that had reached our shed by turn t
        N = int(avail[-1])
        if N == 0:
            continue
        lo = int(min(B.min(), Bp.min())) - 5; hi = int(max(B.max(), Bp.max())) + N + int(r.max()) + 5
        TAB = np.array([F6.price(p, L) for L in range(lo, hi + 1)], float)     # exact prices on the needed range
        price = lambda L: TAB[int(L) - lo]
        Pv = lambda L: TAB[np.asarray(L, int) - lo]
        tau = np.searchsorted(avail, np.arange(N) + 1)                  # first turn unit n (0-based) was available
        t_act = np.repeat(np.arange(T), x_act)                          # turn the agent sold its n-th unit
        if len(t_act) < N:                                              # units never sold (overflow/end): sell at the end
            t_act = np.concatenate([t_act, np.full(N - len(t_act), min(T - 1, LAST))])
        t_act = t_act[:N]
        last = min(T - 1, LAST)
        # suffix argmin of Bp (earliest on ties): best turn to sell at or after t
        idx = np.arange(T); best = np.empty(T, int); cur = last
        for t in range(last, -1, -1):
            if Bp[t] <= Bp[cur]:
                cur = t
            best[t] = cur
        def revenue(tn):
            tn = np.sort(tn)
            return float(Pv(Bp[tn] + np.arange(len(tn))).sum())
        def rival_rev(tn):
            tn = np.sort(tn)
            C = np.searchsorted(tn, np.arange(T), side="left" if ORDER == "rival_first" else "right")   # ours ahead of theirs
            return float(sum(sum(price(B[t] + C[t] + k) for k in range(int(r[t]))) for t in range(T) if r[t]))
        t_hold = best[np.minimum(tau, last)]
        day_end = np.minimum((tau // 24) * 24 + 23, last)
        lim = np.maximum(day_end, t_act)
        t_same = np.empty(N, int)
        for n in range(N):                                               # best turn in [tau, lim]
            a, b = int(min(tau[n], last)), int(lim[n])
            t_same[n] = a + int(np.argmin(Bp[a:b + 1]))
        # margin view: harm(t) = what one extra unit in the inventory before turn t costs the rival over the rest of the game
        harm = np.zeros(T + 1)
        for t in range(T - 1, -1, -1):
            h = price(B[t] + x_act[:t].sum()) - price(B[t] + x_act[:t].sum() + r[t]) if r[t] else 0.0
            harm[t] = harm[t + 1] + h
        t_marg = np.empty(N, int)
        for n in range(N):                                               # our unit at t is priced after the rival's
            a = int(min(tau[n], last))                                   # same-turn units: it only hurts later sells
            vals = Pv(Bp[a:last + 1] + n) + harm[a + 1:last + 2]
            t_marg[n] = a + int(np.argmax(vals))
        # knowable-only planner: pick the turn by the TOWN's consumption alone (no rival knowledge), within the same window
        Btown = R["inv"][0, j] - np.concatenate([[0], np.cumsum(R["drain"][:T - 1, j])])     # inventory path from drain only
        t_town = np.empty(N, int)
        for n in range(N):
            a, b = int(min(tau[n], last)), int(lim[n])
            t_town[n] = a + int(np.argmin(Btown[a:b + 1]))
        base_own, base_riv = revenue(t_act), rival_rev(t_act)
        res = dict(eng=float(R["rev_eng"][j]), act=base_own, N=N, sold=int(x_act.sum()))
        for name, tn in (("sameday", t_same), ("town", t_town), ("hold", t_hold), ("margin", t_marg)):
            own, riv = revenue(tn), rival_rev(tn)
            tn_s = np.sort(tn); held = avail - np.searchsorted(tn_s, np.arange(T), side="right")
            held_act = avail - np.cumsum(x_act)
            res[name] = (own - base_own, riv - base_riv, int(max(0, (held - held_act)[23::24].max())) if T > 24 else 0)
        out[p] = res
    return dict(path=path, bank=R["bank"], prod=out)


def report(label, rows):
    rows = [r for r in rows if r]
    n = len(rows); print(f"\n=== {label}: {n} games", flush=True)
    tot = {k: np.zeros(3) for k in ("sameday", "town", "hold", "margin")}; eng = act = 0.0
    per = {p: {k: np.zeros(3) for k in tot} for p in PURE}
    for r in rows:
        for p, v in r["prod"].items():
            eng += v["eng"]; act += v["act"]
            for k in tot:
                tot[k][:2] += np.array(v[k][:2]); per[p][k][:2] += np.array(v[k][:2]); tot[k][2] = max(tot[k][2], v[k][2])
    print(f"  model check: 7-product sales revenue per game, engine {eng/n:,.0f} vs model (rival first) {act/n:,.0f}")
    print(f"  (same-turn order: {ORDER})")
    for k, lab in (("sameday", "hindsight: best turn within the day"), ("town", "KNOWABLE: town schedule only, same day"),
                   ("hold", "upper bound: hold any time"), ("margin", "margin-optimal timing")):
        d_own, d_riv = tot[k][0] / n, tot[k][1] / n
        print(f"  {lab:36s} own {d_own:+8,.0f}  rival {d_riv:+8,.0f}  MARGIN {d_own - d_riv:+8,.0f} $/game   (max extra units held overnight in one product: {int(tot[k][2])})")
    print("  within-day hindsight gain by product ($/game own):", ", ".join(f"{p} {per[p]['sameday'][0]/n:+,.0f}" for p in PURE))
    print("  knowable (town-only) gain by product ($/game margin):", ", ".join(f"{p} {(per[p]['town'][0]-per[p]['town'][1])/n:+,.0f}" for p in PURE))


if __name__ == "__main__":
    specs = sys.argv[1:]
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "9"))) as pool:
        for spec in specs:
            parts = spec.split(":"); label, tdir = parts[0], parts[1]
            paths = sorted(glob.glob(f"{tdir}/*.json.gz"))
            if len(parts) > 2:
                keep = {str(r["ep"]) for r in json.load(open(parts[2])) if r.get("res") in ("W", "L", "T")}
                paths = [p for p in paths if os.path.basename(p).split(".")[0] in keep]
            report(label, pool.map(analyse, paths, chunksize=2))
