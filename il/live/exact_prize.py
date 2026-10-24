"""Exact (slot-level) market re-scoring of sell-timing planners on live games  (Sep 25).

Engine rule (engine/src_v2/sim_trace.hpp process_market): orders run slot by slot; within a slot the two farms' units
trade in lockstep rounds - each round BOTH farms' next unit is priced at the same pre-round inventory, then each
committed unit adds 1 to the inventory (not for $1 sales); a farm that runs out drops out and the other continues.
Our production and the rival's orders (slot, executed units) are held fixed; only when OUR units of each of the 7
sale-only products are sold changes.  Planners (all within the unit's day, or as late as the agent really held it):
  hindsight  lowest true pre-market inventory from others (perfect foresight of the rival)
  forecast   knowable at the unit's arrival: market inventory then, the town's fixed consumption, the rival repeating
             yesterday's sales at the same hours
Every schedule - including the agent's ACTUAL one (with its real slots, as a check against the engine's receipts) - is
scored by exact slot-level simulation; planned sales go in slot 0.
Usage: exact_prize.py <label>:<tape_dir>[:<rows_json>] ...     env NP
"""
import sys, os, json, gzip, glob
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
PURE = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
LAST = 718


def replay(path):
    from harness import kagsim_trace, PASS
    d = json.load(gzip.open(path, "rt")); A = d["actions"]; me = int(d["seat"])
    g = kagsim_trace.Game(int(d["seed"]), 720, True)
    inv, shed, drain, ours, rival, eng = [], [], [], [], [], {p: [0.0, 0.0] for p in PURE}
    for t in range(720):
        a0 = A[0][t] if isinstance(A[0][t], dict) else dict(PASS); a1 = A[1][t] if isinstance(A[1][t], dict) else dict(PASS)
        g.step(a0, a1); tr = g.trace_step(g.trace_len() - 1)
        inv.append(tr["after_actor"]["minv"][:9]); shed.append(tr["after_actor"]["shed"][me][:9])
        drain.append([max(0, a - b) for a, b in zip(tr["after_market"]["minv"][:9], tr["after_town"]["minv"][:9])])
        o, r = {}, {}
        for rc in tr.get("receipts", []):
            if rc.get("op") == "SELL" and rc.get("executed") and rc.get("item") in PURE:
                side = o if rc["seat"] == me else r
                side.setdefault(rc["item"], []).append((int(rc["slot"]), int(rc["executed"])))
                eng[rc["item"]][0 if rc["seat"] == me else 1] += (rc.get("money_after") or 0) - (rc.get("money_before") or 0)
        ours.append(o); rival.append(r)
        if g.done:
            break
    return dict(inv=np.array(inv, np.int64), shed=np.array(shed, np.int64), drain=np.array(drain, np.int64),
                ours=ours, rival=rival, eng=eng, T=len(inv))


def turn(I, ours, rival, price):
    """Exact slot-level trading of one product in one turn -> (our revenue, rival revenue, inventory after)."""
    slots = sorted({s for s, _ in ours} | {s for s, _ in rival}); ro = rr = 0.0
    for s in slots:
        a = sum(q for sl, q in ours if sl == s); b = sum(q for sl, q in rival if sl == s)
        while a > 0 or b > 0:
            pr = price(I); add = 0
            if a > 0:
                ro += pr; a -= 1; add += pr > 1
            if b > 0:
                rr += pr; b -= 1; add += pr > 1
            I += add
    return ro, rr, I


def analyse(path):
    import features6 as F6
    try:
        R = replay(path)
    except Exception:
        return None
    T = R["T"]; out = {}
    for p in PURE:
        j = ITEMS.index(p)
        o_turn = [R["ours"][t].get(p, []) for t in range(T)]; r_turn = [R["rival"][t].get(p, []) for t in range(T)]
        x_act = np.array([sum(q for _, q in L) for L in o_turn]); r = np.array([sum(q for _, q in L) for L in r_turn])
        C_prev = np.concatenate([[0], np.cumsum(x_act)[:-1]])
        B = R["inv"][:T, j] - C_prev                                       # others-only pre-market inventory path
        avail = np.maximum.accumulate(C_prev + R["shed"][:T, j]); N = int(avail[-1])
        lo = int(B.min()) - 200; hi = int(B.max()) + N + int(r.max()) * 2 + 200
        TAB = np.array([F6.price(p, L) for L in range(lo, hi + 1)], float)
        price = lambda L: TAB[min(max(int(L) - lo, 0), len(TAB) - 1)]
        def score(sched):                                                  # sched[t] = list of (slot, units) for us
            own = riv = 0.0; C = 0
            for t in range(T):
                ro, rr, _ = turn(B[t] + C, sched[t], r_turn[t], price)
                own += ro; riv += rr; C += sum(q for _, q in sched[t])
            return own, riv
        own_act, riv_act = score(o_turn)
        res = dict(eng=tuple(R["eng"][p]), act=(own_act, riv_act))
        if N == 0:
            out[p] = res; continue
        tau = np.searchsorted(avail, np.arange(N) + 1); last = min(T - 1, LAST)
        t_act = np.repeat(np.arange(T), x_act)
        if len(t_act) < N:
            t_act = np.concatenate([t_act, np.full(N - len(t_act), last)])
        t_act = t_act[:N]
        lim = np.maximum(np.minimum((tau // 24) * 24 + 23, last), t_act)
        dr = R["drain"][:T, j]; r_fc = np.concatenate([np.zeros(24, int), r[:-24]])[:T]
        Bp = B + r
        picks = {"hindsight": [], "forecast": [], "town": []}
        for n in range(N):
            a, b = int(min(tau[n], last)), int(lim[n])
            picks["hindsight"].append(a + int(np.argmin(Bp[a:b + 1])))
            cum_dr = np.concatenate([[0], np.cumsum(dr[a:b])]); cum_rf = np.concatenate([[0], np.cumsum(r_fc[a:b])])
            picks["forecast"].append(a + int(np.argmin(B[a] + cum_rf - cum_dr + r_fc[a:b + 1])))
            picks["town"].append(a + int(np.argmin(B[a] - cum_dr)))
        for k, tn in picks.items():
            cnt = np.bincount(np.array(tn), minlength=T)
            sched = [[(0, int(cnt[t]))] if cnt[t] else [] for t in range(T)]
            own, riv = score(sched)
            res[k] = (own - own_act, riv - riv_act)
        # the agent's own timing but every sale moved to slot 0 (isolates the slot effect from the timing effect)
        own0, riv0 = score([[(0, int(x_act[t]))] if x_act[t] else [] for t in range(T)])
        res["slot0"] = (own0 - own_act, riv0 - riv_act)
        out[p] = res
    return out


def report(label, rows):
    rows = [x for x in rows if x]; n = len(rows)
    E = np.zeros(2); A = np.zeros(2); M = {k: np.zeros(2) for k in ("hindsight", "forecast", "town", "slot0")}
    Mp = {p: np.zeros(2) for p in PURE}
    for x in rows:
        for p, v in x.items():
            E += v["eng"]; A += v["act"]
            for k in M:
                if k in v:
                    M[k] += v[k]
            if "forecast" in v:
                Mp[p] += v["forecast"]
    print(f"\n=== {label}: {n} games", flush=True)
    print(f"  exactness check (7 products, $/game): ours engine {E[0]/n:,.0f} vs model {A[0]/n:,.0f} | rival engine {E[1]/n:,.0f} vs model {A[1]/n:,.0f}")
    for k, lab in (("slot0", "same timing, all sales moved to slot 0"), ("hindsight", "hindsight timing (perfect foresight)"),
                   ("forecast", "FORECAST timing (knowable)"), ("town", "town schedule only")):
        own, riv = M[k] / n
        print(f"  {lab:40s} own {own:+8,.0f}  rival {riv:+8,.0f}  MARGIN {own - riv:+8,.0f} $/game")
    print("  forecast margin by product: " + ", ".join(f"{p} {(Mp[p][0] - Mp[p][1])/n:+,.0f}" for p in PURE))


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "9"))) as pool:
        for spec in sys.argv[1:]:
            parts = spec.split(":"); label, tdir = parts[0], parts[1]
            paths = sorted(glob.glob(f"{tdir}/*.json.gz"))
            if len(parts) > 2:
                keep = {str(x["ep"]) for x in json.load(open(parts[2])) if x.get("res") in ("W", "L", "T")}
                paths = [q for q in paths if os.path.basename(q).split(".")[0] in keep]
            report(label, pool.map(analyse, paths, chunksize=2))
