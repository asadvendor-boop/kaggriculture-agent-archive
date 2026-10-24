"""Randomised test of the closed-form JAX market phase against the literal lockstep port (market_ref.py).

Scenarios are biased toward the hard cases: both players on the same item in the same slot (SELL/SELL, BUY/BUY,
SELL vs BUY), inventories at / around every price floor (incl. FERTILIZER above its floor = $1 sales, which is the
two-phase SELL-vs-BUY case), tiny cash, full sheds, huge counts, HIRE at the unit cap, BUY_LAND exhaustion.
usage: env1.sh test_market.py [--n 20000] [--seed 0]
"""
import argparse
import copy
import random
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import consts as C
import engine as E
import market_ref as MR

FLOOR = C.build_price_tables()[3]


def rand_inv(r, item):
    u = r.random()
    if u < 0.35 and FLOOR[item] < C.INV_HI:
        return int(FLOOR[item]) + r.randint(-6, 3)
    if u < 0.5:
        return C.I0 + r.randint(-5, 5)
    if u < 0.7:
        return C.I0 + r.randint(-3000, 800)
    if u < 0.8 and item in (C.WHEAT, C.FERTILIZER):
        return int(FLOOR[item]) + r.randint(-150, 3) if FLOOR[item] < C.INV_HI else C.I0 + r.randint(-3000, 3000)
    return C.I0 + r.randint(-400, 400)


def rand_count(r):
    return r.choice([1, 1, 2, 3, 4, 5, 7, 10, 20, 37, 50, 99, 100, 101, 150, 500, 1000000])


def rand_order(r, other=None):
    u = r.random()
    if other is not None and other[0] in (C.M_SELL, C.M_BUY_PRODUCT) and u < 0.6:
        it = other[1]
        op = r.choice([C.M_SELL, C.M_BUY_PRODUCT])
        if op == C.M_BUY_PRODUCT and r.random() < 0.8:
            it = r.choice([C.WHEAT, C.FERTILIZER])
        return (op, it, rand_count(r))
    if u < 0.05:
        return (C.M_NONE, 0, 0)
    if u < 0.12:
        return (C.M_HIRE, 0, 1)
    if u < 0.16:
        return (C.M_BUY_LAND, 0, 1)
    if u < 0.26:
        return (C.M_BUY_SEED, r.choice(list(range(5)) + [7, 9, 255]), rand_count(r))
    if u < 0.33:
        return (C.M_BUY_ANIMAL, r.choice([9, 10, 11, 11, 3, 255]), rand_count(r))
    if u < 0.55:
        return (C.M_BUY_PRODUCT, r.choice([0, 0, 8, 8, 8, 1, 6, 255]), rand_count(r))
    return (C.M_SELL, r.choice(list(range(9)) + [8, 8, 9, 255]), rand_count(r))


def rand_scenario(r):
    st = {"money": [], "shed": [], "seeds": [], "minv": [rand_inv(r, j) for j in range(9)], "hires": [],
          "n_units": [], "ux": [], "uy": [], "n_quads": []}
    for p in range(2):
        u = r.random()
        st["money"].append(r.choice([0, 1, 5, 24, 25, 26, 99, 100, 101]) if u < 0.2 else
                           r.randint(0, 600) if u < 0.5 else r.randint(0, 20000) if u < 0.9 else
                           r.randint(0, 400000))
        shed = [0] * 12
        tot = r.choice([0, 3, 30, 90, 97, 99, 100, 100])
        for _ in range(tot):
            shed[r.choice([0, 0, 8, 8, r.randrange(12)])] += 1
        st["shed"].append(shed)
        st["seeds"].append([r.randint(0, 5) for _ in range(5)])
        st["hires"].append(r.choice([0, 0, 1, 2, 5, 10, 20]))
        nu = r.choice([1, 2, 5, 12, 38, 39, 40])
        st["n_units"].append(nu)
        xs, ys = [], []
        for k in range(C.U):
            if r.random() < 0.6:
                x, y = C.SHED_ACCESS[r.randrange(4)]
            else:
                x, y = r.randrange(10), r.randrange(10)
            xs.append(x)
            ys.append(y)
        st["ux"].append(xs)
        st["uy"].append(ys)
        st["n_quads"].append(r.choice([1, 1, 2, 3, 4]))
    orders = [[], []]
    nslots = r.randint(1, 10)
    for i in range(nslots):
        o0 = rand_order(r)
        o1 = rand_order(r, o0)
        if r.random() < 0.5:
            o0, o1 = o1, o0
        orders[0].append(o0)
        orders[1].append(o1)
    return st, orders


def to_arrays(scen):
    keys = ["money", "shed", "seeds", "minv", "hires", "n_units", "ux", "uy", "n_quads"]
    out = {k: np.array([s[0][k] for s in scen], np.int32) for k in keys}
    mt = np.zeros((len(scen), 2, C.NSLOT), np.int32)
    mi = np.zeros_like(mt)
    mn = np.zeros_like(mt)
    for b, (_, orders) in enumerate(scen):
        for p in range(2):
            for i, (op, it, n) in enumerate(orders[p]):
                mt[b, p, i], mi[b, p, i], mn[b, p, i] = op, it, n
    return out, mt, mi, mn


COUNT_SEARCH = "--count" in sys.argv


@jax.jit
def run_jax(tb, money, shed, seeds, minv, hires, n_units, ux, uy, n_quads, mt, mi, mn):
    f = jax.vmap(lambda *a: E._market_phase(tb, *a, smax=C.NSLOT, count_search=COUNT_SEARCH))
    return f(money, shed, seeds, minv, hires, n_units, ux, uy, n_quads, mt, mi, mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--count", action="store_true", help="test the prefix-count search variant")
    args = ap.parse_args()
    r = random.Random(args.seed)
    tb = E.tables()
    bad = 0
    cases = {"ss": 0, "bb": 0, "mx": 0, "mx_floor": 0}
    t0 = time.time()
    for c0 in range(0, args.n, args.chunk):
        scen = [rand_scenario(r) for _ in range(min(args.chunk, args.n - c0))]
        arr, mt, mi, mn = to_arrays(scen)
        out = run_jax(tb, arr["money"], arr["shed"], arr["seeds"], arr["minv"], arr["hires"], arr["n_units"],
                      arr["ux"], arr["uy"], arr["n_quads"], mt, mi, mn)
        out = [np.asarray(x) for x in out]
        names = ["money", "shed", "seeds", "minv", "hires", "n_units", "ux", "uy", "n_quads"]
        for b, (st, orders) in enumerate(scen):
            for i in range(len(orders[0])):
                o0, o1 = orders[0][i], orders[1][i]
                if o0[0] in (C.M_SELL, C.M_BUY_PRODUCT) and o1[0] in (C.M_SELL, C.M_BUY_PRODUCT) and o0[1] == o1[1]:
                    k = "ss" if o0[0] == o1[0] == C.M_SELL else "bb" if o0[0] == o1[0] else "mx"
                    cases[k] += 1
                    if k == "mx" and o0[1] == C.FERTILIZER and st["minv"][C.FERTILIZER] >= FLOOR[C.FERTILIZER] - 1:
                        cases["mx_floor"] += 1
            ref = copy.deepcopy(st)
            MR.process_market(ref, orders)
            for k, name in enumerate(names):
                mine = out[k][b]
                want = np.array(ref[name], np.int32)
                if name in ("ux", "uy"):
                    nu = ref["n_units"]
                    ok = all(np.array_equal(mine[p][:nu[p]], want[p][:nu[p]]) for p in range(2))
                else:
                    ok = np.array_equal(mine, want)
                if not ok:
                    bad += 1
                    if bad <= 5:
                        print("MISMATCH", name, "mine", mine.tolist(), "ref", want.tolist())
                        print("   start", {kk: st[kk] for kk in ("money", "minv", "hires", "n_units", "n_quads")},
                              "shed", st["shed"])
                        print("   orders", orders)
                    break
        print(f"{c0 + len(scen)} scenarios, {bad} mismatches, {time.time() - t0:.1f}s", flush=True)
    print("same-slot same-item pairs exercised:", cases)
    print("RESULT", "PASS" if bad == 0 else "FAIL", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
