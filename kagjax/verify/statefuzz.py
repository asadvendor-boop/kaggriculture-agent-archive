"""State-level differential fuzz: JAX engine vs the C++ reference (the spec), starting from random VALID game states.

The reference build in cov_build/ adds Game.set_state(obs0, obs1) (rules code untouched), so each scenario is:
  random state (my generator, independent of A's code) -> loaded into C++ -> C++ observe() dicts are the canonical
  state -> JAX state = state.obs_to_game(those dicts) -> K steps of random (often adversarial / malformed) actions on
  both engines -> the full observe() dict of BOTH seats compared after every step (state.obs_diff, key order incl.).
Scenarios of one batch share the step index t (the JAX batch shares t); batches cycle over interesting t values
(town steps, center steps, day ends incl. shop-unlock days, odd / even, the last step 718).

Families (weights by --mix): unit (crowded units, shed near cap, insertion orders, same-tile conflicts, atomic
PLANT), market (both seats trading the same item in the same slot, inventories at / around the $1 floors, cash and
shed limits, huge counts), eod (plants / animals at every age and state, inventory drops into a nearly full shed),
all (everything random at once).

usage (server, cwd kagjax/verify):  python statefuzz.py --batches 40 --B 64 --steps 3 --seed 0 [--cfg seq,par,gpu]
Writes out/statefuzz_<seed>.json (summary + the first mismatches with full reproduction data).
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KAGJAX = os.path.dirname(HERE)
sys.path.insert(0, KAGJAX)
sys.path.insert(0, os.environ.get("KAG_ENGINE", os.path.join(HERE, "cov_build", "opt")))
import kagsim_trace as K  # noqa: E402

assert "verify_cov" in K.__version__
import numpy as np  # noqa: E402
import jax  # noqa: E402

import consts as C  # noqa: E402
import engine as E  # noqa: E402
import rng as R  # noqa: E402
import state as S  # noqa: E402

ITEMS = C.ITEM_NAMES
PRODUCTS = ITEMS[:9]
CROPS = ITEMS[:5]
ANIMALS = ITEMS[9:]
SHOPS = C.SHOP_NAMES
QUADS = C.QUAD_NAMES
CROPDEF = {"WHEAT": (2, 4, 6, 0), "CARROT": (2, 3, 4, 0), "TOMATO": (8, 8, 4, 1), "STRAWBERRY": (10, 10, 4, 2),
           "MELON": (10, 12, 6, 0)}          # fyd, maxyd, max_yield, interval (0 = not ongoing)
ANIMALDEF = {"GOOSE": ("COOP", 4, 1, 4), "COW": ("PASTURE", 8, 2, 6), "SHEEP": ("PASTURE", 6, 3, 6)}
FLOOR = {}                                    # first inventory with price 1 (products that have one in range)
for j, p in enumerate(PRODUCTS):
    for v in range(10000, 12000):
        if K.market_price(j, v) == 1:
            FLOOR[p] = v
            break


# ------------------------------------------------------------------------------------------------ random states
def quad_of(x, y):
    return (0 if y < 5 else 2) + (0 if x < 5 else 1)


def rand_plant(r, day):
    c = r.choice(CROPS)
    fyd, maxyd, maxy, itv = CROPDEF[c]
    age = r.choice([0, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, r.randint(0, 20)])
    pday = max(0, day - age)
    if itv == 0:
        mls = (pday + maxyd + 1) * 24
        y = r.randint(1, maxy)
    else:
        mls = r.choice([-1, -1, -1, day * 24, (day + 1) * 24, max(0, day - 1) * 24])
        y = r.randint(0, maxy)
    return {"kind": "PLANT", "crop": c, "planted_day": pday, "watered_today": r.random() < 0.5,
            "consecutive_unwatered": r.choice([0, 0, 1]), "yield_units": y, "max_lifespan_step": mls,
            "fertilized_until_day": r.choice([-1, -1, -1, day - 1, day, day + 1, day + 2])}


def rand_structure(r, day, animal_p=0.7):
    kind = r.choice(["COOP", "PASTURE"])
    d = {"kind": kind}
    if r.random() < animal_p:
        a = "GOOSE" if kind == "COOP" else r.choice(["COW", "SHEEP"])
        _, fyd, itv, mh = ANIMALDEF[a]
        d.update({"animal": a, "placed_day": max(0, day - r.randint(0, 16)), "yield_units": r.randint(0, mh),
                  "consecutive_unfed": r.choice([0, 0, 1]), "fed_today": r.random() < 0.5,
                  "cared_today": r.random() < 0.5, "fertilizer_available": r.random() < 0.5,
                  "pending_care_bonus": r.choice([0, 0, 1, 2, 3, 5])})
    return d


def rand_tiles(r, day, nq, fam):
    rows = []
    for y in range(10):
        row = []
        for x in range(10):
            if quad_of(x, y) >= nq:
                row.append("LOCKED")
                continue
            u = r.random()
            if fam == "eod":
                row.append(None if u < 0.2 else {"kind": "WEED"} if u < 0.25 else rand_plant(r, day) if u < 0.65
                           else rand_structure(r, day))
            else:
                row.append(None if u < 0.35 else {"kind": "WEED"} if u < 0.45 else rand_plant(r, day) if u < 0.75
                           else rand_structure(r, day))
        rows.append(row)
    return rows


def rand_inv(r, big=False):
    d = {}
    if r.random() < 0.35:
        return d
    for it in r.sample(ITEMS, r.randint(1, 5)):
        d[it] = r.choice([1, 1, 2, 3, 5, 10, 20, 37, 60, 99]) if big else r.randint(1, 12)
    return d


def rand_farm(r, day, fam, rich):
    nq = r.choice([1, 2, 3, 4, 4])
    n_units = r.choice([1, 1, 2, 3, 5, 8, 12, 16, 17, 25, 40]) if fam in ("unit", "all", "eod") else r.choice([1, 2, 5])
    hot = [(4, 4), (5, 4), (4, 5), (5, 5), (r.randint(0, 9), r.randint(0, 9)), (r.randint(0, 9), r.randint(0, 9))]
    pos = [list(r.choice(hot)) if r.random() < 0.7 else [r.randint(0, 9), r.randint(0, 9)] for _ in range(n_units)]
    money = r.choice([0, 1, 7, 50, 99, 300, 999, 1000, 2000, 3000, 4000, 7777, 25000, 150000]) if not rich else \
        r.choice([20000, 60000, 150000, 150000])   # (> $327k lets BUY_SEED 1e6 overflow the C++ int16 seed counter)
    return {"money": float(money), "tiles": rand_tiles(r, day, nq, fam), "farmer": pos[0], "hands": pos[1:],
            "unlocked_quadrants": QUADS[:nq], "hires_today": n_units - 1}


def rand_shed(r, fam):
    tot = r.choice([0, 3, 20, 60, 90, 95, 98, 99, 100, 100]) if fam != "market" else r.choice([0, 5, 50, 99, 100, 100, 60])
    shed = {it: 0 for it in ITEMS}
    pool = r.sample(ITEMS, r.randint(1, 6))
    for _ in range(tot):
        shed[r.choice(pool)] += 1
    return shed


def rand_minv(r, fam):
    inv = {}
    for p in PRODUCTS:
        u = r.random()
        if p in FLOOR and (u < 0.45 or fam == "market" and u < 0.6):
            inv[p] = FLOOR[p] + r.randint(-4, 2)
        elif u < 0.7:
            inv[p] = 10000 + r.randint(-40, 40)
        else:
            inv[p] = 10000 + r.randint(-3000, 1500)
    return inv


def rand_state(r, t, fam):
    day = t // 24
    rich = fam == "market" and r.random() < 0.4
    farms = [rand_farm(r, day, fam, rich), rand_farm(r, day, fam, rich)]
    n_shops = min(8, r.randint(0, 8))
    shops = [r.choice(SHOPS) for _ in range(n_shops)]
    minv = rand_minv(r, fam)
    obs = []
    for p in range(2):
        nu = 1 + len(farms[p]["hands"])
        seeds = {c: r.choice([0, 0, 1, 1, 2, 3, 5, 40]) for c in CROPS}
        invs = [rand_inv(r, big=(fam == "eod" or r.random() < 0.3)) for _ in range(nu)]
        obs.append({"remainingOverageTime": 60, "step": t, "player": p, "farms": farms,
                    "market": {"inventory": dict(minv), "prices": {}}, "town": {"unlocked_shops": shops},
                    "day": day, "hour": t % 24,
                    "private": {"shed": rand_shed(r, fam), "seeds": seeds, "inventories": invs}})
    return obs[0], obs[1]


# ------------------------------------------------------------------------------------------------ random actions
JUNK_N = [0, -1, -7, 1, 2, 3, 5, 12, 50, 99, 100, 101, 1000, 30000, 40000, 10 ** 6, 10 ** 7, "4", " 2 ", "x", 2.5,
          True, None]


def rand_unit_cmd(r, obs, k, fam):
    farm = obs["farms"][obs["player"]]
    u = r.random()
    if u < 0.03:
        return r.choice([[], ["FOO"], "NORTH", None, ["PLANT"], ["PICKUP"], ["PLACE"], ["PICKUP", 5], ["PLACE", None, 2],
                         ["DROP", "WHEAT", 5], ["HARVEST", "WHEAT"], ("NORTH",)])
    ops = (["PICKUP", "DROP", "PLACE", "PLACE", "PICKUP", "PLANT", "HARVEST", "WATER", "FEED", "CARE",
            "COLLECT_FERTILIZER", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FERTILIZE", "NORTH", "SOUTH", "EAST",
            "WEST", "PASS"] if fam != "market" else ["PASS", "DROP", "PICKUP", "HARVEST", "NORTH"])
    op = r.choice(ops)
    if op in ("PICKUP", "PLACE"):
        inv = obs["private"]["inventories"][k] if k < len(obs["private"]["inventories"]) else {}
        pool = ITEMS + list(inv.keys()) * 3 + ["FOO"]
        it = r.choice(pool)
        cmd = [op, it]
        if r.random() < 0.9:
            cmd.append(r.choice(JUNK_N) if r.random() < 0.25 else r.randint(1, 120))
        return cmd
    if op == "PLANT":
        return [op, r.choice(CROPS + ["EGG", "FOO"]) if r.random() < 0.1 else r.choice(CROPS[:2] + [r.choice(CROPS)])]
    return [op]


def rand_order(r, fam, item_hint=None):
    u = r.random()
    if u < 0.04:
        return r.choice([[], ["SELL"], ["SELL", "WHEAT"], ["SELL", "WHEAT", 0], ["SELL", "WHEAT", -3], ["SELL", "FOO", 3],
                         ["SELL", 3, 3], "HIRE", None, ["NONE", "WHEAT", 3], ["BUY_SEED", "WHEAT", "2"],
                         ["BUY_SEED", "WHEAT", 2.9], ["BUY_PRODUCT", "MELON", 3], ["BUY_ANIMAL", "EGG", 1],
                         ["SELL", "GOOSE", 1], ["BUY_SEED", "WHEAT", 10 ** 30], ["SELL", "WHEAT", "abc"],
                         ["BUY_SEED", "EGG", 2], ["BUY_ANIMAL", "COW", 0]])
    if u < 0.10:
        return ["HIRE"]
    if u < 0.13:
        return ["BUY_LAND"]
    if u < 0.20:
        return ["BUY_SEED", r.choice(CROPS), r.choice([1, 2, 5, 40, 1000])]   # (> 32767 seeds overflow the C++ int16)
    if u < 0.26:
        return ["BUY_ANIMAL", r.choice(ANIMALS), r.choice([1, 2, 3, 50, 10 ** 6])]
    it = item_hint or r.choice(PRODUCTS)
    n = r.choice([1, 2, 3, 5, 10, 17, 40, 99, 100, 101, 250, 1000, 10 ** 6])
    if u < 0.55:
        return ["BUY_PRODUCT", r.choice(["WHEAT", "FERTILIZER", it]), n]
    return ["SELL", it, n]


def rand_action(r, obs, fam):
    nu = 1 + len(obs["farms"][obs["player"]]["hands"])
    extra = r.choice([0, 0, 0, 1, 3, -1])
    cmds = [rand_unit_cmd(r, obs, k, fam) for k in range(max(1, nu + extra))]
    if r.random() < 0.15:                                   # PLANT burst (atomic seed check)
        c = r.choice(CROPS)
        for k in range(len(cmds)):
            if r.random() < 0.6:
                cmds[k] = ["PLANT", c]
    if r.random() < 0.2:                                    # shed traffic burst
        for k in range(len(cmds)):
            if r.random() < 0.5:
                cmds[k] = r.choice([["DROP"], ["PICKUP", r.choice(ITEMS), r.choice([1, 5, 50, 1000])],
                                    ["PLACE", r.choice(ITEMS), r.choice([1, 5, 50, 1000])]])
    nord = r.choice([0, 1, 2, 3, 5, 8, 10, 11, 12]) if fam in ("market", "all") else r.choice([0, 0, 1, 2, 4])
    shed = obs["private"]["shed"]
    have = [k for k, v in shed.items() if v > 0 and k in PRODUCTS]
    mk = [rand_order(r, fam, r.choice(have) if have and r.random() < 0.6 else None) for _ in range(nord)]
    return {"farmer": cmds[0], "hands": cmds[1:], "market": mk}


def entangle(r, a0, a1):
    """Same slot, same item, both seats: SELL/SELL, BUY/BUY, SELL vs BUY (either seat order)."""
    m0, m1 = a0["market"], a1["market"]
    for i in range(min(len(m0), 10)):
        if r.random() < 0.6:
            it = r.choice(["WHEAT", "FERTILIZER", "FERTILIZER", "WOOL", "MILK", "STRAWBERRY", "MELON", r.choice(PRODUCTS)])
            o0 = [r.choice(["SELL", "SELL", "BUY_PRODUCT"]), it, r.choice([1, 3, 10, 40, 100, 10 ** 6])]
            o1 = [r.choice(["SELL", "BUY_PRODUCT", "BUY_PRODUCT"]), it, r.choice([1, 2, 7, 40, 100, 10 ** 6])]
            m0[i] = o0
            while len(m1) <= i:
                m1.append(rand_order(r, "market"))
            m1[i] = o1


# ------------------------------------------------------------------------------------------------ run
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
T_CHOICES = [0, 4, 23, 24, 47, 71, 95, 100, 119, 143, 167, 191, 215, 238, 239, 263, 287, 311, 335, 359, 383, 400,
             431, 455, 479, 503, 527, 551, 575, 599, 600, 623, 647, 671, 695, 700, 713, 716, 717, 718]


def same_obs(a, b):
    """Plain structural equality (dict key order ignored) + per-unit inventory key order + bool/int types."""
    if a != b:
        return False
    ia, ib = a["private"]["inventories"], b["private"]["inventories"]
    if [list(x.keys()) for x in ia] != [list(x.keys()) for x in ib]:
        return False
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True).replace(".0,", ",") or \
        json.dumps(a, sort_keys=True).replace(".0,", ",").replace(".0}", "}") == \
        json.dumps(b, sort_keys=True).replace(".0,", ",").replace(".0}", "}")


def cxx_obs_pair(g):
    return g.observe(0), g.observe(1)


def run_batch(r, B, t, steps, fams, cfgs, keep_fail):
    seeds, refs, starts, acts = [], [], [], []
    for b in range(B):
        fam = r.choice(fams)
        seed = r.choice([0, 1, 7, 4294, r.randrange(2 ** 31)])
        o0, o1 = rand_state(r, t, fam)
        g = K.Game(seed, 720, False)
        g.set_state(o0, o1)
        c0, c1 = cxx_obs_pair(g)                   # canonical state
        seeds.append(seed)
        starts.append((c0, c1, fam))
        acts.append([])
        refs.append(g)
    # actions are generated from the reference's own observations step by step (same for every cfg)
    for k in range(steps):
        for b in range(B):
            o0, o1 = cxx_obs_pair(refs[b])
            a0, a1 = rand_action(r, o0, starts[b][2]), rand_action(r, o1, starts[b][2])
            if r.random() < (0.7 if starts[b][2] == "market" else 0.3):
                entangle(r, a0, a1)
            if r.random() < 0.01:
                a0 = r.choice([None, 7, "PASS", {"farmer": "NORTH"}, {"market": "SELL"}, {"hands": [], "market": None}])
            acts[b].append((a0, a1))
            refs[b].step(a0, a1)
    # reference trajectory: re-run (fresh games) and record observations after every step
    ref_obs = []
    for b in range(B):
        g = K.Game(seeds[b], 720, False)
        g.set_state(starts[b][0], starts[b][1])
        seq = []
        for (a0, a1) in acts[b]:
            g.step(a0, a1)
            seq.append((g.observe(0), g.observe(1), g.reward(0), g.reward(1), g.done))
        ref_obs.append(seq)
    words_all = {}
    fails = []
    for cname, cfg in cfgs:
        st = E.from_games(S.stack_games([S.obs_to_game(starts[b][0], starts[b][1])[0] for b in range(B)]), t)
        bad = [None] * B
        for k in range(steps):
            tt = t + k
            day = min(tt // 24, 29)
            if day not in words_all:
                words_all[day] = np.stack([R.day_words(s, day) for s in seeds])
            a0 = S.encode_actions([acts[b][k][0] for b in range(B)])
            a1 = S.encode_actions([acts[b][k][1] for b in range(B)])
            st = E.step(st, a0, a1, words_all[day], cfg=cfg)
            g = jax.tree_util.tree_map(np.asarray, st.g)
            tn = int(st.t)
            for b in range(B):
                if bad[b] is not None:
                    continue
                for seat in (0, 1):
                    mine = S.game_to_obs(g, b, seat, tn)
                    d = S.obs_diff(mine, ref_obs[b][k][seat])
                    if not d and not same_obs(mine, ref_obs[b][k][seat]):      # independent comparator
                        d = [("independent comparator", "differs", "")]
                    if d:
                        bad[b] = (k, seat, d[:10])
                        break
                if bad[b] is None and [int(g.money[b, 0]), int(g.money[b, 1])] != [ref_obs[b][k][2], ref_obs[b][k][3]]:
                    bad[b] = (k, -1, [("reward", [int(g.money[b, 0]), int(g.money[b, 1])], ref_obs[b][k][2:4])])
        for b in range(B):
            if bad[b] is not None:
                k, seat, d = bad[b]
                fails.append({"cfg": cname, "t": t, "step_in_scenario": k, "seat": seat, "family": starts[b][2],
                              "seed": seeds[b], "diff": [list(map(str, x)) for x in d],
                              "start_obs": [starts[b][0], starts[b][1]] if len(fails) < keep_fail else None,
                              "actions": acts[b][:k + 1] if len(fails) < keep_fail else None})
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", default="unit,market,eod,all")
    ap.add_argument("--cfg", default="seq")
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = random.Random(args.seed)
    cfgs = [(c, {"seq": None, "par": E.CFG_PAR, "gpu": E.CFG_GPU,
                 "static": E.LoopCfg(unit_static=True, slot_static=True, count_search=True, no_cond=True)}[c])
            for c in args.cfg.split(",")]
    fams = args.mix.split(",")
    allf, n = [], 0
    t0 = time.time()
    for i in range(args.batches):
        t = T_CHOICES[(args.seed * 7 + i) % len(T_CHOICES)] if i % 3 else r.choice(T_CHOICES)
        f = run_batch(r, args.B, t, args.steps, fams, cfgs, args.keep)
        n += args.B
        allf += f
        print(f"batch {i} t={t}: {len(f)} mismatching scenario-cfgs (total {len(allf)} of {n} scenarios x "
              f"{len(cfgs)} cfgs) {time.time() - t0:.0f}s", flush=True)
        for x in f[:3]:
            print("   MISMATCH", x["cfg"], "t", x["t"], "k", x["step_in_scenario"], "seat", x["seat"], x["family"],
                  x["diff"][:4], flush=True)
    out = args.out or os.path.join(HERE, "out", f"statefuzz_{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"scenarios": n, "cfgs": [c for c, _ in cfgs], "steps": args.steps, "mismatches": len(allf),
               "fails": allf[:50]}, open(out, "w"))
    print(f"RESULT {n} scenarios x {len(cfgs)} cfgs, {args.steps} steps each: {len(allf)} mismatches -> {out}")


if __name__ == "__main__":
    main()
