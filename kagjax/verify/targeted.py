"""Hand-built targeted scenarios: JAX engine vs the C++ reference, starting from a REAL mid-game state (a recorded
top game replayed in C++) edited to set up one rule condition, then 1-3 steps of chosen actions; full observe() of
both seats compared after every step, for every loop config (seq = trainer default, par, gpu = CFG_GPU, static),
each scenario alone (B=1) and all scenarios of a start step together in one batch (mixed n_units / slots).

usage (server, cwd kagjax/verify): KAG_ENGINE=cov_build/cov GCOV_PREFIX=gcda/targeted python targeted.py
"""
import copy
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("KAG_ENGINE", os.path.join(HERE, "cov_build", "opt")))
import kagsim_trace as K  # noqa: E402
assert "verify_cov" in K.__version__
import numpy as np  # noqa: E402
import jax  # noqa: E402
import engine as E  # noqa: E402
import rng as R  # noqa: E402
import state as S  # noqa: E402

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]
FLOOR = {}
for j, p in enumerate(ITEMS[:9]):
    for v in range(10000, 12000):
        if K.market_price(j, v) == 1:
            FLOOR[p] = v
            break


def real_state(tape_idx, step):
    f = sorted(glob.glob("/Users/asad.ali/kaggriculture/data/tapes_top12/*.json.gz"))[tape_idx]
    t = json.load(gzip.open(f))
    acts = [[x if isinstance(x, dict) else PASS for x in t["actions"][k]] for k in (0, 1)]
    g = K.Game(int(t["seed"]), 720, False)
    for s in range(step):
        g.step(acts[0][s], acts[1][s])
    return int(t["seed"]), g.observe(0), g.observe(1)


def edit(base, fn):
    seed, o0, o1 = base
    o0, o1 = copy.deepcopy(o0), copy.deepcopy(o1)
    o1["farms"] = o0["farms"]                   # one shared public part
    o1["market"] = o0["market"]
    o1["town"] = o0["town"]
    fn(o0, o1)
    return seed, o0, o1


def set_units(o, p, pos, invs):
    """units of seat p at positions pos (list of [x,y]) with inventories invs (list of ordered dicts)."""
    f = o[0]["farms"][p]
    f["farmer"], f["hands"] = list(pos[0]), [list(x) for x in pos[1:]]
    f["hires_today"] = len(pos) - 1
    o[p]["private"]["inventories"] = [dict(x) for x in invs]


def set_shed(o, p, d):
    o[p]["private"]["shed"] = {it: int(d.get(it, 0)) for it in ITEMS}


def tile(o, p, x, y, v):
    o[0]["farms"][p]["tiles"][y][x] = v


def quads(o, p, nq):
    """seat p owns the first nq quadrants: tiles of the others LOCKED, newly owned LOCKED tiles EMPTY (consistent)."""
    f = o[0]["farms"][p]
    f["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"][:nq]
    for y in range(10):
        for x in range(10):
            q = (0 if y < 5 else 2) + (0 if x < 5 else 1)
            if q >= nq:
                f["tiles"][y][x] = "LOCKED"
            elif f["tiles"][y][x] == "LOCKED":
                f["tiles"][y][x] = None


# ------------------------------------------------------------------------------------------------ scenarios
def scenarios(b310, b71):
    """b71 = the day-end base state (start step 287)."""
    sc = []

    def add(name, target, base, fn, acts):
        sc.append((name, target, edit(base, lambda o0, o1: fn([o0, o1])), acts))

    # 1. BUY_ANIMAL with an unknown item name (sim_trace.hpp:29 is_animal(255), the only reachable rule branch the
    #    corpus never executed)
    def s1(o):
        o[0]["farms"][0]["money"] = 5000.0; o[0]["farms"][1]["money"] = 5000.0
        set_shed(o, 0, {"WHEAT": 3}); set_shed(o, 1, {})
    add("buy_animal_unknown_item", "hpp:29 is_animal(255); hpp:823 quote fails", b310, s1,
        [({"farmer": ["PASS"], "market": [["BUY_ANIMAL", "FOO", 1], ["BUY_ANIMAL", "GOOSE", 2], ["BUY_ANIMAL", "PIG", 3]]},
          {"farmer": ["PASS"], "market": [["BUY_ANIMAL", "SHEEP", 1], ["BUY_ANIMAL", "FOO", 1], ["BUY_ANIMAL", 7, 1]]})])

    # 2. HIRE at the 40-unit cap with cash (inconsistent hires_today on purpose: the cap branch hpp:911)
    def s2(o):
        set_units(o, 0, [[4, 4]] * 40, [{} for _ in range(40)])
        o[0]["farms"][0]["hires_today"] = 0
        o[0]["farms"][0]["money"] = 10000.0
    add("hire_at_40_units", "hpp:911 n_units >= MAX_UNITS", b310, s2,
        [({"farmer": ["PASS"], "market": [["HIRE"], ["HIRE"]]}, {"farmer": ["PASS"], "market": [["HIRE"]] * 12})])

    # 3. DROP into a nearly full shed: partial take in insertion order, the rest discarded; two units in sequence
    def s3(o):
        set_shed(o, 0, {"WHEAT": 50, "SHEEP": 47})
        set_units(o, 0, [[4, 4], [5, 5], [4, 5]], [{"MILK": 5, "WHEAT": 3, "EGG": 2}, {"EGG": 4, "MILK": 1}, {"WOOL": 2}])
    add("drop_full_shed_insertion_order", "hpp:604-612 capped DROP", b310, s3,
        [({"farmer": ["DROP"], "hands": [["DROP"], ["PLACE", "WOOL", 5]], "market": []}, PASS),
         ({"farmer": ["PICKUP", "SHEEP", 2], "hands": [["PICKUP", "EGG", 99], ["DROP"]], "market": [["SELL", "WHEAT", 7]]}, PASS)])

    # 4. PLACE animal: matching empty structure under the unit (animal branch, even at a shed tile) vs shed path
    def s4(o):
        quads(o, 0, 4)
        tile(o, 0, 4, 4, {"kind": "COOP"})
        tile(o, 0, 5, 5, {"kind": "PASTURE"})
        set_shed(o, 0, {"WHEAT": 99})
        set_units(o, 0, [[4, 4], [4, 4], [5, 5], [5, 5]], [{"GOOSE": 2}, {"GOOSE": 1, "WHEAT": 3}, {"COW": 1}, {"SHEEP": 2}])
    add("place_animal_structure_vs_shed", "hpp:624-647 PLACE branches", b310, s4,
        [({"farmer": ["PLACE", "GOOSE", 2], "hands": [["PLACE", "GOOSE", 1], ["PLACE", "SHEEP", 1], ["PLACE", "COW", 1]],
           "market": []}, PASS),
         ({"farmer": ["PLACE", "GOOSE", 1], "hands": [["FEED"], ["DIG"], ["PLACE", "SHEEP", 3]], "market": []}, PASS)])

    # 5. atomic PLANT: demand counted over commands for units that do not exist
    def s5(o):
        o[0]["private"]["seeds"] = {"WHEAT": 2, "CARROT": 5, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 1}
        for (x, y) in ((0, 0), (1, 0), (2, 0), (3, 0)):
            tile(o, 0, x, y, None)
        set_units(o, 0, [[0, 0], [1, 0]], [{}, {}])
    add("atomic_plant_ghost_units", "hpp:452-461 demand > seeds", b310, s5,
        [({"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"], ["PLANT", "WHEAT"]], "market": []}, PASS),
         ({"farmer": ["PLANT", "MELON"], "hands": [["PLANT", "CARROT"], ["PLANT", "CARROT"], ["PLANT", "MELON"]],
           "market": []}, PASS)])

    # 6-10. market: same slot, same item, both seats, around the $1 floor
    def mk(item, inv, sh0, sh1, m0, m1):
        def f(o):
            o[0]["market"]["inventory"][item] = inv
            set_shed(o, 0, sh0); set_shed(o, 1, sh1)
            o[0]["farms"][0]["money"] = float(m0); o[0]["farms"][1]["money"] = float(m1)
        return f
    add("sell_sell_wool_floor", "SELL/SELL +2 per pair while price > 1", b310,
        mk("WOOL", FLOOR["WOOL"] - 3, {"WOOL": 10}, {"WOOL": 6}, 100, 100),
        [({"farmer": ["PASS"], "market": [["SELL", "WOOL", 10]]}, {"farmer": ["PASS"], "market": [["SELL", "WOOL", 1000]]})])
    add("buy_buy_wheat_cash_room", "BUY/BUY stride-2, one stops on cash, one on shed room", b310,
        mk("WHEAT", 9950, {"MILK": 60}, {"MILK": 95}, 300, 5000),
        [({"farmer": ["PASS"], "market": [["BUY_PRODUCT", "WHEAT", 50]]}, {"farmer": ["PASS"], "market": [["BUY_PRODUCT", "WHEAT", 50]]})])
    add("sell_buy_fert_floor_descent_seller0", "SELL vs BUY at the FERTILIZER $1 floor, descent completes", b310,
        mk("FERTILIZER", FLOOR["FERTILIZER"] + 1, {"FERTILIZER": 10}, {}, 0, 3000),
        [({"farmer": ["PASS"], "market": [["SELL", "FERTILIZER", 10]]}, {"farmer": ["PASS"], "market": [["BUY_PRODUCT", "FERTILIZER", 10]]})])
    add("sell_buy_fert_floor_descent_seller1", "same, seller in seat 1", b310,
        mk("FERTILIZER", FLOOR["FERTILIZER"] + 1, {}, {"FERTILIZER": 10}, 3000, 0),
        [({"farmer": ["PASS"], "market": [["BUY_PRODUCT", "FERTILIZER", 10]]}, {"farmer": ["PASS"], "market": [["SELL", "FERTILIZER", 10]]})])
    add("sell_buy_fert_floor_descent_incomplete", "buyer cannot finish the descent (cash / count)", b310,
        mk("FERTILIZER", FLOOR["FERTILIZER"] + 1, {"FERTILIZER": 10}, {}, 0, 1),
        [({"farmer": ["PASS"], "market": [["SELL", "FERTILIZER", 10], ["SELL", "FERTILIZER", 3]]},
          {"farmer": ["PASS"], "market": [["BUY_PRODUCT", "FERTILIZER", 10], ["BUY_PRODUCT", "FERTILIZER", 1]]})])

    # 11. land: all three purchases, the 4th fails; units standing on locked tiles; unlocked tiles usable next step
    def s11(o):
        for p in (0, 1):
            o[0]["farms"][p]["money"] = 7500.0
            quads(o, p, 1)
        set_units(o, 0, [[9, 9], [0, 9], [9, 0]], [{}, {}, {}])
    add("buy_land_all_quadrants", "do_buy_land incl. extra >= 3", b310, s11,
        [({"farmer": ["PASS"], "market": [["BUY_LAND"]] * 4}, {"farmer": ["PASS"], "market": [["BUY_LAND"], ["BUY_LAND"]]}),
         ({"farmer": ["BUILD_COOP"], "hands": [["DIG"], ["BUILD_PASTURE"]], "market": [["BUY_LAND"]]}, PASS)])

    # 12. decay on an even step: non-ongoing crop past max_lifespan_step reaches 0 -> WEED; another loses 1
    def s12(o):
        tile(o, 0, 0, 0, {"kind": "PLANT", "crop": "CARROT", "planted_day": 8, "watered_today": True,
                          "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": 288,
                          "fertilized_until_day": -1})
        tile(o, 0, 1, 0, {"kind": "PLANT", "crop": "WHEAT", "planted_day": 7, "watered_today": False,
                          "consecutive_unwatered": 0, "yield_units": 4, "max_lifespan_step": 288,
                          "fertilized_until_day": -1})
    add("decay_to_weed_even_step", "hpp:959-976 decay", b310, s12, [(PASS, PASS), (PASS, PASS), (PASS, PASS)])

    # 13. malformed / edge actions of every kind (parser + rules)
    add("malformed_actions", "conv_* + rules on odd input", b310, lambda o: set_shed(o, 0, {"WHEAT": 5, "GOOSE": 1}),
        [({"farmer": ["PLANT"], "hands": [["PICKUP", "WHEAT", -5], ["PLACE", "FOO", 3], ["PICKUP", "WHEAT", "3"],
                                          ["DROP", "X"], ["PICKUP"], ["NOPE"], None, "EAST", ["PICKUP", "GOOSE", 2.9]],
           "market": [["SELL"], ["HIRE", "x"], ["BUY_SEED", "WHEAT", "2"], None, ["SELL", "WHEAT", True],
                      ["BUY_SEED", "EGG", 1], ["BUY_PRODUCT", "MILK", 2], ["BUY_SEED", "WHEAT", 2.9], ["SELL", 3, 1],
                      ["BUY_LAND", 5], ["HIRE"], ["HIRE"]]},
          {"farmer": ["PICKUP", "WHEAT", " 2 "], "market": [["BUY_SEED", "CARROT", 10 ** 30], ["SELL", "WHEAT", "x"]]})])

    # 14. more than 39 hand commands (C++ keeps the first 39)
    def s14(o):
        set_units(o, 1, [[4, 4]] * 40, [{"WHEAT": 1} for _ in range(40)])
        set_shed(o, 1, {})
    add("hands_over_39", "conv_action nu >= MAX_UNITS", b310, s14,
        [(PASS, {"farmer": ["DROP"], "hands": [["DROP"]] * 45, "market": []})])

    # --- end-of-day group (t = 287: day 11 ends, next_day 12 = shop unlock day)
    def s15(o):
        tile(o, 0, 0, 0, {"kind": "PLANT", "crop": "CARROT", "planted_day": 10, "watered_today": False,
                          "consecutive_unwatered": 1, "yield_units": 1, "max_lifespan_step": 336, "fertilized_until_day": -1})
        tile(o, 0, 1, 0, {"kind": "PLANT", "crop": "TOMATO", "planted_day": 1, "watered_today": True,
                          "consecutive_unwatered": 0, "yield_units": 3, "max_lifespan_step": -1, "fertilized_until_day": 11})
        tile(o, 0, 4, 0, {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 0, "watered_today": True,
                          "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": -1, "fertilized_until_day": -1})
        tile(o, 0, 2, 0, {"kind": "PASTURE", "animal": "COW", "placed_day": 0, "yield_units": 2, "consecutive_unfed": 1,
                          "fed_today": False, "cared_today": True, "fertilizer_available": False, "pending_care_bonus": 2})
        tile(o, 0, 3, 0, {"kind": "COOP", "animal": "GOOSE", "placed_day": 0, "yield_units": 1, "consecutive_unfed": 0,
                          "fed_today": True, "cared_today": True, "fertilizer_available": False, "pending_care_bonus": 2})
        set_shed(o, 0, {"WHEAT": 60, "SHEEP": 35})
        set_units(o, 0, [[4, 4], [5, 4], [9, 9]], [{"MILK": 3, "EGG": 4}, {"WOOL": 1, "WHEAT": 2}, {"MELON": 7}])
    add("eod_refresh_and_capped_midnight_drop", "daily_refresh_*, escape, care bonus, drop_inventories cap", b71, s15,
        [({"farmer": ["PASS"], "hands": [["PASS"], ["PASS"]], "market": []}, PASS), (PASS, PASS)])

    def s16(o):
        o[0]["town"]["unlocked_shops"] = ["BAKERY"] * 8
    add("shop_unlock_with_8_shops", "hpp:1018 n_shops < 8 false", b71, s16, [(PASS, PASS), (PASS, PASS)])

    def s17(o):
        o[0]["town"]["unlocked_shops"] = ["YARN_STORE", "PET_CAFE", "YARN_STORE", "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "SMOOTHIE_SHOP"]
    add("shop_unlock_7th_to_8th", "8th shop unlock + duplicate-shop town drain", b71, s17, [(PASS, PASS), (PASS, PASS), (PASS, PASS)])
    return sc


# ------------------------------------------------------------------------------------------------ runner
CFGS = [("seq", None), ("par", E.CFG_PAR), ("gpu", E.CFG_GPU),
        ("static", E.LoopCfg(unit_static=True, slot_static=True, count_search=True, no_cond=True))]


def run_group(items, cfg):
    """items: [(name, target, (seed, o0, o1), acts)] with a common start step. -> {name: None | first diff}."""
    t = items[0][2][1]["step"]
    refs, starts = [], []
    for name, _, (seed, o0, o1), _ in items:
        g = K.Game(seed, 720, False)
        g.set_state(o0, o1)
        starts.append((g.observe(0), g.observe(1)))
        refs.append(g)
    st = E.from_games(S.stack_games([S.obs_to_game(a, b)[0] for a, b in starts]), t)
    n = max(len(x[3]) for x in items)
    res = {x[0]: None for x in items}
    for k in range(n):
        acts = [x[3][k] if k < len(x[3]) else (PASS, PASS) for x in items]
        words = np.stack([R.day_words(x[2][0], min((t + k) // 24, 29)) for x in items])
        st = E.step(st, S.encode_actions([a for a, _ in acts]), S.encode_actions([b for _, b in acts]), words, cfg=cfg)
        g = jax.tree_util.tree_map(np.asarray, st.g)
        for i, x in enumerate(items):
            refs[i].step(*acts[i])
            if res[x[0]] is not None:
                continue
            for seat in (0, 1):
                d = S.obs_diff(S.game_to_obs(g, i, seat, int(st.t)), refs[i].observe(seat))
                if d:
                    res[x[0]] = {"step": k, "seat": seat, "diff": [list(map(str, y)) for y in d[:8]]}
                    break
    return res


def main():
    b310 = real_state(700, 310)
    b71 = real_state(1200, 287)
    sc = scenarios(b310, b71)
    out = {"scenarios": {}, "batched": {}}
    for cname, cfg in CFGS:
        for x in sc:                                           # alone (B = 1)
            r = run_group([x], cfg)[x[0]]
            out["scenarios"].setdefault(x[0], {"target": x[1]})[cname] = "exact" if r is None else r
        for t in sorted({x[2][1]["step"] for x in sc}):        # all scenarios of a start step in one batch
            grp = [x for x in sc if x[2][1]["step"] == t]
            r = run_group(grp, cfg)
            out["batched"][f"{cname}_t{t}_B{len(grp)}"] = {k: ("exact" if v is None else v) for k, v in r.items()}
        print(cname, "done", flush=True)
    bad = [(n, c) for n, d in out["scenarios"].items() for c, v in d.items() if c != "target" and v != "exact"]
    bad += [(g, n) for g, d in out["batched"].items() for n, v in d.items() if v != "exact"]
    out["mismatches"] = bad
    json.dump(out, open(os.path.join(HERE, "out", "targeted.json"), "w"), indent=1)
    for n, d in out["scenarios"].items():
        print(f"{n:45s} {d['target'][:60]:60s} " + " ".join(f"{c}={'OK' if d[c] == 'exact' else 'MISMATCH'}" for c, _ in CFGS))
    print("batched:", {g: all(v == "exact" for v in d.values()) for g, d in out["batched"].items()})
    print("RESULT mismatches:", bad)


if __name__ == "__main__":
    main()
