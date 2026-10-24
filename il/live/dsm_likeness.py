"""DSM-likeness report (operator-approved plan step 3, Sep 25): the same farm measures for DSM (its own recorded games)
and for our agents (their live games; wins and losses separately), from exact C++ replays:
land days, hires/day (days 12-27), animals bought (cow/sheep/goose), fertilizer applied (all / on wheat), plants by crop,
wheat bought ($), fertilizer bought, cash at the end of day 8, unit-turns/day and their mix, bank.
Usage: dsm_likeness.py      env NP (8), NDSM (120)"""
import sys, os, json, gzip, glob, collections
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
R = "/Users/asad.ali/kaggriculture"
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def profile(item):
    from harness import kagsim_trace, PASS
    path, s = item
    try:
        d = json.load(gzip.open(path, "rt"))
        A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][k]] for k in (0, 1)]
        g = kagsim_trace.Game(int(d["seed"]), 720, True)
    except Exception:
        return None
    P = collections.Counter(); land_days = []; hires = collections.Counter(); units = collections.Counter(); mix = collections.Counter()
    cash8 = None
    for t in range(720):
        o = g.observe(s); tiles = o["farms"][s]["tiles"]
        if t == 216:
            cash8 = o["farms"][s]["money"]
        g.step(A[0][t], A[1][t]); tr = g.trace_step(g.trace_len() - 1); day = t // 24
        for rc in tr.get("receipts", []):
            if rc.get("seat") != s or not rc.get("executed"):
                continue
            op, it, n = rc.get("op"), rc.get("item"), rc["executed"]; dm = (rc.get("money_after") or 0) - (rc.get("money_before") or 0)
            if op == "BUY_LAND":
                land_days.append(day)
            elif op == "HIRE":
                hires[day] += n
            elif op == "BUY_ANIMAL":
                P["animal:" + str(it)] += n
            elif op == "BUY_PRODUCT" and it == "WHEAT":
                P["wheat_bought$"] -= dm
            elif op == "BUY_PRODUCT" and it == "FERTILIZER":
                P["fert_bought"] += n
            elif op == "SELL":
                P["rev:" + str(it)] += dm
        for u in tr.get("unit_receipts", []):
            if u.get("seat") != s:
                continue
            op = u.get("op"); units[day] += 1
            if 12 <= day <= 27:
                mix["move" if op in MOVES else op] += 1
            if not u.get("executed"):
                continue
            if op == "FERTILIZE":
                P["fertilize"] += 1
                x, y = u.get("x"), u.get("y")
                try:
                    tc = tiles[y][x]; crop = tc.get("crop") if isinstance(tc, dict) else None
                except Exception:
                    crop = None
                if crop == "WHEAT":
                    P["fertilize_wheat"] += 1
            elif op == "PLANT":
                P["plant:" + str(u.get("eff_item") or u.get("arg"))] += int(u.get("eff_n") or 1)
        if g.done:
            break
    P["bank"] = g.reward(s); P["margin"] = g.reward(s) - g.reward(1 - s)
    P["hires_12_27"] = sum(hires[k] for k in range(12, 28)) / 16.0
    P["unit_turns_12_27"] = sum(units[k] for k in range(12, 28)) / 16.0
    tot = max(1, sum(mix.values()))
    for k in ("move", "WATER", "HARVEST", "COLLECT_FERTILIZER", "FERTILIZE"):
        P["mix:" + k] = mix[k] / tot
    return dict(P=P, land_days=sorted(land_days), cash8=cash8)


def summarize(label, res):
    res = [r for r in res if r]
    if not res:
        print(f"{label:34s} no games"); return
    P = lambda k: np.mean([r["P"][k] for r in res])
    lands = collections.Counter(tuple(r["land_days"]) for r in res).most_common(2)
    print(f"{label:34s} n {len(res):3d} | bank {P('bank')/1e3:5.1f}k margin {P('margin'):+7,.0f} | land days {lands[0][0]} ({lands[0][1]}/{len(res)})"
          f" | hires/day {P('hires_12_27'):4.1f} | cows {P('animal:COW'):4.1f} sheep {P('animal:SHEEP'):4.1f} geese {P('animal:GOOSE'):4.1f}"
          f" | fertilize {P('fertilize'):5.0f} (wheat {P('fertilize_wheat'):4.0f}) | plants W {P('plant:WHEAT'):4.0f} C {P('plant:CARROT'):3.0f} T {P('plant:TOMATO'):3.0f}"
          f" S {P('plant:STRAWBERRY'):3.0f} M {P('plant:MELON'):3.0f} | wheat bought ${P('wheat_bought$'):5.0f} fert bought {P('fert_bought'):3.0f}"
          f" | cash day8 ${np.mean([r['cash8'] or 0 for r in res]):5.0f} | unit-turns/day {P('unit_turns_12_27'):4.0f}"
          f" (move {P('mix:move'):.0%} water {P('mix:WATER'):.0%} harvest {P('mix:HARVEST'):.0%} collect-fert {P('mix:COLLECT_FERTILIZER'):.0%} fertilize {P('mix:FERTILIZE'):.0%})", flush=True)


if __name__ == "__main__":
    groups = collections.OrderedDict()
    idx = json.load(open(f"{R}/data/tapes_top12/index.json"))
    for team in os.environ.get("TEAMS", "DSM").split(","):
        groups[f"{team[:20]} (own games)"] = [(f"{R}/data/tapes_top12/{e}.json.gz", v["teams"].index(team)) for e, v in sorted(idx.items(), reverse=True)
                                             if team in (v.get("teams") or [])][:int(os.environ.get("NDSM", "120"))]
    for lab, d_ in ([] if os.environ.get("TEAMS_ONLY") else [("NEURAL_10M_S4", "tapes_n10m"), ("NEURAL_4M_S4", "tapes_n4m")]):
        W, L, O = [], [], []
        for f in sorted(glob.glob(f"{R}/data/{d_}/*.json.gz")):
            t = json.load(gzip.open(f, "rt"))
            if t.get("res") not in ("W", "L"):
                continue
            (W if t["res"] == "W" else L).append((f, int(t["seat"]))); O.append((f, 1 - int(t["seat"])))
        groups[f"{lab} wins"] = W; groups[f"{lab} losses"] = L; groups[f"{lab}'s opponents"] = O
    items = [(k, it) for k, v in groups.items() for it in v]
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "8"))) as pool:
        res = pool.map(profile, [it for _, it in items], chunksize=2)
    by = collections.defaultdict(list)
    for (k, _), r in zip(items, res):
        by[k].append(r)
    for k in groups:
        summarize(k, by[k])
