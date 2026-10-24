"""Tape-to-planner handover: a clean planning state from any Kaggriculture observation.

Standard library only, pure Python, no imports from the rest of the package (so it can be inlined into a
Kaggle submission).  One call costs about 1 ms.

    from planner.handover import extract_state, inflight, commitment_tasks
    st = extract_state(obs)          # works for any seat, any step (0..718)

Engine facts used here (all checked against the official 1.32.7 source and the bit-exact engine by
tools/planner/test_handover.py):
  * 720 episode steps, but the last applied action is step 718 (DONE fires after it); the last end-of-day
    rollover is the one at the end of day 28 (step 695).  Day 29 runs hours 0..22 and has no rollover.
  * At hour 0 of every day: no hands, all unit inventories empty (they were dropped into the shed, overflow
    discarded), farmer on (4,4), watered/fed/cared flags cleared.  This is the clean handover point.
  * Plants: a plant with consecutive_unwatered >= 1 that is not watered today becomes a WEED at the end of
    the day (a fresh plant starts at 1, so it must be watered on its planting day).
    Single-harvest crops (WHEAT, CARROT, MELON) start with 1 unit and gain 1 unit per watering (2 while
    fertilized) on ages (max_yield_day+1)//2 .. max_yield_day, capped at max_yield; harvestable from age
    first_yield_day; from step (planted+max_yield_day+1)*24 they lose 1 unit every 2 steps (a HARVEST in that
    first step still gets the full amount: unit actions resolve before decay) and become a WEED at 0.
    That first decay step is hour 0, when every unit is back on the shed (farmer on (4,4)), so for any other
    tile the last reachable full-value harvest is hour 23 of the day before (commitment_tasks uses that).
    Ongoing crops (TOMATO, STRAWBERRY) produce at the end of day e when k = e+1-planted-first_yield_day >= 0,
    k % interval == 0 and k//interval+1 <= 4, +1 unit (+2 if watered that day while fertilized), held cap 4;
    after the 4th production they decay from the start of day e+2.
  * Animals escape at the end of a day on which consecutive_unfed reaches 2 (fresh animal: 0).  They produce
    at the end of day e when k = e+1-placed-first_yield_day >= 0 and k % interval == 0: 1 unit plus the
    banked care bonus if fed on e (an unfed production day wipes the bank); a fed+cared day adds 1 to the
    bank after that day's production.  Held cap: goose 4, cow/sheep 6.
  * Town drain: every step with step % 4 == 0 each unlocked shop instance removes 1 unit of each of its
    products (2 if the shop has a single product); every step with step % 24 == 0 the town centre removes 1
    of every product except FERTILIZER.  Shop n (1-based) unlocks at the start of day 3n, n <= 8.
  * Hire n (0-based, per day) costs fib(n) = 1,1,2,3,5,8,...; land costs 1000, 2000, 4000 (NE, SW, SE).
  * Shed capacity 100 items, animals included.
  * A care bank is paid out capped at max_held: a cow fed+cared daily from placement produces 6, not 8, first.

HANDOVER PROTOCOL (measured by tools/planner/handover_study.py on 120 games; tables in runs/planner/handover_report.txt)
  1. Switch with make_hybrid(tape_agent, planner_factory, D) at hour 0 of day D (step 24*D): the only clean instant
     (no hands, no cargo, farmer on the shed tile, daily flags reset; 6,600/6,600 recorded hour-0 states checked).
     The planner needs nothing from the tape: extract_state(obs) at that step is the full inherited state.
  2. Recommended D = 12 (step 288).  The bar a planner must clear is new_margin(D) = what the tape's own activity
     from day D on adds to the (own - rival) bank margin.  It falls from about 60k (D=7) to 35-39k (D=9) and 27-29k
     (D=11), then to 7k against leader tapes (11.6k against V94) at D=12.  The day-11 program (second land + 13
     strawberries) is worth about 18-20k of margin, far more than its ~4k cost.  At D=12 every large tape
     investment is paid for and working, the melon race is over, and cash is about 15k.
     Move D earlier only once the planner beats the tape's day-11 block, or give it that block as commitments
     (in the raw route the strawberry count matches the realized one in 97% of games, land in 72%).
  3. Obligations on day D: commitment_tasks(st), i.e. FEED animals that escape tonight (about 3.4 per game at D=12),
     WATER plants that die tonight (none at D=12, 24 at D=11), HARVEST plants on their last full-value day and holders
     at their cap (deadline: end of the production day the cap would waste), PLANT seeds already bought,
     PLACE animals already bought.  At hour 0 almost nothing is paid-but-unused: seeds held about 0.3 per game,
     unplaced animals 0 (about 0.75 at D=10), and honouring them is worth under $0.7k.
  4. Not inherited: every future purchase (land, seeds, animals), the hires, and the sale timing.  At D=12 the tape
     would still plant about 38 late carrots (days 21-27), about 110 wheat, and in ~15% of games do a day-18
     land + tomato expansion.  The planner must replace these with at least as much margin.
  5. Objective: margin, not own bank.  Up to half of the tape's post-D value is denial (lower rival prices).
     Freezing at D=7 raises V94's bank by 33k and a fixed leader tape's by 21k, with identical rival actions.
"""

BOARD = 10
HALF = 5
TURNS = 24
EPISODE_STEPS = 720
LAST_ACTION_STEP = 718          # last step whose action is applied
LAST_ROLLOVER_DAY = 28          # last day whose end-of-day processing happens
LAST_DAY = LAST_ACTION_STEP // TURNS   # 29
SHED_CAP = 100
MAX_SHOPS = 8
SHOP_UNLOCK_INTERVAL = 3
SHOP_TICK = 4
CENTER_TICK = 24

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
TOWN_CENTER_PRODUCTS = tuple(p for p in PRODUCTS if p != "FERTILIZER")

# crop: seed price, first_yield_day, max_yield_day, interval, max_yield, ongoing
CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
SHED_ITEMS = PRODUCTS + tuple(ANIMALS)
SHOPS = {
    "BAKERY":         ("EGG", "WHEAT"),
    "PIZZA_SHOP":     ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT":    ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE":     ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE":       ("CARROT",),
    "SMOOTHIE_SHOP":  ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
LAND_ORDER = ("NE", "SW", "SE")
LAND_PRICES = (1000, 2000, 4000)
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
SPAWN = (4, 4)


def fib(n):
    """fib(0)=1, fib(1)=1, fib(2)=2, ... (the engine's hire-cost sequence)."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def quadrant(x, y):
    return ("N" if y < HALF else "S") + ("W" if x < HALF else "E")


def _g(o, k, d=None):
    if isinstance(o, dict):
        return o.get(k, d)
    return getattr(o, k, d)


# ----------------------------------------------------------------------------------------- forecasts
def crop_outlook(crop, planted_day, day, watered_today, yield_units, fertilized_until_day, max_lifespan_step, step=None):
    """What a plant still gives if it is watered every remaining day from `day` on (no new fertilizer).

    Returns dict(units=total units the plant yields from now on incl. the units it holds now,
                 ready_day=first day a HARVEST is legal, full_step=last step a HARVEST gets the full amount,
                 last_gain_day=day of the last unit gain, gains=[(day, +units)...]).
    For single-harvest crops `units` is the final harvest; for ongoing crops it is held + future productions."""
    cd = CROPS[crop]
    first, mxd, cap = cd["first_yield_day"], cd["max_yield_day"], cd["max_yield"]
    ready_day = planted_day + first
    gains = []
    if not cd["ongoing"]:
        ws = (mxd + 1) // 2
        y = yield_units
        start = day + 1 if watered_today else day
        for d in range(start, min(planted_day + mxd, LAST_DAY) + 1):
            age = d - planted_day
            if ws <= age <= mxd:
                inc = 2 if fertilized_until_day >= d else 1
                new = min(cap, y + inc)
                if new > y:
                    gains.append((d, new - y))
                y = new
        full_step = max_lifespan_step if max_lifespan_step >= 0 else (planted_day + mxd + 1) * TURNS
        if step is not None and full_step < step:
            full_step = step        # already decaying: harvesting now is the best there is
        return {"units": y, "ready_day": ready_day, "full_step": min(full_step, LAST_ACTION_STEP),
                "last_gain_day": gains[-1][0] if gains else None, "gains": gains}
    iv = cd["interval"]
    units = yield_units
    last_e = None
    for e in range(day, LAST_ROLLOVER_DAY + 1):
        k = e + 1 - planted_day - first
        if k < 0 or k % iv:
            continue
        if k // iv + 1 > cap:
            break
        inc = 2 if fertilized_until_day >= e else 1
        gains.append((e + 1, inc))
        units += inc
        last_e = e
    if last_e is not None:
        full_step = (last_e + 2) * TURNS
    else:
        full_step = max_lifespan_step if max_lifespan_step >= 0 else LAST_ACTION_STEP
        if step is not None and full_step < step:
            full_step = step
    return {"units": units, "ready_day": ready_day, "full_step": min(full_step, LAST_ACTION_STEP),
            "last_gain_day": gains[-1][0] if gains else None, "gains": gains}


def animal_outlook(species, placed_day, day, fed_today, cared_today, pending_care_bonus, yield_units):
    """Future production of an animal that is kept alive.

    base: units from now to the end if it is only fed often enough not to escape (no care bonus);
    cared: units if it is fed and cared every remaining day (today included, whatever was done so far) and
           harvested before every production; a production is capped at max_held, so a care bank beyond
           max_held-1 is wasted (a cow cared daily from placement loses 2 units on its first production).
    prod_days: end-of-day indices e that produce (units appear at the start of e+1).
    Held stock (yield_units) is NOT included."""
    a = ANIMALS[species]
    first, iv = a["first_yield_day"], a["interval"]
    prod = [e for e in range(day, LAST_ROLLOVER_DAY + 1)
            if e + 1 - placed_day - first >= 0 and (e + 1 - placed_day - first) % iv == 0]
    pend = pending_care_bonus
    cared_units = 0
    for e in range(day, LAST_ROLLOVER_DAY + 1):
        if e in prod:
            cared_units += min(a["max_held"], 1 + pend)
            pend = 0
        pend += 1
    return {"base": len(prod), "cared": cared_units, "prod_days": prod,
            "next_prod_day": prod[0] if prod else None}


# ----------------------------------------------------------------------------------------- farm parse
def _farm(farm, day, step):
    tiles = _g(farm, "tiles") or []
    plants, animals, structures, weeds, empty, locked = [], [], [], [], [], []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if t is None:
                empty.append((x, y))
            elif t == "LOCKED":
                locked.append((x, y))
            elif isinstance(t, dict):
                kind = t.get("kind")
                if kind == "PLANT":
                    crop = t["crop"]
                    cd = CROPS[crop]
                    pd = int(t["planted_day"])
                    cu = int(t.get("consecutive_unwatered", 0))
                    wt = bool(t.get("watered_today", False))
                    yu = int(t.get("yield_units", 0))
                    fud = int(t.get("fertilized_until_day", -1))
                    mls = int(t.get("max_lifespan_step", -1))
                    age = day - pd
                    out = crop_outlook(crop, pd, day, wt, yu, fud, mls, step)
                    if cd["ongoing"]:
                        done = 0
                        for e in range(pd, day):
                            k = e + 1 - pd - cd["first_yield_day"]
                            if k >= 0 and k % cd["interval"] == 0 and k // cd["interval"] + 1 <= cd["max_yield"]:
                                done += 1
                    else:
                        done = None
                    plants.append({
                        "x": x, "y": y, "crop": crop, "planted_day": pd, "age": age, "ongoing": cd["ongoing"],
                        "watered_today": wt, "consecutive_unwatered": cu,
                        "must_water_today": (not wt) and cu >= 1,
                        "yield": yu, "harvestable": yu > 0 and age >= cd["first_yield_day"],
                        "fertilized_until_day": fud, "fert_active": fud >= day,
                        "decay_step": mls if mls >= 0 else None,
                        "decaying": mls >= 0 and step >= mls,
                        "productions_done": done,
                        "outlook": out,
                    })
                elif "animal" in t:
                    sp = t["animal"]
                    a = ANIMALS[sp]
                    pdy = int(t["placed_day"])
                    cu = int(t.get("consecutive_unfed", 0))
                    fed = bool(t.get("fed_today", False))
                    cared = bool(t.get("cared_today", False))
                    pend = int(t.get("pending_care_bonus", 0))
                    yu = int(t.get("yield_units", 0))
                    animals.append({
                        "x": x, "y": y, "species": sp, "product": a["product"], "structure": kind,
                        "placed_day": pdy, "age": day - pdy,
                        "fed_today": fed, "cared_today": cared, "consecutive_unfed": cu,
                        "must_feed_today": (not fed) and cu >= 1,
                        "yield": yu, "max_held": a["max_held"], "pending_care_bonus": pend,
                        "fertilizer_available": bool(t.get("fertilizer_available", False)),
                        "outlook": animal_outlook(sp, pdy, day, fed, cared, pend, yu),
                    })
                elif kind == "WEED":
                    weeds.append((x, y))
                elif kind in ("COOP", "PASTURE"):
                    structures.append({"x": x, "y": y, "kind": kind})
    unlocked = list(_g(farm, "unlocked_quadrants") or ["NW"])
    n_land = len(unlocked) - 1
    fp = _g(farm, "farmer") or list(SPAWN)
    hands = [tuple(h) for h in (_g(farm, "hands") or [])]
    return {
        "money": float(_g(farm, "money", 0.0)),
        "unlocked": unlocked, "n_land": n_land,
        "next_land": LAND_ORDER[n_land] if n_land < len(LAND_ORDER) else None,
        "next_land_price": LAND_PRICES[n_land] if n_land < len(LAND_PRICES) else None,
        "units_pos": [tuple(fp)] + hands,
        "hires_today": int(_g(farm, "hires_today", 0)),
        "plants": plants, "animals": animals, "structures": structures,
        "weeds": weeds, "empty": empty, "locked": locked,
    }


def drain_per_tick(shops):
    """Units removed from the market inventory by the unlocked shops at every step with step % 4 == 0."""
    d = {p: 0 for p in PRODUCTS}
    for s in shops:
        prods = SHOPS.get(s, ())
        m = 2 if len(prods) == 1 else 1
        for p in prods:
            d[p] += m
    return d


def town_drain(shops, step_from, step_to):
    """Total town drain over steps step_from..step_to inclusive (shops fixed)."""
    t = drain_per_tick(shops)
    n_tick = sum(1 for s in range(step_from, step_to + 1) if s % SHOP_TICK == 0)
    n_ctr = sum(1 for s in range(step_from, step_to + 1) if s % CENTER_TICK == 0)
    return {p: t[p] * n_tick + (n_ctr if p in TOWN_CENTER_PRODUCTS else 0) for p in PRODUCTS}


def shops_at_day(day):
    """Number of shop instances unlocked at the start of `day`."""
    return min(MAX_SHOPS, max(0, day) // SHOP_UNLOCK_INTERVAL)


# ----------------------------------------------------------------------------------------- main entry
def extract_state(obs):
    """Clean planning state from any observation (own seat; the rival's private data is not visible).

    Keys: step, day, hour, player, rival_seat, steps_left (actions still to submit incl. this one),
    rollovers_left (end-of-day events still to come), money, rival_money,
    me / rival (farm dicts: money, unlocked, n_land, next_land(_price), units_pos, hires_today, plants,
    animals, structures (empty COOP/PASTURE), weeds, empty, locked),
    units (own: idx, pos, inv, load), next_hire_cost, shed (all 12 items), shed_total, shed_room, seeds,
    carried (sum of unit inventories), market (inventory, prices, shops, n_shops, drain_per_tick,
    drain_per_day, next_shop_day), clean (True at a hand-over-able instant: no hands, nothing carried)."""
    step = int(_g(obs, "step", 0))
    day = step // TURNS
    hour = step % TURNS
    player = int(_g(obs, "player", 0))
    farms = _g(obs, "farms") or []
    rival_seat = 1 - player
    priv = _g(obs, "private") or {}
    me = _farm(farms[player], day, step) if player < len(farms) else None
    rv = _farm(farms[rival_seat], day, step) if rival_seat < len(farms) else None

    shed_raw = _g(priv, "shed") or {}
    shed = {k: int(shed_raw.get(k, 0)) for k in SHED_ITEMS}
    for k, v in shed_raw.items():
        if k not in shed:
            shed[k] = int(v)
    shed_total = sum(shed.values())
    seeds_raw = _g(priv, "seeds") or {}
    seeds = {c: int(seeds_raw.get(c, 0)) for c in CROPS}
    invs = list(_g(priv, "inventories") or [{}])
    units = []
    carried = {}
    pos_list = me["units_pos"] if me else [SPAWN]
    for i, pos in enumerate(pos_list):
        inv = {k: int(v) for k, v in (invs[i] if i < len(invs) else {}).items() if int(v) > 0}
        for k, v in inv.items():
            carried[k] = carried.get(k, 0) + v
        units.append({"idx": i, "pos": tuple(pos), "inv": inv, "load": sum(inv.values())})

    market = _g(obs, "market") or {}
    town = _g(obs, "town") or {}
    shops = list(_g(town, "unlocked_shops") or [])
    per_tick = drain_per_tick(shops)
    per_day = {p: per_tick[p] * (TURNS // SHOP_TICK) + (1 if p in TOWN_CENTER_PRODUCTS else 0) for p in PRODUCTS}
    n_shops = len(shops)
    next_shop_day = SHOP_UNLOCK_INTERVAL * (n_shops + 1) if n_shops < MAX_SHOPS else None
    hires_today = me["hires_today"] if me else 0
    return {
        "step": step, "day": day, "hour": hour, "player": player, "rival_seat": rival_seat,
        "steps_left": max(0, LAST_ACTION_STEP - step + 1),
        "rollovers_left": max(0, LAST_ROLLOVER_DAY - day + 1),
        "money": me["money"] if me else 0.0, "rival_money": rv["money"] if rv else 0.0,
        "me": me, "rival": rv,
        "units": units, "next_hire_cost": fib(hires_today),
        "shed": shed, "shed_total": shed_total, "shed_room": max(0, SHED_CAP - shed_total),
        "seeds": seeds, "carried": carried,
        "market": {
            "inventory": {p: int((_g(market, "inventory") or {}).get(p, 0)) for p in PRODUCTS},
            "prices": {p: int((_g(market, "prices") or {}).get(p, 0)) for p in PRODUCTS},
            "shops": shops, "n_shops": n_shops,
            "drain_per_tick": per_tick, "drain_per_day": per_day, "next_shop_day": next_shop_day,
        },
        "clean": len(units) == 1 and not carried,
    }


# ----------------------------------------------------------------------------------------- handover views
def inflight(st, farm_key="me"):
    """Everything already paid for / already growing that the incoming planner inherits, with counts and a
    value at CURRENT prices (a yardstick, not a forecast: it ignores the price impact of selling)."""
    f = st[farm_key]
    pr = st["market"]["prices"]
    day = st["day"]
    crops = {}
    for p in f["plants"]:
        c = crops.setdefault(p["crop"], {"tiles": 0, "units": 0, "value": 0.0, "must_water": 0, "planted_days": []})
        c["tiles"] += 1
        u = p["outlook"]["units"]
        if p["outlook"]["ready_day"] > LAST_DAY or (not p["ongoing"] and p["planted_day"] + CROPS[p["crop"]]["first_yield_day"] > LAST_DAY):
            u = 0
        c["units"] += u
        c["value"] += u * pr.get(p["crop"], 0)
        c["must_water"] += p["must_water_today"]
        c["planted_days"].append(p["planted_day"])
    herd = {}
    for a in f["animals"]:
        h = herd.setdefault(a["species"], {"n": 0, "held": 0, "base_units": 0, "cared_units": 0, "value_base": 0.0,
                                           "value_cared": 0.0, "must_feed": 0, "placed_days": []})
        h["n"] += 1
        h["held"] += a["yield"]
        h["base_units"] += a["outlook"]["base"]
        h["cared_units"] += a["outlook"]["cared"]
        h["value_base"] += (a["outlook"]["base"] + a["yield"]) * pr.get(a["product"], 0)
        h["value_cared"] += (a["outlook"]["cared"] + a["yield"]) * pr.get(a["product"], 0)
        h["must_feed"] += a["must_feed_today"]
        h["placed_days"].append(a["placed_day"])
    out = {"crops": crops, "herd": herd,
           "structures_empty": {k: sum(1 for s in f["structures"] if s["kind"] == k) for k in ("COOP", "PASTURE")},
           "empty_tiles": len(f["empty"]), "weeds": len(f["weeds"]), "n_land": f["n_land"], "money": f["money"]}
    if farm_key == "me":
        shed = st["shed"]
        out["seeds_held"] = {c: n for c, n in st["seeds"].items() if n}
        out["seed_value_paid"] = sum(n * CROPS[c]["seed"] for c, n in st["seeds"].items())
        out["animals_unplaced"] = {a: shed.get(a, 0) + st["carried"].get(a, 0) for a in ANIMALS
                                   if shed.get(a, 0) + st["carried"].get(a, 0)}
        out["goods"] = {p: shed.get(p, 0) + st["carried"].get(p, 0) for p in PRODUCTS
                        if shed.get(p, 0) + st["carried"].get(p, 0)}
        out["goods_value"] = sum(n * pr.get(p, 0) for p, n in out["goods"].items())
    return out


def _task(kind, x, y, earliest, deadline, value, item=None, n=1, needs=None):
    return {"kind": kind, "x": x, "y": y, "item": item, "n": n, "earliest": earliest, "deadline": deadline,
            "value": float(value), "needs": dict(needs or {})}


def commitment_tasks(st):
    """The obligations a planner inherits at this instant, as Task dicts of the shared planner interface.

    WATER  every plant that dies tonight without water (value = its remaining crop at current prices);
    FEED   every animal that escapes tonight without feed (value = held + base future units), needs WHEAT;
    HARVEST every harvestable plant (single-harvest or ongoing) whose last reachable full-value step falls today:
           decay starts at hour 0 of the decay step (after unit actions), but every unit is back on the shed at
           hour 0, so the last full-value harvest of a remote tile is hour 23 of the day before (deadline
           decay_step - 1); a plant already decaying gets deadline = now.  On day 29 every single-harvest crop
           (deadline 718).  And every animal or ongoing crop at its held cap, with deadline = the end of its next
           production day (the production that the cap would waste).
    PLACE  every unplaced animal already bought, onto an empty matching structure (value = base output);
    PLANT  one task per seed already bought (item = crop; x,y = -1: the planner picks the tile).
    Deadlines are engine steps (inclusive).  Values use current prices and ignore price impact."""
    day, step = st["day"], st["step"]
    eod = min(day * TURNS + TURNS - 1, LAST_ACTION_STEP)
    pr = st["market"]["prices"]
    me = st["me"]
    tasks = []
    for p in me["plants"]:
        val = p["outlook"]["units"] * pr.get(p["crop"], 0)
        if p["must_water_today"] and day <= LAST_ROLLOVER_DAY:
            tasks.append(_task("WATER", p["x"], p["y"], step, eod, val))
        if p["harvestable"]:
            ds = p["decay_step"]
            last_full = None
            if ds is not None:
                last_full = step if ds <= step else ds - 1
            if not p["ongoing"] and day == LAST_DAY:
                last_full = min(last_full if last_full is not None else LAST_ACTION_STEP, LAST_ACTION_STEP)
            if last_full is not None and last_full <= eod:
                tasks.append(_task("HARVEST", p["x"], p["y"], step, max(step, last_full), p["yield"] * pr.get(p["crop"], 0), item=p["crop"], n=p["yield"]))
            elif p["ongoing"] and p["yield"] >= CROPS[p["crop"]]["max_yield"] and p["outlook"].get("gains"):
                prod_day = p["outlook"]["gains"][0][0] - 1          # gains hold (day the unit appears, +n)
                dl = min(prod_day * TURNS + TURNS - 1, LAST_ACTION_STEP)
                tasks.append(_task("HARVEST", p["x"], p["y"], step, max(step, dl), p["yield"] * pr.get(p["crop"], 0), item=p["crop"], n=p["yield"]))
    for a in me["animals"]:
        val = (a["yield"] + a["outlook"]["base"]) * pr.get(a["product"], 0)
        if a["must_feed_today"] and day <= LAST_ROLLOVER_DAY:
            tasks.append(_task("FEED", a["x"], a["y"], step, eod, val, needs={"WHEAT": 1}))
        if a["yield"] >= a["max_held"] and a["outlook"]["next_prod_day"] is not None:
            dl = min(a["outlook"]["next_prod_day"] * TURNS + TURNS - 1, LAST_ACTION_STEP)
            tasks.append(_task("HARVEST", a["x"], a["y"], step, max(step, dl), a["yield"] * pr.get(a["product"], 0), item=a["product"], n=a["yield"]))
    free = {"COOP": [s for s in me["structures"] if s["kind"] == "COOP"],
            "PASTURE": [s for s in me["structures"] if s["kind"] == "PASTURE"]}
    for sp, A in ANIMALS.items():
        n = st["shed"].get(sp, 0) + st["carried"].get(sp, 0)
        for i in range(n):
            out = animal_outlook(sp, day, day, False, False, 0, 0)
            val = out["base"] * pr.get(A["product"], 0)
            spot = free[A["structure"]].pop(0) if free[A["structure"]] else None
            tasks.append(_task("PLACE", spot["x"] if spot else -1, spot["y"] if spot else -1, step, eod, val,
                               item=sp, needs={sp: 1}))
    for c, n in st["seeds"].items():
        for i in range(n):
            out = crop_outlook(c, day, day, False, 1 if not CROPS[c]["ongoing"] else 0, -1, -1)
            ok = day + CROPS[c]["first_yield_day"] <= LAST_DAY
            tasks.append(_task("PLANT", -1, -1, step, eod, (out["units"] * pr.get(c, 0)) if ok else 0.0, item=c))
    return tasks


def summary(st, farm_key="me"):
    """Flat numbers for tables: counts by crop / species, structures, land, money."""
    f = st[farm_key]
    s = {"day": st["day"], "money": f["money"], "n_land": f["n_land"], "empty": len(f["empty"]), "weeds": len(f["weeds"]),
         "coop_empty": sum(1 for x in f["structures"] if x["kind"] == "COOP"),
         "pasture_empty": sum(1 for x in f["structures"] if x["kind"] == "PASTURE")}
    for c in CROPS:
        s["plant_" + c] = sum(1 for p in f["plants"] if p["crop"] == c)
    for a in ANIMALS:
        s["animal_" + a] = sum(1 for x in f["animals"] if x["species"] == a)
    if farm_key == "me":
        for c in CROPS:
            s["seed_" + c] = st["seeds"][c]
        for a in ANIMALS:
            s["shed_" + a] = st["shed"].get(a, 0)
        s["shed_total"] = st["shed_total"]
    return s


_GLYPH = {"WHEAT": "w", "CARROT": "c", "TOMATO": "t", "STRAWBERRY": "s", "MELON": "m", "GOOSE": "G", "COW": "K", "SHEEP": "S"}


def render(st, farm_key="me"):
    """10x10 text grid of a farm (debug)."""
    f = st[farm_key]
    g = [["." for _ in range(BOARD)] for _ in range(BOARD)]
    for (x, y) in f["locked"]:
        g[y][x] = "#"
    for (x, y) in f["weeds"]:
        g[y][x] = "x"
    for s in f["structures"]:
        g[s["y"]][s["x"]] = "C" if s["kind"] == "COOP" else "P"
    for p in f["plants"]:
        g[p["y"]][p["x"]] = _GLYPH[p["crop"]]
    for a in f["animals"]:
        g[a["y"]][a["x"]] = _GLYPH[a["species"]]
    return "\n".join(" ".join(r) for r in g)


# ----------------------------------------------------------------------------------------- the switch
def make_hybrid(tape_agent, planner_factory, handover_day=9, safe_fallback=True):
    """Kaggle agent: `tape_agent` plays every step before hour 0 of `handover_day`; at that step
    planner_factory(extract_state(obs), obs) builds the planner (any callable obs -> action), which plays the rest.

    The tape is not called after the switch (nothing it holds is needed: at hour 0 there are no hands and no
    cargo, and every commitment is visible in the observation - see inflight() / commitment_tasks()).
    If building the planner raises (planner_factory or extract_state), the tape keeps playing the whole game
    (it has seen every step so far, so it continues exactly as without the switch); box["build_error"] says why.
    If the planner raises, the step falls back to PASS for every unit (hands padded) so the game goes on.
    With safe_fallback=False both errors propagate.
    agent.handover holds diagnostics: step of the switch, planner build time, build error, planner errors."""
    start = handover_day * TURNS
    box = {"planner": None, "step": None, "build_s": None, "build_error": None, "errors": 0, "last_error": None}
    code = getattr(tape_agent, "__code__", None)
    n_args = code.co_argcount if code is not None and hasattr(code, "co_argcount") else 2

    def tape(obs, configuration):
        # the way kaggle_environments calls an agent: (obs, configuration) trimmed to the agent's co_argcount
        return tape_agent(*[obs, configuration][:n_args])

    def agent(obs, configuration=None):
        step = int(_g(obs, "step", 0))
        if step < start or box["build_error"] is not None:
            return tape(obs, configuration)
        if box["planner"] is None:
            import time as _t
            t0 = _t.perf_counter()
            try:
                pl = planner_factory(extract_state(obs), obs)
                if not callable(pl):
                    raise TypeError(f"planner_factory returned {type(pl).__name__}, not a callable")
                box["planner"] = pl
            except Exception as ex:      # never crash the game: the tape carries on
                box["build_error"] = f"{type(ex).__name__}: {ex}"[:200]
                if not safe_fallback:
                    raise
                return tape(obs, configuration)
            box["step"] = step
            box["build_s"] = _t.perf_counter() - t0
        try:
            return box["planner"](obs)
        except Exception as ex:          # never crash the game
            box["errors"] += 1
            box["last_error"] = f"{type(ex).__name__}: {ex}"[:200]
            if not safe_fallback:
                raise
            farms = _g(obs, "farms") or []
            me = farms[int(_g(obs, "player", 0))] if farms else {}
            return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_g(me, "hands") or [])], "market": []}
    agent.handover = box
    return agent


def audit_transition(st0, st1, farm_key="me"):
    """Losses between two states of the same farm (typically hour 0 of consecutive days): plants that became WEED
    (missed watering or decay; a plant harvested and its tile re-weeded overnight is counted too - weeds spawn on
    empty tiles with p=0.005), animals that escaped (exact: a placed animal can only leave by escaping), and held
    units that vanished without a harvest cannot be seen here (harvests are not in the state)."""
    f0, f1 = st0[farm_key], st1[farm_key]
    weeds1 = set(map(tuple, f1["weeds"]))
    struct1 = {(s["x"], s["y"]) for s in f1["structures"]}
    died = [{"x": p["x"], "y": p["y"], "crop": p["crop"], "planted_day": p["planted_day"], "yield": p["yield"],
             "was_must_water": p["must_water_today"]} for p in f0["plants"] if (p["x"], p["y"]) in weeds1]
    escaped = [{"x": a["x"], "y": a["y"], "species": a["species"], "held": a["yield"], "placed_day": a["placed_day"]}
               for a in f0["animals"] if (a["x"], a["y"]) in struct1]
    return {"plants_to_weed": died, "animals_escaped": escaped,
            "n_plants_to_weed": len(died), "n_escaped": len(escaped)}
