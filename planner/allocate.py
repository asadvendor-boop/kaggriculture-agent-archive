"""planner/allocate.py -- the planner's daily allocation: what to plant, buy, feed, care and fertilize today.

    from planner.allocate import plan_day
    plan = plan_day(obs)                                  # at hour 0 of every day from day 12
    plan = plan_day(obs, state, forecast, labour_budget)  # full signature (all but obs optional)
    plan = plan_day(state, obs, tracker=tracker)          # the argument order planner/planner.py uses also works

Standard library only; one call takes ~40-110 ms.  Needs planner/model.py: imported as a sibling (`from . import
model`, so it works inside any package, e.g. the private package of tools/planner/test_planner.py build), else as
planner.model, else -- inlined flat after model.py's code -- from the module globals (or call set_model(ns)).
planner/targets.py is optional: the leaders' herd table it would provide is embedded (herd_prior).
Tests and measurements: tools/planner/test_allocate.py (reports runs/planner/allocate_*report.txt).

RETURNS a dict (all coordinates are board (x, y); hours are hours of today):
  plant      [PlantItem]  each unpacks like the tuple (x, y, crop, hour) and reads like the dict {"x", "y", "crop",
             "hour", "fertilize", "value", "urgency", "dig"}: PLANT at or after `hour` (1 on a tile free at hour 0 --
             seeds bought at hour 0 arrive at step+1; hour+2 on a tile whose crop is harvested today -- HARVEST first);
             "fertilize" = the program fertilizes it on its first window / production day; "value" = the executor
             priority (value_mode "net": max(urgency, the planting's own realized net margin before labour), >= 1);
             "urgency" = what skipping this planting today loses (program value minus the value of starting a day
             later); "dig" = a WEED / finished ongoing crop must be dug first (also listed in `dig`).  Empty coops /
             pastures get no crop unless params["crops_on_structures"] (they need an explicit DIG).
  seeds      {crop: n}             BUY_SEED today (plantings minus seeds already held)
  animals    {species: n}          BUY_ANIMAL today (each one goes to a `place` tile)
  build      [(x, y, "COOP"|"PASTURE")]
  place      [(x, y, species)]
  feed       {species: bool}       any animal of that species to feed today; the exact list is feed_tiles
  care       {species: bool}       idem, care_tiles
  fertilize  [(x, y)]              existing plants to FERTILIZE today (ongoing crops: must also be WATERed today, the
                                   bonus needs a watered production day; WHEAT / CARROT / MELON: before today's WATER)
  land       bool                  BUY_LAND today (its tiles' plantings are already in `plant`)
  extras     feed_tiles, care_tiles [(x, y)]; dig [(x, y)]; buy {"FERTILIZER": n, "WHEAT": n} (BUY_PRODUCT today, only
             the items in params["buy_items"]); shortfall {"FERTILIZER": n, "WHEAT": n} = what must be bought today for
             the plan's FERTILIZE / FEED (whoever buys it; its cost is inside the plan's cash check);
             wheat_reserve (wheat to keep in the shed tonight = tomorrow's feed); info (model value of the whole plan,
             programs per tile, costs, op cost, land evaluation, dropped items, timing).

HOW IT DECIDES (all values are MARGIN dollars = our revenue + the rival's revenue loss, see MarginBook)
  * Market: a model.Forecast holding the rival's forecast sales (model.rival_sales_schedule) and our planned
    sales.  If `forecast` is None it is built here and loaded with our farm's projected output (plants harvested
    when ripe, animals fed + cared, wheat net of feed, fertilizer net of use), sold in lots after the drain ticks
    of the day it becomes available.  A passed forecast is assumed to already hold our existing supply; after the
    call it holds that plus today's decisions (params["commit"]).
  * Margin MR: selling one more unit at step t earns its price there and shifts the inventory of every LATER step
    by +1 (the town drain is fixed, so the effect is permanent): our later sales lose (a cost), the rival's later
    sales lose (a gain for the margin, weighted by `denial`, 1.0 = pure margin); the shift is absorbed where the
    path reaches the $1 floor.  Exact for the forecast's flows (tested against Forecast.marginal_revenue and brute
    force).  Decisions weight a unit sold d days ahead by (1 - horizon_discount x d) (the forecast cannot see the
    rival's future plantings, so late prices are optimistic), and scale revenue by the leaders' measured
    realization (actual / modelled units: wheat 0.85 ... melon 0.97, animals 0.75-0.85).
  * Crops: every tile of the farm gets a PROGRAM = a sequence of plantings until day 29, chosen by a per-tile DP
    over plant days (two states: the tile still needs a DIG or is clean): value(crop, day, fertilize) = realized
    margin MR of its units (sold after the harvest, same day) - seed - fertilizer at its margin value - unit-ops x
    op_cost, + the value of the tile after it frees.  Tiles are allocated one after another and each program's
    sales are committed to the forecast before the next tile is valued (no over-planting of one crop); occupied
    tiles plan their continuation from their free day, so today's choices see the future supply of the whole farm;
    a second pass re-optimises today's free tiles against everything else.  Only today's first plantings are
    executed; the plan is redone every day.  Late cutoffs come from the engine model (care_plan): a planting that
    cannot be harvested by step 716 has no units; partial ongoing crops keep the productions refreshed by day 28
    (tomato full crop <= day 18, strawberry <= 13, melon <= 19, wheat / carrot to days 25-27 with partial yields).
  * Labour: every unit-op (PLANT, WATER, FERTILIZE, HARVEST, DIG, FEED, CARE, COLLECT, BUILD, PLACE) costs op_cost
    = the next hire's cost per op at today's workload (fib(hands) / ops_per_unit, ~$6.6 at 10 hands, $10.7 at 11),
    or labour_budget["op_cost"].  Land uses the full convex hire curve for its extra workload.
  * Animals: bought only while the leaders' herd target (targets.py / embedded table, keyed on the visible shops)
    is above the herd AND the realized margin value (sales + fertilizer - price - feed wheat - ops) exceeds
    animal_min_value and beats the tile's crop program.  Feed / care (feed_policy "rule", default): feed while a
    production at an end-of-day <= 28 remains, care while one after today remains (the zero-value rule; lesson 1:
    supply cuts lose margin); "value" uses model.animal_op_values against the op cost.
  * Fertilize: an existing plant is fertilized today when the extra units (exact tile simulation; ongoing crops
    assume every later uncovered production is fertilized too) are worth more than one fertilizer at its margin
    value + one op, from our own stock + today's collections only; a purchase (FERTILIZER has no town drain, so a
    buy raises the price of every later rival sale) needs a gain of fert_buy_min_gain over its margin cost, at
    most fert_buy_cap a day.
  * Land: only quadrants in land_quadrants (default NE, SW: the leaders never buy SE); value = the new tiles'
    programs - price - extra hires on the fib curve > land_margin.
  * Cash: seeds + animals + land + today's FERTILIZER / feed-WHEAT shortfall at the exact ask quotes <= money - reserve
    (hires today + a margin); the lowest value-per-dollar
    items are dropped.  Labour cap: labour_budget["ops"] (or a plain number) limits today's NEW ops.

ENGINE-MEASURED EXECUTION (tools/planner/test_allocate.py --engine 120 --pin DIR; hybrid LiveRepair days 0-11 + planner/
  planner.py with use_allocate=True, paired against its built-in plan; runs/planner/allocate_engine_pinned_report.txt)
  * units harvested per planting / template units at the better fertilize option (20 games, planner 861c9a56):
    wheat 0.63-0.79, carrot 0.62-0.82, tomato planted days 18-22 0.59 (31% died dry), melon 0.88 but only 65% of
    melon units were sold (day-29 harvests), new strawberries 0.81; 10-25% of all new plantings died unharvested.
    params["realization"] = REALIZATION_ENGINE uses these (+ sales share); the default keeps the leaders' factors.
  * with the planner's executor the fib hire cost is not the binding constraint (both plans hire the same); halving
    op_cost was worse (-$490 / game vs op_cost 1.0 over 120 games).
  * plan["value"] on the urgency scale (10-25) lost PLANT tasks to the executor's other work, and crops planned on
    empty coops / pastures never ran (no DIG task): hence value_mode "net" and crops_on_structures False.
  * returning the hour-0 BUY_PRODUCTs double-bought with planner.py's own shortfall buys: buy_items () (the amounts
    are in plan["shortfall"]); animal_min_value 400 (bought animals partly escape; 400 >= 100 on the latest planner).
  * result (120 paired games, planner.py e0aa9dab, these defaults): margin +1,084 / game vs planner.py's built-in
    plan [-350, +2,661] (leader tapes +333, V94 +1,835), win flips +11 / -9; it plants more tomato / melon /
    strawberry and ~45 fewer wheat + carrot, and buys $2.5k more feed wheat (the main remaining leak).  Earlier planner 861c9a56: +1,551
    [+68, +3,050] (buy_items (), min value 100).

labour_budget: None | number (max new ops today) | {"ops": cap or None, "op_cost": $/op, "workload": ops today}.
fixed (evaluation): force today's decisions ({"plant": [...], "animals": [(x, y, sp)], "land": bool}); info["value"]
then scores them with the same continuation DP (tools/planner/test_allocate.py B2).
"""
import math
import time

try:                                                  # a sibling in any package (planner/, or a private package
    from . import model as M                          # built by tools/planner/test_planner.py build)
except ImportError:
    try:
        from planner import model as M
    except ImportError:                               # inlined into one flat file after model.py's code
        import types as _types
        M = _types.SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith("__")})


def set_model(ns):
    """Use `ns` (a module or namespace with model.py's names) as the engine model."""
    global M
    M = ns


REALIZATION_ENGINE = {"WHEAT": 0.74, "CARROT": 0.72, "TOMATO": 0.59, "STRAWBERRY": 0.8, "MELON": 0.6,
                      "GOOSE": 0.85, "COW": 0.8, "SHEEP": 0.75}   # measured with planner.py's executor (see docstring)
TPD = 24
LAST_STEP = 718
LAST_DAY = 29
LAST_REFRESH_DAY = 28
CROP_LIST = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_LIST = ("GOOSE", "COW", "SHEEP")
SALE_TICK_PHASE = 1          # leaders sell just after the drain tick (step % 4 == 1)

DEFAULTS = {
    "denial": 1.0,           # weight of the rival's revenue loss (1.0 = margin objective)
    "op_cost": None,         # $ per unit-op; None: marginal hire cost at today's workload
    "ops_per_unit": 13.5,    # useful ops per unit per day (executor benches: ~133 tasks + 12 support ops, ~10.8 units)
    "horizon_discount": 0.01,  # decisions weight a unit sold d days ahead by (1 - d x this): the forecast cannot see
                             # the rival's future plantings, so late prices are optimistic (plan values are unweighted)
    "visit_hour": 6,         # hour of the daily care visit in crop programs
    "replant_hour": 8,       # hour a tile freed by a harvest is replanted
    "first_hour": 1,         # earliest PLANT on a tile free at hour 0 (seeds bought at hour 0 arrive at hour 1)
    "sell_lag": 2,           # a harvest must leave this many steps to sell before step 718
    "sale_delay": 1,         # units are sold from this many steps after the harvest, at the drain-tick phase
    "passes": 2,             # 1 = one sweep; 2 = re-optimise today's free tiles against the whole plan
    "crops": CROP_LIST,
    "animals": True,
    "herd_extra": 0,         # animals allowed above the leaders' herd target
    "targets": None,         # a planner.targets.Targets (or a callable(species, shops, day) -> target); None: load
    "land": True,
    "land_quadrants": ("NE", "SW"),  # quadrants the planner may buy: the leaders buy NE (day 6) and SW (day 9) and
                             # never SE (targets.py); add "SE" to let the model decide it (info["land"] has its value)
    "land_last_day": 22,
    "land_margin": 500.0,    # land must be worth this much more than its price + extra hires
    "cash_reserve": 400.0,   # kept on top of today's hire cost
    "fert_use_share": 0.5,   # base forecast: share of collected fertilizer used on crops (not sold)
    "min_value": 1.0,        # a program must be worth more than this (margin $) to be planted
    "commit": "today",       # what stays in a passed forecast: "today" (today's decisions), "all", "none"
    "base_supply": None,     # None: add our farm's supply iff the forecast is built here
    "fert_buy_cap": 6,       # FERTILIZER bought per day at most (it has no town drain: a buy raises the price of every
                             # later rival sale; engine test: large buys lost margin)
    "fert_buy_min_gain": 40.0,  # a bought fertilizer must beat its margin cost (price + rival gain) by this much
    "fert_collect_share": 0.0,  # share of today's collectible fertilizer counted as available for today's FERTILIZE
                             # (0: shed stock only -- a consumer that buys every FERTILIZE shortfall at hour 0, before
                             # collections, turns anything more into purchases: engine test)
    "fert_crops": CROP_LIST,  # crops whose existing plants may be fertilized
    "op_cost_scale": 1.0,    # multiplies the labour price (the executor's real capacity may be tighter than the hire
                             # curve: missed optional waterings then kill plants)
    "animal_min_value": 400.0,  # an animal is bought only if its margin value (after realization) exceeds this (engine,
                             # 120 paired games: 400 >= 100 on the latest planner; bought animals partly escape)
    "realization": {"WHEAT": 0.85, "CARROT": 0.9, "TOMATO": 0.93, "STRAWBERRY": 0.9, "MELON": 0.97,
                    "GOOSE": 0.85, "COW": 0.8, "SHEEP": 0.75},
                             # expected share of the model's units actually sold: leaders' actual / modelled units
                             # (test_model: crops by fertilize option, animals from I2); scales revenue in decisions
    "feed_policy": "rule",   # "rule": feed / care while a production at an end-of-day <= 28 remains (leaders' zero-value
                             # rule; supply cuts lose margin, lesson 1); "value": model.animal_op_values vs op cost
    "value_mode": "net",     # PlantItem["value"] (the executor's task priority): "net" = max(urgency, the planting's own
                             # net margin before labour -- the scale of planner.py's built-in plan); "urgency" = what
                             # skipping today loses (engine: urgency values of 10-25 lost their PLANT tasks to the
                             # executor's other work, so tiles stayed empty; the urgency is kept in item["urgency"])
    "buy_items": (),         # items whose hour-0 BUY_PRODUCT is returned in plan["buy"] ("FERTILIZER", "WHEAT"); off: planner.py
                             # buys its own shortfalls in the same step, so returning them double-bought (engine: >= as good)
    "crops_on_structures": False,  # plan crops on EMPTY coops / pastures (they need an explicit DIG first; planner.py
                             # creates DIG tasks only for plant tiles and the executor auto-DIGs only weeds, so such
                             # PLANTs never ran in engine games); the tiles stay available for PLACE
}


# ----------------------------------------------------------------------------------------------- helpers
class PlantItem(dict):
    """One planting.  Unpacks like the tuple (x, y, crop, hour) -- `for x, y, c, h in plan["plant"]` -- and reads
    like the dict {"x", "y", "crop", "hour", "fertilize", "value", "dig"} that planner/planner.py consumes."""
    __slots__ = ()

    def _t(self):
        return (dict.__getitem__(self, "x"), dict.__getitem__(self, "y"), dict.__getitem__(self, "crop"),
                dict.__getitem__(self, "hour"))

    def __iter__(self):
        return iter(self._t())

    def __getitem__(self, k):
        if isinstance(k, int):
            return self._t()[k]
        return dict.__getitem__(self, k)


def _g(o, k, d=None):
    if isinstance(o, dict):
        return o.get(k, d)
    try:
        return o[k]
    except Exception:
        return getattr(o, k, d)


def _tiles(farm):
    return _g(farm, "tiles") or []


def _day_ticks(day, from_step):
    """Sale steps of `day` at the tick phase from `from_step` on (never past LAST_STEP)."""
    end = min(day * TPD + TPD - 1, LAST_STEP)
    s = max(from_step, day * TPD)
    first = s + ((SALE_TICK_PHASE - s) % 4)
    out = list(range(first, end + 1, 4))
    if not out:
        out = [min(max(s, day * TPD), LAST_STEP)] if s <= LAST_STEP else []
    return out


def sale_steps(step, units, delay=1):
    """Steps at which `units` harvested at `step` are sold: one per drain tick of that day after the harvest,
    round-robin (leaders sell in lots of about one per-tick drain just after each tick)."""
    if units <= 0 or step > LAST_STEP:
        return []
    ticks = _day_ticks(step // TPD, step + delay)
    if not ticks:
        return [LAST_STEP] * units
    return [ticks[k % len(ticks)] for k in range(units)]


def _fibc(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_curve(workload, ops_per_unit=13.5):
    """(hands, daily hire cost, marginal $ per op) for `workload` unit-ops in a day (the farmer is free)."""
    units = max(1, int(math.ceil(workload / ops_per_unit)))
    hands = units - 1
    cost = _fibc(hands + 1) - 1
    return hands, cost, _fibc(hands) / ops_per_unit


def _next_prod_day(species, placed_day, day):
    a = M.ANIMALS[species]
    k = day + 1 - placed_day - a["first_yield_day"]
    if k < 0:
        return placed_day + a["first_yield_day"] - 1
    return day + (-k) % a["interval"]


def feed_rule(species, placed_day, day):
    """Leaders' stop rule: feed while a production at an end-of-day <= 28 is still to come."""
    return day <= LAST_REFRESH_DAY and _next_prod_day(species, placed_day, day) <= LAST_REFRESH_DAY


def care_rule(species, placed_day, day):
    return day < LAST_REFRESH_DAY and _next_prod_day(species, placed_day, day + 1) <= LAST_REFRESH_DAY


# ----------------------------------------------------------------------------------------------- margin book
class MarginBook:
    """Margin view of a model.Forecast.

    mmr(P)[i] = margin change from selling ONE more unit of P at step step0+i (after the rival's and our planned
    units of that step): its price + the change of all our later revenue (<= 0) - denial x the change of the
    rival's later revenue (their forecast units x the price drop).  Negative-unit flows (buys) telescope the same
    way.  Inventories are evaluated at the nearest integer (sub-unit rival rates) -- tested against
    Forecast.marginal_revenue.  commit()/uncommit() change our flows in the forecast itself."""

    def __init__(self, fc, denial=1.0):
        self.fc = fc
        self.denial = float(denial)
        self.s0 = fc.step0
        self.n = fc.n
        self._mmr = {}
        self._tot = {}
        self._pc = {P: {} for P in M.PRODUCTS}

    def _compute(self, P):
        """One forward pass (same order as Forecast._run: rival flow, our flow, town drain) recording, per step,
        the effect of a +1 inventory shift entering that step; one backward pass sums the effects.  A shift is
        ABSORBED at a step where the shifted path reaches the $1 floor (a unit sold at $1 adds no inventory), so
        the result is exact with floors for integer flows (fractional rival rates: absorbed when >= 0.5 unit)."""
        fc = self.fc
        n, s0 = self.n, self.s0
        inv = float(fc.inv0[P])
        rival = fc.rival[P]
        ours = fc.ours[P]
        drain = fc.drain[P]
        pc = self._pc[P]
        mp = M.market_price

        def pr(x):
            k = int(x + 0.5) if x >= 0 else -int(-x + 0.5)
            v = pc.get(k)
            if v is None:
                v = mp(P, k)
                pc[k] = v
            return v
        eff = [0.0] * n
        absorb = [False] * n
        top = [0] * n
        rev_ours = 0.0
        rev_riv = 0.0
        den = self.denial
        for i in range(n):
            s = s0 + i
            shift = 1.0
            e = 0.0
            r = rival.get(s)
            if r:
                if r > 0 and r.__class__ is int:
                    x, xs, rv, rvs = inv, inv + 1, 0, 0
                    for _ in range(r):
                        p = pr(x)
                        rv += p
                        if p > 1:
                            x += 1
                        p = pr(xs)
                        rvs += p
                        if p > 1:
                            xs += 1
                    rev_riv += rv
                    e -= den * (rvs - rv)
                    shift = xs - x
                    inv = x
                elif r > 0:
                    p0 = pr(inv)
                    p1 = pr(inv + 1)
                    rev_riv += r * p0
                    e -= den * r * (p1 - p0)
                    if p0 > 1:
                        inv += r
                        if p1 <= 1 and r >= 0.5:
                            shift = 0.0
                else:
                    if r.__class__ is int:
                        for _ in range(-r):
                            inv -= 1
                            p0 = pr(inv)
                            rev_riv -= p0
                            e -= den * (p0 - pr(inv + 1))
                    else:
                        p0 = pr(inv - 1)
                        rev_riv += r * p0
                        e -= den * r * (pr(inv) - p0)
                        inv += r
            o = ours.get(s)
            if o:
                if o > 0:
                    x, xs, rv, rvs = inv, inv + shift, 0, 0
                    for _ in range(o):
                        p = pr(x)
                        rv += p
                        if p > 1:
                            x += 1
                        if shift > 0.5:
                            p = pr(xs)
                            rvs += p
                            if p > 1:
                                xs += 1
                    rev_ours += rv
                    if shift > 0.5:
                        e += rvs - rv
                        shift = xs - x
                    inv = x
                else:
                    for _ in range(-o):
                        inv -= 1
                        p0 = pr(inv)
                        rev_ours -= p0
                        if shift > 0.5:
                            e += p0 - pr(inv + 1)
            top[i] = pr(inv)
            eff[i] = e
            absorb[i] = shift < 0.5
            inv -= drain[i]
        out = [0.0] * n
        suf = 0.0
        for i in range(n - 1, -1, -1):
            p = top[i]
            out[i] = (p + suf) if p > 1 else 1.0
            suf = eff[i] + (0.0 if absorb[i] else suf)
        self._mmr[P] = out
        self._tot[P] = (rev_ours, rev_riv)

    def mmr(self, P):
        v = self._mmr.get(P)
        if v is None:
            self._compute(P)
            v = self._mmr[P]
        return v

    def set_discount(self, day0, per_day):
        """Decision weights: mmr_w(P)[i] = mmr(P)[i] x (1 - per_day x days after day0)."""
        self._disc = (day0, float(per_day))
        self._mmrw = {}

    def mmr_w(self, P):
        disc = getattr(self, "_disc", None)
        if not disc or disc[1] == 0.0:
            return self.mmr(P)
        m = self.mmr(P)
        c = self._mmrw.get(P)
        if c is not None and c[0] is m:
            return c[1]
        d0, k = disc
        s0 = self.s0
        w = [max(0.0, 1.0 - k * max(0, (s0 + i) // TPD - d0)) for i in range(self.n)]
        out = [a * b if a > 0 else a for a, b in zip(m, w)]
        self._mmrw[P] = (m, out)
        return out

    def totals(self, P):
        """(our revenue, rival revenue) of P under the current flows (forecast, $)."""
        if P not in self._tot:
            self._compute(P)
        return self._tot[P]

    def margin_total(self, products=None):
        s = 0.0
        for P in (products or M.PRODUCTS):
            a, b = self.totals(P)
            s += a - self.denial * b
        return s

    def value(self, P, steps, weighted=False):
        m = self.mmr_w(P) if weighted else self.mmr(P)
        s0, n = self.s0, self.n
        v = 0.0
        for s in steps:
            i = s - s0
            if 0 <= i < n:
                v += m[i]
        return v

    def commit(self, P, steps, sign=1):
        if not steps:
            return
        for s in steps:
            self.fc.add_ours(P, s, sign)
        self._mmr.pop(P, None)
        self._tot.pop(P, None)

    def uncommit(self, P, steps):
        self.commit(P, steps, -1)

    def add_flow(self, P, step, units):
        if units:
            self.fc.add_ours(P, step, units)
            self._mmr.pop(P, None)
            self._tot.pop(P, None)

    def snapshot(self):
        return {P: dict(d) for P, d in self.fc.ours.items()}

    def restore(self, snap):
        for P in M.PRODUCTS:
            self.fc.set_ours(P, snap.get(P, {}))
        self._mmr.clear()
        self._tot.clear()

    def mean_value(self, P, day_from, day_to, hour_from=6):
        """Mean margin MR of P over the tick-phase sale steps of days day_from..day_to."""
        m = self.mmr(P)
        tot, k = 0.0, 0
        for d in range(max(day_from, self.s0 // TPD), min(day_to, LAST_DAY) + 1):
            for s in _day_ticks(d, d * TPD + hour_from):
                i = s - self.s0
                if 0 <= i < self.n:
                    tot += m[i]
                    k += 1
        return tot / k if k else 0.0


# ----------------------------------------------------------------------------------------------- crop templates
_TEMPLATES = {}


def crop_template(crop, day, hour, fert, visit_hour=6, sell_lag=2, delay=1):
    """Engine-exact program of one planting (model.crop_outcome) reduced to what the DP needs:
    sale steps (one per unit), unit-ops (by day), fertilizers, seed, first day the tile is free again."""
    key = (crop, day, hour, fert, visit_hour, sell_lag, delay)
    t = _TEMPLATES.get(key)
    if t is not None:
        return t
    res = M.crop_outcome(crop, day, hour, fert, visit_hour, LAST_STEP, sell_lag)
    plan = res["plan"]
    ok = bool(plan.get("feasible")) and res["units"] > 0
    steps = []
    for s, u in res["harvests"]:
        steps.extend(sale_steps(s, u, delay))
    ops_by_day = {d: len(v) for d, v in plan["visits"].items()} if ok else {}
    free = plan.get("tile_free_day") if ok else None
    t = {"crop": crop, "day": day, "hour": hour, "fert": fert, "ok": ok, "units": res["units"] if ok else 0,
         "steps": steps, "ops": sum(ops_by_day.values()), "ops_by_day": ops_by_day,
         "ferts": res["ferts"] if ok else 0, "seed": M.CROPS[crop]["seed"],
         "free_day": free if free is not None else LAST_DAY + 1,
         "harvests": res["harvests"] if ok else []}
    _TEMPLATES[key] = t
    return t


# ----------------------------------------------------------------------------------------------- farm reading
def _tile_kind(t):
    if t is None:
        return "E"
    if t == "LOCKED":
        return "L"
    k = t.get("kind") if isinstance(t, dict) else None
    if k == "PLANT":
        return "P"
    if k == "WEED":
        return "W"
    if k in ("COOP", "PASTURE"):
        return "A" if "animal" in t else ("C" if k == "COOP" else "S")
    return "?"


def plant_free_day(t, day, step):
    """(first day the tile can take a new crop, hour) for an existing plant cared for as usual."""
    crop = t["crop"]
    cd = M.CROPS[crop]
    pd = int(t["planted_day"])
    if not cd["ongoing"]:
        ws, we = (cd["max_yield_day"] + 1) // 2, cd["max_yield_day"]
        y = int(t.get("yield_units", 0))
        fu = int(t.get("fertilized_until_day", -1))
        hd = None
        for d in range(day, LAST_DAY + 1):
            age = d - pd
            if ws <= age <= we and not (d == day and t.get("watered_today")):
                y = min(cd["max_yield"], y + (2 if fu >= d else 1))
            if age >= cd["first_yield_day"] and (y >= cd["max_yield"] or age >= we or d == LAST_DAY):
                hd = d
                break
        if hd is None:
            return LAST_DAY + 1, 0
        return hd, 0
    # ongoing: free two days after its 4th production (the plant decays into a WEED -> DIG), else never
    first = pd + cd["first_yield_day"] - 1
    last_e = first + (cd["max_yield"] - 1) * cd["interval"]
    if last_e > LAST_REFRESH_DAY:
        return LAST_DAY + 1, 0
    return max(day, last_e + 2), 0


def _plant_future(t, day, fert_today, policy=True):
    """Remaining life of an existing plant watered every day, harvested when ripe (single-harvest) or every day
    (ongoing); fertilized today if fert_today, and (ongoing, policy) on every later production day not covered.
    Returns ([(day, units)], ferts used)."""
    tile = dict(t)
    crop = tile["crop"]
    cd = M.CROPS[crop]
    ongoing = cd["ongoing"]
    pd = int(tile["planted_day"])
    we = cd["max_yield_day"]
    out = []
    ferts = 0
    for d in range(day, LAST_DAY + 1):
        if not M.is_plant(tile):
            break
        if d == day and fert_today:
            M.plant_fertilize(tile, d)
            ferts += 1
        elif d > day and policy and ongoing and d <= LAST_REFRESH_DAY:
            k = d + 1 - pd - cd["first_yield_day"]
            if k >= 0 and k % cd["interval"] == 0 and k // cd["interval"] + 1 <= cd["max_yield"] \
                    and tile.get("fertilized_until_day", -1) < d:
                M.plant_fertilize(tile, d)
                ferts += 1
        if d <= LAST_REFRESH_DAY or not ongoing:
            M.plant_water(tile, d)
        age = d - pd
        if ongoing:
            if tile["yield_units"] > 0 and age >= cd["first_yield_day"]:
                out.append((d, tile["yield_units"]))
                tile["yield_units"] = 0
        else:
            if age >= cd["first_yield_day"] and (tile["yield_units"] >= cd["max_yield"] or age >= we or d == LAST_DAY):
                out.append((d, tile["yield_units"]))
                break
        if d <= LAST_REFRESH_DAY:
            tile = M.plant_refresh(tile, d)
    return out, ferts


# ----------------------------------------------------------------------------------------------- base supply
def _spread_day(day, units, hour_from):
    """Sale steps for `units` becoming available on `day` from `hour_from`."""
    if units <= 0:
        return []
    return sale_steps(day * TPD + hour_from - 1, units, 1)


def add_base_supply(book, obs, params=None):
    """Load our farm's projected sales into the book: shed goods today, plants' harvests, animal products,
    wheat net of feed (a reserve of tomorrow's feed is kept; a shortfall is bought), fertilizer net of use.
    Returns {P: units} added (buys negative)."""
    A = dict(DEFAULTS)
    A.update(params or {})
    step = int(_g(obs, "step"))
    day = step // TPD
    me = int(_g(obs, "player"))
    farm = _g(obs, "farms")[me]
    priv = _g(obs, "private") or {}
    shed = dict(_g(priv, "shed") or {})
    fdict = _as_dict(farm)
    proj = M.project_farm_output(fdict, day, LAST_DAY, True)
    added = {P: 0 for P in M.PRODUCTS}
    for P in M.PRODUCTS:
        if P in ("WHEAT", "FERTILIZER"):
            continue
        row = proj[P]
        for i, u in enumerate(row):
            k = int(round(u))
            if i == 0:
                k += int(shed.get(P, 0))
            steps = _spread_day(day + i, k, 7 if i else max(1, step % TPD + 1))
            for s in steps:
                book.add_flow(P, s, 1)
            added[P] += len(steps)
    # wheat: production - feed, keep tomorrow's feed in the shed
    feed = [0] * (LAST_DAY - day + 2)
    for y, row in enumerate(_tiles(fdict)):
        for x, t in enumerate(row):
            if M.is_animal(t):
                for d in range(day, LAST_DAY + 1):
                    if feed_rule(t["animal"], int(t["placed_day"]), d):
                        feed[d - day] += 1
    bal = int(shed.get("WHEAT", 0))
    for i in range(LAST_DAY - day + 1):
        d = day + i
        bal += int(round(proj["WHEAT"][i])) - feed[i]
        keep = feed[i + 1] if d < LAST_DAY else 0
        if bal - keep > 0:
            k = bal - keep
            for s in _spread_day(d, k, 7):
                book.add_flow("WHEAT", s, 1)
            added["WHEAT"] += k
            bal = keep
        elif bal < 0:
            s = max(step, d * TPD - 1)
            book.add_flow("WHEAT", s, bal)
            added["WHEAT"] += bal
            bal = 0
    # fertilizer: collected + stock, a share used on crops, the rest sold
    stock = int(shed.get("FERTILIZER", 0))
    for i in range(LAST_DAY - day + 1):
        u = proj["FERTILIZER"][i] + (stock if i == 0 else 0)
        k = int(round(u * (1.0 - A["fert_use_share"])))
        for s in _spread_day(day + i, k, 7):
            book.add_flow("FERTILIZER", s, 1)
        added["FERTILIZER"] += k
    return added


def _as_dict(o):
    """Deep plain-dict copy of a (possibly kaggle Struct) observation fragment."""
    if isinstance(o, dict):
        return {k: _as_dict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_as_dict(v) for v in o]
    if hasattr(o, "keys") and hasattr(o, "__getitem__") and not isinstance(o, str):
        try:
            return {k: _as_dict(o[k]) for k in o.keys()}
        except Exception:
            return o
    return o


def build_forecast(obs, tracker=None, params=None, with_supply=True):
    """Forecast from obs with the rival's schedule (model.rival_sales_schedule) and, if with_supply, our farm's
    projected sales.  Returns (forecast, book)."""
    A = dict(DEFAULTS)
    A.update(params or {})
    o = _as_dict(obs)
    sched = M.rival_sales_schedule(o, tracker)
    fc = M.forecast_from_obs(o, rival_schedule=sched)
    book = MarginBook(fc, A["denial"])
    if with_supply:
        add_base_supply(book, o, A)
    return fc, book


# ----------------------------------------------------------------------------------------------- the tile DP
class _Tile:
    __slots__ = ("x", "y", "start", "hour", "dig", "today", "kind", "program", "value", "new", "idle", "nocrop")

    def __init__(self, x, y, start, hour, dig, kind, new=False, nocrop=False):
        self.x, self.y = x, y
        self.start = start          # first day a new crop may be planted
        self.hour = hour            # plant hour on the start day
        self.dig = dig              # a DIG op is needed before the first planting
        self.kind = kind            # E free, W weed, C/S empty coop/pasture, P plant (occupied), L new land
        self.today = False
        self.program = []           # [template, ...]
        self.value = 0.0
        self.idle = 0.0             # value of the tile's best program if nothing is planted on its start day
        self.new = new
        self.nocrop = nocrop        # no crop program (an empty structure kept for animals)


class _Ctx:
    def __init__(self, book, day, A, op_cost, fert_value):
        self.book = book
        self.day = day
        self.A = A
        self.op_cost = op_cost
        self.fert_value = fert_value
        self.crops = tuple(A["crops"])
        self.real = dict(A.get("realization") or {})


def _tmpl_value(ctx, t, mm):
    m = mm[t["crop"]]
    s0, n = ctx.book.s0, ctx.book.n
    v = 0.0
    for s in t["steps"]:
        i = s - s0
        if 0 <= i < n:
            v += m[i]
    return v * ctx.real.get(t["crop"], 1.0) - t["seed"] - t["ferts"] * ctx.fert_value - t["ops"] * ctx.op_cost


def _dp(ctx, tile):
    """Best program for one tile given the book: returns (value, [templates]) and sets tile.idle (the value if nothing
    is planted on the start day).  Two states per day: the tile still needs a DIG (a WEED / empty structure: the first
    planting costs one more op) or is clean (after any harvest).  O(days x crops x 2)."""
    A = ctx.A
    book = ctx.book
    mm = {c: book.mmr_w(c) for c in ctx.crops}
    n = LAST_DAY + 3
    V = [0.0] * n                 # clean tile free at hour `replant` of day d
    choice = [None] * n
    dig = bool(tile.dig)
    Vd = [0.0] * n if dig else V  # tile that still needs a DIG
    choice_d = [None] * n if dig else choice
    start = max(tile.start, ctx.day)
    oc = ctx.op_cost
    for d in range(LAST_DAY, start - 1, -1):
        best, bc = V[d + 1], None
        bestd, bcd = Vd[d + 1], None
        h = tile.hour if d == start else A["replant_hour"]
        for c in ctx.crops:
            for f in (False, True):
                t = crop_template(c, d, h, f, A["visit_hour"], A["sell_lag"], A["sale_delay"])
                if not t["ok"]:
                    continue
                v = _tmpl_value(ctx, t, mm)
                fd = t["free_day"]
                if fd <= LAST_DAY:
                    v += V[fd]
                if v > best + 1e-9:
                    best, bc = v, t
                if dig and v - oc > bestd + 1e-9:
                    bestd, bcd = v - oc, t
        V[d], choice[d] = best, bc
        if dig:
            Vd[d], choice_d[d] = bestd, bcd
    prog = []
    d = start
    ch = choice_d
    while d <= LAST_DAY:
        t = ch[d]
        if t is None:
            d += 1
            continue
        prog.append(t)
        d = t["free_day"]
        ch = choice
    tile.idle = Vd[start + 1] if start + 1 < n else 0.0
    return Vd[start], prog


def _commit_program(ctx, prog, sign=1):
    for t in prog:
        ctx.book.commit(t["crop"], t["steps"], sign)


# ----------------------------------------------------------------------------------------------- animals
def animal_program(species, day, book, op_cost, fert_value, wheat_value, build=True, A=None):
    """Margin value of buying one `species` today and keeping it (fed / cared by the leaders' rules, harvested
    daily at the visit hour, fertilizer collected daily).  Returns dict(value, steps, units, ops, ...)."""
    A = A or DEFAULTS
    days = range(day, LAST_DAY + 1)
    fed = [d for d in days if feed_rule(species, day, d)]
    cared = [d for d in fed if care_rule(species, day, d)]
    r = M.simulate_animal(species, day, fed_days=fed, cared_days=cared, harvest_days=None, collect_days=None,
                          until_day=LAST_DAY)
    P = M.ANIMALS[species]["product"]
    steps = []
    for d, u in r["harvests"]:
        steps.extend(sale_steps(d * TPD + A["visit_hour"] + 1, u, A["sale_delay"]))
    n_fert = len(r["fertilizer"])
    ops = len(fed) + len(cared) + len(r["harvests"]) + n_fert + 1 + (1 if build else 0)
    rev = book.value(P, steps, weighted=True) * float((A.get("realization") or {}).get(species, 1.0))
    cost = M.ANIMALS[species]["cost"] + len(fed) * wheat_value + ops * op_cost
    return {"species": species, "product": P, "steps": steps, "units": len(steps), "ferts": n_fert,
            "feeds": len(fed), "ops": ops, "revenue": rev, "value": rev + n_fert * fert_value - cost}


# Leaders' herd on the farm at hour 0 of each day 0..29: a + b x drain4 of the product (units drained per 4-step tick by
# the visible shops), fitted on 1,937 top-10 seats (planner/targets.json "stock", archive flavour) -- embedded so the
# prior also works in a single-file build without targets.json; Targets.herd_target is used when importable.
_HERD_PRIOR = {
    "GOOSE": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.081, -0.004), (0.2, 0.034), (0.2, 0.034), (0.216, -0.002), (1.014, 0.865),
              (1.087, 0.979), (1.096, 0.787), (1.269, 0.942), (1.335, 1.002), (1.744, 0.928), (1.814, 0.976), (1.833, 0.987),
              (1.874, 0.775), (1.876, 0.791), (1.877, 0.798), (1.817, 0.702), (1.811, 0.714), (1.811, 0.717), (1.866, 0.594),
              (1.852, 0.596), (1.837, 0.6), (1.818, 0.528), (1.8, 0.526), (1.78, 0.526), (1.759, 0.53), (1.725, 0.535),
              (1.55, 0.543)],
    "COW": [(0.0, 0.0), (2.19, 0.0), (2.185, 0.0), (2.255, 0.022), (2.38, 0.164), (2.448, 0.203), (2.521, 0.096), (4.182, 1.741),
            (4.28, 1.942), (4.293, 1.635), (4.991, 2.319), (5.055, 2.542), (5.033, 2.026), (4.923, 2.249), (4.909, 2.279),
            (4.799, 1.858), (4.758, 1.904), (4.733, 1.922), (4.613, 1.648), (4.538, 1.685), (4.209, 1.779), (4.119, 1.537),
            (3.656, 1.64), (3.499, 1.675), (3.225, 1.52), (3.1, 1.535), (2.729, 1.599), (2.593, 1.618), (2.286, 1.657),
            (1.529, 1.757)],
    "SHEEP": [(0.0, 0.0), (2.727, 0.0), (2.727, 0.0), (2.721, 0.029), (2.728, 0.118), (2.728, 0.197), (2.734, 0.113), (2.883, 1.617),
              (2.885, 1.851), (3.02, 1.633), (3.114, 2.144), (3.105, 2.822), (3.037, 2.605), (3.045, 2.979), (3.05, 3.101),
              (3.076, 2.483), (3.104, 2.623), (3.083, 2.672), (3.135, 2.215), (3.105, 2.269), (3.01, 2.305), (2.94, 2.016),
              (2.833, 2.03), (2.693, 2.043), (2.512, 1.784), (2.347, 1.794), (2.173, 1.795), (2.07, 1.798), (1.225, 1.95),
              (0.59, 1.093)],
}
_TARGETS = []                     # module cache: [Targets instance or False]


def herd_prior(species, shops, day):
    """Leaders' herd size of `species` at hour 0 of `day` for the visible shops (embedded targets.json table)."""
    day = max(0, min(LAST_DAY, int(day)))
    vis = list(shops)[:min(8, day // 3)]
    P = M.ANIMALS[species]["product"]
    d4 = 0
    for sh in vis:
        prods = M.SHOPS.get(sh, ())
        if P in prods:
            d4 += 2 if len(prods) == 1 else 1
    a, b = _HERD_PRIOR[species][day]
    return max(0.0, a + b * d4)


def _herd_target(A, species, shops, day):
    tg = A.get("targets")
    if tg is None:
        if not _TARGETS:
            T = False
            for how in ("rel", "abs"):
                try:
                    if how == "rel":
                        from .targets import Targets
                    else:
                        from planner.targets import Targets
                    T = Targets.load()
                    break
                except Exception:
                    T = False
            _TARGETS.append(T)
        tg = _TARGETS[0]
    try:
        if tg is False or tg is None:
            return herd_prior(species, shops, day)
        if callable(tg) and not hasattr(tg, "herd_target"):
            return float(tg(species, shops, day))
        return float(tg.herd_target(species, shops, day))
    except Exception:
        return herd_prior(species, shops, day)


# ----------------------------------------------------------------------------------------------- main entry
def plan_day(obs, state=None, forecast=None, labour_budget=None, tracker=None, params=None, fixed=None):
    """Today's allocation (see the module docstring).  `fixed` (evaluation): {"plant": [(x, y, crop, hour)],
    "animals": [(x, y, species)], "land": bool} forces today's new decisions instead of choosing them (tiles not
    listed stay empty today, continuations are still optimised); info["value"] then scores those decisions."""
    t_start = time.perf_counter()
    if _g(obs, "farms") is None and state is not None and _g(state, "farms") is not None:
        obs, state = state, obs                        # called as plan_day(state, obs, ...)
    A = dict(DEFAULTS)
    A.update(params or {})
    o = _as_dict(obs)
    step = int(o["step"])
    day = step // TPD
    hour = step % TPD
    me = int(o["player"])
    farm = o["farms"][me]
    priv = o.get("private") or {}
    shed = {k: int(v) for k, v in (priv.get("shed") or {}).items()}
    seeds_held = {c: int(v) for c, v in (priv.get("seeds") or {}).items()}
    money = float(farm.get("money", 0.0))
    shops = list((o.get("town") or {}).get("unlocked_shops") or [])
    tiles = farm["tiles"]

    # ---- market book
    built_here = forecast is None
    if built_here:
        fc = M.forecast_from_obs(o, rival_schedule=M.rival_sales_schedule(o, tracker))
    else:
        fc = forecast
    book = MarginBook(fc, A["denial"])
    snap0 = book.snapshot()
    base_supply = A["base_supply"] if A["base_supply"] is not None else built_here
    if base_supply:
        add_base_supply(book, o, A)
    snap_base = book.snapshot()
    margin_base = {P: book.margin_total([P]) for P in M.PRODUCTS}
    book.set_discount(day, A["horizon_discount"])

    # ---- labour price
    n_plants = sum(1 for row in tiles for t in row if M.is_plant(t))
    n_anim = sum(1 for row in tiles for t in row if M.is_animal(t))
    LB = labour_budget if isinstance(labour_budget, dict) else ({"ops": labour_budget} if labour_budget is not None else {})
    workload = float(LB.get("workload") or (1.3 * n_plants + 4.0 * n_anim + 5.0))
    hands0, hire_cost0, mop = hire_curve(workload, A["ops_per_unit"])
    op_cost = float(LB.get("op_cost") if LB.get("op_cost") is not None else (A["op_cost"] if A["op_cost"] is not None else mop))
    op_cost *= float(A["op_cost_scale"])
    ops_cap = LB.get("ops")

    fert_value = max(1.0, book.mean_value("FERTILIZER", day, min(LAST_DAY, day + 2)))
    wheat_value = max(1.0, book.mean_value("WHEAT", day, min(LAST_DAY, day + 3)))
    ctx = _Ctx(book, day, A, op_cost, fert_value)

    # ---- tiles
    T = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            k = _tile_kind(t)
            h_free = min(21, max(A["first_hour"], hour + 1))
            if k == "E":
                T.append(_Tile(x, y, day, h_free, False, "E"))
            elif k == "W":
                T.append(_Tile(x, y, day, h_free, True, "W"))
            elif k in ("C", "S"):
                T.append(_Tile(x, y, day, h_free, True, k, nocrop=not A["crops_on_structures"]))
            elif k == "P":
                fd, _ = plant_free_day(t, day, step)
                if fd <= LAST_DAY:
                    # an ongoing crop leaves a WEED (DIG); a harvested single-harvest crop leaves an empty tile
                    T.append(_Tile(x, y, fd, A["replant_hour"] if fd > day else min(21, max(2, hour + 2)),
                                   bool(M.CROPS[t["crop"]]["ongoing"]), "P"))
    for tl in T:
        tl.today = tl.start == day
    T.sort(key=lambda tl: (tl.start, tl.hour, tl.dig, abs(tl.x - 4.5) + abs(tl.y - 4.5)))

    fixed_plant = None
    if fixed is not None:
        fixed_plant = {(int(x), int(y)): (c, int(h)) for x, y, c, h in (fixed.get("plant") or [])}

    def solve(tl):
        if tl.nocrop and not (fixed_plant is not None and tl.today and (tl.x, tl.y) in fixed_plant):
            tl.idle = 0.0
            return 0.0, []
        if fixed_plant is not None and tl.today:
            fp = fixed_plant.get((tl.x, tl.y))
            if fp is None:
                # forced idle today: continuation from tomorrow
                saved = (tl.start, tl.hour, tl.dig)
                tl.start, tl.hour = day + 1, A["replant_hour"]
                v, prog = _dp(ctx, tl)
                tl.start, tl.hour, tl.dig = saved
                return v, prog
            c, h = fp
            saved = (tl.start, tl.hour, tl.dig)
            tl.start, tl.hour = day + 1, A["replant_hour"]
            idle_v, _ = _dp(ctx, tl)
            tl.start, tl.hour, tl.dig = saved
            best = None
            for f in (False, True):
                t0 = crop_template(c, day, max(h, hour), f, A["visit_hour"], A["sell_lag"], A["sale_delay"])
                if not t0["ok"]:
                    continue
                mm = {c: book.mmr_w(c)}
                v0 = _tmpl_value(ctx, t0, mm) - (op_cost if tl.dig else 0.0)
                cont = 0.0
                cprog = []
                if t0["free_day"] <= LAST_DAY:
                    saved = (tl.start, tl.hour, tl.dig)
                    tl.start, tl.hour, tl.dig = t0["free_day"], A["replant_hour"], False
                    cont, cprog = _dp(ctx, tl)
                    tl.start, tl.hour, tl.dig = saved
                if best is None or v0 + cont > best[0]:
                    best = (v0 + cont, [t0] + cprog)
            tl.idle = idle_v
            return best if best is not None else (0.0, [])
        return _dp(ctx, tl)

    # pass 1: every tile in order of its free day, committing its program
    for tl in T:
        v, prog = solve(tl)
        tl.value, tl.program = v, prog
        _commit_program(ctx, prog)
    # pass 2: today's free tiles re-optimised against the whole plan
    if A["passes"] >= 2:
        for tl in T:
            if not tl.today:
                continue
            _commit_program(ctx, tl.program, -1)
            v, prog = solve(tl)
            tl.value, tl.program = v, prog
            _commit_program(ctx, prog)

    # ---- animals: the leaders' herd target as the prior, bought only at a positive margin value
    counts = {a: 0 for a in ANIMAL_LIST}
    for row in tiles:
        for t in row:
            if M.is_animal(t):
                counts[t["animal"]] += 1
    buys = []                                      # (x, y, species, build, value)
    fixed_animals = None if fixed is None else list(fixed.get("animals") or [])
    if fixed_animals is not None:
        for x, y, sp in fixed_animals:
            tl = next((q for q in T if q.x == x and q.y == y and q.today), None)
            k = _tile_kind(tiles[y][x]) if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else "?"
            build = k not in ("C", "S")
            ap = animal_program(sp, day, book, op_cost, fert_value, wheat_value, build, A)
            if tl is not None:
                _commit_program(ctx, tl.program, -1)
                tl.program, tl.value = [], 0.0
            book.commit(ap["product"], ap["steps"])
            buys.append((x, y, sp, build, ap["value"]))
    elif A["animals"] and day <= 24:
        for sp in ANIMAL_LIST:
            tgt = _herd_target(A, sp, shops, day)
            if tgt is None:
                continue
            want = int(round(tgt)) + int(A["herd_extra"]) - counts[sp]
            struct = M.ANIMALS[sp]["structure"]
            for _ in range(max(0, want)):
                cands = [tl for tl in T if tl.today and tl.kind in ("E", "W", "C", "S")
                         and (tl.kind in ("E", "W") or (tl.kind == "C") == (struct == "COOP"))]
                if not cands:
                    break
                best = None
                for tl in cands:
                    build = tl.kind in ("E", "W")
                    ap = animal_program(sp, day, book, op_cost, fert_value, wheat_value, build, A)
                    gain = ap["value"] - (op_cost if tl.kind == "W" else 0.0) - tl.value
                    if best is None or gain > best[0]:
                        best = (gain, tl, ap, build)
                if best is None or best[0] <= 0 or best[2]["value"] <= A["animal_min_value"]:
                    break
                gain, tl, ap, build = best
                _commit_program(ctx, tl.program, -1)
                tl.program, tl.value = [], ap["value"]
                tl.kind = "A"
                book.commit(ap["product"], ap["steps"])
                buys.append((tl.x, tl.y, sp, build, ap["value"]))
                counts[sp] += 1

    # ---- land
    land = False
    land_info = None
    n_land = len(farm.get("unlocked_quadrants") or ["NW"]) - 1
    nl = M.next_land(farm.get("unlocked_quadrants") or ["NW"])
    want_land_eval = (fixed is None and A["land"] and nl is not None and day <= A["land_last_day"]
                      and nl[0] in tuple(A["land_quadrants"]))
    if fixed is not None and fixed.get("land") and nl is not None:
        want_land_eval = True
    new_tiles = []
    if want_land_eval:
        q, price = nl
        spend_so_far = _spend(T, buys, seeds_held, day)
        if money - spend_so_far - hire_cost0 - A["cash_reserve"] >= price or fixed is not None:
            snap_l = book.snapshot()
            cand = []
            for y in range(len(tiles)):
                for x in range(len(tiles[y])):
                    if tiles[y][x] == "LOCKED" and M.quadrant_of(x, y) == q:
                        cand.append(_Tile(x, y, day, min(21, max(A["first_hour"], hour + 1)), False, "L", new=True))
            cand.sort(key=lambda tl: abs(tl.x - 4.5) + abs(tl.y - 4.5))
            tot = 0.0
            ops_by_day = {}
            for tl in cand:
                tl.today = True
                v, prog = solve(tl)
                tl.value, tl.program = v, prog
                _commit_program(ctx, prog)
                tot += v
                for t in prog:
                    for d, k in t["ops_by_day"].items():
                        ops_by_day[d] = ops_by_day.get(d, 0) + k
            # the DP charged op_cost per op; the real extra hands follow the convex fib curve
            extra = 0.0
            for d, k in ops_by_day.items():
                _, c0, _ = hire_curve(workload, A["ops_per_unit"])
                _, c1, _ = hire_curve(workload + k, A["ops_per_unit"])
                extra += (c1 - c0) - k * op_cost
            net = tot - price - extra
            land_info = {"quadrant": q, "price": price, "program_value": round(tot, 1), "extra_hire": round(extra, 1),
                         "net": round(net, 1)}
            forced = fixed is not None and fixed.get("land")
            if forced or (fixed is None and net > A["land_margin"]):
                land = True
                new_tiles = cand
                land_info["tiles_planted_today"] = sum(1 for tl in cand if tl.program and tl.program[0]["day"] == day)
            else:
                book.restore(snap_l)

    # ---- today's decisions
    plant, dig, build_l, place_l = [], [], [], []
    today_items = []                                   # (value, cash, ops, kind, payload)
    for tl in T + new_tiles:
        if not tl.today or not tl.program:
            continue
        t0 = tl.program[0]
        if t0["day"] != day:
            continue
        v0 = tl.value - tl.idle                        # urgency: what skipping today's planting would lose
        ops0 = 2 + (1 if tl.dig else 0)                # PLANT + planting-day WATER (+ DIG)
        today_items.append([v0, t0["seed"], ops0, "plant", (tl, t0)])
    prio = {}                                          # executor priority of today's plantings (value_mode)
    for it in today_items:
        tl, t0 = it[4]
        net0 = _tmpl_value(ctx, t0, {t0["crop"]: book.mmr_w(t0["crop"])}) + t0["ops"] * op_cost
        prio[(tl.x, tl.y)] = max(it[0], net0) if A["value_mode"] == "net" else it[0]
    for (x, y, sp, build, v) in buys:
        today_items.append([v, M.ANIMALS[sp]["cost"], 2 if build else 1, "animal", (x, y, sp, build)])

    # ---- fertilize existing plants
    fert_cands = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if not M.is_plant(t) or day > LAST_REFRESH_DAY or t["crop"] not in A["fert_crops"]:
                continue
            if t.get("fertilized_until_day", -1) >= day + 2:
                continue
            P = t["crop"]
            h1, f1 = _plant_future(t, day, True)
            h0, f0 = _plant_future(t, day, False)
            if sum(u for _, u in h1) <= sum(u for _, u in h0) and f1 >= f0:
                continue

            rf = ctx.real.get(P, 1.0)

            def val(hs, nf):
                v = 0.0
                for d, u in hs:
                    v += book.value(P, sale_steps(d * TPD + A["visit_hour"] + 1, u, A["sale_delay"]), weighted=True)
                return v * rf - nf * fert_value
            gain = val(h1, f1) - val(h0, f0) - op_cost
            if gain > 0:
                fert_cands.append((gain, x, y))
    fert_cands.sort(reverse=True)
    collectible = sum(1 for row in tiles for t in row if M.is_animal(t) and t.get("fertilizer_available"))
    fert_avail = int(shed.get("FERTILIZER", 0)) + int(A["fert_collect_share"] * collectible)
    fert_buy = 0
    fertilize = []
    fbuy_price = M.market_price("FERTILIZER", int(o["market"]["inventory"]["FERTILIZER"]) - 1)
    i_now = min(book.n - 1, max(0, step + 1 - book.s0))
    fbuy_cost = max(float(fbuy_price), book.mmr("FERTILIZER")[i_now]) if book.n else float(fbuy_price)
    for gain, x, y in fert_cands:
        if len(fertilize) < fert_avail:
            fertilize.append((x, y))
        elif (gain + fert_value - fbuy_cost > A["fert_buy_min_gain"] and fert_buy < A["fert_buy_cap"]
              and day < LAST_REFRESH_DAY):
            fertilize.append((x, y))
            fert_buy += 1

    # ---- feed / care
    feed_tiles, care_tiles = [], []
    uval = {}
    for P in ("EGG", "MILK", "WOOL"):
        uval[P] = max(0.0, book.mean_value(P, day, LAST_DAY))
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if not M.is_animal(t):
                continue
            P = M.ANIMALS[t["animal"]]["product"]
            sp, pdy = t["animal"], int(t["placed_day"])
            if A["feed_policy"] == "value":
                ov = M.animal_op_values(t, day, uval[P], fert_value, wheat_value, LAST_REFRESH_DAY)
                do_feed = ov.get("FEED", 0.0) > op_cost
                do_care = ov.get("CARE", 0.0) > op_cost
            else:
                do_feed = feed_rule(sp, pdy, day)
                do_care = care_rule(sp, pdy, day)
            if (not t.get("fed_today")) and do_feed:
                feed_tiles.append((x, y))
            if (not t.get("cared_today")) and do_care and ((x, y) in feed_tiles or t.get("fed_today")):
                care_tiles.append((x, y))
    feed = {a: False for a in ANIMAL_LIST}
    care = {a: False for a in ANIMAL_LIST}
    for x, y in feed_tiles:
        feed[tiles[y][x]["animal"]] = True
    for x, y in care_tiles:
        care[tiles[y][x]["animal"]] = True
    feed_tomorrow = sum(1 for row in tiles for t in row if M.is_animal(t)
                        and feed_rule(t["animal"], int(t["placed_day"]), day + 1)) + sum(1 for b in buys)
    wheat_need_today = len(feed_tiles) + sum(1 for b in buys if feed_rule(b[2], day, day))
    ripe_wheat = 0
    for row in tiles:
        for t in row:
            if M.is_plant(t) and t["crop"] == "WHEAT":
                fd, _ = plant_free_day(t, day, step)
                if fd == day:
                    ripe_wheat += int(t.get("yield_units", 0)) + 1
    wheat_buy = max(0, wheat_need_today - int(shed.get("WHEAT", 0)) - ripe_wheat // 2)

    # ---- feasibility: cash and today's labour
    inv_m = o["market"]["inventory"]
    short_cash = {it: float(sum(M.buy_quotes(it, int(inv_m[it]), n)[0])) if n > 0 else 0.0
                  for it, n in (("WHEAT", wheat_buy), ("FERTILIZER", fert_buy))}
    cash_free = money - hire_cost0 - A["cash_reserve"] - short_cash["WHEAT"] - short_cash["FERTILIZER"]
    if land:
        cash_free -= nl[1]
    ops_extra = len(fertilize)
    today_items.sort(key=lambda it: -(it[0] / max(1.0, it[1])))
    dropped = {"cash": 0, "ops": 0, "value": 0}
    kept = []
    seeds_left = dict(seeds_held)
    cash_used = 0.0
    ops_used = ops_extra
    for it in today_items:
        v, cash, ops, kind, pl = it
        if kind == "plant" and v < A["min_value"] and fixed is None:
            dropped["value"] += 1
            continue
        c_need = cash
        if kind == "plant":
            crop = pl[1]["crop"]
            if seeds_left.get(crop, 0) > 0:
                c_need = 0.0
        if cash_used + c_need > cash_free and fixed is None:
            dropped["cash"] += 1
            continue
        if ops_cap is not None and ops_used + ops > ops_cap and fixed is None:
            dropped["ops"] += 1
            continue
        if kind == "plant" and c_need == 0.0:
            seeds_left[pl[1]["crop"]] -= 1
        cash_used += c_need
        ops_used += ops
        kept.append(it)
    seeds_buy = {}
    animals_buy = {}
    kept_tiles = set()
    for v, cash, ops, kind, pl in kept:
        if kind == "plant":
            tl, t0 = pl
            plant.append(PlantItem(x=tl.x, y=tl.y, crop=t0["crop"], hour=int(t0["hour"]), fertilize=bool(t0["fert"]),
                                   value=round(max(1.0, prio.get((tl.x, tl.y), v)), 1), urgency=round(v, 1),
                                   dig=bool(tl.dig)))
            kept_tiles.add((tl.x, tl.y))
            if tl.dig and tl.kind in ("W", "C", "S"):
                dig.append((tl.x, tl.y))
        else:
            x, y, sp, build = pl
            animals_buy[sp] = animals_buy.get(sp, 0) + 1
            if build:
                if _tile_kind(tiles[y][x]) == "W":
                    dig.append((x, y))
                build_l.append((x, y, M.ANIMALS[sp]["structure"]))
            place_l.append((x, y, sp))
    need = {}
    for x, y, c, h in plant:
        need[c] = need.get(c, 0) + 1
    for c, n in need.items():
        k = n - seeds_held.get(c, 0)
        if k > 0:
            seeds_buy[c] = k
    # plans dropped for cash / labour: take their sales out of the book
    for it in today_items:
        if it in kept:
            continue
        if it[3] == "plant":
            tl, t0 = it[4]
            _commit_program(ctx, tl.program, -1)
            tl.program, tl.value = [], 0.0
        else:
            x, y, sp, build = it[4]
            ap = animal_program(sp, day, book, op_cost, fert_value, wheat_value, build, A)
            book.uncommit(ap["product"], ap["steps"])

    # ---- value of today's decisions (model margin) and what stays in the forecast
    real = dict(A.get("realization") or {})
    real_p = {P: real.get(P, real.get({"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}.get(P, ""), 1.0))
              for P in M.PRODUCTS}
    d_margin = {P: book.margin_total([P]) - margin_base[P] for P in M.PRODUCTS}
    gross = sum(d_margin.values())
    gross_real = sum(d_margin[P] * real_p[P] for P in M.PRODUCTS)
    cost_all = 0.0
    ops_all = 0
    for tl in T + new_tiles:
        for t in tl.program:
            cost_all += t["seed"] + t["ferts"] * fert_value + t["ops"] * op_cost
            ops_all += t["ops"]
        if tl.program and tl.dig:
            cost_all += op_cost
    animal_cost = 0.0
    for v, cash, ops, kind, pl in kept:
        if kind == "animal":
            sp = pl[2]
            ap = animal_program(sp, day, book, op_cost, fert_value, wheat_value, pl[3], A)
            animal_cost += M.ANIMALS[sp]["cost"] + ap["feeds"] * wheat_value + ap["ops"] * op_cost - ap["ferts"] * fert_value
    land_cost = nl[1] + (land_info["extra_hire"] if land_info else 0.0) if land else 0.0
    plan_value = gross_real - cost_all - animal_cost - land_cost

    info = {
        "day": day, "value": round(plan_value, 1), "margin_gross": round(gross, 1),
        "margin_gross_realized": round(gross_real, 1),
        "costs": {"crops": round(cost_all, 1), "animals": round(animal_cost, 1), "land": round(land_cost, 1)},
        "op_cost": round(op_cost, 2), "fert_value": round(fert_value, 2), "wheat_value": round(wheat_value, 2),
        "workload": workload, "hands": hands0, "program_ops": ops_all,
        "ops_today": ops_used, "cash_today": round(cash_used + (nl[1] if land else 0) + short_cash["WHEAT"]
                                                  + short_cash["FERTILIZER"], 1),
        "cash_free": round(cash_free, 1), "dropped": dropped, "land": land_info,
        "programs": [((tl.x, tl.y), [(t["crop"], t["day"], t["fert"]) for t in tl.program], round(tl.value, 1))
                     for tl in T + new_tiles if tl.program],
        "today_values": sorted([(round(it[0], 1), it[3], (it[4][0].x, it[4][0].y, it[4][1]["crop"]) if it[3] == "plant"
                                 else it[4][:3]) for it in kept], reverse=True),
        "fert_candidates": len(fert_cands), "fert_buy": fert_buy,
    }
    commit = A["commit"]
    if commit == "none" or (commit == "today" and not built_here):
        # keep only today's decisions in the caller's forecast
        today_steps = []
        for tl in T + new_tiles:
            if tl.program and tl.program[0]["day"] == day and (tl.x, tl.y) in kept_tiles:
                today_steps.append((tl.program[0]["crop"], tl.program[0]["steps"]))
        book.restore(snap_base if base_supply and not built_here else snap0)
        if commit == "today":
            for P, st in today_steps:
                book.commit(P, st)
            for v, cash, ops, kind, pl in kept:
                if kind == "animal":
                    ap = animal_program(pl[2], day, book, op_cost, fert_value, wheat_value, pl[3], A)
                    book.commit(ap["product"], ap["steps"])
    info["ms"] = round(1000 * (time.perf_counter() - t_start), 1)
    return {
        "day": day,
        "plant": plant, "seeds": seeds_buy, "animals": animals_buy, "build": build_l, "place": place_l,
        "feed": feed, "care": care, "fertilize": fertilize, "land": land,
        "feed_tiles": feed_tiles, "care_tiles": care_tiles, "dig": dig,
        "buy": {k: v for k, v in (("FERTILIZER", fert_buy), ("WHEAT", wheat_buy)) if v and k in A["buy_items"]},
        "shortfall": {k: v for k, v in (("FERTILIZER", fert_buy), ("WHEAT", wheat_buy)) if v},
        "wheat_reserve": feed_tomorrow,
        "info": info,
    }


def _spend(T, buys, seeds_held, day):
    need = {}
    for tl in T:
        if tl.today and tl.program and tl.program[0]["day"] == day:
            c = tl.program[0]["crop"]
            need[c] = need.get(c, 0) + 1
    s = sum(max(0, n - seeds_held.get(c, 0)) * M.CROPS[c]["seed"] for c, n in need.items())
    s += sum(M.ANIMALS[b[2]]["cost"] for b in buys)
    return s
