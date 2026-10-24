"""planner/model.py -- engine facts and market forecaster for Kaggriculture.

Ground truth: kaggle_environments 1.32.7, envs/kaggriculture/kaggriculture.py.
Pure Python, standard library only.  Every engine rule restated here is checked
against the official interpreter and the bit-exact kagsim_trace engine by
tools/planner/test_model.py; forecasts are labelled as such.

TIME (verified)
  * The agent sees obs["step"] = s and the interpreter applies that action at
    step s with day = s // 24, hour = s % 24.
  * Order inside step s: unit actions (farmer, then hands in list order; seat 0
    then seat 1) -> market orders (slot by slot, lockstep between seats) ->
    town drain (every unlocked shop if s % 4 == 0, town centre if s % 24 == 0)
    -> price refresh -> plant decay(s) -> end-of-day refresh if s % 24 == 23.
  * The last applied step is 718 (day 29, hour 22).  There is NO end-of-day
    refresh for day 29; the last refresh is the end of day 28.  Anything still
    in a shed or a unit's hands after step 718 is worth nothing.

TILES are the engine's own dicts (the same keys as obs["farms"][i]["tiles"]),
so every stepper here can run directly on (copies of) observed tiles.

API INDEX
  prices     market_price(P, inv) [exact], price(P, inv) [cached], buy_price, sell_quotes, buy_quotes,
             lockstep_trade (both seats' same-slot orders), inventory_for_price
  costs      hire_cost(n_already_today) = fib(n): 1,1,2,3,5,...; hires_total_cost(k) = fib(k+1)-1;
             next_land(unlocked) -> (NE,1000),(SW,2000),(SE,4000)
  shed       shed_room, drop_into_shed (DROP / day-end drop: overflow DISCARDED), place_into_shed (PLACE: no loss),
             spawn_tile (where a hired hand appears), SHED_ACCESS, blocked_plant_crops (atomic PLANT rule)
  plants     new_plant, plant_water, plant_fertilize, harvest, plant_decay, plant_refresh (engine steppers),
             window, production_refresh_days, dry_death_day, simulate_plant (exact tile life),
             care_plan / crop_outcome (minimal-labour schedule and its exact yield), plan_tasks (Task dicts)
  animals    new_animal, animal_feed/care/collect, animal_refresh, animal_production_days, simulate_animal,
             animal_outcome, animal_steady_rate, herd_for_demand, animal_op_values (dollars at stake today)
  town       shop_tick_drain, drain_at_step [exact], active_shop_count, shop_unlock_day, daily_drain,
             EXPECTED_SLOT_DRAIN, drain_path, expected_future_drain, shop_scenarios
  forecast   Forecast (inventory/price path given our flows + rival flows; marginal_revenue(_all); allocate),
             forecast_from_obs, FlowTracker (exact rival net flow per step), project_farm_output,
             estimate_rival_sales (next days), rival_sales_schedule (per day to day 29)
  valuation  value_of_crop_on_tile, rank_crops
"""
import math

# ---------------------------------------------------------------------------
# constants (verbatim from the engine; test_model.py compares them)
# ---------------------------------------------------------------------------
TPD = 24                      # turnsPerDay
EPISODE_STEPS = 720
LAST_STEP = 718               # last step whose actions are applied
LAST_DAY = 29                 # day of LAST_STEP
LAST_REFRESH_DAY = 28         # last day that gets an end-of-day refresh
BOARD = 10
HALF = BOARD // 2
SHED_CAP = 100
START_MONEY = 3000
MAX_ORDERS = 10               # maxMarketOrdersPerTurn
WEED_CHANCE = 0.005           # per empty unlocked tile, per end of day
SHOP_UNLOCK_INTERVAL = 3      # a shop unlocks at the end of days 2, 5, ..., 23
SHOP_TICK = 4                 # townShopSellInterval
CENTER_TICK = 24              # townCenterSellInterval
MAX_SHOP_INSTANCES = 8
MARKET_I0 = 10000
PRICE_FLOOR = 1
HINGE_GAIN = 8.0

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
FARM_HAND_COST_MULT = 1
SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
SHOP_TYPES = sorted(SHOPS)                      # the engine draws rng.choice(sorted(SHOPS))
TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]
SHED_ACCESS = [(HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF)]  # NWSE order
DEFAULT_SPAWN = (HALF - 1, HALF - 1)            # farmer respawns here every morning
PRODUCT_OF = {a: ANIMALS[a]["product"] for a in ANIMALS}
WEED = "WEED"


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------
def day_of(step):
    return step // TPD


def hour_of(step):
    return step % TPD


def step_of(day, hour=0):
    return day * TPD + hour


# ---------------------------------------------------------------------------
# market price (exact engine arithmetic, including Python's round-half-even)
# ---------------------------------------------------------------------------
def _shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory, params=None):
    """Engine price of `item` at market inventory `inventory` (int or float).  Floor 1."""
    p = (params or MARKET_PARAMS)[item]
    base = p["base"]
    I0 = p["I0"]
    T = p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


_PRICE_CACHE = {P: {} for P in PRODUCTS}


def price(item, inventory):
    """Cached market_price for integer inventories (floats are computed directly)."""
    if inventory.__class__ is int:
        c = _PRICE_CACHE[item]
        v = c.get(inventory)
        if v is None:
            v = market_price(item, inventory)
            c[inventory] = v
        return v
    return market_price(item, inventory)


def buy_price(item, inventory):
    """Engine quote for BUY_PRODUCT (WHEAT/FERTILIZER only): the post-buy inventory's price."""
    return price(item, inventory - 1)


def sell_quotes(item, inventory, n):
    """Prices of n units sold one after another by one player (no rival order in the same slot).
    Returns (prices, inventory_after).  Sales at $1 do not add market supply (engine rule)."""
    out = []
    inv = inventory
    for _ in range(n):
        p = price(item, inv)
        out.append(p)
        if p > 1:
            inv += 1
    return out, inv


def buy_quotes(item, inventory, n):
    """Prices of n units bought one after another (WHEAT/FERTILIZER).  Returns (prices, inventory_after)."""
    out = []
    inv = inventory
    for _ in range(n):
        out.append(price(item, inv - 1))
        inv -= 1
    return out, inv


def lockstep_trade(item, inventory, qa, qb):
    """Two players' orders for the same item in the SAME slot index (engine lockstep): each round both
    are quoted at the same pre-commit inventory, then seat a commits, then seat b.  qa/qb are signed
    unit counts (+ sell, - buy; buys only for WHEAT/FERTILIZER).  Money/shed limits are not modelled.
    Returns (prices_a, prices_b, inventory_after)."""
    pa, pb = [], []
    inv = inventory
    ra, rb = abs(qa), abs(qb)
    while ra > 0 or rb > 0:
        qa_p = qb_p = None
        if ra > 0:
            qa_p = price(item, inv) if qa > 0 else price(item, inv - 1)
        if rb > 0:
            qb_p = price(item, inv) if qb > 0 else price(item, inv - 1)
        if qa_p is not None:
            pa.append(qa_p)
            if qa > 0:
                if qa_p > 1:
                    inv += 1
            else:
                inv -= 1
            ra -= 1
        if qb_p is not None:
            pb.append(qb_p)
            if qb > 0:
                if qb_p > 1:
                    inv += 1
            else:
                inv -= 1
            rb -= 1
    return pa, pb, inv


def inventory_for_price(item, target_price, lo=0, hi=40000):
    """Largest inventory whose price is >= target_price (prices fall as inventory rises).
    None if even inventory `lo` is below the target; None too if the price never drops below it."""
    if price(item, lo) < target_price:
        return None
    while price(item, hi) >= target_price:
        if hi > 10 ** 8:
            return None
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if price(item, mid) >= target_price:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# hires, land, shed, geometry
# ---------------------------------------------------------------------------
def fib(n):
    """Engine _fib: fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5 ..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(n_already_today, mult=FARM_HAND_COST_MULT):
    """Cost of the next HIRE when `n_already_today` hires were already made today (farm['hires_today'])."""
    return mult * fib(n_already_today)


def hires_total_cost(k, already=0, mult=FARM_HAND_COST_MULT):
    """Cost of k more hires today after `already` hires.  k hires from zero cost fib(k+1)-1."""
    return sum(hire_cost(already + i, mult) for i in range(k))


def next_land(unlocked_quadrants):
    """(quadrant, price) of the next BUY_LAND, or None when everything is bought."""
    n = len(unlocked_quadrants) - 1
    if n >= len(LAND_ORDER):
        return None
    return LAND_ORDER[n], LAND_PRICES[n]


def quadrant_of(x, y):
    return ("N" if y < HALF else "S") + ("W" if x < HALF else "E")


def is_shed_access(x, y):
    return (x, y) in SHED_ACCESS


def spawn_tile(farmer_pos, hand_positions):
    """Where the next hired hand appears: the least-occupied shed-access tile (ties: NWSE order),
    counting the farmer and all hands at their positions after this step's unit actions."""
    occ = {t: 0 for t in SHED_ACCESS}
    for p in [tuple(farmer_pos)] + [tuple(h) for h in hand_positions]:
        if p in occ:
            occ[p] += 1
    return min(SHED_ACCESS, key=lambda t: (occ[t], SHED_ACCESS.index(t)))


def shed_used(shed):
    return sum(shed.values())


def shed_room(shed, cap=SHED_CAP):
    return max(0, cap - sum(shed.values()))


def drop_into_shed(shed, carried, cap=SHED_CAP):
    """DROP (and the end-of-day auto-drop): every carried item goes to the shed up to the cap, in the
    unit's inventory order; the overflow is DISCARDED.  Returns (new_shed, discarded dict)."""
    shed = dict(shed)
    lost = {}
    for item, n in list(carried.items()):
        if n <= 0:
            continue
        room = max(0, cap - sum(shed.values()))
        take = min(n, room)
        if take > 0:
            shed[item] = shed.get(item, 0) + take
        if n - take > 0:
            lost[item] = n - take
    return shed, lost


def place_into_shed(shed, item, n, carried_n, cap=SHED_CAP):
    """PLACE <item> <n> on a shed-access tile: stores min(n, carried, room); nothing is lost."""
    if n <= 0:
        return 0
    return max(0, min(n, carried_n, cap - sum(shed.values())))


# ---------------------------------------------------------------------------
# plant tile stepper (mirrors the engine; mutates and/or returns the tile)
# ---------------------------------------------------------------------------
def new_plant(crop, day):
    cd = CROPS[crop]
    return {
        "kind": "PLANT", "crop": crop, "planted_day": day, "watered_today": False,
        "consecutive_unwatered": 1,  # the planting day counts as unwatered
        "yield_units": 0 if cd["ongoing"] else 1,
        "max_lifespan_step": (-1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * TPD),
        "fertilized_until_day": -1,
    }


def window(crop):
    """Ages (days since planting) at which WATER adds yield for a non-ongoing crop: [start, end]."""
    cd = CROPS[crop]
    if cd["ongoing"]:
        return None
    return (cd["max_yield_day"] + 1) // 2, cd["max_yield_day"]


def is_plant(tile):
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_animal(tile):
    return isinstance(tile, dict) and "animal" in tile


def plant_water(tile, day):
    """WATER.  Returns True when it changed the tile."""
    if not is_plant(tile) or tile["watered_today"]:
        return False
    tile["watered_today"] = True
    cd = CROPS[tile["crop"]]
    if not cd["ongoing"]:
        age = day - tile["planted_day"]
        ws = (cd["max_yield_day"] + 1) // 2
        if ws <= age <= cd["max_yield_day"]:
            bonus = 2 if tile["fertilized_until_day"] >= day else 1
            tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + bonus)
    return True


def plant_fertilize(tile, day):
    """FERTILIZE (the unit must carry a FERTILIZER, which is consumed).  Active for day, day+1, day+2."""
    if not is_plant(tile):
        return False
    tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), day + 2)
    return True


def can_harvest(tile, day):
    if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
        return False
    if tile.get("kind") == "PLANT":
        return day - tile["planted_day"] >= CROPS[tile["crop"]]["first_yield_day"]
    return "animal" in tile


def harvest(tile, day):
    """HARVEST.  Returns (units, product, tile_after); a non-ongoing crop's tile becomes None (empty)."""
    if not can_harvest(tile, day):
        return 0, None, tile
    units = tile["yield_units"]
    tile["yield_units"] = 0
    if tile.get("kind") == "PLANT":
        if not CROPS[tile["crop"]]["ongoing"]:
            return units, tile["crop"], None
        return units, tile["crop"], tile
    return units, PRODUCT_OF[tile["animal"]], tile


def plant_decay(tile, step):
    """Per-step decay after max_lifespan_step: -1 unit on steps mls, mls+2, ...; 0 units -> WEED."""
    if not is_plant(tile):
        return tile
    mls = tile["max_lifespan_step"]
    if mls < 0 or step < mls or (step - mls) % 2 != 0:
        return tile
    tile["yield_units"] -= 1
    if tile["yield_units"] <= 0:
        return {"kind": WEED}
    return tile


def plant_refresh(tile, day):
    """End-of-day refresh of one plant at the end of `day`.  Returns the tile (or a WEED dict)."""
    if not is_plant(tile):
        return tile
    was_watered = tile["watered_today"]
    if was_watered:
        tile["consecutive_unwatered"] = 0
    else:
        tile["consecutive_unwatered"] += 1
    tile["watered_today"] = False
    if tile["consecutive_unwatered"] >= 2:
        return {"kind": WEED}
    cd = CROPS[tile["crop"]]
    if not cd["ongoing"]:
        return tile
    next_day = day + 1
    dsf = next_day - tile["planted_day"] - cd["first_yield_day"]
    if dsf < 0 or dsf % cd["interval"] != 0:
        return tile
    count = dsf // cd["interval"] + 1
    if count > cd["max_yield"]:
        return tile
    fert = was_watered and tile.get("fertilized_until_day", -1) >= day
    tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + (2 if fert else 1))
    if count == cd["max_yield"]:
        tile["max_lifespan_step"] = (next_day + 1) * TPD
    return tile


def dry_death_day(plant_day, watered_days, until_day=LAST_REFRESH_DAY):
    """Day whose end-of-day refresh turns the plant into a WEED for lack of water, or None.
    The planting day counts as unwatered, so an unwatered planting day kills at its own end; afterwards two
    consecutive dry days kill at the end of the second one."""
    w = set(watered_days)
    cu = 1
    for d in range(plant_day, until_day + 1):
        cu = 0 if d in w else cu + 1
        if cu >= 2:
            return d
    return None


def blocked_plant_crops(seeds, action):
    """Engine's atomic PLANT validation for one player's action dict: crops whose PLANT requests this turn
    (farmer + every entry of 'hands', even for hands that do not exist) exceed the seeds held -- ALL PLANTs
    of such a crop are turned into PASS."""
    if not isinstance(action, dict):
        return set()
    units = [action.get("farmer", ["PASS"])] + list(action.get("hands", []) if isinstance(action.get("hands", []), list) else [])
    demand = {}
    for a in units:
        if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
            demand[a[1]] = demand.get(a[1], 0) + 1
    return {c for c, n in demand.items() if n > seeds.get(c, 0)}


def production_refresh_days(crop, plant_day):
    """Ongoing crop: the days whose END-of-day refresh adds yield (units usable from the next day)."""
    cd = CROPS[crop]
    if not cd["ongoing"]:
        return []
    return [plant_day + cd["first_yield_day"] - 1 + k * cd["interval"] for k in range(cd["max_yield"])]


# ---------------------------------------------------------------------------
# animal tile stepper
# ---------------------------------------------------------------------------
def new_animal(animal, day):
    a = ANIMALS[animal]
    return {"kind": a["structure"], "animal": animal, "placed_day": day, "yield_units": 0,
            "consecutive_unfed": 0, "fed_today": False, "cared_today": False,
            "fertilizer_available": False, "pending_care_bonus": 0}


def animal_feed(tile):
    """FEED (the unit must carry WHEAT, 1 is consumed when this returns True)."""
    if not is_animal(tile) or tile["fed_today"]:
        return False
    tile["fed_today"] = True
    return True


def animal_care(tile):
    if not is_animal(tile) or tile["cared_today"]:
        return False
    tile["cared_today"] = True
    return True


def animal_collect(tile):
    """COLLECT_FERTILIZER -> True when a FERTILIZER was taken."""
    if not is_animal(tile) or not tile["fertilizer_available"]:
        return False
    tile["fertilizer_available"] = False
    return True


def animal_refresh(tile, day):
    """End-of-day refresh at the end of `day`.  Returns the tile, or the bare structure if it escaped."""
    if not is_animal(tile):
        return tile
    if tile["fed_today"]:
        tile["consecutive_unfed"] = 0
    else:
        tile["consecutive_unfed"] += 1
    if tile["consecutive_unfed"] >= 2:
        return {"kind": ANIMALS[tile["animal"]]["structure"]}
    a = ANIMALS[tile["animal"]]
    dsf = day + 1 - tile["placed_day"] - a["first_yield_day"]
    if dsf >= 0 and dsf % a["interval"] == 0:
        bonus = tile.pop("pending_care_bonus", 0) if tile["fed_today"] else 0
        tile["yield_units"] = min(a["max_held"], tile["yield_units"] + 1 + bonus)
        tile["pending_care_bonus"] = 0
    if tile["cared_today"] and tile["fed_today"]:
        tile["pending_care_bonus"] = tile.get("pending_care_bonus", 0) + 1
    tile["fertilizer_available"] = True
    tile["fed_today"] = False
    tile["cared_today"] = False
    return tile


def animal_production_days(animal, placed_day, until_day=LAST_REFRESH_DAY):
    """Days whose END-of-day refresh produces (base 1 + banked care if fed that day)."""
    a = ANIMALS[animal]
    first = placed_day + a["first_yield_day"] - 1
    return list(range(first, until_day + 1, a["interval"]))


# ---------------------------------------------------------------------------
# generic tile op (used by the tests to replay receipts; mirrors _apply_unit_action's tile part)
# ---------------------------------------------------------------------------
def apply_tile_op(tile, op, arg, day, has_item=True):
    """Apply one unit op to a tile.  `has_item` says whether the unit's carried/seed precondition holds
    (seed for PLANT, FERTILIZER for FERTILIZE, WHEAT for FEED, the animal for PLACE).
    Returns (tile_after, executed, harvested_units)."""
    if tile == "LOCKED":
        return tile, False, 0
    if op == "PLANT":
        if arg in CROPS and tile is None and has_item:
            return new_plant(arg, day), True, 0
        return tile, False, 0
    if op == "WATER":
        return tile, plant_water(tile, day), 0
    if op == "HARVEST":
        units, _, after = harvest(tile, day)
        return after, units > 0, units
    if op == "FERTILIZE":
        if is_plant(tile) and has_item:
            return tile, plant_fertilize(tile, day), 0
        return tile, False, 0
    if op == "DIG":
        if tile is None or is_animal(tile):
            return tile, False, 0
        return None, True, 0
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        if tile is not None:
            return tile, False, 0
        return {"kind": "COOP" if op == "BUILD_COOP" else "PASTURE"}, True, 0
    if op == "FEED":
        if is_animal(tile) and not tile["fed_today"] and has_item:
            return tile, animal_feed(tile), 0
        return tile, False, 0
    if op == "CARE":
        return tile, animal_care(tile), 0
    if op == "COLLECT_FERTILIZER":
        return tile, animal_collect(tile), 0
    if op == "PLACE":
        if (arg in ANIMALS and isinstance(tile, dict) and tile.get("kind") == ANIMALS[arg]["structure"]
                and "animal" not in tile and has_item):
            return new_animal(arg, day), True, 0
        return tile, False, 0
    return tile, False, 0


# ---------------------------------------------------------------------------
# crop timelines
# ---------------------------------------------------------------------------
def simulate_plant(crop, plant_step, events, end_step=LAST_STEP):
    """Exact single-tile simulation from the step the crop is planted.
    events: {step: [op, ...]} with ops in 'WATER', 'FERTILIZE', 'HARVEST' (applied in list order at that
    step, before that step's decay; the PLANT itself happens at plant_step before its events).
    Returns dict(harvests=[(step, units)], death_step, death_cause ('dry'|'decay'|None), waterings,
    ferts, tile=final tile (None if harvested away), produced (ongoing: [(refresh_day, units)]))."""
    day0 = day_of(plant_step)
    tile = new_plant(crop, day0)
    harvests = []
    produced = []
    waterings = ferts = 0
    death_step = cause = None
    for s in range(plant_step, end_step + 1):
        d = s // TPD
        for op in events.get(s, ()):
            if op == "WATER":
                if plant_water(tile, d):
                    waterings += 1
            elif op == "FERTILIZE":
                if plant_fertilize(tile, d):
                    ferts += 1
            elif op == "HARVEST":
                units, _, tile = harvest(tile, d)
                if units:
                    harvests.append((s, units))
            if tile is None:
                break
        if tile is None:
            break
        tile = plant_decay(tile, s)
        if not is_plant(tile):
            death_step, cause = s, "decay"
            break
        if s % TPD == TPD - 1:
            before = tile["yield_units"]
            tile = plant_refresh(tile, d)
            if not is_plant(tile):
                death_step, cause = s, "dry"
                break
            if tile["yield_units"] != before:
                produced.append((d, tile["yield_units"] - before))
    return {"harvests": harvests, "death_step": death_step, "death_cause": cause, "waterings": waterings,
            "ferts": ferts, "tile": tile, "produced": produced,
            "units": sum(u for _, u in harvests)}


def _survival_waterings(start_day, last_day, forced):
    """Minimal set of watering days on [start_day, last_day] containing start_day and `forced` with no two
    consecutive dry days (greedy: water when forced or when yesterday was already dry)."""
    out = [start_day]
    last = start_day
    for d in range(start_day + 1, last_day + 1):
        if d in forced or d - last >= 2:
            out.append(d)
            last = d
    return out


def care_plan(crop, plant_day, plant_hour=0, fertilize=False, visit_hour=6, end_step=LAST_STEP, sell_lag=0,
              harvest_age=None):
    """Minimal-labour care schedule for one tile.

    Non-ongoing (WHEAT/CARROT/MELON): water on the planting day, keep it alive (never two dry days in a
    row), water every window day until the yield cap (or harvest age) is reached, optionally FERTILIZE on the
    first window day (covers 3 days), harvest at the earliest age >= first_yield_day that has the best
    reachable yield (WHEAT 4, CARROT 3, MELON 10), or earlier if the season ends.
    Ongoing (TOMATO/STRAWBERRY): keep alive until the last production refresh; with fertilizer, water on every
    production day and FERTILIZE on the first uncovered one (2 ferts per plant); harvest whenever the held
    stock would overflow max_yield, and finally on the day after the last production; then DIG the WEED.
    Only productions refreshed by the end of day 28 and harvests at steps <= end_step - sell_lag count.

    Ops of one visit are put on consecutive steps (FERTILIZE, WATER, HARVEST) starting at visit_hour.
    Returns dict(events={step:[ops]}, visits={day:[ops]}, plant_step, harvest_days, fert_days, water_days,
    last_harvest_day, tile_free_day (first day the tile can be replanted), dig_day)."""
    cd = CROPS[crop]
    d = plant_day
    plant_step = step_of(d, plant_hour)
    last_ok_step = end_step - sell_lag           # a harvest after this cannot be sold
    last_ok_day = day_of(last_ok_step)
    water_days, fert_days, harvest_days = [], [], []
    dig_day = None
    if not cd["ongoing"]:
        ws, we = window(crop)
        fyd = cd["first_yield_day"]
        fert_cover = set(range(ws, ws + 3)) if fertilize else set()

        def yield_at(age, watered):
            y = 1
            for a in range(ws, min(age, we) + 1):
                if a in watered:
                    y = min(cd["max_yield"], y + (2 if a in fert_cover else 1))
            return y
        full = set(range(ws, we + 1))
        if harvest_age is None:
            best = yield_at(we, full)
            harvest_age = next(a for a in range(fyd, we + 1) if yield_at(a, full) >= best)
        harvest_age = max(fyd, min(harvest_age, last_ok_day - d))
        if harvest_age < fyd or d + harvest_age > last_ok_day:
            return {"events": {}, "visits": {}, "plant_step": plant_step, "harvest_days": [], "fert_days": [],
                    "water_days": [], "last_harvest_day": None, "tile_free_day": d, "dig_day": None,
                    "feasible": False}
        # yield waterings: window days up to the harvest age, stopping once the cap is reached
        forced = set()
        y = 1
        for a in range(ws, min(harvest_age, we) + 1):
            if y >= cd["max_yield"]:
                break
            forced.add(d + a)
            y = min(cd["max_yield"], y + (2 if a in fert_cover else 1))
        # survival until the harvest day (the harvest happens before that day's refresh)
        water_days = _survival_waterings(d, d + harvest_age - 1, forced)
        if d + harvest_age in forced:
            water_days.append(d + harvest_age)
        if fertilize:
            fert_days = [d + ws]
        harvest_days = [d + harvest_age]
        tile_free_day = d + harvest_age
    else:
        R = [r for r in production_refresh_days(crop, d) if r <= LAST_REFRESH_DAY]
        # a production is only worth it if its units can be harvested (the next day) in time
        R = [r for r in R if r + 1 <= last_ok_day]
        if not R:
            return {"events": {}, "visits": {}, "plant_step": plant_step, "harvest_days": [], "fert_days": [],
                    "water_days": [], "last_harvest_day": None, "tile_free_day": d, "dig_day": None,
                    "feasible": False}
        forced = set(R) if fertilize else set()
        if fertilize:
            covered = -1
            for r in R:
                if r > covered:
                    fert_days.append(r)
                    covered = r + 2
        water_days = _survival_waterings(d, R[-1], forced)
        bonus = 2 if fertilize else 1
        held = 0
        for i, r in enumerate(R):
            held += bonus
            nxt = R[i + 1] if i + 1 < len(R) else None
            if nxt is None:
                harvest_days.append(r + 1)
            elif held + bonus > cd["max_yield"]:
                harvest_days.append(nxt)          # harvest on the next production day, before its refresh
                held = 0
        full_run = len(R) == cd["max_yield"]
        # After the last production the plant decays from the start of day R[-1]+2; harvested to 0 it
        # becomes a WEED at that step, so the tile needs a DIG from day R[-1]+2 on.
        dig_day = R[-1] + 2 if full_run and R[-1] + 2 <= LAST_DAY else None
        tile_free_day = (R[-1] + 2) if full_run else None
    events = {}
    visits = {}

    def put(day, ops):
        if not ops:
            return
        h = plant_hour + 1 if day == d else visit_hour
        h = min(h, TPD - len(ops))
        if day == d and h <= plant_hour:          # squeeze: plant at plant_hour, care right after
            h = plant_hour + 1 if plant_hour + len(ops) < TPD else plant_hour
        for i, op in enumerate(ops):
            s = step_of(day, h + i)
            if day == LAST_DAY and op == "HARVEST":
                s = min(s, last_ok_step)
            events.setdefault(s, []).append(op)
        visits.setdefault(day, []).extend(ops)
    alldays = sorted(set(water_days) | set(fert_days) | set(harvest_days))
    visits[d] = ["PLANT"]
    for day in alldays:
        ops = []
        if day in fert_days:
            ops.append("FERTILIZE")
        if day in water_days:
            ops.append("WATER")
        if day in harvest_days:
            ops.append("HARVEST")
        put(day, ops)
    if dig_day is not None:
        visits.setdefault(dig_day, []).append("DIG")
    return {"events": events, "visits": visits, "plant_step": plant_step, "harvest_days": harvest_days,
            "fert_days": fert_days, "water_days": water_days,
            "last_harvest_day": harvest_days[-1] if harvest_days else None,
            "tile_free_day": tile_free_day, "dig_day": dig_day, "feasible": True}


def crop_outcome(crop, plant_day, plant_hour=0, fertilize=False, visit_hour=6, end_step=LAST_STEP, sell_lag=0,
                 harvest_age=None):
    """care_plan + simulate_plant: the exact yield of following the plan.  Adds 'plan' to the result."""
    plan = care_plan(crop, plant_day, plant_hour, fertilize, visit_hour, end_step, sell_lag, harvest_age)
    if not plan["feasible"]:
        return {"harvests": [], "units": 0, "death_step": None, "death_cause": None, "waterings": 0, "ferts": 0,
                "tile": None, "produced": [], "plan": plan}
    res = simulate_plant(crop, plan["plant_step"], plan["events"], end_step)
    res["plan"] = plan
    return res


# ---------------------------------------------------------------------------
# animal timelines
# ---------------------------------------------------------------------------
def simulate_animal(animal, placed_day, fed_days=None, cared_days=None, harvest_days=None, collect_days=None,
                    until_day=LAST_DAY, held=None):
    """Day-level exact simulation of one placed animal (placed on `placed_day`).
    fed_days / cared_days / collect_days / harvest_days: iterables of days (None = every day).  Within a day
    the order is HARVEST, COLLECT, FEED, CARE (harvest before the end-of-day refresh).  Pass `held` to start from
    an observed tile dict (then placed_day is taken from it and simulation starts at `placed_day` argument day).
    Returns dict(harvests=[(day, units)], produced=[(refresh_day, units_added)], lost=[(refresh_day, units)],
    escaped_day, fertilizer=[days collected], tile)."""
    tile = dict(held) if held is not None else new_animal(animal, placed_day)
    all_days = range(placed_day, until_day + 1)
    fed = set(all_days) if fed_days is None else set(fed_days)
    cared = set(all_days) if cared_days is None else set(cared_days)
    hv = set(all_days) if harvest_days is None else set(harvest_days)
    col = set(all_days) if collect_days is None else set(collect_days)
    out = {"harvests": [], "produced": [], "lost": [], "escaped_day": None, "fertilizer": [], "tile": tile}
    a = ANIMALS[animal]
    for d in all_days:
        if d in hv:
            units, _, tile = harvest(tile, d)
            if units:
                out["harvests"].append((d, units))
        if d in col and animal_collect(tile):
            out["fertilizer"].append(d)
        if d in fed:
            animal_feed(tile)
        if d in cared:
            animal_care(tile)
        if d > LAST_REFRESH_DAY:
            continue
        before = tile["yield_units"]
        dsf = d + 1 - tile["placed_day"] - a["first_yield_day"]
        is_prod = dsf >= 0 and dsf % a["interval"] == 0
        bonus = tile.get("pending_care_bonus", 0) if tile["fed_today"] else 0
        tile = animal_refresh(tile, d)
        if not is_animal(tile):
            out["escaped_day"] = d
            break
        if is_prod:
            added = tile["yield_units"] - before
            out["produced"].append((d, added))
            if 1 + bonus > added:
                out["lost"].append((d, 1 + bonus - added))
    out["tile"] = tile
    return out


def animal_steady_rate(animal, fed=True, cared=True):
    """Long-run units per day with daily feed (+care): GOOSE 2.0, COW 1.5, SHEEP 1.333; without care 1/interval."""
    a = ANIMALS[animal]
    per = 1 + (a["interval"] if (fed and cared) else 0)
    return per / a["interval"]


def animal_outcome(animal, placed_day, last_feed_day=LAST_REFRESH_DAY, care=True):
    """Units a newly placed animal yields over the rest of the season when fed (+cared) every day up to
    last_feed_day and harvested daily; last day of harvest is LAST_DAY.  Returns simulate_animal's dict plus
    'units' and 'fertilizer_units'."""
    days = range(placed_day, last_feed_day + 1)
    r = simulate_animal(animal, placed_day, fed_days=days, cared_days=(days if care else ()), until_day=LAST_DAY)
    r["units"] = sum(u for _, u in r["harvests"])
    r["fertilizer_units"] = len(r["fertilizer"])
    return r


def herd_for_demand(animal, daily_units, care=True):
    """Number of animals whose steady output matches `daily_units` per day."""
    return daily_units / animal_steady_rate(animal, True, care)


# ---------------------------------------------------------------------------
# town drain
# ---------------------------------------------------------------------------
def shop_tick_drain(shops):
    """Units per shop tick (every 4 steps) drained by a list of unlocked shop instances."""
    out = {P: 0 for P in PRODUCTS}
    for name in shops:
        prods = SHOPS[name]
        mult = 2 if len(prods) == 1 else 1
        for P in prods:
            out[P] += mult
    return out


def active_shop_count(step):
    """How many shop instances drain at `step` (unlocks happen at the end of days 2, 5, ..., 23)."""
    return min(MAX_SHOP_INSTANCES, day_of(step) // SHOP_UNLOCK_INTERVAL)


def shop_unlock_day(slot):
    """First day on which shop slot `slot` (0-based) drains (3, 6, ..., 24)."""
    return (slot + 1) * SHOP_UNLOCK_INTERVAL


def drain_at_step(step, shops):
    """Exact units removed from market inventory by the town at `step` (after that step's orders).
    `shops` = the unlocked list as observed during that step (obs['town']['unlocked_shops'])."""
    out = {P: 0 for P in PRODUCTS}
    if step % SHOP_TICK == 0:
        for name in shops:
            prods = SHOPS[name]
            mult = 2 if len(prods) == 1 else 1
            for P in prods:
                out[P] += mult
    if step % CENTER_TICK == 0:
        for P in TOWN_CENTER_PRODUCTS:
            out[P] += 1
    return out


# expected per-tick drain of one shop slot whose type is not drawn yet (uniform over 8 types)
EXPECTED_SLOT_DRAIN = {P: sum((2 if len(SHOPS[s]) == 1 else 1) for s in SHOP_TYPES if P in SHOPS[s]) / len(SHOP_TYPES)
                       for P in PRODUCTS}


def daily_drain(shops, include_center=True):
    """Units per full day drained by the given shop instances (6 ticks) plus the town centre (1)."""
    tick = shop_tick_drain(shops)
    return {P: TPD // SHOP_TICK * tick[P] + (1 if include_center and P in TOWN_CENTER_PRODUCTS else 0)
            for P in PRODUCTS}


def drain_path(step0, end_step, shops, future_shops=None):
    """Per-product list of the town drain at every step in [step0, end_step].
    Known slots drain exactly; slots not yet unlocked use `future_shops` (a list for the remaining slots, e.g.
    an oracle or a scenario) or else EXPECTED_SLOT_DRAIN (fractional)."""
    n = end_step - step0 + 1
    known = len(shops)
    slot_tick = []
    for k in range(MAX_SHOP_INSTANCES):
        if k < known:
            slot_tick.append(shop_tick_drain([shops[k]]))
        elif future_shops is not None and k - known < len(future_shops):
            slot_tick.append(shop_tick_drain([future_shops[k - known]]))
        else:
            slot_tick.append(EXPECTED_SLOT_DRAIN)
    out = {P: [0] * n for P in PRODUCTS}
    for i in range(n):
        s = step0 + i
        if s % SHOP_TICK == 0:
            na = active_shop_count(s)
            for k in range(na):
                t = slot_tick[k]
                for P in PRODUCTS:
                    v = t[P]
                    if v:
                        out[P][i] += v
        if s % CENTER_TICK == 0:
            for P in TOWN_CENTER_PRODUCTS:
                out[P][i] += 1
    return out


def expected_future_drain(step0, end_step, shops):
    """Total expected drain per product over [step0, end_step] (known shops exact, future slots expected)."""
    dp = drain_path(step0, end_step, shops)
    return {P: sum(dp[P]) for P in PRODUCTS}


# ---------------------------------------------------------------------------
# market forecast
# ---------------------------------------------------------------------------
class Forecast:
    """Market inventory / price forecast for every product from `step0` to `end_step`.

    inv[i] is the inventory at the START of step step0+i (what obs shows at that step).  Inside a step the
    rival's flow is applied first (pessimistic), then ours, then the town drain.  Flows are signed units
    (+ sold to the market, - bought).  Our flows must be integers; the rival's may be fractional rates.
    With future_shops (oracle) and exact rival flows the path is exact (tested)."""

    def __init__(self, inventory, step0, shops, future_shops=None, end_step=LAST_STEP, apply_floor=True):
        self.apply_floor = apply_floor   # False: flows are 'effective' units (already net of $1 sales)
        self.inv0 = {P: inventory[P] for P in PRODUCTS}
        self.step0 = step0
        self.end = end_step
        self.n = max(0, end_step - step0 + 1)
        self.drain = drain_path(step0, end_step, list(shops), future_shops)
        self.ours = {P: {} for P in PRODUCTS}
        self.rival = {P: {} for P in PRODUCTS}
        self._cache = {}

    # -- flows ------------------------------------------------------------
    def add_ours(self, P, step, units):
        if step < self.step0 or step > self.end or not units:
            return
        d = self.ours[P]
        d[step] = d.get(step, 0) + units
        if d[step] == 0:
            del d[step]
        self._cache.pop(P, None)

    def set_ours(self, P, schedule):
        self.ours[P] = {s: u for s, u in schedule.items() if self.step0 <= s <= self.end and u}
        self._cache.pop(P, None)

    def add_rival(self, P, step, units):
        if step < self.step0 or step > self.end or not units:
            return
        d = self.rival[P]
        d[step] = d.get(step, 0) + units
        self._cache.pop(P, None)

    def set_rival_schedule(self, P, per_day_list, day0):
        """Rival net sales per day (list starting at day0), each day's units spread evenly over its 24 steps."""
        for i, u in enumerate(per_day_list):
            if u:
                d = day0 + i
                self.set_rival_rate(P, u, step_of(d, 0), step_of(d, TPD - 1))

    def set_rival_rate(self, P, per_day, from_step=None, to_step=None):
        """Spread `per_day` units/day of rival net sales evenly over the steps (fractional)."""
        a = self.step0 if from_step is None else max(self.step0, from_step)
        b = self.end if to_step is None else min(self.end, to_step)
        r = per_day / TPD
        d = self.rival[P]
        for s in range(a, b + 1):
            d[s] = d.get(s, 0) + r
        self._cache.pop(P, None)

    # -- core simulation ---------------------------------------------------
    def _run(self, P, extra_step=None, extra=0):
        """Returns (inv_start list, our revenue per step dict, total our revenue)."""
        inv = self.inv0[P]
        ours = self.ours[P]
        rival = self.rival[P]
        drain = self.drain[P]
        s0 = self.step0
        starts = [0] * self.n
        rev = 0
        revs = {}
        for i in range(self.n):
            s = s0 + i
            starts[i] = inv
            r = rival.get(s)
            if r:
                if r > 0 and not self.apply_floor:
                    inv += r
                elif r > 0:
                    if r.__class__ is int:
                        for _ in range(r):
                            if price(P, inv) > 1:
                                inv += 1
                    elif market_price(P, inv) > 1:
                        inv += r
                else:
                    inv += r
            o = ours.get(s, 0)
            if extra_step == s:
                o += extra
            if o:
                if o > 0:
                    got = 0
                    for _ in range(o):
                        p = price(P, inv) if inv.__class__ is int else market_price(P, inv)
                        got += p
                        if p > 1 or not self.apply_floor:
                            inv += 1
                else:
                    got = 0
                    for _ in range(-o):
                        inv -= 1
                        got -= price(P, inv) if inv.__class__ is int else market_price(P, inv)
                revs[s] = got
                rev += got
            inv -= drain[i]
        return starts, revs, rev

    def _path(self, P):
        c = self._cache.get(P)
        if c is None:
            c = self._run(P)
            self._cache[P] = c
        return c

    # -- queries -----------------------------------------------------------
    def inventory_path(self, P):
        """Inventory at the start of each step step0..end."""
        return self._path(P)[0]

    def inventory_at(self, P, step):
        return self._path(P)[0][step - self.step0]

    def price_path(self, P):
        """Price the first extra unit we sell at each step would fetch (after the rival's flow at that step)."""
        starts = self._path(P)[0]
        out = []
        for i, inv in enumerate(starts):
            r = self.rival[P].get(self.step0 + i, 0)
            out.append(market_price(P, inv + r) if r else (price(P, inv) if inv.__class__ is int else market_price(P, inv)))
        return out

    def price_at(self, P, step):
        return self.price_path(P)[step - self.step0]

    def our_revenue(self, P):
        return self._path(P)[2]

    def marginal_revenue(self, P, t, units=1):
        """Exact change of our total revenue in P from selling `units` more at step t (negative units = buy),
        including the depressing effect on all our later planned sales.  O(steps)."""
        if t < self.step0 or t > self.end:
            return 0
        return self._run(P, t, units)[2] - self._path(P)[2]

    def marginal_revenue_all(self, P):
        """MR of one extra unit at every step step0..end in one O(steps) pass (telescoped: exact whenever no
        price along our planned sales is at the $1 floor)."""
        starts, revs, _ = self._path(P)
        ours = self.ours[P]
        rival = self.rival[P]
        n = self.n
        s0 = self.step0
        # delta[i]: change of our revenue at step i if the inventory entering step i is 1 higher
        delta = [0] * n
        after_ours = [0] * n
        for i in range(n):
            s = s0 + i
            inv_r = starts[i] + rival.get(s, 0)
            o = ours.get(s, 0)
            # sells: sum_k [p(I+1+k) - p(I+k)] = p(I+o) - p(I); buys telescope to the same expression
            if o:
                delta[i] = price(P, inv_r + o) - price(P, inv_r)
            after_ours[i] = inv_r + o
        out = [0] * n
        suffix = 0
        for i in range(n - 1, -1, -1):
            out[i] = price(P, after_ours[i]) + suffix
            suffix += delta[i]
        return out

    def allocate(self, P, supply, candidate_steps=None, commit=True):
        """Greedy sale plan for units becoming available: supply = [(avail_step, units), ...].
        Each unit goes to the step >= its availability with the highest marginal revenue given everything
        already planned.  Returns [(step, units)] and (if commit) adds them to our flows."""
        if candidate_steps is None:
            candidate_steps = [s for s in range(self.step0, self.end + 1)]
        cand = sorted(s for s in candidate_steps if self.step0 <= s <= self.end)
        plan = {}
        units = []
        for a, u in supply:
            units.extend([a] * int(u))
        units.sort()
        added = []
        for a in units:
            mr = self.marginal_revenue_all(P)
            best, best_v = None, None
            for s in cand:
                if s < a:
                    continue
                v = mr[s - self.step0]
                if best_v is None or v > best_v:
                    best, best_v = s, v
            if best is None:
                continue
            self.add_ours(P, best, 1)
            added.append(best)
            plan[best] = plan.get(best, 0) + 1
        if not commit:
            for s in added:
                self.add_ours(P, s, -1)
        return sorted(plan.items())


def forecast_from_obs(obs, future_shops=None, rival_rates=None, end_step=LAST_STEP, rival_schedule=None):
    """Forecast starting at obs['step'] from the observed market and town.  The rival's future sales come from
    rival_schedule ({P: per-day list from today}, e.g. rival_sales_schedule(obs, tracker)) or flat per-day
    rival_rates ({P: units/day}); neither = the rival sells nothing."""
    fc = Forecast(obs["market"]["inventory"], int(obs["step"]), obs["town"]["unlocked_shops"], future_shops, end_step)
    if rival_schedule:
        d0 = day_of(int(obs["step"]))
        for P, row in rival_schedule.items():
            fc.set_rival_schedule(P, row, d0)
    elif rival_rates:
        for P, r in rival_rates.items():
            if r:
                fc.set_rival_rate(P, r)
    return fc


# ---------------------------------------------------------------------------
# rival flow inference and supply projection
# ---------------------------------------------------------------------------
class FlowTracker:
    """Infers the rival's net market flow per product per step from consecutive observations:
        inv(t+1) - inv(t) = our_net(t) + rival_net(t) - drain(t)
    exactly (except sales at the $1 floor, which leave no trace).  Call update() once per observation with our
    own executed net flow at the previous step (sold - bought; the executor knows it)."""

    def __init__(self):
        self.prev = None          # (step, inventory dict, shops list)
        self.history = []         # [(step, {P: rival net units})]
        self.total = {P: 0 for P in PRODUCTS}

    def update(self, obs, our_net_prev=None):
        s = int(obs["step"])
        inv = obs["market"]["inventory"]
        shops = list(obs["town"]["unlocked_shops"])
        res = None
        if self.prev is not None and self.prev[0] == s - 1:
            t, pinv, pshops = self.prev
            dr = drain_at_step(t, pshops)
            our = our_net_prev or {}
            res = {}
            for P in PRODUCTS:
                v = inv[P] - pinv[P] + dr[P] - our.get(P, 0)
                res[P] = v
                self.total[P] += v
            self.history.append((t, res))
        self.prev = (s, dict(inv), shops)
        return res

    def rate(self, P, window_steps=48, now=None):
        """Rival net sales per day over the last `window_steps` observed steps."""
        if not self.history:
            return 0.0
        now = self.history[-1][0] if now is None else now
        tot = 0
        n = 0
        for t, r in reversed(self.history):
            if t <= now - window_steps:
                break
            tot += r[P]
            n += 1
        return tot * TPD / max(1, n)


def project_farm_output(farm, day, until_day=LAST_DAY, fertilize_ongoing=None):
    """Expected units per product becoming harvestable per day for a farm's visible tiles, assuming normal
    care from `day` on (plants watered daily and harvested at maturity; animals fed + cared daily and
    harvested daily; 1 fertilizer per animal per day).  fertilize_ongoing: None = assume fertilized iff the
    tile shows fertilized_until_day >= 0.  Returns {P: [units on day d for d in day..until_day]}.
    This is a forecast (it cannot see future plantings or animal purchases)."""
    n = until_day - day + 1
    out = {P: [0.0] * n for P in PRODUCTS}
    for row in farm["tiles"]:
        for t in row:
            if is_plant(t):
                cd = CROPS[t["crop"]]
                pd = t["planted_day"]
                if not cd["ongoing"]:
                    ws, we = window(t["crop"])
                    y = t["yield_units"]
                    fert_until = t.get("fertilized_until_day", -1)
                    hd = None
                    for dd in range(day, until_day + 1):
                        age = dd - pd
                        if ws <= age <= we and not (dd == day and t["watered_today"]):
                            y = min(cd["max_yield"], y + (2 if fert_until >= dd else 1))
                        if age >= cd["first_yield_day"] and (y >= cd["max_yield"] or age >= we):
                            hd = dd
                            break
                    if hd is not None:
                        out[t["crop"]][hd - day] += y
                else:
                    fert = (t.get("fertilized_until_day", -1) >= 0) if fertilize_ongoing is None else fertilize_ongoing
                    for r in production_refresh_days(t["crop"], pd):
                        if r >= day and r <= min(LAST_REFRESH_DAY, until_day - 1):
                            out[t["crop"]][r + 1 - day] += 2 if fert else 1
                    if t["yield_units"] > 0:
                        out[t["crop"]][0] += t["yield_units"]
            elif is_animal(t):
                a = t["animal"]
                P = PRODUCT_OF[a]
                r = simulate_animal(a, day, held=t, until_day=until_day)
                # held units at `day` are harvested on `day`; later production the day after its refresh
                for dd, u in r["harvests"]:
                    out[P][dd - day] += u
                for dd in r["fertilizer"]:
                    out["FERTILIZER"][dd - day] += 1
    return out


def count_animals(farm):
    c = {a: 0 for a in ANIMALS}
    for row in farm["tiles"]:
        for t in row:
            if is_animal(t):
                c[t["animal"]] += 1
    return c


CROP_PRODUCTS = ("CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_PRODUCTS = ("EGG", "MILK", "WOOL")


def estimate_rival_sales(obs, tracker=None, horizon_days=3, method="calibrated", rival_seat=None):
    """Forecast of the rival's NET market sales (units/day, + sold, - bought) per product over the next
    `horizon_days` days, from what is visible: their farm tiles (project_farm_output) and their recent market
    flow (FlowTracker, if given).  Calibrated on 25 archive games (tools/planner/test_model.py, report I1):
      crops (CARROT/TOMATO/STRAWBERRY/MELON): farm projection (they sell what ripens);
      EGG/MILK/WOOL: mean of projection and the last-48-step rate (they hold premium goods a while);
      FERTILIZER: last-48-step rate (they spend part on their crops; the farm cannot show how much);
      WHEAT: mean of the rate and (projection - 1 feed per animal per day).
    method: 'calibrated' (above), 'projection', or 'rate'."""
    day = day_of(int(obs["step"]))
    seat = (1 - me(obs)) if rival_seat is None else rival_seat
    farm = obs["farms"][seat]
    until = min(LAST_DAY, day + max(1, horizon_days) - 1)
    ndays = until - day + 1
    proj = project_farm_output(farm, day, until)
    have_rate = tracker is not None and bool(tracker.history)
    feed = sum(count_animals(farm).values())
    out = {}
    for P in PRODUCTS:
        pj = sum(proj[P]) / ndays
        if P == "WHEAT":
            pj -= feed
        rt = tracker.rate(P, 48) if have_rate else None
        if method == "projection" or (rt is None and method != "rate"):
            v = pj if P != "FERTILIZER" else 0.0
        elif method == "rate":
            v = rt if rt is not None else 0.0
        elif P in CROP_PRODUCTS:
            v = pj
        elif P in ANIMAL_PRODUCTS or P == "WHEAT":
            v = 0.5 * pj + 0.5 * rt
        else:
            v = rt
        out[P] = v
    return out


def rival_sales_schedule(obs, tracker=None, rival_seat=None, near_days=3, animal_tail="proj", momentum=0.0,
                         fertilize_ongoing=True):
    """Per-day forecast of the rival's net sales for every remaining day: {P: [units on day d for d=today..29]}.
      days today..today+near_days-1: estimate_rival_sales (calibrated);
      later days -- TOMATO/STRAWBERRY/MELON: the harvest timing of their visible plants, future tomato/strawberry
      productions assumed fertilized (top teams fertilize ~95% of them); CARROT: visible plants, never less
      than 0.8 x the near-term rate (short replant cycles); EGG/MILK/WOOL: their herd's projected output
      (animal_tail='proj'; 'rate' holds the near-term rate, 'blend' averages); WHEAT/FERTILIZER: the near-term
      rate.  momentum > 0 continues their last-2-day planting pace of long crops with geometric decay (did not
      help; off).  Nothing is sold after day 29.  Chosen by report H3 (price error at +1..+14 days)."""
    day = day_of(int(obs["step"]))
    seat = (1 - me(obs)) if rival_seat is None else rival_seat
    near = estimate_rival_sales(obs, tracker, near_days, rival_seat=seat)
    proj = project_farm_output(obs["farms"][seat], day, LAST_DAY, fertilize_ongoing)
    n = LAST_DAY - day + 1
    if momentum > 0:
        # continue the rival's planting pace of the last 2 days (visible planted_day), decaying geometrically
        recent = {c: 0 for c in ("TOMATO", "STRAWBERRY", "MELON")}
        for _, _, t in plants_on(obs["farms"][seat]):
            if t["crop"] in recent and day - 2 <= t["planted_day"] <= day - 1:
                recent[t["crop"]] += 1
        for c, k in recent.items():
            rate = k / 2.0
            for j in range(0, n):
                w = rate * momentum ** (j + 1)
                if w < 0.05:
                    break
                for hs, u in crop_outcome(c, day + j, 1, True)["harvests"]:
                    i = day_of(hs) - day
                    if 0 <= i < n:
                        proj[c][i] += w * u
    out = {}
    for P in PRODUCTS:
        row = []
        for i in range(n):
            if i < near_days:
                v = near[P]
            elif P in ("TOMATO", "STRAWBERRY", "MELON"):
                v = proj[P][i]
            elif P == "CARROT":
                v = max(proj[P][i], 0.8 * near[P])
            elif P in ANIMAL_PRODUCTS and animal_tail != "rate":
                v = proj[P][i] if animal_tail == "proj" else 0.5 * (proj[P][i] + near[P])
            else:
                v = near[P]
            row.append(v)
        out[P] = row
    return out


def shop_scenarios(shops_known, k=8, seed=0):
    """k equally-likely completions of the shop list (uniform over the 8 types, with replacement), as lists of
    the FUTURE slots only -- pass each as Forecast(..., future_shops=...) to get scenario price paths.
    Stratified on the first unknown slot when k is a multiple of 8, so every type appears there."""
    import random as _r
    rng = _r.Random(seed * 7919 + len(shops_known))
    m = MAX_SHOP_INSTANCES - len(shops_known)
    out = []
    for i in range(k):
        sc = [rng.choice(SHOP_TYPES) for _ in range(m)]
        if m and k % len(SHOP_TYPES) == 0:
            sc[0] = SHOP_TYPES[i % len(SHOP_TYPES)]
        out.append(sc)
    return out


# ---------------------------------------------------------------------------
# crop valuation for the allocator
# ---------------------------------------------------------------------------
DEFAULT_ASSUMPTIONS = {
    "fertilize": "auto",      # True / False / "auto" (use it when it raises net value)
    "fert_value": None,       # $ per fertilizer used (None: current FERTILIZER market price = its sale value)
    "plant_hour": None,       # None: now if planting today, else 1
    "visit_hour": 6,
    "sell_lag": 4,            # steps from harvest to the sale (walk to the shed / next morning)
    "hold": True,             # sell each unit at its best step (small lots) instead of at harvest
    "forecast": None,         # a Forecast with our other planned sales (built from obs if None)
    "rival_rates": None,      # {P: units/day} used when the forecast is built here
    "rival_schedule": None,   # {P: per-day list} (rival_sales_schedule) used when the forecast is built here
    "candidate_every": 4,     # sale step grid when hold=True
}


def value_of_crop_on_tile(crop, plant_day, obs, assumptions=None):
    """Expected net dollars of planting `crop` on a free tile on `plant_day` and following care_plan:
    revenue at forecast marginal prices (the units are added to our planned sales) minus seed minus the
    fertilizer's sale value.  Also returns the labour it needs (visits by day).
    The Forecast passed in assumptions['forecast'] is not modified."""
    A = dict(DEFAULT_ASSUMPTIONS)
    if assumptions:
        A.update(assumptions)
    now = int(obs["step"])
    if plant_day < day_of(now):
        raise ValueError("plant_day is in the past")
    ph = A["plant_hour"]
    if ph is None:
        ph = hour_of(now) if plant_day == day_of(now) else 1
    fc = A["forecast"] or forecast_from_obs(obs, rival_rates=A["rival_rates"], rival_schedule=A["rival_schedule"])
    fert_value = A["fert_value"]
    if fert_value is None:
        fert_value = market_price("FERTILIZER", obs["market"]["inventory"]["FERTILIZER"])
    options = [False, True] if A["fertilize"] == "auto" else [bool(A["fertilize"])]
    best = None
    for fert in options:
        res = crop_outcome(crop, plant_day, ph, fert, A["visit_hour"], LAST_STEP, A["sell_lag"])
        plan = res["plan"]
        sales = []
        for s, u in res["harvests"]:
            sales.append((min(LAST_STEP, s + A["sell_lag"]), u))
        revenue = _added_revenue(fc, crop, sales, A["hold"], A["candidate_every"])
        seed = CROPS[crop]["seed"]
        fert_cost = res["ferts"] * fert_value
        net = revenue - seed - fert_cost
        visits = plan["visits"]
        labour = {d: len(v) for d, v in sorted(visits.items())}
        occ_end = plan["tile_free_day"] if plan["tile_free_day"] is not None else LAST_DAY + 1
        r = {"crop": crop, "plant_day": plant_day, "fertilize": fert, "feasible": plan["feasible"],
             "units": res["units"], "harvests": res["harvests"], "revenue": revenue, "seed_cost": seed,
             "ferts": res["ferts"], "fert_cost": fert_cost, "net": net, "visits": visits, "labour": labour,
             "labour_ops": sum(labour.values()), "waterings": res["waterings"],
             "tile_days": max(1, occ_end - plant_day), "tile_free_day": plan["tile_free_day"],
             "dig_day": plan["dig_day"], "death_step": res["death_step"], "outcome": res}
        r["net_per_tile_day"] = net / r["tile_days"]
        r["net_per_op"] = net / max(1, r["labour_ops"])
        if best is None or r["net"] > best["net"]:
            best = r
    if not best["feasible"]:
        best["net"] = 0.0 if best["units"] == 0 else best["net"]
    return best


def _added_revenue(fc, P, sales, hold, every):
    """Revenue our plan gains by adding `sales` [(avail_step, units)] of P to forecast fc (fc restored)."""
    if not sales:
        return 0
    base = fc.our_revenue(P)
    saved = dict(fc.ours[P])
    if hold:
        cands = [s for s in range(fc.step0, fc.end + 1) if (s - fc.step0) % every == 0 or s == fc.end]
        # also allow the exact availability steps
        cands = sorted(set(cands) | {s for s, _ in sales if fc.step0 <= s <= fc.end})
        fc.allocate(P, sales, cands, commit=True)
    else:
        for s, u in sales:
            fc.add_ours(P, s, u)
    gained = fc.our_revenue(P) - base
    fc.set_ours(P, saved)
    return gained


def _task(kind, x, y, earliest, deadline, value, item=None, n=1, needs=None):
    return {"kind": kind, "x": x, "y": y, "item": item, "n": n, "earliest": int(earliest), "deadline": int(deadline),
            "value": float(value), "needs": dict(needs or {})}


def plan_tasks(crop, x, y, outcome, unit_value, sell_lag=2, dig_value=0.0):
    """Executor Task dicts (SHARED INTERFACE) for a new crop plan (crop_outcome(...) result) on tile (x, y).
    Each task's value = dollars lost if that one task is skipped (leave-one-out exact simulation x unit_value;
    PLANT carries the whole plan's value minus the seed).  Same-day precedence is encoded in the time windows:
    PLANT < planting-day WATER; for WHEAT/CARROT/MELON FERTILIZE < WATER < HARVEST inside a day (fertilizer only
    doubles a watering done after it; a harvest takes the yield of that day's watering).  Ongoing crops need no
    order inside a day.  Harvest deadlines leave `sell_lag` steps to sell before step 718."""
    plan = outcome["plan"]
    if not plan.get("feasible"):
        return []
    events = plan["events"]
    ps = plan["plant_step"]
    d0 = day_of(ps)
    base = outcome["units"]
    ongoing = CROPS[crop]["ongoing"]
    # PLANT by hour 21-22 so the planting-day WATER still fits; a plan made at hour 23 keeps its own step (never an
    # empty window) -- it then needs a SECOND unit to WATER in the same step, after the planting unit (engine: a
    # crop planted at hour 23 and not watered that step is a WEED at midnight).
    plant_deadline = max(ps, min(step_of(d0, TPD - 2), max(ps, step_of(d0, 21))))
    tasks = [_task("PLANT", x, y, ps, plant_deadline, base * unit_value - CROPS[crop]["seed"], item=crop)]
    by_day = {}
    for s, ops in events.items():
        by_day.setdefault(day_of(s), []).extend((s, op) for op in ops)
    for d, items in sorted(by_day.items()):
        ops_today = [op for _, op in items]
        for s, op in items:
            ev2 = {k: [o for o in v] for k, v in events.items()}
            ev2[s].remove(op)
            lost = base - simulate_plant(crop, ps, ev2)["units"]
            lo, hi = step_of(d, 0), step_of(d, TPD - 1)
            if d == d0:
                lo = ps + 1 if ps + 1 <= hi else ps
            if not ongoing:
                slots = [o for o in ("FERTILIZE", "WATER", "HARVEST") if o in ops_today]
                k = slots.index(op)
                span = (hi - lo + 1) // len(slots)
                lo, hi = lo + k * span, (lo + (k + 1) * span - 1) if k < len(slots) - 1 else hi
            if op == "HARVEST":
                hi = min(hi, LAST_STEP - sell_lag)
            needs = {"FERTILIZER": 1} if op == "FERTILIZE" else None
            tasks.append(_task(op, x, y, lo, hi, lost * unit_value, needs=needs))
    if plan.get("dig_day") is not None:
        dd = plan["dig_day"]
        tasks.append(_task("DIG", x, y, step_of(dd, 0), step_of(dd, TPD - 1), dig_value))
    return tasks


def animal_op_values(tile, day, unit_value, fert_value=0.0, wheat_value=0.0, last_feed_day=LAST_REFRESH_DAY):
    """Dollars at stake TODAY for each op on one placed animal (an observed tile at `day`), assuming full care
    (feed+care+harvest+collect every day, feeding through last_feed_day) from tomorrow on:
      {'FEED': ..., 'CARE': ..., 'HARVEST': ..., 'COLLECT_FERTILIZER': ...}
    Exact via simulate_animal (escape after 2 unfed days, care bank lost on an unfed production day, held
    cap overflow, 1 fertilizer/day).  FEED is net of the wheat it uses; an op already done today is worth 0.
    Feeding on day 29 (no refresh follows) is always worth -wheat_value."""
    if not is_animal(tile):
        return {}
    a = tile["animal"]
    days = set(range(day, LAST_DAY + 1))
    feed = set(range(day, last_feed_day + 1))

    def run(skip):
        r = simulate_animal(a, day, feed - ({day} if skip == "FEED" else set()),
                            feed - ({day} if skip == "CARE" else set()),
                            days - ({day} if skip == "HARVEST" else set()),
                            days - ({day} if skip == "COLLECT_FERTILIZER" else set()),
                            until_day=LAST_DAY, held=tile)
        return sum(u for _, u in r["harvests"]) * unit_value + len(r["fertilizer"]) * fert_value
    base = run(None)
    out = {}
    for op, done in (("FEED", tile["fed_today"]), ("CARE", tile["cared_today"]),
                     ("HARVEST", tile["yield_units"] <= 0), ("COLLECT_FERTILIZER", not tile["fertilizer_available"])):
        if done:
            out[op] = 0.0
        else:
            out[op] = base - run(op) - (wheat_value if op == "FEED" else 0.0)
    return out


def rank_crops(plant_day, obs, assumptions=None, crops=None):
    """value_of_crop_on_tile for each crop, best net first."""
    A = dict(assumptions or {})
    if A.get("forecast") is None:
        A["forecast"] = forecast_from_obs(obs, rival_rates=A.get("rival_rates"), rival_schedule=A.get("rival_schedule"))
    res = [value_of_crop_on_tile(c, plant_day, obs, A) for c in (crops or list(CROPS))]
    res.sort(key=lambda r: -r["net"])
    return res


# ---------------------------------------------------------------------------
# observation helpers
# ---------------------------------------------------------------------------
def me(obs):
    return int(obs["player"])


def my_farm(obs):
    return obs["farms"][me(obs)]


def rival_farm(obs):
    return obs["farms"][1 - me(obs)]


def shed_of(obs):
    return obs["private"]["shed"]


def carried_total(obs):
    return sum(sum(inv.values()) for inv in obs["private"]["inventories"])


def plants_on(farm):
    """[(x, y, tile)] for every plant tile of a farm."""
    return [(x, y, t) for y, row in enumerate(farm["tiles"]) for x, t in enumerate(row) if is_plant(t)]


def animals_on(farm):
    return [(x, y, t) for y, row in enumerate(farm["tiles"]) for x, t in enumerate(row) if is_animal(t)]
