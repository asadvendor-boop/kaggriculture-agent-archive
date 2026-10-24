"""Static game data for the JAX Kaggriculture engine (1.32.7 rules, as in engine/src_v2/sim_trace.hpp).

Everything here is plain numpy / Python so it can be imported without JAX. Ids follow the C++ enums exactly:
items WHEAT..SHEEP = 0..11 (first 5 are crops, first 9 are products), shops in sorted(SHOPS) order, unit ops
PASS..CARE = 0..17 (18 = INVALID, which behaves like PASS), market ops NONE..SELL = 0..6.
"""
import math
import numpy as np

# ------------------------------------------------------------------ ids
ITEM_NAMES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER",
              "GOOSE", "COW", "SHEEP"]
ITEM_ID = {n: i for i, n in enumerate(ITEM_NAMES)}
WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER, GOOSE, COW, SHEEP = range(12)
N_ITEMS, N_PRODUCTS, N_CROPS, N_ANIMALS = 12, 9, 5, 3
INVALID_ITEM = 255                     # unknown / non-string item name (C++ canon "__INVALID_ITEM__")

UNIT_OPS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE", "PLANT", "WATER", "HARVEST",
            "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"]
UNIT_OP_ID = {n: i for i, n in enumerate(UNIT_OPS)}
(OP_PASS, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST, OP_PICKUP, OP_DROP, OP_PLACE, OP_PLANT, OP_WATER, OP_HARVEST,
 OP_FERTILIZE, OP_DIG, OP_BUILD_COOP, OP_BUILD_PASTURE, OP_FEED, OP_COLLECT_FERTILIZER, OP_CARE) = range(18)
OP_INVALID = 18                        # unknown op string: a no-op, exactly like PASS

MARKET_OPS = ["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]
MARKET_OP_ID = {n: i for i, n in enumerate(MARKET_OPS)}
M_NONE, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL = range(7)

SHOP_NAMES = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP",
              "SMOOTHIE_SHOP", "YARN_STORE"]     # sorted(SHOPS): the order rng.choice draws from
SHOP_ID = {n: i for i, n in enumerate(SHOP_NAMES)}
QUAD_NAMES = ["NW", "NE", "SW", "SE"]

# ------------------------------------------------------------------ sizes / config (defaults of the real game)
BOARD = 10
NT = BOARD * BOARD                     # tiles per farm, index = y * 10 + x
U = 40                                 # max units per farm (C++ MAX_UNITS: farmer + 39 hands)
NSLOT = 10                             # market orders processed per player per turn
EPISODE_STEPS = 720
TURNS_PER_DAY = 24
LAST_STEP = EPISODE_STEPS - 2          # step index 718 is the last one executed; done afterwards
STARTING_MONEY = 3000
SHED_CAP = 100
MAX_SHOPS = 8
SHOP_UNLOCK_INTERVAL = 3
SHOP_SELL_INTERVAL = 4
CENTER_SELL_INTERVAL = 24
I0 = 10000
NW_WORDS = 624                         # MT19937 outputs per end-of-day generator used by the engine

# ------------------------------------------------------------------ tiles
K_EMPTY, K_LOCKED, K_WEED, K_COOP, K_PASTURE, K_PLANT = range(6)
KIND_NAMES = ["EMPTY", "LOCKED", "WEED", "COOP", "PASTURE", "PLANT"]
# tile field columns of tiles[..., 100, NF]
(F_KIND, F_WHAT, F_ANIMAL, F_WATERED, F_FED, F_CARED, F_FERTAV, F_DRY, F_YIELD, F_CARE, F_DAY, F_MLS,
 F_FERTU) = range(13)
NF = 13
FIELD_NAMES = ["kind", "what", "has_animal", "watered_today", "fed_today", "cared_today", "fertilizer_available",
               "consecutive_dry", "yield_units", "pending_care_bonus", "day", "max_lifespan_step",
               "fertilized_until_day"]
FRESH_TILE = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1], np.int32)   # C++ Tile{} (kind set separately)

SHED_ACCESS = [(4, 4), (5, 4), (4, 5), (5, 5)]    # NWSE order, (x, y)
SPAWN_X, SPAWN_Y = 4, 4
LAND_PRICES = np.array([1000, 2000, 4000], np.int32)
# quadrant id of each tile: 0=NW 1=NE 2=SW 3=SE; land unlocks quadrants 1, 2, 3 in that order
QUAD_OF_TILE = np.array([(0 if y < 5 else 2) + (0 if x < 5 else 1) for y in range(BOARD) for x in range(BOARD)],
                        np.int32)

# ------------------------------------------------------------------ crops / animals
#                    seed  fyd  maxyd  interval  max_yield  ongoing
CROP_TABLE = np.array([[10, 2, 4, 0, 6, 0],     # WHEAT
                       [20, 2, 3, 0, 4, 0],     # CARROT
                       [50, 8, 8, 1, 4, 1],     # TOMATO
                       [100, 10, 10, 2, 4, 1],  # STRAWBERRY
                       [80, 10, 12, 0, 6, 0]],  # MELON
                      np.int32)
CROP_SEED, CROP_FYD, CROP_MAXYD, CROP_INTERVAL, CROP_MAXY, CROP_ONGOING = (CROP_TABLE[:, k].copy() for k in range(6))
#                      cost  structure(kind)  fyd  interval  max_held  product
ANIMAL_TABLE = np.array([[300, K_COOP, 4, 1, 4, EGG],        # GOOSE
                         [400, K_PASTURE, 8, 2, 6, MILK],    # COW
                         [500, K_PASTURE, 6, 3, 6, WOOL]],   # SHEEP
                        np.int32)
ANIMAL_COST, ANIMAL_KIND, ANIMAL_FYD, ANIMAL_INTERVAL, ANIMAL_MAXHELD, ANIMAL_PRODUCT = (
    ANIMAL_TABLE[:, k].copy() for k in range(6))

# ------------------------------------------------------------------ town
_SHOP_ITEMS = {"BAKERY": [EGG, WHEAT], "BRUNCH_SPOT": [EGG, WHEAT, STRAWBERRY],
               "FARMERS_MARKET": [WHEAT, CARROT, TOMATO, STRAWBERRY], "ICE_CREAM_SHOP": [STRAWBERRY, MILK, WHEAT],
               "PET_CAFE": [CARROT], "PIZZA_SHOP": [MILK, TOMATO, WHEAT], "SMOOTHIE_SHOP": [STRAWBERRY, MILK],
               "YARN_STORE": [WOOL]}
SHOP_DRAIN = np.zeros((8, N_PRODUCTS), np.int32)  # units of each product one shop instance removes every 4 steps
for _s, _n in enumerate(SHOP_NAMES):
    _its = _SHOP_ITEMS[_n]
    for _it in _its:
        SHOP_DRAIN[_s, _it] = 2 if len(_its) == 1 else 1
CENTER_DRAIN = np.array([1] * 8 + [0], np.int32)  # every 24 steps, all products but FERTILIZER


def fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


FIB = np.array([fib(n) for n in range(U + 2)], np.int64)
assert FIB[U] < 2 ** 31
FIB = FIB.astype(np.int32)                         # hire cost = FIB[hires_today]

# ------------------------------------------------------------------ market price function (bit-exact)
# base, I0, T, below_f, below_t, above_f, above_t  (sim_trace.hpp MARKET[], 1.32.7)
MARKET = [(25, 10000, 400, "sqrt", 0.80, "log", 0.20),
          (35, 10000, 450, "hinge", 1.00, "sqrt", 0.70),
          (60, 10000, 200, "hinge", 0.40, "sqrt", 0.60),
          (120, 10000, 100, "sqrt", 0.70, "linear", 1.60),
          (250, 10000, 300, "log", 0.20, "sq", 3.60),
          (50, 10000, 332, "hinge", 0.40, "log", 0.20),
          (160, 10000, 122, "sqrt", 0.60, "linear", 1.60),
          (200, 10000, 105, "log", 0.20, "sq", 3.20),
          (100, 10000, 200, "linear", 0.40, "linear", 0.40)]


def _shape(f, x, T=0.0):
    x = float(x)
    if x < 0.0:
        x = 0.0
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "log":
        return math.log(1.0 + x)
    if f == "hinge":
        if T <= 0.0:
            return x
        u = x / T
        e = u - 1.0
        if e < 0.0:
            e = 0.0
        return u + 8.0 * e * e
    raise ValueError(f)


_AMPS = [(bt * float(base) / _shape(bf, float(T), float(T)), at * float(base) / _shape(af, float(T), float(T)))
         for (base, _i0, T, bf, bt, af, at) in MARKET]


def market_price(item, inv):
    """Exact port of C++ market_price (and of the official Python): max(1, round-half-even(price))."""
    base, i0, T, bf, bt, af, at = MARKET[item]
    a_below, a_above = _AMPS[item]
    if inv < i0:
        price = float(base) + a_below * _shape(bf, float(i0 - inv), float(T))
    else:
        price = float(base) - a_above * _shape(af, float(inv - i0), float(T))
    v = int(round(price))                      # Python round == nearbyint (half to even)
    return 1 if v < 1 else v


# Tables over market inventory v in [INV_LO, INV_HI]. Index = v - INV_LO. Reachable inventories are far inside:
# non-buyable products only fall through town demand (<= ~3k over a season) and floored products cannot rise past
# their floor + 1 (a $1 sale adds no supply); WHEAT / FERTILIZER can fall through BUY_PRODUCT but every unit costs
# >= $25 / $100, and WHEAT / EGG can only rise by selling real production. Leaving the range sets GameState.oob.
INV_LO = -10000
INV_HI = 50000
NTAB = INV_HI - INV_LO + 1 + 2       # +2 so exclusive sums can be read one/two past INV_HI


def build_price_tables():
    """Returns (P, CS, E2, FLOOR):
    P[j, v-LO]   = market_price(j, v)                      (int32; clamped at the ends)
    CS[j, v-LO]  = sum_{LO <= w < v} P(j, w)          mod 2^32 (int32 two's complement; differences are exact)
    E2[j, v-LO]  = sum_{LO <= w < v, w = v mod 2} P(j, w)  mod 2^32
    FLOOR[j]     = smallest v >= I0 with price 1 (prices are 1 for all v >= FLOOR, > 1 below); INV_HI+10 if none.
    """
    n = INV_HI - INV_LO + 1
    P = np.zeros((N_PRODUCTS, NTAB), np.int64)
    for j in range(N_PRODUCTS):
        row = [market_price(j, v) for v in range(INV_LO, INV_HI + 1)]
        P[j, :n] = row
        P[j, n:] = row[-1]
    # monotone non-increasing in inventory (needed by the closed-form market)
    assert np.all(np.diff(P[:, :n], axis=1) <= 0)
    CS = np.zeros((N_PRODUCTS, NTAB), np.int64)
    CS[:, 1:] = np.cumsum(P[:, :-1], axis=1)
    E2 = np.zeros((N_PRODUCTS, NTAB), np.int64)
    for k in range(2, NTAB):
        E2[:, k] = E2[:, k - 2] + P[:, k - 2]
    FLOOR = np.full(N_PRODUCTS, INV_HI + 10, np.int64)
    for j in range(N_PRODUCTS):
        ones = np.nonzero(P[j, :n] == 1)[0]
        if len(ones):
            FLOOR[j] = INV_LO + int(ones[0])
            assert FLOOR[j] >= I0
    wrap = lambda a: ((a + 2 ** 31) % 2 ** 32 - 2 ** 31).astype(np.int32)
    return P.astype(np.int32), wrap(CS), wrap(E2), FLOOR.astype(np.int32)


# Weed spawn: random() < 0.005 with random() = (a * 2^26 + b) / 2^53, a = w0 >> 5, b = w1 >> 6 (CPython).
# 0.005 * 2^53 = 45035996273704.96..., so the test is the integer compare a * 2^26 + b <= WEED_X.
WEED_X = 45035996273704
WEED_A = WEED_X >> 26
WEED_B = WEED_X & ((1 << 26) - 1)
assert int(0.005 * 2.0 ** 53) == WEED_X and WEED_X < 0.005 * 2.0 ** 53 < WEED_X + 1
