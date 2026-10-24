"""Constants of the v5/v6 encoder and the action vocabulary (mirrors il/features.py and il/features6.py; the tests
assert equality with those modules), plus exact lookup tables.

Why lookup tables: the reference encoder computes in float64 Python math and then stores float32 / uint8.  In JAX,
float32 math is not enough for bit-exactness: XLA rewrites x / c into x * (1/c) (not correctly rounded: 1 ulp off in
~28% of step/720), log1p(money) is 1 ulp off in ~35% of values, the price function rounds half-to-even a float64
expression, and the uint8 continuous planes truncate x*255+0.5 (FMA-sensitive).  Every such feature is an integer
function on a bounded domain, so it is tabulated here with the reference arithmetic itself (numpy float64 / float32,
Python floats): the JAX encoder is exact by construction, on any backend.
"""
import os
import math
import numpy as np

B = 10
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ITEMS = PRODUCTS + ANIMALS                      # 12; same order as the C++ engine's ITEM_NAMES
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
CROP_INFO = {"WHEAT": (2, 4, 6, False), "CARROT": (2, 3, 4, False), "TOMATO": (8, 8, 4, True),
             "STRAWBERRY": (10, 10, 4, True), "MELON": (10, 12, 6, False)}   # first_yield_day, max_yield_day, max_yield, ongoing
ANIMAL_INFO = {"GOOSE": (4, 4), "COW": (8, 4), "SHEEP": (6, 4)}              # first_yield_day, max_held
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
ANIMAL_HOME = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
KIND_NAMES = ["EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE"]         # features.KIND codes 0..5 (EMPTY = tile None)
K_EMPTY, K_LOCKED, K_WEED, K_PLANT, K_COOP, K_PASTURE = range(6)
SHED_ACCESS = [(4, 4), (5, 4), (4, 5), (5, 5)]
SPAWN = (4, 4)

SIMPLE = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP",
          "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
OPS = SIMPLE + ["PLANT_" + c for c in CROPS] + ["PICKUP_" + i for i in ITEMS] + ["PLACE_" + i for i in ITEMS]
OP_INDEX = {o: i for i, o in enumerate(OPS)}
N_OPS = len(OPS)                                # 44
N_QTY = 12
MAX_UNITS = 16                                  # units the network sees (farmer + 15 hands)
OP_PLANT0, OP_PICKUP0, OP_PLACE0 = OP_INDEX["PLANT_WHEAT"], OP_INDEX["PICKUP_WHEAT"], OP_INDEX["PLACE_WHEAT"]   # 15, 20, 32

MARKET_HEADS = ([("SELL:" + p, 18) for p in PRODUCTS] + [("BUY_SEED:" + c, 8) for c in CROPS] +
                [("BUY_PRODUCT:WHEAT", 18), ("BUY_PRODUCT:FERTILIZER", 4)] + [("BUY_ANIMAL:" + a, 4) for a in ANIMALS] +
                [("HIRE", 11), ("BUY_LAND", 2)])
HEAD_INDEX = {h: i for i, (h, _) in enumerate(MARKET_HEADS)}
N_HEADS = len(MARKET_HEADS)                     # 21
HEAD_SIZES = [n for _, n in MARKET_HEADS]
HEAD_OFF = np.cumsum([0] + HEAD_SIZES)          # [22]; total 249
MAX_CLASSES = max(HEAD_SIZES)                   # 18
BIG = {"SELL": 1000, "BUY_PRODUCT:WHEAT": 35}   # decoded value of class 17

C_OWN, C_RIVAL = 31, 16
C = C_OWN + C_RIVAL                             # 47 planes
G5 = 98
HORIZONS = (4, 12, 24, 48)
E6 = 9 * len(HORIZONS) + 9 + 36                 # 81 future features
G6 = G5 + E6                                    # 179
U = MAX_UNITS + 2 + len(ITEMS) + 1 + 1 + N_OPS  # 76 unit features (incl. previous-op one-hot)

# features6 market model (MKT) and town drain
MKT = {"WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20), "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
       "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60), "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
       "MELON": (250, 300, "log", 0.20, "sq", 3.60), "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
       "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60), "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
       "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40)}
I0, FLOOR, HINGE = 10000, 1, 8.0
SHOP_ITEMS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
              "YARN_STORE": ["WOOL"], "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
              "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
# SHOP_RATE[s, p]: units of product p one instance of shop s takes every 4th turn (2 for single-product shops)
SHOP_RATE = np.zeros((len(SHOPS), len(PRODUCTS)), np.int32)
for _s, _sh in enumerate(SHOPS):
    for _it in SHOP_ITEMS[_sh]:
        SHOP_RATE[_s, PRODUCTS.index(_it)] = 2 if len(SHOP_ITEMS[_sh]) == 1 else 1
TOWN_RATE = np.array([0 if p == "FERTILIZER" else 1 for p in PRODUCTS], np.int32)   # every 24th turn

# per-crop / per-animal arrays (index = CROPS / ANIMALS order)
CROP_FIRST = np.array([CROP_INFO[c][0] for c in CROPS], np.int32)
CROP_MAXDAY = np.array([CROP_INFO[c][1] for c in CROPS], np.int32)
CROP_MAXY = np.array([CROP_INFO[c][2] for c in CROPS], np.int32)
CROP_ONGOING = np.array([CROP_INFO[c][3] for c in CROPS], bool)
ANIMAL_FIRST = np.array([ANIMAL_INFO[a][0] for a in ANIMALS], np.int32)
ANIMAL_PROD = np.array([PRODUCTS.index(ANIMAL_PRODUCT[a]) for a in ANIMALS], np.int32)
ANIMAL_HOME_KIND = np.array([K_COOP if ANIMAL_HOME[a] == "COOP" else K_PASTURE for a in ANIMALS], np.int32)
BASE_ARR = np.array([BASE[p] for p in PRODUCTS], np.int32)

# ---- lookup tables ---------------------------------------------------------------------------------------------
YMAX = 16            # yield_units clipped to this for the ratio plane (ratio saturates at 1 from max_yield / max_held on)
AGE_LO, AGE_HI = -2, 40
PRICE_LO, PRICE_N = -4096, 40960   # price(item, inv) tabulated for inv in [PRICE_LO, PRICE_LO + PRICE_N) (the forecast
                                    # subtracts up to ~500 of town drain); real games stay within ~[9500, 10500]
LOG_N = 1 << 20      # log1p(money)/12 and money/1e5 tabulated for integer money in [0, LOG_N); above: float32 fallback
MDIFF = 60001        # (m_own - m_riv)/2e4 tabulated for |diff| <= MDIFF (clipped to +-3 afterwards, exact beyond)
OBS_PRICE_N = 4096   # observed price / base tabulated for prices in [0, OBS_PRICE_N); above: float32 fallback
# Integer / constant quotients of the encoder, name -> (divisor, lo, hi): table[i] = float32(float64(lo + i) / divisor)
# for i in [0, hi - lo].  Indices are clipped into [lo, hi]; every feature using a clipped table saturates (min / clip)
# at or inside the range, so clipping is exact.  (XLA rewrites x / c into x * (1/c), which is not correctly rounded:
# measured 222 of 800 values of step/720 off by 1 ulp, so no float division is used anywhere in the encoder.)
QUOT = {"q720": (720.0, 0, 1023), "q29": (29.0, 0, 63), "q10": (10.0, 0, 20), "q50": (50.0, 0, 100),
        "q100": (100.0, 0, 4095), "qinv": (100.0, -301, 301), "q2": (2.0, 0, 63), "q8": (8.0, 0, 63),
        "q15": (15.0, -1, 63), "q72": (72.0, 0, 127), "q30": (30.0, 0, 60), "q9": (9.0, 0, 15), "q20": (20.0, 0, 60),
        "q255": (255.0, 0, 255)}


def _u8(vals):
    """Reference plane quantization: values stored in a float32 array, then np.clip(p*255.0+0.5, 0, 255).astype(uint8)."""
    a = np.zeros(len(vals), np.float32)
    for i, v in enumerate(vals):
        a[i] = v
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _shape(func, x, T):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    u = x / T
    return u + HINGE * max(0.0, u - 1.0) ** 2


def price(item, inv):
    """features6.price (the engine's market_price): int."""
    base, T, bf, bt, af, at = MKT[item]
    if inv < I0:
        p = base + bt * base / _shape(bf, T, T) * _shape(bf, I0 - inv, T)
    else:
        p = base - at * base / _shape(af, T, T) * _shape(af, inv - I0, T)
    return max(FLOOR, int(round(p)))


def build_tables():
    t = {}
    # plane 16 (yield ratio): rows 0..4 crops (max_yield), 5..7 animals (max_held)
    den = [CROP_INFO[c][2] for c in CROPS] + [ANIMAL_INFO[a][1] for a in ANIMALS]
    t["lut_yield"] = np.stack([_u8([min(1.0, yu / float(d)) for yu in range(YMAX + 1)]) for d in den])        # [8,17]
    # plane 17 (age ratio) per crop, age in [AGE_LO, AGE_HI]
    t["lut_age"] = np.stack([_u8([min(1.0, age / (1.5 * (CROP_INFO[c][1] + 1))) for age in range(AGE_LO, AGE_HI + 1)])
                             for c in CROPS])                                                                  # [5,43]
    # planes 28 / 46 (units on a cell): the reference adds 1/3 per unit in a float32 array, capped at 1
    cnt = np.zeros(1, np.float32); vals = [0.0]
    for _ in range(41):
        cnt[0] = min(1.0, cnt[0] + 1 / 3.0); vals.append(float(cnt[0]))
    t["lut_units"] = _u8(vals)                                                                                 # [42]
    # price(item, inv) / base as float32 for inv in [PRICE_LO, PRICE_LO + PRICE_N)
    pt = np.zeros((len(PRODUCTS), PRICE_N), np.float32)
    for i, p in enumerate(PRODUCTS):
        pt[i] = [price(p, x) / MKT[p][0] for x in range(PRICE_LO, PRICE_LO + PRICE_N)]
    t["price_ratio"] = pt
    # log1p(money)/12 and money/1e5 for integer money (float64 math stored as float32, as the reference)
    m = np.arange(LOG_N, dtype=np.float64)
    t["log_money"] = (np.log1p(m) / 12).astype(np.float32)
    t["money_q"] = (m / 1e5).astype(np.float32)
    d = np.arange(-MDIFF, MDIFF + 1, dtype=np.float64)
    t["mdiff_q"] = (d / 2e4).astype(np.float32)
    t["obs_price_q"] = np.stack([(np.arange(OBS_PRICE_N, dtype=np.float64) / BASE[p]).astype(np.float32) for p in PRODUCTS])
    for name, (dv, lo, hi) in QUOT.items():
        t[name] = (np.arange(lo, hi + 1, dtype=np.float64) / dv).astype(np.float32)
    return t


# per platform: the tables embed float64 libm results (log1p, log, sqrt) of the machine that builds them, so each
# machine uses its own (Linux x86_64 = the Kaggle runtime's platform)
import platform as _platform
_TABLES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"_tables_v3_{_platform.system()}_{_platform.machine()}.npz")
_TABLES = None


def tables():
    """Cached lookup tables (numpy).  Rebuilt when the cache file is missing."""
    global _TABLES
    if _TABLES is None:
        if os.path.exists(_TABLES_FILE):
            z = np.load(_TABLES_FILE)
            _TABLES = {k: z[k] for k in z.files}
        else:
            _TABLES = build_tables()
            try:
                np.savez(_TABLES_FILE, **_TABLES)
            except OSError:
                pass
    return _TABLES


# ---- action vocab metadata (for act.py) --------------------------------------------------------------------------
# C++ engine unit op codes (engine/src_v2 UOP_NAMES) and market op codes (MOP_NAMES)
UOP = {n: i for i, n in enumerate(["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE", "PLANT", "WATER",
                                   "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED",
                                   "COLLECT_FERTILIZER", "CARE"])}
MOP = {n: i for i, n in enumerate(["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"])}
# op class (44) -> (engine uop code, item index or -1)
OP_UOP = np.zeros(N_OPS, np.int32)
OP_ITEM = np.full(N_OPS, -1, np.int32)
for _k, _o in enumerate(OPS):
    if _o.startswith("PLANT_"):
        OP_UOP[_k] = UOP["PLANT"]; OP_ITEM[_k] = ITEMS.index(_o[6:])
    elif _o.startswith("PICKUP_"):
        OP_UOP[_k] = UOP["PICKUP"]; OP_ITEM[_k] = ITEMS.index(_o[7:])
    elif _o.startswith("PLACE_"):
        OP_UOP[_k] = UOP["PLACE"]; OP_ITEM[_k] = ITEMS.index(_o[6:])
    else:
        OP_UOP[_k] = UOP[_o]
