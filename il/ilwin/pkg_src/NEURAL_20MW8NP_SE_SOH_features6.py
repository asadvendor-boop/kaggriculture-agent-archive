"""v6 encoder = the v5 encoder (features.py) + "if nobody acts" future features, the idea every top Orbit Wars solution
used.  All exact, from the official engine's rules (kaggle_environments/envs/kaggriculture/kaggriculture.py:
MARKET_PARAMS, market_price, _town_consume, SHOPS):

  price forecast   price / base of each product after 4, 12, 24, 48 more turns of the town's drain with no sales
                   (every 4 turns each unlocked shop takes 1 of each of its products, 2 if it sells one product; every
                   24 turns the town center takes 1 of every product except fertilizer)             9 x 4 = 36
  dump price       average price / base if we sold our whole shed stock of the product now             9
  pending supply   per farm (ours, rival's): units ready now (mature crop yields, held animal products,
                   uncollected fertilizer) and units maturing within a day                           2 x 2 x 9 = 36

Glob layout: the 98 v5 features, then these 81 (G = 179).  Kept in its own module: RL pool workers re-import
features.py from disk, so the v5 encoder must never change in place.
"""
import math
import numpy as np
from features import *                      # noqa: F401,F403  (v5 constants, labels, decoders, rule layer)
import features as _F

MKT = {  # base, T, below_func, below_target, above_func, above_target
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20), "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60), "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60), "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60), "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40)}
I0, FLOOR, HINGE = 10000, 1, 8.0
SHOP_ITEMS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
              "YARN_STORE": ["WOOL"], "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
              "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
HORIZONS = (4, 12, 24, 48)
E = 9 * len(HORIZONS) + 9 + 36
G5 = _F.G
G = G5 + E


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
    u = x / T                                    # hinge
    return u + HINGE * max(0.0, u - 1.0) ** 2


def price(item, inv):
    base, T, bf, bt, af, at = MKT[item]
    if inv < I0:
        p = base + bt * base / _shape(bf, T, T) * _shape(bf, I0 - inv, T)
    else:
        p = base - at * base / _shape(af, T, T) * _shape(af, inv - I0, T)
    return max(FLOOR, int(round(p)))


def drain(step, k, shops):
    """Units of each product the town consumes over the next k turns (turns step+1 .. step+k)."""
    d = dict.fromkeys(PRODUCTS, 0)
    for s in range(step + 1, step + k + 1):
        if s % 4 == 0:
            for sh in shops:
                items = SHOP_ITEMS.get(sh, ())
                m = 2 if len(items) == 1 else 1
                for it in items:
                    d[it] += m
        if s % 24 == 0:
            for it in PRODUCTS:
                if it != "FERTILIZER":
                    d[it] += 1
    return d


def future_features(obs):
    step = int(obs["step"]); me = int(obs["player"]); day = step // 24
    inv = obs["market"]["inventory"]; shops = list(obs["town"].get("unlocked_shops") or [])
    x = np.zeros(E, np.float32); i = 0
    per_shop = [(SHOP_ITEMS.get(sh, ()), 2 if len(SHOP_ITEMS.get(sh, ())) == 1 else 1) for sh in shops]
    dr = dict.fromkeys(PRODUCTS, 0); s_ = step
    for k in HORIZONS:                          # cumulative drain, one pass over the next 48 turns
        while s_ < step + k:
            s_ += 1
            if s_ % 4 == 0:
                for items, m in per_shop:
                    for it in items:
                        dr[it] += m
            if s_ % 24 == 0:
                for it in PRODUCTS:
                    if it != "FERTILIZER":
                        dr[it] += 1
        for p in PRODUCTS:
            x[i] = price(p, int(inv.get(p, I0)) - dr[p]) / MKT[p][0]; i += 1
    shed = obs["private"].get("shed") or {}
    for p in PRODUCTS:                         # midpoint of the dump: average price of selling the whole stock now
        q = int(shed.get(p, 0))
        x[i] = price(p, int(inv.get(p, I0)) + q // 2) / MKT[p][0]; i += 1
    for who in (me, 1 - me):
        now = dict.fromkeys(PRODUCTS, 0); soon = dict.fromkeys(PRODUCTS, 0)
        for row in obs["farms"][who]["tiles"]:
            for tl in row:
                if not isinstance(tl, dict):
                    continue
                if tl.get("kind") == "PLANT" and tl.get("crop") in CROP_INFO:
                    first = CROP_INFO[tl["crop"]][0]; age = day - int(tl.get("planted_day", day))
                    n = int(tl.get("yield_units", 0) or 0)
                    if age >= first:
                        now[tl["crop"]] += n
                    elif age + 1 >= first:
                        soon[tl["crop"]] += max(1, n)
                elif tl.get("animal") in ANIMAL_PRODUCT:
                    now[ANIMAL_PRODUCT[tl["animal"]]] += int(tl.get("yield_units", 0) or 0)
                    if tl.get("fertilizer_available"):
                        now["FERTILIZER"] += 1
                    elif tl.get("fed_today"):
                        soon["FERTILIZER"] += 1
        for p in PRODUCTS:
            x[i] = min(3.0, now[p] / 20.0); x[i + 9] = min(3.0, soon[p] / 20.0); i += 1
        i += 9
    assert i == E, i
    return x


def encode_obs(obs, prev_ops=None):
    planes, g, pos, uf = _F.encode_obs(obs, prev_ops)
    return planes, np.concatenate([g, future_features(obs)]).astype(np.float32), pos, uf
