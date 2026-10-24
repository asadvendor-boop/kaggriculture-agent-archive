"""Observation / action encoding for the imitation-learning agent (pure Python + numpy).

Used identically by the dataset builder (from engine replays of leader games), the trainer, and the runtime agent,
so the network always sees the same features.  Board is 10x10, farm tiles are None | "LOCKED" | dict.

Per step the encoder produces
  planes : uint8 [C, 10, 10]   own + rival farm state, scaled to 0..255
  glob   : float32 [G]         clock, money, seeds, shed, prices, market inventory, shops, hires, land
  units  : int16 [N, 2] positions (x, y) and float32 [N, U] per-unit features (index, inventory, previous op)
Actions are encoded per unit as an op class (OPS, 44 classes) plus a quantity class for PICKUP/PLACE, and the
market list as 21 quantity heads (MARKET_HEADS) decoded back in the leader's canonical order.
"""
import numpy as np

B = 10
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ITEMS = PRODUCTS + ANIMALS                       # 12 shed items
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
CROP_INFO = {  # first_yield_day, max_yield_day, max_yield, ongoing
    "WHEAT": (2, 4, 6, False), "CARROT": (2, 3, 4, False), "TOMATO": (8, 8, 4, True),
    "STRAWBERRY": (10, 10, 4, True), "MELON": (10, 12, 6, False)}
ANIMAL_INFO = {"GOOSE": (4, 4), "COW": (8, 4), "SHEEP": (6, 4)}   # first_yield_day, max_held
KIND = {None: 0, "LOCKED": 1, "WEED": 2, "PLANT": 3, "COOP": 4, "PASTURE": 5}
SHED_ACCESS = {(4, 4), (5, 4), (4, 5), (5, 5)}
SPAWN = (4, 4)
QUADS = ["NE", "SW", "SE"]

# ---- unit op vocabulary -----------------------------------------------------------------------------------------
SIMPLE = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP",
          "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
OPS = SIMPLE + ["PLANT_" + c for c in CROPS] + ["PICKUP_" + i for i in ITEMS] + ["PLACE_" + i for i in ITEMS]
OP_INDEX = {o: i for i, o in enumerate(OPS)}
N_OPS = len(OPS)                                  # 44
N_QTY = 12                                        # quantity classes 1..12 (class k -> k+1)
MAX_UNITS = 16

# ---- market heads (name, n_classes) -------------------------------------------------------------------------------
MARKET_HEADS = ([("SELL:" + p, 18) for p in PRODUCTS] + [("BUY_SEED:" + c, 8) for c in CROPS] +
                [("BUY_PRODUCT:WHEAT", 18), ("BUY_PRODUCT:FERTILIZER", 4)] + [("BUY_ANIMAL:" + a, 4) for a in ANIMALS] +
                [("HIRE", 11), ("BUY_LAND", 2)])
HEAD_INDEX = {h: i for i, (h, _) in enumerate(MARKET_HEADS)}
N_HEADS = len(MARKET_HEADS)                       # 21
BIG = {"SELL": 1000, "BUY_PRODUCT:WHEAT": 35}     # decoded value of the top class (17)

C_OWN, C_RIVAL = 31, 16
C = C_OWN + C_RIVAL
G = 98
U = MAX_UNITS + 2 + len(ITEMS) + 1 + 1 + N_OPS   # index one-hot, x, y, inventory, carrying, at shed, previous op


def _tile_kind(t):
    if t is None or t == "LOCKED":
        return t
    return t.get("kind") if isinstance(t, dict) else None


def encode_obs(obs, prev_ops=None):
    """obs: the engine / Kaggle observation dict for the acting player.  prev_ops: list of op-class ints (by unit
    index) this player issued last step, or None.  Returns (planes, glob, pos, ufeat)."""
    me = int(obs["player"])
    step = int(obs["step"])
    day = step // 24
    hour = step % 24
    farms = obs["farms"]
    own, riv = farms[me], farms[1 - me]
    planes = np.zeros((C, B, B), np.float32)
    # ---- own farm ----
    for y in range(B):
        row = own["tiles"][y]
        for x in range(B):
            t = row[x]
            k = _tile_kind(t)
            planes[KIND.get(k, 0), y, x] = 1
            if not isinstance(t, dict):
                continue
            if k == "PLANT":
                crop = t.get("crop")
                ci = CROPS.index(crop) if crop in CROPS else -1
                if ci >= 0:
                    planes[6 + ci, y, x] = 1
                    fy, my, mxy, ongoing = CROP_INFO[crop]
                    age = day - int(t.get("planted_day", day))
                    yu = int(t.get("yield_units", 0))
                    planes[14, y, x] = 1 if t.get("watered_today") else 0
                    planes[15, y, x] = 1 if (not t.get("watered_today") and int(t.get("consecutive_unwatered", 0)) >= 1) else 0
                    planes[16, y, x] = min(1.0, yu / float(mxy))
                    planes[17, y, x] = min(1.0, age / (1.5 * (my + 1)))
                    planes[18, y, x] = 1 if (yu > 0 and age >= fy) else 0
                    planes[19, y, x] = 1 if int(t.get("fertilized_until_day", -1)) >= day else 0
                    if not ongoing and (my + 1) // 2 <= age <= my:
                        planes[20, y, x] = 1
                    mls = int(t.get("max_lifespan_step", -1))
                    if mls >= 0 and step >= mls - 12:
                        planes[21, y, x] = 1
            elif "animal" in t:
                a = t.get("animal")
                ai = ANIMALS.index(a) if a in ANIMALS else -1
                if ai >= 0:
                    planes[11 + ai, y, x] = 1
                    fy, mh = ANIMAL_INFO[a]
                    planes[16, y, x] = min(1.0, int(t.get("yield_units", 0)) / float(mh))
                    planes[22, y, x] = 1 if t.get("fed_today") else 0
                    planes[23, y, x] = 1 if t.get("cared_today") else 0
                    planes[24, y, x] = 1 if t.get("fertilizer_available") else 0
                    planes[25, y, x] = 1 if (not t.get("fed_today") and int(t.get("consecutive_unfed", 0)) >= 1) else 0
                    planes[26, y, x] = 1 if int(t.get("pending_care_bonus", 0)) > 0 else 0
                    planes[27, y, x] = 1 if day - int(t.get("placed_day", day)) >= fy - 1 else 0
                    planes[18, y, x] = 1 if int(t.get("yield_units", 0)) > 0 else 0
    positions = [own["farmer"]] + list(own.get("hands") or [])
    for p in positions:
        planes[28, int(p[1]), int(p[0])] = min(1.0, planes[28, int(p[1]), int(p[0])] + 1 / 3.0)
    for (x, y) in SHED_ACCESS:
        planes[29, y, x] = 1
    planes[30, SPAWN[1], SPAWN[0]] = 1
    # ---- rival farm (subset) ----
    o = C_OWN
    for y in range(B):
        row = riv["tiles"][y]
        for x in range(B):
            t = row[x]
            k = _tile_kind(t)
            planes[o + KIND.get(k, 0), y, x] = 1
            if isinstance(t, dict):
                if k == "PLANT" and t.get("crop") in CROPS:
                    planes[o + 6 + CROPS.index(t["crop"]), y, x] = 1
                    planes[o + 14, y, x] = 1 if int(t.get("yield_units", 0)) > 0 else 0
                elif t.get("animal") in ANIMALS:
                    planes[o + 11 + ANIMALS.index(t["animal"]), y, x] = 1
                    planes[o + 14, y, x] = 1 if int(t.get("yield_units", 0)) > 0 else 0
    for p in [riv["farmer"]] + list(riv.get("hands") or []):
        planes[o + 15, int(p[1]), int(p[0])] = min(1.0, planes[o + 15, int(p[1]), int(p[0])] + 1 / 3.0)
    planes_u8 = np.clip(planes * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # ---- global vector ----
    g = np.zeros(G, np.float32)
    i = 0
    g[i] = step / 720.0; g[i + 1] = day / 29.0; i += 2
    g[i + hour] = 1; i += 24
    m_own, m_riv = float(own.get("money", 0)), float(riv.get("money", 0))
    g[i] = m_own / 1e5; g[i + 1] = m_riv / 1e5; g[i + 2] = np.clip((m_own - m_riv) / 2e4, -3, 3); g[i + 3] = np.log1p(max(0.0, m_own)) / 12; i += 4
    priv = obs["private"]
    seeds = priv.get("seeds", {}) or {}
    for c in CROPS:
        g[i] = min(2.0, seeds.get(c, 0) / 10.0); i += 1
    shed = priv.get("shed", {}) or {}
    for it in ITEMS:
        g[i] = min(2.0, shed.get(it, 0) / 50.0); i += 1
    g[i] = sum(shed.values()) / 100.0; i += 1
    prices = obs["market"]["prices"]; inv = obs["market"]["inventory"]
    for p in PRODUCTS:
        g[i] = prices.get(p, BASE[p]) / BASE[p]; i += 1
    for p in PRODUCTS:
        g[i] = np.clip((inv.get(p, 10000) - 10000) / 100.0, -3, 3); i += 1
    shops = list(obs["town"].get("unlocked_shops", []) or [])
    for s in SHOPS:
        g[i] = shops.count(s) / 2.0; i += 1
    g[i] = len(shops) / 8.0; i += 1
    g[i] = int(own.get("hires_today", 0)) / 15.0; g[i + 1] = len(own.get("hands") or []) / 15.0
    g[i + 2] = int(riv.get("hires_today", 0)) / 15.0; g[i + 3] = len(riv.get("hands") or []) / 15.0; i += 4
    oq, rq = own.get("unlocked_quadrants") or [], riv.get("unlocked_quadrants") or []
    for q in QUADS:
        g[i] = 1 if q in oq else 0; g[i + 3] = 1 if q in rq else 0; i += 1
    i += 3
    g[i] = ((3 - day % 3) * 24 - hour) / 72.0; i += 1
    invs = priv.get("inventories", []) or []
    for it in ITEMS:
        g[i] = min(2.0, sum(v.get(it, 0) for v in invs) / 30.0); i += 1
    assert i == G, i

    # ---- units ----
    n = min(MAX_UNITS, len(positions))
    pos = np.zeros((n, 2), np.int16)
    uf = np.zeros((n, U), np.float32)
    for k in range(n):
        x, y = int(positions[k][0]), int(positions[k][1])
        pos[k] = (x, y)
        uf[k, k] = 1
        uf[k, MAX_UNITS] = x / 9.0; uf[k, MAX_UNITS + 1] = y / 9.0
        iv = invs[k] if k < len(invs) else {}
        for j, it in enumerate(ITEMS):
            uf[k, MAX_UNITS + 2 + j] = min(2.0, iv.get(it, 0) / 10.0)
        uf[k, MAX_UNITS + 2 + len(ITEMS)] = 1 if any(v > 0 for v in iv.values()) else 0
        uf[k, MAX_UNITS + 3 + len(ITEMS)] = 1 if (x, y) in SHED_ACCESS else 0
        if prev_ops is not None and k < len(prev_ops) and prev_ops[k] >= 0:
            uf[k, MAX_UNITS + 4 + len(ITEMS) + prev_ops[k]] = 1
    return planes_u8, g, pos, uf


def unit_op_label(cmd):
    """(op_class, qty_class or -1) for one unit command list."""
    if not isinstance(cmd, list) or not cmd:
        return OP_INDEX["PASS"], -1
    op = cmd[0]
    if op in ("PLANT",):
        c = cmd[1] if len(cmd) > 1 else None
        return (OP_INDEX["PLANT_" + c], -1) if c in CROPS else (OP_INDEX["PASS"], -1)
    if op in ("PICKUP", "PLACE"):
        it = cmd[1] if len(cmd) > 1 else None
        if it not in ITEMS:
            return OP_INDEX["PASS"], -1
        try:
            q = int(cmd[2]) if len(cmd) > 2 else 1
        except Exception:
            q = 1
        return OP_INDEX[op + "_" + it], max(0, min(N_QTY, q) - 1)
    return OP_INDEX.get(op, OP_INDEX["PASS"]), -1


def unit_command(op_class, qty_class=0):
    o = OPS[op_class]
    if o.startswith("PLANT_"):
        return ["PLANT", o[6:]]
    if o.startswith("PICKUP_"):
        return ["PICKUP", o[7:], int(qty_class) + 1]
    if o.startswith("PLACE_"):
        return ["PLACE", o[6:], int(qty_class) + 1]
    return [o]


def market_labels(market):
    """Market order list -> int8 [N_HEADS] quantity classes."""
    lab = np.zeros(N_HEADS, np.int8)
    q = {}
    for o in market or []:
        if not (isinstance(o, list) and o):
            continue
        if o[0] == "HIRE":
            q["HIRE"] = q.get("HIRE", 0) + 1
        elif o[0] == "BUY_LAND":
            q["BUY_LAND"] = 1
        elif o[0] in ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL") and len(o) >= 3:
            key = o[0] + ":" + str(o[1])
            if key in HEAD_INDEX:
                try:
                    q[key] = q.get(key, 0) + max(0, int(o[2]))
                except Exception:
                    pass
    for key, v in q.items():
        h = HEAD_INDEX[key]
        lab[h] = min(v, MARKET_HEADS[h][1] - 1)
    return lab


SELL_ORDER = "held"                     # "canonical" (leaders' fixed order) | "base" | "held" | "value" (tested)


def decode_market(classes, obs=None):
    """[N_HEADS] ints -> market order list in the leader's canonical order (sells, hires, land, animals, seeds,
    product buys), at most 10 orders."""
    out = []
    for p in ["FERTILIZER", "WHEAT", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO", "MELON"]:
        k = int(classes[HEAD_INDEX["SELL:" + p]])
        if k > 0:
            out.append(["SELL", p, BIG["SELL"] if k == 17 else k])
    out += [["HIRE"]] * int(classes[HEAD_INDEX["HIRE"]])
    if int(classes[HEAD_INDEX["BUY_LAND"]]):
        out.append(["BUY_LAND"])
    for a in ANIMALS:
        k = int(classes[HEAD_INDEX["BUY_ANIMAL:" + a]])
        if k > 0:
            out.append(["BUY_ANIMAL", a, k])
    for c in CROPS:
        k = int(classes[HEAD_INDEX["BUY_SEED:" + c]])
        if k > 0:
            out.append(["BUY_SEED", c, k])
    k = int(classes[HEAD_INDEX["BUY_PRODUCT:WHEAT"]])
    if k > 0:
        out.append(["BUY_PRODUCT", "WHEAT", BIG["BUY_PRODUCT:WHEAT"] if k == 17 else k])
    k = int(classes[HEAD_INDEX["BUY_PRODUCT:FERTILIZER"]])
    if k > 0:
        out.append(["BUY_PRODUCT", "FERTILIZER", k])
    if SELL_ORDER != "canonical":            # slot order decides who sells first when both farms sell a product
        sells = [o for o in out if o[0] == "SELL"]; rest = [o for o in out if o[0] != "SELL"]
        if SELL_ORDER == "base":
            sells.sort(key=lambda o: -BASE[o[1]])
        elif SELL_ORDER == "held" and obs is not None:   # Sep 28: products we hold (shed or carried) first, premium first;
            priv = obs.get("private") or {}; sh = priv.get("shed") or {}   # phantom sells (nothing held) last (they
            held = {k: int(v) for k, v in sh.items()}                      # cost slot priority in the lockstep)
            for inv in priv.get("inventories") or []:
                for k, v in (inv or {}).items():
                    held[k] = held.get(k, 0) + int(v)
            sells.sort(key=lambda o: (0 if held.get(o[1], 0) > 0 else 1, -BASE[o[1]]))
        elif SELL_ORDER == "value" and obs is not None:
            pr = obs["market"]["prices"]; sh = (obs.get("private") or {}).get("shed") or {}
            sells.sort(key=lambda o: -pr.get(o[1], BASE[o[1]]) * min(int(o[2]), int(sh.get(o[1], 0)) or int(o[2])))
        out = sells + rest
    if len(out) > 10:                      # engine executes 10 orders: drop trailing SELLs (they can wait an hour),
        other = [o for o in out if o[0] != "SELL"][:10]          # never a HIRE / purchase (hour-0 hires were being cut)
        out = [o for o in out if o[0] == "SELL"][:10 - len(other)] + other
    return out


SHED_CAP = 100
FINAL_FIX = 1


HOMING = 1


def _homing(obs, action, left):
    """Last day: goods still carried at the final turn (step 718) are lost.  A unit carrying products walks to the
    nearest shed tile as late as possible (when its Manhattan distance >= turns left - 1) so it can DROP there by the
    final turn, where the final-turn rule sells everything."""
    priv = obs["private"]; invs = priv.get("inventories") or []
    farm = obs["farms"][int(obs["player"])]
    positions = [farm["farmer"]] + list(farm.get("hands") or [])
    units = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
    changed = False
    for i in range(min(len(units), len(positions), len(invs))):
        if not any(k in PRODUCTS and int(v) > 0 for k, v in invs[i].items()):
            continue
        x, y = int(positions[i][0]), int(positions[i][1])
        tx, ty = min(SHED_ACCESS, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
        d = abs(tx - x) + abs(ty - y)
        if d == 0:
            if left <= 1:
                units[i] = ["DROP"]; changed = True
            continue
        if d >= left - 1 and d <= left:
            units[i] = ["EAST"] if tx > x else ["WEST"] if tx < x else ["SOUTH"] if ty > y else ["NORTH"]
            changed = True
    return dict(action, farmer=units[0], hands=units[1:]) if changed else action


def _final_fix(obs, action):
    """Game's last turn: goods are worth nothing afterwards, so every unit standing at the shed with products drops
    them (units act before the market) and every product in the shed is sold (units elsewhere cannot reach it)."""
    priv = obs["private"]; invs = priv.get("inventories") or []
    farm = obs["farms"][int(obs["player"])]
    positions = [farm["farmer"]] + list(farm.get("hands") or [])
    stock = {k: int(v) for k, v in (priv.get("shed") or {}).items() if k in PRODUCTS}
    units = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
    for i in range(min(len(units), len(positions), len(invs))):
        if (int(positions[i][0]), int(positions[i][1])) in SHED_ACCESS and any(k in PRODUCTS and int(v) > 0 for k, v in invs[i].items()):
            units[i] = ["DROP"]
            for k, v in invs[i].items():
                if k in PRODUCTS:
                    stock[k] = stock.get(k, 0) + int(v)
    if FINAL_ORDER == "base":                 # Sep 28: the last turn is the biggest same-turn sale race of the game
        key = lambda kv: -BASE.get(kv[0], 0)
    elif FINAL_ORDER == "value":
        pr = (obs.get("market") or {}).get("prices") or {}
        key = lambda kv: -float(pr.get(kv[0], BASE.get(kv[0], 0)) or 0) * kv[1]
    else:
        key = lambda kv: -kv[1]
    market = [["SELL", k, n] for k, n in sorted(stock.items(), key=key) if n > 0][:10]
    return dict(action, farmer=units[0], hands=units[1:], market=market)


FINAL_ORDER = "qty"   # slot order of the last turn's sell-everything list: "qty" (largest lot first) | "base" | "value"


SE_GUARD = 1          # rule layer (Sep 26 live-loss study): buy the 3rd land (SE) from day 10 hour SE_HOUR (0 = off)
SE_HOUR = 12
SE_LAST_DAY = 13
SE_RESERVE = 300
SE_SHOPS = 0          # light planning (Sep 28, il/live/se_world_features.py): with SE_GUARD, buy the third field only when at
                      # least SE_SHOPS of the first three shops (unlocked on days 3, 6, 9) buy crops (PIZZA_SHOP / FARMERS_MARKET).
                      # New-plan net, 750 paired worlds: 0 such shops -> SE costs $3.6k, 1 -> $1.1k, 2 -> SE +$7.4k, 3 -> +$21.3k
                      # (0 = the guard applies in every world, as before)
CROP_SHOPS = ("PIZZA_SHOP", "FARMERS_MARKET")
SHEEP_YARN = 0        # light planning (Sep 28, live weakness W1: NPS lost -$11.5k to Aaweg with 9 sheep vs 13 in a 2-yarn-store
                      # world, wool -$17k): keep the flock (placed + shed + carried) at SHEEP_BASE + SHEEP_YARN per YARN_STORE
SHEEP_BASE = 3        # open so far (DSM's table ~3 + 7 per store; PLANNING_FORENSICS 3.3), capped at SHEEP_MAX, while
SHEEP_MAX = 18        # (no cross-farm cap: TOPVSTOP_A, supply lowers the rival's wool price) the wool price clears the
SHEEP_LAST_DAY = 15   # payback floor for the day (break-even $56-83 up to day 20, x1.5), until SHEEP_LAST_DAY; at most 2 a turn,
                      # ordered before the seed buys (0 = off)
TOMATO_SHOPS = 0      # light planning (Sep 28, live weakness W6: NPS lost to Planned Economy with 0 tomato tiles in a PIZZA x3 +
TOMATO_PER_SHOP = 8   # FARMERS_MARKET town; they sold 200 tomatoes at $195): with >= TOMATO_SHOPS tomato buyers (PIZZA_SHOP /
TOMATO_LAST_DAY = 18  # FARMERS_MARKET) open and tomatoes >= $55, keep TOMATO_PER_SHOP tomato tiles per such shop (cap 32): the
                      # policy's own CARROT (and WHEAT, while the shed keeps >= 25 wheat for feed) PLANT orders become TOMATO while
                      # seeds last; seeds ($50) bought ahead. A tile yields 4 (8 fertilized) from 8 days after planting (0 = off)
TOMATO_FROM_DAY = 2   # first day of the tomato rule (TOPVSTOP_B: DSM plants ~12 tomato tiles on days 15-18 in place of wheat in
                      # crop-shop towns, +$8.8k: the DSM-calibrated variant is TOMATO_FROM_DAY=15, TOMATO_PER_SHOP=4)
COW_MILK = 0          # light planning (Sep 28, live weakness W7: NPS lost to Orbital Terraformer in a 4-smoothie town, milk 59 vs 133
COW_BASE = 5          # with 5 cows each): keep COW_BASE + COW_MILK cows per milk buyer (SMOOTHIE_SHOP / ICE_CREAM_SHOP / PIZZA_SHOP)
COW_MAX = 15          # open (DSM's table ~5.7 + 3.3 per milk shop), cap COW_MAX, while milk >= $100 (<= day 15) / $130, until
COW_LAST_DAY = 18     # COW_LAST_DAY; at most 2 a turn, ordered before the seed buys (0 = off)
SHEEP_FLOOR = 150       # Sep 28 night (fresh3 study of sy7g1: late sheep bought at $90-150 wool after a day-15 yarn store lose $15-19k,
                      # early sheep at $200+ wool win $21-36k): one fixed wool-price floor for SHEEP_YARN buys (0 = old 90 / 125 / 150)
SHEEP_EARLY = 0       # Sep 28 night (wool study, fresh3 sy7g1 by first-store day: d11 -$5.3k, d14 -$8.8k, d17 -$3.0k a seat; leaders
                      # add ~2-3 sheep for a d11-14 store, ~0 later): only yarn stores open by this day count for the SHEEP_YARN table
                      # (the town's i-th shop opens on day 3i + 3); later stores add 0 (0 = every store counts)
SHEEP_MIRROR = 1      # Sep 28 night (wool study: live wool losses 3 vs 7-11 sheep with one d8/d11 store): with a yarn store open, the
                      # flock target is at least the rival's placed sheep + SHEEP_MIRROR - 1 (1 = match; wool margin ~0 instead of
                      # the rival's whole wool edge) and at least SHEEP_BASE; own floor / last day below, same cash rules (0 = off)
SHEEP_MFLOOR = 0      # wool-price floor of the mirror target (0 = SHEEP_FLOOR's rule); the SHEEP_YARN table keeps SHEEP_FLOOR
SHEEP_MLAST = 0       # last day of the mirror target (0 = SHEEP_LAST_DAY); the SHEEP_YARN table keeps SHEEP_LAST_DAY
SHEEP_MGAP = 4        # the mirror switches on (for the rest of the game, per game state) once the rival's placed flock leads ours by
                      # >= SHEEP_MGAP sheep (wool-loss replays: leads of 4-12 sheep = +$62.6k over 7 worlds; leads of 0-1 lost) (0 = any lead)
SHEEP_MINY = 0        # Sep 28 night (wool study: one-store wool markets flood by days 16-25 at any flock size - wool $12-110 in the
                      # control worlds - and extra sheep lost there; 2+ store markets stay $200+): the sheep rules act only with at
                      # least SHEEP_MINY yarn stores open (0 = any)
SHEEP_KEEP = 1        # Sep 28 night bug check (mirror test: kibuna world 16 rule sheep bought, 17 escaped - the policy feeds ~10 sheep a
                      # day; an escape lowers our count and the rule re-bought = a buy-escape loop; dev worlds with 3+ mid-game flock
                      # drops -$1.3k a seat, others +$0.2k): 1 = the rule counts our flock as the most sheep we have held this game
                      # (per game state, like the straddle's), so a lost sheep is never replaced by a rule buy (0 = off)
SHEEP_HOLD = 1        # Sep 28 night bug check (spy runs): a rule buy tips the POLICY into buying sheep itself (kibuna: the rule added 1
                      # sheep on day 13, the policy then bought 3 a day on days 15-18 = 12, 17 escaped, -$12.3k; toge: rule 6, policy
                      # 11 more, flock 19 vs 11): once the rule has bought in a game, the policy's own sheep buys are capped at the
                      # rule's target (mirror / table) + SHEEP_HOLD - 1 for the rest of the game (0 = off)
_SHEEP = {}           # per (game tag, player): step, most sheep held (SHEEP_KEEP), mirror on (SHEEP_MGAP), rule bought (SHEEP_HOLD)


def _sheep_state(obs):
    key = (obs.get("_seed_tag", 0), int(obs["player"]))
    st = _SHEEP.get(key); step = int(obs["step"])
    if st is None or step < st["step"]:                  # first call or a new game (live_rollout reuses processes)
        if len(_SHEEP) > 64:
            for k in [k for k, v in _SHEEP.items() if v["step"] >= 718]:
                del _SHEEP[k]
        st = _SHEEP[key] = {"step": -1, "max": 0, "mir": False, "rule": False}
    st["step"] = step
    return st


STRAW_UP = 0          # light planning (Sep 28, TOPVSTOP_A: Vadim beat DSM in a mirror game with 7 more strawberry tiles on days
STRAW_MIN = 150       # 11-18): on days 10-14, with strawberries >= STRAW_MIN, up to STRAW_UP of the policy's own WHEAT (while the
                      # shed keeps >= 25) / CARROT PLANT orders become STRAWBERRY while seeds last; seeds ($100) bought ahead (0 = off)
MILK_SHOPS = ("SMOOTHIE_SHOP", "ICE_CREAM_SHOP", "PIZZA_SHOP")
TOMATO_BUYERS = ("PIZZA_SHOP", "FARMERS_MARKET")
SHEEP_CAP = 0         # rule layer: no sheep beyond SHEEP_CAP until the 3rd land is bought (0 = off)
LAND2_DAY = 0         # DSM's Sep 26 plan (il/live/PLANNING_DESIGN.md): the 2nd extra land (SW, $2,000) on this day from hour
LAND2_HOUR = 5        # LAND2_HOUR (DSM Sep 25 and our agent: day 9) (0 = off)
NO_LAND3 = 0          # DSM's Sep 26 plan: never the 3rd extra land (SE, $4,000): BUY_LAND is dropped with 3 quadrants (0 = off;
                      # use with SE_GUARD=0)
EARLY_COWS = 0        # DSM's Sep 26 plan (+3.3 cows on days 2-5, ours 0; we buy them on day 6): a cow floor of EARLY_COW_BASE +
                      # min(day - 1, EARLY_COWS) from day 2 hour EARLY_COW_HOUR, one $400 cow per turn ordered before the
                      # seed buys (the policy builds the pasture and places it from the shed) (0 = off)
EARLY_COW_HOUR = 6
EARLY_COW_BASE = 2    # cows the policy buys on day 0 (both leader and our games); the floor counts on top of them
MELON_WAVE = 0        # DSM's Sep 26 plan: a second melon wave of MELON_WAVE tiles on days MELON_DAY..MELON_DAY + 1 (NE land,
MELON_DAY = 6         # bought ~6.1): the policy's own PLANT orders of MELON_FROM crops become MELON (same tile, same labour)
MELON_FROM = ("WHEAT", "CARROT")   # while seeds last; melon seeds ($80) bought from hour 22 of the day before (0 = off)
MELON_WHEAT_MIN = 0   # Sep 28 evening bug check (m10 lost 6-10 wins): WHEAT plantings become melons only while the shed keeps >= this
                      # much wheat for feed (tomato / strawberry rules use 25); 0 = old behaviour (always)
WASH = 0              # rule layer (Sep 26 re-test of the wash trade Boey / Fourth Quadrant use): sell WASH extra units of
WASH_MODE = "sells"   # WHEAT / FERTILIZER from the shed and buy them back at the end of the same turn's order list;
                      # "sells" = only on turns we already sell that item, "all" = every turn with shed stock (0 = off)
SEED_BUDGET = 0       # rule layer (Sep 26, Codex's JointSeed): the engine drops EVERY PLANT of a crop when more units ask
                      # to plant it than seeds held; keep the first `seeds` requesters (unit order), the rest PASS (0 = off)


GOOSE_EGG = 2         # light planning (Sep 29, live weakness W12 eggs: ST2 lost -$23.0k to My second life with 3 geese vs 10, SOH
                      # -$3.8k to Ebi 5 vs 13, -$14.6k to IsaiahP 3 vs 8, eggs +$10-14k them): in egg-demand towns (BAKERY /
GOOSE_BASE = 5        # BRUNCH_SPOT open) keep the geese (placed + shed + carried) at GOOSE_BASE + GOOSE_EGG per egg shop open
GOOSE_MAX = 13        # (top-7 teams, 3,357 games: 5.1 + 2.3 per egg shop open on day 9; our net's mean is already on that table,
GOOSE_FROM_DAY = 10   # the rule is a floor for its low tail), capped at GOOSE_MAX, from GOOSE_FROM_DAY hour GOOSE_HOUR to the end
GOOSE_LAST_DAY = 13   # of GOOSE_LAST_DAY (a goose yields 2 eggs a day cared + fed, from 4 days after placing, at a
GOOSE_FLOOR = 45      # flat ~$41-53), while eggs >= GOOSE_FLOOR; GOOSE_BATCH a turn, ordered before the seed buys and paid only
GOOSE_RIVAL = 0       # from the cash the policy's own orders of the turn leave (+ SE_RESERVE + LAND_GUARD's reserve), only with
                      # a free tile (empty or empty coop) for each; 1 = the target is at least the rival's placed geese (0 = off)
GOOSE_YARN = 2        # target cut in yarn towns (a YARN_STORE open): the top-7 keep 3.7 / 5.8 / 7.8 geese with 0 / 1 / 2 egg shops
                      # there vs 5.9 / 7.7 / 9.4 in milk towns and ~11 in towns with neither (sheep take the labour and cash)
GOOSE_HOUR = 16       # first hour on GOOSE_FROM_DAY (the net orders its own geese on days 6-11 at hours 4-15 only; a floor that
                      # fires inside that window front-runs them: Mac control test, days 6-11 from hour 0: -$1.0k/game)
GOOSE_GAP = 3         # act only when the flock is at least GOOSE_GAP geese short of the target (small top-ups, mostly escape
GOOSE_BATCH = 6       # replacements, lost money in the Mac control test), then buy up to GOOSE_BATCH a turn toward the target
EGG_SHOPS = ("BAKERY", "BRUNCH_SPOT")
_UNIT_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500, "WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}


def _turn_cash(obs, farm, market, nq):
    """Cash left after this turn's own orders: money + 90% of the sells' value (from shed stock; sells run first)
    - hires (Fibonacci) - land - animals - seeds - product buys."""
    priv = obs.get("private") or {}; shed = priv.get("shed") or {}
    prices = (obs.get("market") or {}).get("prices") or {}
    cash = float(farm.get("money", 0)); hires = int(farm.get("hires_today", 0) or 0)
    for o in market[:10]:
        if not isinstance(o, (list, tuple)) or not o:
            continue
        q = 0
        if len(o) > 2:
            try:
                q = max(0, int(o[2]))
            except Exception:
                q = 0
        if o[0] == "SELL":
            cash += 0.9 * min(q, int(shed.get(o[1], 0))) * float(prices.get(o[1], 0) or 0)
        elif o[0] == "HIRE":
            a_, b_ = 1, 1
            for _ in range(hires):
                a_, b_ = b_, a_ + b_
            cash -= a_; hires += 1
        elif o[0] == "BUY_LAND":
            cash -= (1000, 2000, 4000)[min(2, max(0, nq - 1))]; nq += 1
        elif o[0] in ("BUY_ANIMAL", "BUY_SEED"):
            cash -= _UNIT_COST.get(o[1], 500) * q
        elif o[0] == "BUY_PRODUCT":
            cash -= (float(prices.get(o[1], 100) or 100) + 15.0) * q
    return cash


def _goose_buy(obs, farm, market, day, nq, res, me):
    """Geese the GOOSE_EGG rule orders this turn (0 = none)."""
    es = sum(x in EGG_SHOPS for x in (((obs.get("town") or {}).get("unlocked_shops")) or []))
    if es < 1:
        return 0
    if float(((obs.get("market") or {}).get("prices") or {}).get("EGG", 0) or 0) < GOOSE_FLOOR:
        return 0
    priv = obs.get("private") or {}; tiles = farm["tiles"]
    loose = int((priv.get("shed") or {}).get("GOOSE", 0)) + sum(int((inv or {}).get("GOOSE", 0)) for inv in (priv.get("inventories") or []))
    mine = sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("animal") == "GOOSE") + loose
    target = GOOSE_BASE + GOOSE_EGG * es
    if GOOSE_YARN and "YARN_STORE" in (((obs.get("town") or {}).get("unlocked_shops")) or []):
        target -= GOOSE_YARN
    if GOOSE_RIVAL:
        target = max(target, sum(1 for row in obs["farms"][1 - me]["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "GOOSE"))
    want = min(target, GOOSE_MAX) - mine
    if want <= 0 or want < GOOSE_GAP:
        return 0
    free = sum(1 for row in tiles for t in row if t is None or (isinstance(t, dict) and t.get("kind") == "COOP" and not t.get("animal")))
    cash = _turn_cash(obs, farm, market, nq) - SE_RESERVE - res
    return max(0, min(want, GOOSE_BATCH, free - loose, int(cash // 300)))


LAND_GUARD = 2        # Sep 28 evening bug check (sheep smoke test, IsaiahP world: SHEEP_YARN's day-5 sheep left the policy
                      # short for its day-6 land -> land on day 9, cows never bought, -$19k): the rule layer's OWN purchases
                      # (sheep, cows, tomato / strawberry / melon seeds, early cows) keep the next land's price in cash:
                      # 1 = the first two extra lands ($1,000 / $2,000) until day 9, 2 = also the SE ($4,000) while the SE guard
                      # would still buy it (0 = off). The policy's own purchases are never touched.


STRAW2 = 4            # Sep 28 night (demand study, W9 / W10): strawberries by MARKET PROJECTION instead of today's price. The price
STRAW2_P = 120        # on days 9-14 is at its peak ($190-210) because the day-7 wave has not produced yet - it says nothing about the
STRAW2_FROM = 11       # harvest window (tetsuya: planted at $208, sold at $1-18). _straw_proj replays the market day by day: every
STRAW2_LAST = 13      # strawberry tile on BOTH farms yields on days +10/+12/+14/+16 after planting, each open buyer (SMOOTHIE, ICE_CREAM,
STRAW2_CUT = 0        # BRUNCH, FARMERS) drains 6 a day + the town 1, shops still to unlock count STRAW2_FUT % of their expected
STRAW2_FUT = 100      # buyers. On days STRAW2_FROM..STRAW2_LAST, up to STRAW2 of the policy's own WHEAT (shed keeps >= 25) / CARROT
STRAW2_PER = 185      # PLANT orders a turn become STRAWBERRY while the projected mean price on the new tiles' yield days stays
STRAW2_SHARE = 0      # >= STRAW2_P with them (seeds bought ahead). STRAW2_CUT > 0: when the policy's own strawberry plantings would
                      # push that projected mean below STRAW2_CUT, the excess plantings become WHEAT / CARROT (seeds permitting) and
                      # new strawberry seed buys are capped. STRAW2_PER = units per yield event x100 (leaders sell 7.2-7.5 per planting,
                      # v13_p3 6.6-6.8). STRAW2_SHARE = 1: margin view (extra supply lowers BOTH farms' prices) - add tiles only while
                      # the rival has >= our strawberry tiles, cut only while we have >= the rival's (0 = off; STRAW2_CUT works
                      # alone too). st2str data (1,150 paired worlds): +4 tiles gained +$0.7-2.0k where this projection (on the
                      # recorded game) read >= $120, lost $0.3-0.5k where it read < $80.
COW2 = 2              # Sep 28 night (demand study, W7 cow count): the blanket COW_MILK rule lost in every town class, even in tight milk
COW2_SHOPS = 2        # markets (paired fresh3 diagnostics: v13_p3 already runs 8-14 cows by day 9 in milk towns; one rule cow on day 6
COW2_FLOOR = 170      # displaced the policy's own cows / land -> +-$15k chaos). This rule only closes a herd GAP to the rival in towns
COW2_LAST_DAY = 12    # with strong EARLY milk demand: on days COW2_FROM..COW2_LAST_DAY, with >= COW2_SHOPS milk buyers open (SMOOTHIE /
COW2_GAP = 1          # ICE_CREAM, + PIZZA unless COW2_SI) and milk >= COW2_FLOOR, keep our herd (placed + shed + carried) >= the
COW2_MAX = 14         # rival's placed herd - COW2_GAP (cap COW2_MAX), <= COW2 a turn, ordered before the seed buys; the LAND_GUARD
COW2_SI = 0           # reserve applies (0 = off)
COW2_FROM = 7         # first day (v13_p3 buys its own cow wave on day 6: earlier buys front-run it)
STRAW_BUYERS = ("SMOOTHIE_SHOP", "ICE_CREAM_SHOP", "BRUNCH_SPOT", "FARMERS_MARKET")


def _straw_price(inv):
    """Exact engine STRAWBERRY price at market inventory inv (sqrt below 10,000, linear above, rounded half-even, floor 1)."""
    import math as _m
    if inv < 10000:
        p = 120.0 + (0.70 * 120.0 / _m.sqrt(100.0)) * _m.sqrt(float(10000 - inv))
    else:
        p = 120.0 - (1.60 * 120.0 / 100.0) * float(inv - 10000)
    v = int(round(p))
    return 1 if v < 1 else v


def _straw_proj(obs, today_new):
    """Projected strawberry price for each day from today to day 29, if `today_new` more tiles (beyond the ones on both
    farms) are planted today. Supply: STRAW2_PER / 100 units per yield event (days +10, +12, +14, +16 after planting) of
    every tile on both farms, our shed + carried stock and every tile's held units (sold at once); demand: 6 a day per open
    strawberry buyer + STRAW2_FUT % of the expected buyers among the shops still to unlock (half the kinds buy
    strawberries; unlocks on days 3, 6, ..., 24) + 1 (town center). Returns {day: price}."""
    step = int(obs["step"]); day = step // 24
    per = STRAW2_PER / 100.0
    ev = {}
    stock = 0
    for f in obs["farms"]:
        for row in f["tiles"]:
            for t in row:
                if isinstance(t, dict) and t.get("crop") == "STRAWBERRY":
                    p = int(t.get("planted_day", day)); stock += int(t.get("yield_units", 0) or 0)
                    for k in (10, 12, 14, 16):
                        if p + k > day:
                            ev[p + k] = ev.get(p + k, 0.0) + per
    if today_new:
        for k in (10, 12, 14, 16):
            ev[day + k] = ev.get(day + k, 0.0) + per * today_new
    priv = obs.get("private") or {}
    stock += int((priv.get("shed") or {}).get("STRAWBERRY", 0)) + sum(int((v or {}).get("STRAWBERRY", 0)) for v in (priv.get("inventories") or []))
    buyers = sum(s in STRAW_BUYERS for s in (((obs.get("town") or {}).get("unlocked_shops")) or []))
    fut = STRAW2_FUT / 100.0 * len(STRAW_BUYERS) / 8.0
    x = float(int(((obs.get("market") or {}).get("inventory") or {}).get("STRAWBERRY", 10000)) + stock)
    frac = 1.0 - (step % 24) / 24.0
    out = {}
    for d in range(day, 30):
        f_ = frac if d == day else 1.0
        n = buyers + fut * sum(1 for u in range(3, 25, 3) if day < u <= d)
        x += ev.get(d, 0.0) - (6.0 * n + 1.0) * f_
        out[d] = _straw_price(x)
    return out


def _straw_window(obs, today_new):
    """Mean projected strawberry price on the yield days (+10, +12, +14, +16) of a tile planted today (days <= 29)."""
    day = int(obs["step"]) // 24
    pr = _straw_proj(obs, today_new)
    w = [pr[d] for d in (day + 10, day + 12, day + 14, day + 16) if d in pr]
    return sum(w) / len(w) if w else 0.0


def _land_reserve(obs, nq, day):
    if not LAND_GUARD:
        return 0.0
    extra = nq - 1
    if extra < 2 and day <= 9:
        return float((1000, 2000)[max(0, extra)])
    if (LAND_GUARD >= 2 and extra == 2 and day <= SE_LAST_DAY and SE_GUARD
            and (not SE_SHOPS or sum(x in CROP_SHOPS for x in (((obs.get("town") or {}).get("unlocked_shops")) or [])[:3]) >= SE_SHOPS)):
        return 4000.0
    return 0.0


def _plan_fix(obs, action):
    """Rule layer from the Sep 26 study of the live neural agents' losses (SE missing in 8 of 14 losses, often after a
    big sheep purchase on day 10).  (a) SE guard: with the first two lands bought and the third not, from day 10 hour
    SE_HOUR to the end of day SE_LAST_DAY, when cash >= $4,000 + SE_RESERVE and no BUY_LAND is ordered, order it right
    after the sells and hires (the leaders' canonical position).  (b) Sheep cap: until the third land is bought, a sheep
    purchase may not raise the herd (placed + shed + carried) above SHEEP_CAP."""
    me = int(obs["player"]); farm = obs["farms"][me]; step = int(obs["step"]); day, hour = divmod(step, 24)
    nq = len(farm.get("unlocked_quadrants") or [])
    res = _land_reserve(obs, nq, day)
    sst = _sheep_state(obs) if (SHEEP_KEEP or SHEEP_MGAP or SHEEP_HOLD) and SHEEP_MIRROR + SHEEP_YARN else None   # every turn
    if SEED_BUDGET:
        seeds = (obs.get("private") or {}).get("seeds") or {}
        units = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
        req = {}
        for i, c in enumerate(units):
            if isinstance(c, (list, tuple)) and len(c) > 1 and c[0] == "PLANT":
                req.setdefault(c[1], []).append(i)
        drop = [i for crop, us in req.items() for i in us[max(0, int(seeds.get(crop, 0))):]]
        if drop:
            units = [list(c) if isinstance(c, (list, tuple)) else ["PASS"] for c in units]
            for i in drop:
                units[i] = ["PASS"]
            action = dict(action, farmer=units[0], hands=units[1:])
    melon_need = 0
    if MELON_WAVE and step < 718 and ((day == MELON_DAY - 1 and hour >= 22) or MELON_DAY <= day <= MELON_DAY + 1):
        seeds = (obs.get("private") or {}).get("seeds") or {}
        late = sum(1 for row in farm["tiles"] for t in row
                   if isinstance(t, dict) and t.get("crop") == "MELON" and int(t.get("planted_day", -1)) >= MELON_DAY)
        units = [list(c) if isinstance(c, (list, tuple)) and c else ["PASS"] for c in [action.get("farmer")] + list(action.get("hands") or [])]
        own = sum(1 for c in units if c[0] == "PLANT" and len(c) > 1 and c[1] == "MELON")
        have = int(seeds.get("MELON", 0)) - own; room = MELON_WAVE - late - own
        if day >= MELON_DAY and room > 0 and have > 0:
            conv = 0
            for c in units:
                if conv < min(room, have) and c[0] == "PLANT" and len(c) > 1 and c[1] in MELON_FROM and (
                        c[1] != "WHEAT" or int(((obs.get("private") or {}).get("shed") or {}).get("WHEAT", 0)) >= MELON_WHEAT_MIN):
                    c[1] = "MELON"; conv += 1
            if conv:
                action = dict(action, farmer=units[0], hands=units[1:]); room -= conv; have -= conv
        if not (day == MELON_DAY + 1 and hour > 18):
            melon_need = max(0, room - max(0, have))
    tomato_need = 0
    shops_now = ((obs.get("town") or {}).get("unlocked_shops")) or []
    if TOMATO_SHOPS and max(2, TOMATO_FROM_DAY - 1) <= day <= TOMATO_LAST_DAY and step < 718:
        ts = sum(x in TOMATO_BUYERS for x in shops_now)
        tprice = float(((obs.get("market") or {}).get("prices") or {}).get("TOMATO", 0) or 0)
        if ts >= TOMATO_SHOPS and tprice >= 55:          # the price lags the shops (Planned Economy loss: $60-71 until day 10,
                                                          # $118 by day 20, sold at $195): trigger on demand, skip a floored market
            priv = obs.get("private") or {}; seeds = priv.get("seeds") or {}
            cur = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "TOMATO")
            units = [list(c) if isinstance(c, (list, tuple)) and c else ["PASS"] for c in [action.get("farmer")] + list(action.get("hands") or [])]
            own = sum(1 for c in units if c[0] == "PLANT" and len(c) > 1 and c[1] == "TOMATO")
            have = int(seeds.get("TOMATO", 0)) - own; room = min(TOMATO_PER_SHOP * ts, 32) - cur - own
            wheat_ok = int((priv.get("shed") or {}).get("WHEAT", 0)) >= 25
            if room > 0 and have > 0 and day >= TOMATO_FROM_DAY:   # seeds are bought from the day before
                conv = 0
                for c in units:
                    if conv < min(room, have) and c[0] == "PLANT" and len(c) > 1 and (c[1] == "CARROT" or (c[1] == "WHEAT" and wheat_ok)):
                        c[1] = "TOMATO"; conv += 1
                if conv:
                    action = dict(action, farmer=units[0], hands=units[1:]); room -= conv; have -= conv
            tomato_need = min(10, max(0, room - max(0, have)))
    straw_need = 0
    if STRAW_UP and 10 <= day <= 14 and step < 718:
        sprice = float(((obs.get("market") or {}).get("prices") or {}).get("STRAWBERRY", 0) or 0)
        if sprice >= STRAW_MIN:
            priv = obs.get("private") or {}; seeds = priv.get("seeds") or {}
            late = sum(1 for row in farm["tiles"] for t in row
                       if isinstance(t, dict) and t.get("crop") == "STRAWBERRY" and int(t.get("planted_day", -1)) >= 10)
            units = [list(c) if isinstance(c, (list, tuple)) and c else ["PASS"] for c in [action.get("farmer")] + list(action.get("hands") or [])]
            own = sum(1 for c in units if c[0] == "PLANT" and len(c) > 1 and c[1] == "STRAWBERRY")
            have = int(seeds.get("STRAWBERRY", 0)) - own; room = STRAW_UP - late - own
            wheat_ok = int((priv.get("shed") or {}).get("WHEAT", 0)) >= 25
            if room > 0 and have > 0:
                conv = 0
                for c in units:
                    if conv < min(room, have) and c[0] == "PLANT" and len(c) > 1 and (c[1] == "CARROT" or (c[1] == "WHEAT" and wheat_ok)):
                        c[1] = "STRAWBERRY"; conv += 1
                if conv:
                    action = dict(action, farmer=units[0], hands=units[1:]); room -= conv; have -= conv
            straw_need = max(0, room - max(0, have))
    straw2_need = 0; straw2_cap = -1
    if (STRAW2 or STRAW2_CUT) and STRAW2_FROM <= day <= STRAW2_LAST and step < 718 \
            and sum(s in STRAW_BUYERS for s in (((obs.get("town") or {}).get("unlocked_shops")) or [])) >= 1:
        priv = obs.get("private") or {}; seeds = priv.get("seeds") or {}
        units = [list(c) if isinstance(c, (list, tuple)) and c else ["PASS"] for c in [action.get("farmer")] + list(action.get("hands") or [])]
        own = sum(1 for c in units if c[0] == "PLANT" and len(c) > 1 and c[1] == "STRAWBERRY")
        conv = 0
        mine_t = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")
        riv_t = sum(1 for row in obs["farms"][1 - me]["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")
        if STRAW2_CUT and (not STRAW2_SHARE or mine_t + own >= riv_t):
            cap_t = 0                                      # tiles planted today the market takes at a mean >= STRAW2_CUT
            while cap_t < own + 12 and _straw_window(obs, cap_t + 1) >= STRAW2_CUT:
                cap_t += 1
            if own > cap_t:                                # the policy's excess plantings become WHEAT / CARROT, within
                w_have = int(seeds.get("WHEAT", 0)) - sum(1 for c in units if c[0] == "PLANT" and len(c) > 1 and c[1] == "WHEAT")
                c_have = int(seeds.get("CARROT", 0)) - sum(1 for c in units if c[0] == "PLANT" and len(c) > 1 and c[1] == "CARROT")
                cut = own - cap_t                          # the seeds held (the engine drops EVERY plant of a crop asked
                for c in reversed(units):                  # beyond the seeds held)
                    if cut > 0 and c[0] == "PLANT" and len(c) > 1 and c[1] == "STRAWBERRY":
                        if w_have > 0:
                            c[1] = "WHEAT"; w_have -= 1; cut -= 1; conv += 1
                        elif c_have > 0:
                            c[1] = "CARROT"; c_have -= 1; cut -= 1; conv += 1
                own -= conv
            straw2_cap = max(0, cap_t - own)               # further tiles (held seeds + new buys) the market can take
        if STRAW2 and not conv and (not STRAW2_SHARE or riv_t >= mine_t + own):
            k = 0                                          # extra tiles (beyond the policy's own) the projection allows
            while k < STRAW2 and _straw_window(obs, own + k + 1) >= STRAW2_P:
                k += 1
            have = int(seeds.get("STRAWBERRY", 0)) - own
            wheat_ok = int((priv.get("shed") or {}).get("WHEAT", 0)) >= 25
            if k > 0 and have > 0:
                for c in units:
                    if conv < min(k, have) and c[0] == "PLANT" and len(c) > 1 and (c[1] == "CARROT" or (c[1] == "WHEAT" and wheat_ok)):
                        c[1] = "STRAWBERRY"; conv += 1
                have -= conv
            straw2_need = max(0, k - conv - max(0, have))
        if conv:
            action = dict(action, farmer=units[0], hands=units[1:])
    market = [list(o) for o in (action.get("market") or []) if isinstance(o, (list, tuple))]
    changed = False
    if straw2_cap >= 0:                                    # STRAW2_CUT: strawberry seed buys beyond what the market can take
        seeds_h = int(((obs.get("private") or {}).get("seeds") or {}).get("STRAWBERRY", 0))
        allow = max(0, straw2_cap - max(0, seeds_h - sum(1 for c in [action.get("farmer")] + list(action.get("hands") or [])
                                                          if isinstance(c, (list, tuple)) and len(c) > 1 and c[0] == "PLANT" and c[1] == "STRAWBERRY")))
        out_ = []
        for o in market:
            if len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "STRAWBERRY" and int(o[2]) > allow:
                changed = True
                if allow <= 0:
                    continue
                o = ["BUY_SEED", "STRAWBERRY", allow]
            out_.append(o)
        market = out_
    if straw2_need and len(market) < 10 and float(farm["money"]) >= 100 * straw2_need + 300 + res:
        o = next((o for o in market if len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "STRAWBERRY"), None)
        if o is not None:
            if int(o[2]) < straw2_need:
                o[2] = straw2_need; changed = True
        else:
            market.append(["BUY_SEED", "STRAWBERRY", straw2_need]); changed = True
    if straw_need and len(market) < 10 and float(farm["money"]) >= 100 * straw_need + 300 + res:
        o = next((o for o in market if len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "STRAWBERRY"), None)
        if o is not None:
            if int(o[2]) < straw_need:
                o[2] = straw_need; changed = True
        else:
            market.append(["BUY_SEED", "STRAWBERRY", straw_need]); changed = True
    if tomato_need and len(market) < 10 and float(farm["money"]) >= 50 * tomato_need + 300 + res:
        o = next((o for o in market if len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "TOMATO"), None)
        if o is not None:
            if int(o[2]) < tomato_need:
                o[2] = tomato_need; changed = True
        else:
            market.append(["BUY_SEED", "TOMATO", tomato_need]); changed = True
    if melon_need:
        o = next((o for o in market if len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "MELON"), None)
        need = melon_need - (max(0, int(o[2])) if o is not None else 0)
        if need > 0 and float(farm["money"]) >= 80 * need + (1000 if nq == 1 else 300) + res:
            if o is not None:
                o[2] = int(o[2]) + need; changed = True
            elif len(market) < 10:
                market.append(["BUY_SEED", "MELON", need]); changed = True
    if SHEEP_CAP and nq < 4 and any(len(o) >= 3 and o[0] == "BUY_ANIMAL" and o[1] == "SHEEP" for o in market):
        priv = obs.get("private") or {}
        herd = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        herd += int((priv.get("shed") or {}).get("SHEEP", 0)) + sum(int((inv or {}).get("SHEEP", 0)) for inv in (priv.get("inventories") or []))
        room = max(0, SHEEP_CAP - herd); out = []
        for o in market:
            if len(o) >= 3 and o[0] == "BUY_ANIMAL" and o[1] == "SHEEP":
                q = min(max(0, int(o[2])), room); room -= q
                if q != int(o[2]):
                    changed = True
                if q <= 0:
                    continue
                o = ["BUY_ANIMAL", "SHEEP", q]
            out.append(o)
        market = out
    if WASH and step < 718:
        shed = (obs.get("private") or {}).get("shed") or {}
        for it in ("WHEAT", "FERTILIZER"):
            have = int(shed.get(it, 0))
            o = next((o for o in market if len(o) >= 3 and o[0] == "SELL" and o[1] == it), None)
            q = int(o[2]) if o is not None else 0
            if o is None and WASH_MODE != "all":
                continue
            if q >= 1000 or have - q <= 0 or len(market) >= (10 if o is not None else 9):
                continue
            w = min(int(WASH), have - q)
            if o is None:                     # a new sell goes in front of the other sells (slot 0)
                market.insert(0, ["SELL", it, w])
            else:
                o[2] = q + w
            market.append(["BUY_PRODUCT", it, w])    # buy-back after every sell of the turn
            changed = True
    if NO_LAND3 and nq >= 3 and any(o and o[0] == "BUY_LAND" for o in market):
        market = [o for o in market if not (o and o[0] == "BUY_LAND")]; changed = True
    if (LAND2_DAY and nq == 2 and ((day == LAND2_DAY and hour >= LAND2_HOUR) or LAND2_DAY < day <= LAND2_DAY + 1)
            and not any(o and o[0] == "BUY_LAND" for o in market) and float(farm["money"]) >= 2000 + SE_RESERVE):
        k = max((i + 1 for i, o in enumerate(market) if o and o[0] in ("SELL", "HIRE")), default=0)
        market.insert(k, ["BUY_LAND"]); changed = True
        if len(market) > 10:
            cand = [i for i, o in enumerate(market) if o and o[0] not in ("SELL", "HIRE", "BUY_LAND")] or \
                [i for i, o in enumerate(market) if o and o[0] == "SELL"]
            market.pop(cand[-1])
    if EARLY_COWS and 2 <= day <= 2 + EARLY_COWS and hour >= EARLY_COW_HOUR and step < 718 \
            and not any(o and o[0] == "BUY_ANIMAL" and o[1] == "COW" for o in market):
        priv = obs.get("private") or {}          # a cow floor: EARLY_COW_BASE (the policy's day-0 pair) + min(day - 1,
        herd = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")   # EARLY_COWS)
        herd += int((priv.get("shed") or {}).get("COW", 0)) + sum(int((inv or {}).get("COW", 0)) for inv in (priv.get("inventories") or []))
        pr = (obs.get("market") or {}).get("prices") or {}; shed = priv.get("shed") or {}   # + this turn's sells (they run first)
        cash = float(farm["money"]) + 0.9 * sum(min(max(0, int(o[2])), int(shed.get(o[1], 0))) * float(pr.get(o[1], 0) or 0)
                                                for o in market if len(o) >= 3 and o[0] == "SELL")
        if herd < EARLY_COW_BASE + min(day - 1, EARLY_COWS) and cash >= 400 + res:
            k = next((i for i, o in enumerate(market) if o and o[0] in ("BUY_SEED", "BUY_PRODUCT")), len(market))
            market.insert(k, ["BUY_ANIMAL", "COW", 1]); changed = True     # before the seed buys: DSM plants fewer
            if len(market) > 10:                                           # strawberries on days 2-5 for these cows
                market.pop()
    sh_last = max(SHEEP_LAST_DAY, (SHEEP_MLAST or SHEEP_LAST_DAY) if SHEEP_MIRROR else 0)
    pol_sheep = any(o and o[0] == "BUY_ANIMAL" and o[1] == "SHEEP" for o in market)
    hold = bool(SHEEP_HOLD and sst is not None and sst["rule"] and pol_sheep)
    if (SHEEP_YARN or SHEEP_MIRROR) and 3 <= day and step < 718 and (hold or (day <= sh_last and len(market) < 10 and not pol_sheep)):
        shops_y = (obs.get("town") or {}).get("unlocked_shops") or []
        yarn = sum(x == "YARN_STORE" for x in shops_y)
        if yarn and yarn >= SHEEP_MINY:
            priv = obs.get("private") or {}; riv = obs["farms"][1 - me]
            mine = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
            mine += int((priv.get("shed") or {}).get("SHEEP", 0)) + sum(int((inv or {}).get("SHEEP", 0)) for inv in (priv.get("inventories") or []))
            if sst is not None:                    # (the state's step moves every turn, so a new game resets it)
                sst["max"] = max(sst["max"], mine)
                if SHEEP_KEEP:
                    mine = sst["max"]
            theirs = sum(1 for row in riv["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
            price = float(((obs.get("market") or {}).get("prices") or {}).get("WOOL", 0) or 0)
            floor = SHEEP_FLOOR if SHEEP_FLOOR else (90 if day <= 15 else (125 if day <= 18 else 150))
            table = SHEEP_BASE + SHEEP_YARN * (sum(1 for i, x in enumerate(shops_y) if x == "YARN_STORE" and 3 * i + 3 <= SHEEP_EARLY)
                                               if SHEEP_EARLY else yarn)
            mtarget = max(SHEEP_BASE, theirs + SHEEP_MIRROR - 1) if SHEEP_MIRROR else 0
            if hold:                               # the policy's own sheep buys after a rule buy: capped at the rule's target
                room = max(0, min(max(table if SHEEP_YARN else 0, mtarget), SHEEP_MAX) + SHEEP_HOLD - 1 - mine); out = []
                for o in market:
                    if len(o) >= 3 and o[0] == "BUY_ANIMAL" and o[1] == "SHEEP":
                        q = min(max(0, int(o[2])), room); room -= q
                        if q != int(o[2]):
                            changed = True
                        if q <= 0:
                            continue
                        o = ["BUY_ANIMAL", "SHEEP", q]
                    out.append(o)
                market = out
            else:
                want = 0
                if SHEEP_YARN and day <= SHEEP_LAST_DAY and price >= floor:
                    want = min(table, SHEEP_MAX) - mine
                if SHEEP_MIRROR and day <= (SHEEP_MLAST or SHEEP_LAST_DAY) and price >= (SHEEP_MFLOOR or floor):
                    if SHEEP_MGAP and theirs - mine >= SHEEP_MGAP:
                        sst["mir"] = True
                    if not SHEEP_MGAP or sst["mir"]:
                        want = max(want, min(mtarget, SHEEP_MAX) - mine)
                # no cross-farm cap (TOPVSTOP_A: one sheep less in a flooded wool market costs margin in 9 of 10 cases)
                n = min(want, 2, int((float(farm["money"]) - SE_RESERVE - res) // 500))
                if n > 0:
                    k = next((i for i, o in enumerate(market) if o and o[0] in ("BUY_SEED", "BUY_PRODUCT")), len(market))
                    market.insert(k, ["BUY_ANIMAL", "SHEEP", n]); changed = True
                    if sst is not None:
                        sst["rule"] = True
    if GOOSE_EGG and ((day == GOOSE_FROM_DAY and hour >= GOOSE_HOUR) or GOOSE_FROM_DAY < day <= GOOSE_LAST_DAY) \
            and step < 718 and len(market) < 10 \
            and not any(o and o[0] == "BUY_ANIMAL" and o[1] == "GOOSE" for o in market):
        n = _goose_buy(obs, farm, market, day, nq, res, me)
        if n > 0:
            k = next((i for i, o in enumerate(market) if o and o[0] in ("BUY_SEED", "BUY_PRODUCT")), len(market))
            market.insert(k, ["BUY_ANIMAL", "GOOSE", n]); changed = True
    if COW_MILK and 3 <= day <= COW_LAST_DAY and step < 718 and len(market) < 10 \
            and not any(o and o[0] == "BUY_ANIMAL" and o[1] == "COW" for o in market):
        ms = sum(x in MILK_SHOPS for x in (((obs.get("town") or {}).get("unlocked_shops")) or []))
        mprice = float(((obs.get("market") or {}).get("prices") or {}).get("MILK", 0) or 0)
        if ms and mprice >= (100 if day <= 15 else 130):
            priv = obs.get("private") or {}
            cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            cows += int((priv.get("shed") or {}).get("COW", 0)) + sum(int((inv or {}).get("COW", 0)) for inv in (priv.get("inventories") or []))
            n = min(min(COW_BASE + COW_MILK * ms, COW_MAX) - cows, 2, int((float(farm["money"]) - SE_RESERVE - res) // 400))
            if n > 0:
                k = next((i for i, o in enumerate(market) if o and o[0] in ("BUY_SEED", "BUY_PRODUCT")), len(market))
                market.insert(k, ["BUY_ANIMAL", "COW", n]); changed = True
    if COW2 and COW2_FROM <= day <= COW2_LAST_DAY and step < 718 and len(market) < 10 \
            and not any(o and o[0] == "BUY_ANIMAL" and o[1] == "COW" for o in market):
        ms = sum(x in (MILK_SHOPS[:2] if COW2_SI else MILK_SHOPS) for x in (((obs.get("town") or {}).get("unlocked_shops")) or []))
        mprice = float(((obs.get("market") or {}).get("prices") or {}).get("MILK", 0) or 0)
        if ms >= COW2_SHOPS and mprice >= COW2_FLOOR:
            priv = obs.get("private") or {}; riv = obs["farms"][1 - me]
            cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            cows += int((priv.get("shed") or {}).get("COW", 0)) + sum(int((inv or {}).get("COW", 0)) for inv in (priv.get("inventories") or []))
            rcows = sum(1 for row in riv["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            n = min(min(rcows - COW2_GAP, COW2_MAX) - cows, COW2, int((float(farm["money"]) - SE_RESERVE - res) // 400))
            if n > 0:
                k = next((i for i, o in enumerate(market) if o and o[0] in ("BUY_SEED", "BUY_PRODUCT")), len(market))
                market.insert(k, ["BUY_ANIMAL", "COW", n]); changed = True
    if (SE_GUARD and nq == 3 and step < 718 and ((day == 10 and hour >= SE_HOUR) or 10 < day <= SE_LAST_DAY)
            and not any(o and o[0] == "BUY_LAND" for o in market) and float(farm["money"]) >= 4000 + SE_RESERVE
            and (not SE_SHOPS or sum(x in CROP_SHOPS for x in (((obs.get("town") or {}).get("unlocked_shops")) or [])[:3]) >= SE_SHOPS)):
        k = max((i + 1 for i, o in enumerate(market) if o and o[0] in ("SELL", "HIRE")), default=0)
        market.insert(k, ["BUY_LAND"])
        if len(market) > 10:                 # the engine executes 10 orders: drop the last purchase (else the last sell)
            cand = [i for i, o in enumerate(market) if o and o[0] not in ("SELL", "HIRE", "BUY_LAND")] or \
                [i for i, o in enumerate(market) if o and o[0] == "SELL"]
            market.pop(cand[-1])
        changed = True
    return dict(action, market=market) if changed else action


FEED_GUARD = 0        # rule layer (Sep 28, il/live/PLANNING_FORENSICS.md R2): an animal unfed yesterday and not fed today
FEED_SLACK = 1        # escapes at midnight (engine: consecutive_dry >= 2); we lose ~1.9 such animals/game vs DSM 0.4 with wheat
                      # in the shed - a routing failure. On the day's last turns the nearest unit feeds it (via the shed for
                      # wheat), as late as possible (when the turns it needs + FEED_SLACK reach the turns left) (0 = off)
ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}


def _toward(p, q):
    """One move from p toward q (x first, like _homing)."""
    return ["EAST"] if q[0] > p[0] else ["WEST"] if q[0] < p[0] else ["SOUTH"] if q[1] > p[1] else ["NORTH"]


def _feed_guard(obs, action):
    me = int(obs["player"]); farm = obs["farms"][me]; step = int(obs["step"])
    if step >= 718:
        return action
    left = 24 - step % 24                                  # turns left today, this one included (refresh after hour 23)
    priv = obs.get("private") or {}; invs = priv.get("inventories") or []
    shed_wheat = int((priv.get("shed") or {}).get("WHEAT", 0))
    prices = (obs.get("market") or {}).get("prices") or {}
    risk = []
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            if not isinstance(t, dict) or not t.get("animal") or t.get("fed_today") or int(t.get("consecutive_unfed", 0) or 0) < 1:
                continue
            prod = ANIMAL_PRODUCT.get(t.get("animal"))
            if prod and prices.get(prod) is not None and float(prices[prod]) < 0.3 * BASE[prod]:
                continue                                   # its product has collapsed: a deliberate cull, let it go
            risk.append((x, y))
    if not risk:
        return action
    positions = [farm["farmer"]] + list(farm.get("hands") or [])
    units = [list(c) if isinstance(c, (list, tuple)) and c else ["PASS"] for c in [action.get("farmer")] + list(action.get("hands") or [])]
    n = min(len(units), len(positions))
    pos = [(int(positions[i][0]), int(positions[i][1])) for i in range(n)]
    wheat = [int((invs[i] if i < len(invs) else {}).get("WHEAT", 0) or 0) for i in range(n)]
    free = set(range(n))
    for i in range(n):                                     # the policy already feeds an animal at risk this turn
        if units[i][0] == "FEED" and pos[i] in risk and wheat[i] >= 1:
            risk.remove(pos[i]); free.discard(i); wheat[i] -= 1
    dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    changed, short = False, 0
    plan = []
    for a in risk:
        best = None
        for i in free:
            if wheat[i] >= 1:
                c = (dist(pos[i], a) + 1, i, None)
            elif shed_wheat >= 1:
                s = min(SHED_ACCESS, key=lambda q: dist(pos[i], q) + dist(q, a))
                c = (dist(pos[i], s) + 1 + dist(s, a) + 1, i, s)
            else:
                continue
            if best is None or c[0] < best[0]:
                best = c
        if best is None:
            short += 1
            continue
        plan.append((best[0], a, best[1], best[2]))
    for cost, a, i, via in sorted(plan, reverse=True):     # the tightest animal first
        if i not in free or cost + FEED_SLACK < left:
            continue                                       # still time: the policy may feed it itself
        free.discard(i); changed = True
        if via is None:
            if pos[i] == a:
                units[i] = ["FEED"]; wheat[i] -= 1
            else:
                units[i] = _toward(pos[i], a)
        elif pos[i] == via:
            units[i] = ["PICKUP", "WHEAT", 1]; shed_wheat -= 1
        else:
            units[i] = _toward(pos[i], via)
    market = [list(o) for o in (action.get("market") or []) if isinstance(o, (list, tuple))]
    if short and left <= 12 and not any(o and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT" for o in market) and len(market) < 10 \
            and float(farm["money"]) > 40 * short:
        market.append(["BUY_PRODUCT", "WHEAT", short]); changed = True     # no wheat anywhere: have some in the shed next turn
    if not changed:
        return action
    return dict(action, farmer=units[0], hands=units[1:], market=market)


COW_CARE = 2          # rule layer (Sep 29 milk study): YIELD-DAY BONUS GUARD. The engine pays an animal's care bonus
                      # (pending_care_bonus, +1 per cared-AND-fed day) at its yield event (end of day: cow every 2nd day from
                      # placed + 7, sheep every 3rd from placed + 5, goose daily from placed + 3) only if the animal was FED that
                      # day - an unfed yield day loses the whole bonus. Our units visit such animals on fertilizer rounds
                      # without wheat (priced misses: visited 89-96%, visitor carried wheat 5-11%): we lose ~4.9 milk / 6.9
                      # wool / 13.5 eggs a game vs the leaders' 0.5 / 3.6 / 7.6. From hour CARE_HOUR, each such animal whose
                      # bonus is worth >= CARE_MIN (pending x product price) gets the cheapest unit (idle / fertilizer
                      # collection / walking first, else the nearest; sticky per target) to fetch wheat and feed it, then
                      # care for it. Kinds served: 1 = cows, 2 = + sheep, 3 = + geese (0 = off)
CARE_MIN = 100        # least bonus value ($) worth a trip
CARE_HOUR = 15        # first hour of the day the guard acts (by then ~97% of our yield-day feeds are done)
CARE_PEN = 4          # extra turns tolerated to spare a unit busy with anything but PASS / moves / COLLECT_FERTILIZER / DIG
CARE_ESC = 0          # 1 = also keep served animals that would escape tonight (unfed yesterday and today) while their product
                      # sells at >= CARE_ESC_MIN % of its base price and >= CARE_ESC_DAYS game days remain (value = days left x
                      # price, so these go first)
CARE_ESC_MIN = 40
CARE_ESC_DAYS = 4
_CARE_KIND = {"COW": 1, "SHEEP": 2, "GOOSE": 3}
_CARE_CYCLE = {"COW": (8, 2, 6), "SHEEP": (6, 3, 6), "GOOSE": (4, 1, 4)}   # engine: first_yield_day, interval, max_held
_CARE_CHEAP = ("PASS", "COLLECT_FERTILIZER", "NORTH", "SOUTH", "EAST", "WEST", "DIG")
_CARE = {}            # per (game tag, player): last step, {target tile: unit} (sticky assignments within a day)


def _cow_care(obs, action):
    me = int(obs["player"]); farm = obs["farms"][me]; step = int(obs["step"])
    day, hour = divmod(step, 24)
    if day >= 29 or hour < CARE_HOUR:                      # no rollover on the last day (the game ends at step 718)
        return action
    key = (obs.get("_seed_tag", 0), me)
    st = _CARE.get(key)
    if st is None or st["step"] != step - 1 or hour == CARE_HOUR:
        if len(_CARE) > 64:
            _CARE.clear()
        st = _CARE[key] = {"step": step, "assign": {}}
    st["step"] = step
    left = 24 - hour                                       # turns left today, this one included
    priv = obs.get("private") or {}; invs = priv.get("inventories") or []
    prices = (obs.get("market") or {}).get("prices") or {}
    need, fed_nocare = [], set()
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            if not isinstance(t, dict) or not t.get("animal") or _CARE_KIND.get(t["animal"], 9) > COW_CARE:
                continue
            a = t["animal"]; price = float(prices.get(ANIMAL_PRODUCT[a], 0) or 0)
            if t.get("fed_today"):
                if not t.get("cared_today") and price >= 0.3 * BASE[ANIMAL_PRODUCT[a]]:
                    fed_nocare.add((x, y))                 # +1 product at its next yield event for one turn
                continue
            fy, iv, mh = _CARE_CYCLE[a]
            since = day + 1 - int(t.get("placed_day", day)) - fy
            pend = int(t.get("pending_care_bonus", 0) or 0)
            val = pend * price if (since >= 0 and since % iv == 0) else 0.0
            if (CARE_ESC and int(t.get("consecutive_unfed", 0) or 0) >= 1 and price >= CARE_ESC_MIN * BASE[ANIMAL_PRODUCT[a]] / 100.0
                    and 29 - day >= CARE_ESC_DAYS):
                val = max(val, price * (29 - day))
            if val >= CARE_MIN:
                need.append((val, (x, y)))
    if not need and not fed_nocare:
        st["assign"] = {}
        return action
    positions = [farm["farmer"]] + list(farm.get("hands") or [])
    units = [list(c) if isinstance(c, (list, tuple)) and c else ["PASS"] for c in [action.get("farmer")] + list(action.get("hands") or [])]
    n = min(len(units), len(positions))
    pos = [(int(positions[i][0]), int(positions[i][1])) for i in range(n)]
    wheat = [int((invs[i] if i < len(invs) else {}).get("WHEAT", 0) or 0) for i in range(n)]
    shed_wheat = int((priv.get("shed") or {}).get("WHEAT", 0))
    for i in range(n):                                     # the policy's own wheat pickups go first (unit order)
        if units[i][0] == "PICKUP" and len(units[i]) > 1 and units[i][1] == "WHEAT":
            try:
                shed_wheat -= max(0, int(units[i][2]) if len(units[i]) > 2 else 1)
            except Exception:
                shed_wheat -= 1
    shed_wheat = max(0, shed_wheat)
    free = set(range(n))
    needy = {a for _, a in need}
    for i in range(n):                                     # the policy already feeds a needy animal this turn
        if units[i][0] == "FEED" and pos[i] in needy and wheat[i] >= 1:
            needy.discard(pos[i]); free.discard(i); wheat[i] -= 1
    need = sorted(((v, a) for v, a in need if a in needy), reverse=True)
    dist = lambda p, q: abs(p[0] - q[0]) + abs(p[1] - q[1])
    changed = False
    fed_nocare -= {pos[i] for i in range(n) if units[i][0] == "CARE"}
    for i in sorted(free):                                 # a unit standing on a fed, uncared, valuable animal cares for it
        if pos[i] in fed_nocare and units[i][0] in _CARE_CHEAP:
            units[i] = ["CARE"]; free.discard(i); fed_nocare.discard(pos[i]); changed = True
    assign = {}
    short = 0
    for k, (val, a) in enumerate(need):
        best = None; dry = True
        for i in free:
            if wheat[i] >= 1:
                c, via = dist(pos[i], a) + 1, None
            elif shed_wheat >= 1:
                s_ = min(SHED_ACCESS, key=lambda q: dist(pos[i], q) + dist(q, a))
                c, via = dist(pos[i], s_) + 1 + dist(s_, a) + 1, s_
            else:
                continue
            dry = False
            if c > left:
                continue                                   # cannot reach it before midnight
            pen = 0 if units[i][0] in _CARE_CHEAP else CARE_PEN
            if st["assign"].get(a) == i:
                pen -= 100                                 # keep the unit already sent there (no thrashing)
            kk = (c + pen, c, i)
            if best is None or kk < best[0]:
                best = (kk, i, via)
        if best is None:
            short += dry                                   # no wheat carried and none in the shed
            continue
        (_, c, _), i, via = best
        free.discard(i); assign[a] = i
        if units[i][0] not in _CARE_CHEAP and left - c >= 2:
            continue                                       # time to spare: the unit does the policy's own work first
        changed = True
        if via is None:
            if pos[i] == a:
                units[i] = ["FEED"]; wheat[i] -= 1
            else:
                units[i] = _toward(pos[i], a)
        elif pos[i] == via:
            q = max(1, min(shed_wheat, 4, sum(1 for _, b in need[k:] if b in needy)))
            units[i] = ["PICKUP", "WHEAT", q]; shed_wheat -= q
        else:
            units[i] = _toward(pos[i], via)
    st["assign"] = assign
    market = [list(o) for o in (action.get("market") or []) if isinstance(o, (list, tuple))]
    if short and left >= 3 and not any(o and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT" for o in market) and len(market) < 10 \
            and float(farm["money"]) > 40 * short + 300:
        market.append(["BUY_PRODUCT", "WHEAT", short]); changed = True     # no wheat anywhere: in the shed next turn
    if not changed:
        return action
    return dict(action, farmer=units[0], hands=units[1:], market=market)


CARE_DEDUP = 0        # rule layer (Sep 29 milk study): the policy decodes each unit alone, so two units on one tile often issue
                      # the same op in the same step and the engine executes only the first (our live games: ~14 CARE,
                      # 7 FEED, 7 COLLECT_FERTILIZER, 2.5 HARVEST duplicates on animals and ~30 WATER / 13 HARVEST on plants a
                      # game; failed unit ops ~85 a game vs the leaders' ~20). A duplicate (a certain no-op) becomes the most
                      # useful op still open on its tile: animal FEED (wheat carried) > CARE > HARVEST > COLLECT_FERTILIZER;
                      # 2 = also plants (HARVEST of a ripe tomato / strawberry, or of a grain / carrot / melon at its last
                      # growth day) (0 = off)


def _care_dedup(obs, action):
    me = int(obs["player"]); farm = obs["farms"][me]; step = int(obs["step"]); day = step // 24
    positions = [farm["farmer"]] + list(farm.get("hands") or [])
    units = [list(c) if isinstance(c, (list, tuple)) and c else ["PASS"] for c in [action.get("farmer")] + list(action.get("hands") or [])]
    n = min(len(units), len(positions))
    invs = (obs.get("private") or {}).get("inventories") or []
    tile_ops = ("CARE", "FEED", "HARVEST", "COLLECT_FERTILIZER", "WATER", "FERTILIZE", "DIG")
    by = {}
    for i in range(n):
        by.setdefault((int(positions[i][0]), int(positions[i][1])), []).append(i)
    changed = False
    for (x, y), us in by.items():
        if len(us) < 2:
            continue
        t = farm["tiles"][y][x]
        if not isinstance(t, dict):
            continue
        animal = bool(t.get("animal"))
        if not animal and (CARE_DEDUP < 2 or t.get("kind") != "PLANT"):
            continue
        taken, dup = set(), []
        for i in us:                                       # engine order: the first unit issuing an op executes it
            op = units[i][0]
            if op in tile_ops:
                if op in taken:
                    dup.append(i)
                else:
                    taken.add(op)
        for i in dup:
            inv = (invs[i] if i < len(invs) else None) or {}
            alt = None
            if animal:
                if not t.get("fed_today") and "FEED" not in taken and int(inv.get("WHEAT", 0) or 0) >= 1:
                    alt = "FEED"
                elif not t.get("cared_today") and "CARE" not in taken:
                    alt = "CARE"
                elif int(t.get("yield_units", 0) or 0) > 0 and "HARVEST" not in taken:
                    alt = "HARVEST"
                elif t.get("fertilizer_available") and "COLLECT_FERTILIZER" not in taken:
                    alt = "COLLECT_FERTILIZER"
            else:
                ci = CROP_INFO.get(t.get("crop"))
                age = day - int(t.get("planted_day", day))
                if (ci and int(t.get("yield_units", 0) or 0) > 0 and "HARVEST" not in taken
                        and ((ci[3] and age >= ci[0]) or (not ci[3] and age >= ci[1]))):
                    alt = "HARVEST"
            if alt:
                units[i] = [alt]; taken.add(alt); changed = True
    if not changed:
        return action
    return dict(action, farmer=units[0], hands=units[1:])

def midnight_fix(obs, action):
    """Rule layer (ported from the scripted track's MIDNIGHT layer): at the end of each day every unit's inventory drops
    into the shed (capacity 100, animals included) and the overflow is destroyed.  On a day's last turn (units act, then
    the market, then the drop) project shed + carried after this turn's own unit actions and orders and sell the excess
    now from the shed, cheapest first.  Never on the game's last turn; any error leaves the action untouched.
    The Sep 26 plan rules (_plan_fix: SE guard, sheep cap) run first when enabled."""
    if (SE_GUARD or SHEEP_CAP or SEED_BUDGET or WASH or LAND2_DAY or NO_LAND3 or EARLY_COWS or MELON_WAVE or SHEEP_YARN or SHEEP_MIRROR
            or TOMATO_SHOPS or COW_MILK or STRAW_UP or GOOSE_EGG or STRAW2 or STRAW2_CUT or COW2) and isinstance(action, dict):
        try:
            action = _plan_fix(obs, action)
        except Exception:
            pass
    if FEED_GUARD and isinstance(action, dict):
        try:
            action = _feed_guard(obs, action)
        except Exception:
            pass
    if COW_CARE and isinstance(action, dict):
        try:
            action = _cow_care(obs, action)
        except Exception:
            pass
    if CARE_DEDUP and isinstance(action, dict):
        try:
            action = _care_dedup(obs, action)
        except Exception:
            pass
    try:
        step = int(obs["step"])
        if step >= 718 and FINAL_FIX and isinstance(action, dict):       # the engine's last turn is step 718
            return _final_fix(obs, action)
        if 690 <= step < 718 and HOMING and isinstance(action, dict):
            action = _homing(obs, action, 718 - step)
        if step % 24 != 23 or step >= 719 or not isinstance(action, dict):
            return action
        me = int(obs["player"]); farm = obs["farms"][me]; priv = obs["private"]
        shed = {k: int(v) for k, v in (priv.get("shed") or {}).items()}
        invs = priv.get("inventories") or []
        carried = sum(int(v) for inv in invs for v in inv.values())
        positions = [farm["farmer"]] + list(farm.get("hands") or [])
        units = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
        gain = 0; stock = dict(shed)
        for i in range(min(len(units), len(positions))):
            c = units[i]
            if not c:
                continue
            x, y = int(positions[i][0]), int(positions[i][1])
            tl = farm["tiles"][y][x]
            if c[0] == "HARVEST" and isinstance(tl, dict):
                gain += max(0, int(tl.get("yield_units", 0) or 0))
            elif c[0] == "COLLECT_FERTILIZER" and isinstance(tl, dict) and tl.get("animal") and tl.get("fertilizer_available"):
                gain += 1
            elif c[0] == "PICKUP" and len(c) > 2:
                stock[c[1]] = max(0, stock.get(c[1], 0) - int(c[2]))
            elif c[0] == "DROP" and i < len(invs):
                for k, v in invs[i].items():
                    stock[k] = stock.get(k, 0) + int(v)
        market = [list(o) for o in (action.get("market") or [])]
        sold, bought = {}, 0
        for o in market:
            if len(o) >= 3 and o[0] == "SELL":
                q = min(max(0, int(o[2])), max(0, stock.get(o[1], 0) - sold.get(o[1], 0)))
                sold[o[1]] = sold.get(o[1], 0) + q
            elif len(o) >= 3 and o[0] in ("BUY_PRODUCT", "BUY_ANIMAL"):
                bought += max(0, int(o[2]))
        over = sum(shed.values()) + carried + gain + bought - sum(sold.values()) - SHED_CAP
        if over <= 0:
            return action
        prices = obs["market"]["prices"]
        left = {k: stock.get(k, 0) - sold.get(k, 0) for k in prices}
        added = 0
        for item in sorted((k for k in prices if left.get(k, 0) > 0), key=lambda k: prices.get(k, 0)):   # cheapest first
                                                                    # (relative-to-base ordering tested: -$91/game, n.s.)
            if over <= 0:
                break
            q = min(over, left[item])
            hit = next((o for o in market if len(o) >= 3 and o[0] == "SELL" and o[1] == item), None)
            if hit is not None:
                hit[2] = int(hit[2]) + q
            elif len(market) < 10:
                market.insert(max((k + 1 for k, o in enumerate(market) if len(o) >= 3 and o[0] == "SELL"), default=0), ["SELL", item, q])
            else:
                continue
            over -= q; added += q
        return dict(action, market=market) if added else action
    except Exception:
        return action


STRADDLE = 2          # rule layer (Sep 28 turnover study, the top players' wheat "drain straddle"): at steps with step % 4 == 0
                      # every open wheat shop takes 1 wheat from the market right after the market phase (the town center 1
                      # more at hour 0), so wheat bought before the drain and sold back the next turn earns ~ drained units x
                      # the buy's price impact (a round trip with no drain in between nets exactly $0: buys are quoted at the
                      # post-buy inventory). 1 = buy in slot 0 at the drain step, sell the same units back in slot 0 the next
                      # turn (+$0.8k/game on our 467 live games, +$4k vs straddling rivals); 2 = both legs after our own orders
                      # unless the rival is seen straddling (slot 0 then). Applied last (straddle_fix); the policy and the other
                      # rules see the farm as if no straddle were open (straddle_obs) (0 = off)
STRADDLE_DAY = 12     # first day (all land bought: the cash comes back first thing next turn)
STRADDLE_CAP = 100    # wheat per straddle, within shed room (our own purchases and next turn's drops keep their space)
STRADDLE_MIN_D = 2    # wheat drained at the step (open wheat shops + 1 at hour 0) needed to straddle
STRADDLE_RESERVE = 300   # cash kept after our own purchases of the turn
STRADDLE_PREBUY = 1   # Sep 29 (Codex's wheat-trading guard): skip our drain buy when the public wheat inventory fell since the last
STRADDLE_PREBUY_MIN = 25   # step by >= STRADDLE_PREBUY_MIN more than the drain + our own buy explain (the rival bought ahead) (0 = off)


def _prebuy_skip(obs, st, inv):
    pv = st.get("prev"); step = int(obs["step"])
    if step % 4 != 0 or not isinstance(pv, (tuple, list)) or len(pv) != 5 or int(pv[0]) != step - 1:
        return False
    return int(pv[1]) - int(inv) - int(pv[4]) - int(pv[2]) >= STRADDLE_PREBUY_MIN
_STRAD_SHOPS = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PIZZA_SHOP")   # the shops that take wheat
_STRAD = {}           # per (game tag, player): step, units held, rival-straddle detection (STRADDLE = 2)


def _strad_price(inv):
    """Exact engine wheat price at market inventory inv (sqrt below 10,000, log above, rounded half-even)."""
    import math as _m
    if inv < 10000:
        p = 25.0 + (0.80 * 25.0 / _m.sqrt(400.0)) * _m.sqrt(float(10000 - inv))
    else:
        p = 25.0 - (0.20 * 25.0 / _m.log(1.0 + 400.0)) * _m.log(1.0 + float(inv - 10000))
    v = int(round(p))
    return 1 if v < 1 else v


def _strad_state(obs):
    key = (obs.get("_seed_tag", 0), int(obs["player"]))
    st = _STRAD.get(key); step = int(obs["step"])
    if st is None or step < st["step"]:                  # first call or a new game (live_rollout reuses processes)
        if len(_STRAD) > 64:
            for k in [k for k, v in _STRAD.items() if v["step"] >= 718]:
                del _STRAD[k]
        st = _STRAD[key] = {"step": -1, "held": 0}
    return st


def straddle_obs(obs):
    """The observation for the policy and the rule layer while a straddle is open (the turn after its buy): our shed
    without the straddle's wheat, cash and the wheat market as right after this turn's sell-back. Unchanged otherwise."""
    if not STRADDLE:
        return obs
    try:
        held = int(_strad_state(obs)["held"])
        if held <= 0:
            return obs
        me = int(obs["player"]); o = dict(obs)
        priv = dict(o["private"]); shed = dict(priv.get("shed") or {})
        shed["WHEAT"] = max(0, int(shed.get("WHEAT", 0)) - held); priv["shed"] = shed; o["private"] = priv
        mkt = dict(o["market"]); inv = dict(mkt["inventory"]); pr = dict(mkt["prices"])
        w = int(inv.get("WHEAT", 10000))
        farms = list(o["farms"]); fm = dict(farms[me])
        fm["money"] = float(fm.get("money", 0)) + sum(_strad_price(w + k) for k in range(held))
        farms[me] = fm; o["farms"] = farms
        inv["WHEAT"] = w + held; pr["WHEAT"] = _strad_price(w + held); mkt["inventory"] = inv; mkt["prices"] = pr; o["market"] = mkt
        return o
    except Exception:
        return obs


def straddle_fix(obs, action):
    """Very last transformation of our action (after midnight_fix), on the REAL observation: opens the straddle at a
    drain step (BUY_PRODUCT WHEAT) and closes it the next turn (SELL WHEAT), in slot 0 (STRADDLE = 2: after our own
    orders unless the rival straddles). Never drops or reorders an order of ours: with 10 orders queued the buy is
    skipped and the sell-back waits a turn. Any error leaves the action untouched."""
    if not STRADDLE or not isinstance(action, dict):
        return action
    try:
        return _straddle(obs, action)
    except Exception:
        return action


def _strad_flows(obs, action, me):
    """(units our units move into the shed during this turn's unit phase, most units they could DROP next turn)."""
    farm = obs["farms"][me]; priv = obs.get("private") or {}; shed = priv.get("shed") or {}
    invs = priv.get("inventories") or []
    pos = [farm["farmer"]] + list(farm.get("hands") or [])
    cmds = [action.get("farmer")] + list(action.get("hands") or [])
    into, nxt = 0, 0
    for i, p in enumerate(pos):
        x, y = int(p[0]), int(p[1])
        c = cmds[i] if i < len(cmds) and isinstance(cmds[i], (list, tuple)) and cmds[i] else ["PASS"]
        iv = (invs[i] if i < len(invs) else None) or {}
        load = sum(int(v) for v in iv.values())
        q = 1
        if len(c) > 2:
            try:
                q = max(0, int(c[2]))
            except Exception:
                q = 1
        if (x, y) in SHED_ACCESS:
            if c[0] == "DROP":
                into += load; load = 0
            elif c[0] == "PLACE" and len(c) > 1 and c[1] in PRODUCTS:
                n = min(q, int(iv.get(c[1], 0))); into += n; load -= n
            elif c[0] == "PICKUP" and len(c) > 1:
                n = min(q, int(shed.get(c[1], 0))); into -= n; load += n
        if c[0] == "HARVEST":
            load += 6                                    # at most one crop's yield
        elif c[0] == "COLLECT_FERTILIZER":
            load += 1
        dx, dy = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}.get(c[0], (0, 0))
        if (min(B - 1, max(0, x + dx)), min(B - 1, max(0, y + dy))) in SHED_ACCESS:
            nxt += load
    return max(0, into), nxt


def _straddle(obs, action):
    st = _strad_state(obs)
    me = int(obs["player"]); step = int(obs["step"]); day, hour = divmod(step, 24)
    farm = obs["farms"][me]; priv = obs.get("private") or {}; shed = priv.get("shed") or {}
    shops = ((obs.get("town") or {}).get("unlocked_shops")) or []
    inv = int(((obs.get("market") or {}).get("inventory") or {}).get("WHEAT", 10000))
    drain = (sum(1 for s in shops if s in _STRAD_SHOPS) if step % 4 == 0 else 0) + (1 if step % 24 == 0 else 0)
    pv = st.get("prev")
    if STRADDLE == 2 and pv and pv[0] == step - 1:       # the rival's net wheat trades of the last turn (bounds from ours)
        delta = inv - pv[1]
        if pv[0] % 4 == 0 and -delta - pv[4] - pv[2] >= 25:
            st["rb"] = pv[0]                             # it bought >= 25 wheat right before a drain ...
        elif pv[0] == st.get("rb", -9) + 1 and delta + pv[4] - pv[3] >= 25:
            st["rs"] = st.get("rs", 0) + 1               # ... and sold them back the next turn
    first = STRADDLE == 1 or st.get("rs", 0) >= 2
    mk = list(action.get("market") or [])
    new = None
    held = int(st["held"])
    if held > 0:                                         # close: sell the straddle's wheat back
        if len(mk) < 10:
            picks = 0
            for c in [action.get("farmer")] + list(action.get("hands") or []):
                if isinstance(c, (list, tuple)) and len(c) > 1 and c[0] == "PICKUP" and c[1] == "WHEAT":
                    try:
                        picks += max(0, int(c[2])) if len(c) > 2 else 1
                    except Exception:
                        picks += 1
            n = min(held, max(0, int(shed.get("WHEAT", 0)) - picks))
            if n > 0:
                new = ([["SELL", "WHEAT", n]] + mk) if first else (mk + [["SELL", "WHEAT", n]])
            st["held"] = 0
    elif (step % 4 == 0 and day >= STRADDLE_DAY and step < 716 and len(mk) < 10 and drain >= STRADDLE_MIN_D):
        into, nxt = _strad_flows(obs, action, me)        # open: buy right before the drain
        own_in, spend = 0, 0.0
        prices = (obs.get("market") or {}).get("prices") or {}
        hires = int(farm.get("hires_today", 0) or 0)
        nq = len(farm.get("unlocked_quadrants") or [])
        for o in mk[:10]:
            if not isinstance(o, (list, tuple)) or not o:
                continue
            q = 0
            if len(o) > 2:
                try:
                    q = max(0, int(o[2]))
                except Exception:
                    q = 0
            if o[0] == "HIRE":
                a_, b_ = 1, 1
                for _ in range(hires):
                    a_, b_ = b_, a_ + b_
                spend += a_; hires += 1
            elif o[0] == "BUY_LAND":
                spend += (1000, 2000, 4000)[min(2, max(0, nq - 1))]; nq += 1
            elif o[0] == "BUY_ANIMAL":
                spend += {"GOOSE": 300, "COW": 400, "SHEEP": 500}.get(o[1], 500) * q; own_in += q
            elif o[0] == "BUY_SEED":
                spend += {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(o[1], 100) * q
            elif o[0] == "BUY_PRODUCT":
                spend += (float(prices.get(o[1], 100) or 100) + 15.0) * q; own_in += q
        room = SHED_CAP - sum(int(v) for v in shed.values()) - into - own_in - nxt - 5
        n = min(STRADDLE_CAP, room)
        cash = float(farm.get("money", 0)) - spend - STRADDLE_RESERVE
        while n >= 10 and sum(_strad_price(inv - 2 * k + 1) for k in range(1, n + 1)) > cash:   # worst case: the
            n -= 5                                                                                 # rival buys too
        if n >= 10 and not (STRADDLE_PREBUY and _prebuy_skip(obs, st, inv)):
            new = ([["BUY_PRODUCT", "WHEAT", n]] + mk) if first else (mk + [["BUY_PRODUCT", "WHEAT", n]])
            st["held"] = n
    out = dict(action, market=new) if new is not None else action
    if STRADDLE == 2:
        bq = sq = 0
        for o in (out.get("market") or [])[:10]:
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] == "WHEAT":
                try:
                    q = max(0, int(o[2]))
                except Exception:
                    continue
                if o[0] == "BUY_PRODUCT":
                    bq += q
                elif o[0] == "SELL":
                    sq += q
        have = int(shed.get("WHEAT", 0)) + sum(int((v or {}).get("WHEAT", 0)) for v in (priv.get("inventories") or []))
        st["prev"] = (step, inv, bq, min(sq, have + bq), drain)
    st["step"] = step
    return out
