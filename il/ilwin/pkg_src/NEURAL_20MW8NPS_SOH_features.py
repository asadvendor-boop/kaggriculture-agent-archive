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


SE_GUARD = 0          # rule layer (Sep 26 live-loss study): buy the 3rd land (SE) from day 10 hour SE_HOUR (0 = off)
SE_HOUR = 12
SE_LAST_DAY = 13
SE_RESERVE = 300
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
WASH = 0              # rule layer (Sep 26 re-test of the wash trade Boey / Fourth Quadrant use): sell WASH extra units of
WASH_MODE = "sells"   # WHEAT / FERTILIZER from the shed and buy them back at the end of the same turn's order list;
                      # "sells" = only on turns we already sell that item, "all" = every turn with shed stock (0 = off)
SEED_BUDGET = 0       # rule layer (Sep 26, Codex's JointSeed): the engine drops EVERY PLANT of a crop when more units ask
                      # to plant it than seeds held; keep the first `seeds` requesters (unit order), the rest PASS (0 = off)


def _plan_fix(obs, action):
    """Rule layer from the Sep 26 study of the live neural agents' losses (SE missing in 8 of 14 losses, often after a
    big sheep purchase on day 10).  (a) SE guard: with the first two lands bought and the third not, from day 10 hour
    SE_HOUR to the end of day SE_LAST_DAY, when cash >= $4,000 + SE_RESERVE and no BUY_LAND is ordered, order it right
    after the sells and hires (the leaders' canonical position).  (b) Sheep cap: until the third land is bought, a sheep
    purchase may not raise the herd (placed + shed + carried) above SHEEP_CAP."""
    me = int(obs["player"]); farm = obs["farms"][me]; step = int(obs["step"]); day, hour = divmod(step, 24)
    nq = len(farm.get("unlocked_quadrants") or [])
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
                if conv < min(room, have) and c[0] == "PLANT" and len(c) > 1 and c[1] in MELON_FROM:
                    c[1] = "MELON"; conv += 1
            if conv:
                action = dict(action, farmer=units[0], hands=units[1:]); room -= conv; have -= conv
        if not (day == MELON_DAY + 1 and hour > 18):
            melon_need = max(0, room - max(0, have))
    market = [list(o) for o in (action.get("market") or []) if isinstance(o, (list, tuple))]
    changed = False
    if melon_need:
        o = next((o for o in market if len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "MELON"), None)
        need = melon_need - (max(0, int(o[2])) if o is not None else 0)
        if need > 0 and float(farm["money"]) >= 80 * need + (1000 if nq == 1 else 300):
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
        if herd < EARLY_COW_BASE + min(day - 1, EARLY_COWS) and cash >= 400:
            k = next((i for i, o in enumerate(market) if o and o[0] in ("BUY_SEED", "BUY_PRODUCT")), len(market))
            market.insert(k, ["BUY_ANIMAL", "COW", 1]); changed = True     # before the seed buys: DSM plants fewer
            if len(market) > 10:                                           # strawberries on days 2-5 for these cows
                market.pop()
    if (SE_GUARD and nq == 3 and step < 718 and ((day == 10 and hour >= SE_HOUR) or 10 < day <= SE_LAST_DAY)
            and not any(o and o[0] == "BUY_LAND" for o in market) and float(farm["money"]) >= 4000 + SE_RESERVE):
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


def midnight_fix(obs, action):
    """Rule layer (ported from the scripted track's MIDNIGHT layer): at the end of each day every unit's inventory drops
    into the shed (capacity 100, animals included) and the overflow is destroyed.  On a day's last turn (units act, then
    the market, then the drop) project shed + carried after this turn's own unit actions and orders and sell the excess
    now from the shed, cheapest first.  Never on the game's last turn; any error leaves the action untouched.
    The Sep 26 plan rules (_plan_fix: SE guard, sheep cap) run first when enabled."""
    if (SE_GUARD or SHEEP_CAP or SEED_BUDGET or WASH or LAND2_DAY or NO_LAND3 or EARLY_COWS or MELON_WAVE) and isinstance(action, dict):
        try:
            action = _plan_fix(obs, action)
        except Exception:
            pass
    if FEED_GUARD and isinstance(action, dict):
        try:
            action = _feed_guard(obs, action)
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
