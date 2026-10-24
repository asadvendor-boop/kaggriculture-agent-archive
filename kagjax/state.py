"""Batched state / action layout of the JAX Kaggriculture engine, plus host-side converters.

See spec.md for the full field documentation. All arrays are int32 unless noted; the leading dim is the batch B.
Converters here are numpy / pure Python (no JAX needed):
  encode_action(dict)            -> per-player action arrays, parsed exactly like kagsim_trace.cpp conv_action
  encode_actions(list_of_dicts)  -> Actions (numpy, batch = len(list))
  obs_to_game(obs0, obs1)        -> single-game GameState (numpy, no batch dim) from two C++ observe() dicts
  stack_games([...])             -> batched GameState
  game_to_obs(g, b, seat, t)     -> the observe(seat) dict the C++ engine would return for batch element b
"""
from typing import NamedTuple, Any
import numpy as np

try:
    from . import consts as C
except ImportError:
    import consts as C

Array = Any


class GameState(NamedTuple):
    money: Array     # int32[B,2]            bank of each player (the C++ keeps a double; it is always integral)
    tiles: Array     # int32[B,2,100,NF]     tile fields, tile index = y*10+x, columns F_* (consts.py)
    ux: Array        # int32[B,2,U]          unit x (0..9); unit 0 = main farmer, 1.. = hands in hire order
    uy: Array        # int32[B,2,U]          unit y (0..9); only units < n_units are meaningful
    n_units: Array   # int32[B,2]            1 + hands hired today (reset to 1 at end of day)
    n_quads: Array   # int32[B,2]            unlocked quadrants (1..4); owned = QUAD_NAMES[:n_quads]
    hires: Array     # int32[B,2]            hires_today (next HIRE costs FIB[hires])
    shed: Array      # int32[B,2,12]         shed contents per item (sum <= 100)
    seeds: Array     # int32[B,2,5]          seed counts per crop
    inv: Array       # int32[B,2,U,12]       per-unit carried inventory
    ins: Array       # int32[B,2,U,12]       insertion stamp of inv[..,item] (dict order = ascending stamp among
                     #                       items with inv>0; value is meaningless when inv==0)
    minv: Array      # int32[B,9]            shared market inventory per product (prices = P[item, minv])
    shops: Array     # int32[B,8]            unlocked shop ids in unlock order (-1 = empty slot)
    n_shops: Array   # int32[B]
    oob: Array       # bool[B]               diagnostic: a market inventory left the exact price-table range
    key: Array       # uint32[B,2]           PRNG key (training-mode randomness only)


class EnvState(NamedTuple):
    t: Array         # int32 scalar: index of the next step to execute (== observation "step"); shared by the batch
    g: GameState


class Actions(NamedTuple):
    """One player's action for every game in the batch."""
    uop: Array       # int32[B,U]     unit op id (consts.UNIT_OPS; 18 = INVALID acts as PASS). index 0 = farmer
    uarg: Array      # int32[B,U]     item id 0..11 (255 = invalid/non-string); 0 when the op had no argument
    un: Array        # int32[B,U]     count for PICKUP / PLACE (default 1, clamped to [-30000, 30000])
    mtype: Array     # int32[B,10]    market op id (consts.MARKET_OPS; 0 = empty or malformed "dead" slot)
    mitem: Array     # int32[B,10]    item id 0..11 (255 = invalid); ignored for HIRE / BUY_LAND
    mn: Array        # int32[B,10]    count (>= 1 for a live order, clamped to 1e6); 1 for HIRE / BUY_LAND


# ---------------------------------------------------------------------------------------------- actions (dicts)
def _py_int(x):
    """kagsim_trace.cpp py_int: int(x) semantics (PyNumber_Long) and it must fit a C long (int64)."""
    try:
        v = int(x)
    except Exception:
        return None
    if v < -(2 ** 63) or v >= 2 ** 63:
        return None
    return v


def _conv_unit(h):
    op, arg, n = C.OP_PASS, 0, 1
    if not isinstance(h, list) or len(h) == 0:
        return op, arg, n
    op = C.UNIT_OP_ID.get(str(h[0]), C.OP_INVALID)
    if len(h) >= 2:
        arg = C.ITEM_ID.get(h[1], C.INVALID_ITEM) if isinstance(h[1], str) else C.INVALID_ITEM
    if len(h) >= 3:
        v = _py_int(h[2])
        n = 1 if v is None else max(-30000, min(30000, v))
    return op, arg, n


_DEAD = (C.M_NONE, 0, 0)


def _conv_order(h):
    if not isinstance(h, list) or len(h) == 0:
        return _DEAD
    op = C.MARKET_OP_ID.get(str(h[0]))
    if op is None or op == C.M_NONE:
        return _DEAD
    if op in (C.M_HIRE, C.M_BUY_LAND):
        return op, 0, 1
    if len(h) < 3:
        return _DEAD
    v = _py_int(h[2])
    if v is None or v <= 0:
        return _DEAD
    item = C.ITEM_ID.get(h[1], C.INVALID_ITEM) if isinstance(h[1], str) else C.INVALID_ITEM
    return op, item, min(v, 1000000)


def encode_action(d):
    """dict action -> (uop[U], uarg[U], un[U], mtype[10], mitem[10], mn[10]) numpy int32, mirroring conv_action.
    Missing units are PASS; missing / malformed market slots are M_NONE (a dead slot has the same effect)."""
    uop = np.zeros(C.U, np.int32)
    uarg = np.zeros(C.U, np.int32)
    un = np.ones(C.U, np.int32)
    mt = np.zeros(C.NSLOT, np.int32)
    mi = np.zeros(C.NSLOT, np.int32)
    mn = np.zeros(C.NSLOT, np.int32)
    if isinstance(d, dict):
        if "farmer" in d:
            uop[0], uarg[0], un[0] = _conv_unit(d["farmer"])
        if "hands" in d:
            hands = d["hands"]
            if not isinstance(hands, list):
                raise TypeError("hands must be a list (the C++ binding raises too)")
            nu = 1
            for hh in hands:
                if nu >= C.U:
                    break
                uop[nu], uarg[nu], un[nu] = _conv_unit(hh)
                nu += 1
        m = d.get("market")
        if isinstance(m, list):
            k = 0
            for oo in m:
                if k >= C.NSLOT:
                    break
                mt[k], mi[k], mn[k] = _conv_order(oo)
                k += 1
    return uop, uarg, un, mt, mi, mn


def encode_actions(dicts):
    """list of B action dicts (one player) -> Actions of numpy arrays with batch dim B."""
    cols = [encode_action(d) for d in dicts]
    return Actions(*[np.stack([c[k] for c in cols]) for k in range(6)])


def pass_actions(B):
    z = np.zeros((B, C.U), np.int32)
    return Actions(z, z.copy(), np.ones((B, C.U), np.int32), np.zeros((B, C.NSLOT), np.int32),
                   np.zeros((B, C.NSLOT), np.int32), np.zeros((B, C.NSLOT), np.int32))


# ---------------------------------------------------------------------------------------------- initial state
def initial_game(key=(0, 0)):
    """Single-game initial GameState (numpy, no batch dim)."""
    tiles = np.zeros((2, C.NT, C.NF), np.int32)
    tiles[:, :] = C.FRESH_TILE
    tiles[:, :, C.F_KIND] = np.where(C.QUAD_OF_TILE == 0, C.K_EMPTY, C.K_LOCKED)[None, :]
    ux = np.zeros((2, C.U), np.int32)
    uy = np.zeros((2, C.U), np.int32)
    ux[:, 0] = C.SPAWN_X
    uy[:, 0] = C.SPAWN_Y
    return GameState(
        money=np.full(2, C.STARTING_MONEY, np.int32), tiles=tiles, ux=ux, uy=uy,
        n_units=np.ones(2, np.int32), n_quads=np.ones(2, np.int32), hires=np.zeros(2, np.int32),
        shed=np.zeros((2, C.N_ITEMS), np.int32), seeds=np.zeros((2, C.N_CROPS), np.int32),
        inv=np.zeros((2, C.U, C.N_ITEMS), np.int32), ins=np.zeros((2, C.U, C.N_ITEMS), np.int32),
        minv=np.full(C.N_PRODUCTS, C.I0, np.int32), shops=np.full(C.MAX_SHOPS, -1, np.int32),
        n_shops=np.zeros((), np.int32), oob=np.zeros((), bool), key=np.asarray(key, np.uint32))


def stack_games(games):
    return GameState(*[np.stack([np.asarray(getattr(g, f)) for g in games]) for f in GameState._fields])


def index_game(g, b):
    return GameState(*[np.asarray(x)[b] for x in g])


# ---------------------------------------------------------------------------------------------- obs <-> state
_KIND_OF = {"WEED": C.K_WEED, "COOP": C.K_COOP, "PASTURE": C.K_PASTURE, "PLANT": C.K_PLANT}


def _tile_from_obs(t):
    row = C.FRESH_TILE.copy()
    if t is None:
        row[C.F_KIND] = C.K_EMPTY
        return row
    if t == "LOCKED":
        row[C.F_KIND] = C.K_LOCKED
        return row
    k = _KIND_OF[t["kind"]]
    row[C.F_KIND] = k
    if k == C.K_PLANT:
        row[C.F_WHAT] = C.ITEM_ID[t["crop"]]
        row[C.F_DAY] = t["planted_day"]
        row[C.F_WATERED] = int(bool(t["watered_today"]))
        row[C.F_DRY] = t["consecutive_unwatered"]
        row[C.F_YIELD] = t["yield_units"]
        row[C.F_MLS] = t["max_lifespan_step"]
        row[C.F_FERTU] = t["fertilized_until_day"]
    elif k in (C.K_COOP, C.K_PASTURE) and "animal" in t:
        row[C.F_WHAT] = C.ITEM_ID[t["animal"]]
        row[C.F_ANIMAL] = 1
        row[C.F_DAY] = t["placed_day"]
        row[C.F_YIELD] = t["yield_units"]
        row[C.F_DRY] = t["consecutive_unfed"]
        row[C.F_FED] = int(bool(t["fed_today"]))
        row[C.F_CARED] = int(bool(t["cared_today"]))
        row[C.F_FERTAV] = int(bool(t["fertilizer_available"]))
        row[C.F_CARE] = t["pending_care_bonus"]
    return row


def obs_to_game(obs0, obs1, key=(0, 0)):
    """Build a single-game GameState from the two seats' C++ observe() dicts (same step). Returns (game, t)."""
    assert obs0["step"] == obs1["step"]
    g = initial_game(key)
    tiles, ux, uy = g.tiles, g.ux, g.uy
    money = np.zeros(2, np.int32)
    n_units = np.zeros(2, np.int32)
    n_quads = np.zeros(2, np.int32)
    hires = np.zeros(2, np.int32)
    for p in range(2):
        f = obs0["farms"][p]
        m = f["money"]
        assert float(m) == int(m)
        money[p] = int(m)
        for y in range(C.BOARD):
            for x in range(C.BOARD):
                tiles[p, y * C.BOARD + x] = _tile_from_obs(f["tiles"][y][x])
        ux[p, 0], uy[p, 0] = f["farmer"]
        for h, (hx, hy) in enumerate(f["hands"]):
            ux[p, h + 1], uy[p, h + 1] = hx, hy
        n_units[p] = 1 + len(f["hands"])
        n_quads[p] = len(f["unlocked_quadrants"])
        assert f["unlocked_quadrants"] == C.QUAD_NAMES[:n_quads[p]]
        hires[p] = f["hires_today"]
    shed = np.zeros((2, C.N_ITEMS), np.int32)
    seeds = np.zeros((2, C.N_CROPS), np.int32)
    inv = np.zeros((2, C.U, C.N_ITEMS), np.int32)
    ins = np.zeros((2, C.U, C.N_ITEMS), np.int32)
    for p, o in enumerate((obs0, obs1)):
        pr = o["private"]
        for k, v in pr["shed"].items():
            shed[p, C.ITEM_ID[k]] = v
        for k, v in pr["seeds"].items():
            seeds[p, C.ITEM_ID[k]] = v
        for u, d in enumerate(pr["inventories"]):
            for r, (k, v) in enumerate(d.items()):      # dict order == insertion order
                inv[p, u, C.ITEM_ID[k]] = v
                ins[p, u, C.ITEM_ID[k]] = r - 100       # before any stamp the engine will write (stamp = step >= 0)
    minv = np.array([obs0["market"]["inventory"][n] for n in C.ITEM_NAMES[:C.N_PRODUCTS]], np.int32)
    shops = np.full(C.MAX_SHOPS, -1, np.int32)
    names = obs0["town"]["unlocked_shops"]
    for i, n in enumerate(names):
        shops[i] = C.SHOP_ID[n]
    g = g._replace(money=money, tiles=tiles, ux=ux, uy=uy, n_units=n_units, n_quads=n_quads, hires=hires,
                   shed=shed, seeds=seeds, inv=inv, ins=ins, minv=minv, shops=shops,
                   n_shops=np.asarray(len(names), np.int32))
    return g, int(obs0["step"])


_PRICE_TABLE = None


def price_table():
    global _PRICE_TABLE
    if _PRICE_TABLE is None:
        _PRICE_TABLE = C.build_price_tables()[0]
    return _PRICE_TABLE


def _tile_to_obs(row):
    k = int(row[C.F_KIND])
    if k == C.K_EMPTY:
        return None
    if k == C.K_LOCKED:
        return "LOCKED"
    if k == C.K_WEED:
        return {"kind": "WEED"}
    if k == C.K_PLANT:
        return {"kind": "PLANT", "crop": C.ITEM_NAMES[int(row[C.F_WHAT])], "planted_day": int(row[C.F_DAY]),
                "watered_today": bool(row[C.F_WATERED]), "consecutive_unwatered": int(row[C.F_DRY]),
                "yield_units": int(row[C.F_YIELD]), "max_lifespan_step": int(row[C.F_MLS]),
                "fertilized_until_day": int(row[C.F_FERTU])}
    d = {"kind": "COOP" if k == C.K_COOP else "PASTURE"}
    if row[C.F_ANIMAL]:
        d.update({"animal": C.ITEM_NAMES[int(row[C.F_WHAT])], "placed_day": int(row[C.F_DAY]),
                  "yield_units": int(row[C.F_YIELD]), "consecutive_unfed": int(row[C.F_DRY]),
                  "fed_today": bool(row[C.F_FED]), "cared_today": bool(row[C.F_CARED]),
                  "fertilizer_available": bool(row[C.F_FERTAV]), "pending_care_bonus": int(row[C.F_CARE])})
    return d


def game_to_obs(g, b, seat, t):
    """The observe(seat) dict of batch element b (g: batched GameState of numpy arrays, t: step count)."""
    P = price_table()
    farms = []
    for p in range(2):
        nu = int(g.n_units[b, p])
        tiles = g.tiles[b, p]
        farms.append({
            "money": float(g.money[b, p]),
            "tiles": [[_tile_to_obs(tiles[y * C.BOARD + x]) for x in range(C.BOARD)] for y in range(C.BOARD)],
            "farmer": [int(g.ux[b, p, 0]), int(g.uy[b, p, 0])],
            "hands": [[int(g.ux[b, p, u]), int(g.uy[b, p, u])] for u in range(1, nu)],
            "unlocked_quadrants": C.QUAD_NAMES[:int(g.n_quads[b, p])],
            "hires_today": int(g.hires[b, p])})
    minv = g.minv[b]
    inv_d = {C.ITEM_NAMES[i]: int(minv[i]) for i in range(C.N_PRODUCTS)}
    pr_d = {C.ITEM_NAMES[i]: int(P[i, min(max(int(minv[i]) - C.INV_LO, 0), P.shape[1] - 1)])
            for i in range(C.N_PRODUCTS)}
    ns = int(g.n_shops[b])
    invs = []
    for u in range(int(g.n_units[b, seat])):
        cnt = g.inv[b, seat, u]
        st = g.ins[b, seat, u]
        items = [i for i in range(C.N_ITEMS) if cnt[i] > 0]
        items.sort(key=lambda i: int(st[i]))
        invs.append({C.ITEM_NAMES[i]: int(cnt[i]) for i in items})
    return {
        "remainingOverageTime": 60, "step": int(t), "player": seat, "farms": farms,
        "market": {"inventory": inv_d, "prices": pr_d},
        "town": {"unlocked_shops": [C.SHOP_NAMES[int(s)] for s in g.shops[b, :ns]]},
        "day": int(t) // C.TURNS_PER_DAY, "hour": int(t) % C.TURNS_PER_DAY,
        "private": {"shed": {C.ITEM_NAMES[i]: int(g.shed[b, seat, i]) for i in range(C.N_ITEMS)},
                    "seeds": {C.ITEM_NAMES[i]: int(g.seeds[b, seat, i]) for i in range(C.N_CROPS)},
                    "inventories": invs}}


def obs_diff(a, b, path="", out=None, limit=20):
    """Recursive diff of two observation-like structures. Dict key ORDER is compared too (inventories)."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a.keys()) != list(b.keys()):
            if set(a.keys()) != set(b.keys()) or path.find("inventories") >= 0:
                out.append((path + " keys", list(a.keys()), list(b.keys())))
                if set(a.keys()) != set(b.keys()):
                    return out
        for k in a:
            if k in b:
                obs_diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + " len", len(a), len(b)))
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            obs_diff(x, y, f"{path}[{i}]", out, limit)
    else:
        if type(a) != type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                       and not isinstance(a, bool) and not isinstance(b, bool)):
            out.append((path + " type", repr(a), repr(b)))
        elif a != b:
            out.append((path, a, b))
    return out
