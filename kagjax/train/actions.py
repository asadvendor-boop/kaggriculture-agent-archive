"""Action arrays for the self-play trainer: policy classes -> engine-native unit commands and market orders, plus the
rule layer of our production agent (il/features.py midnight_fix: _plan_fix SE guard / sheep cap / seed budget,
_final_fix, _homing, the midnight overflow sale) as pure array functions of ONE game (vmap them over the batch).

Engine-native action format (mirrors the C++ engine's Action struct, engine/src_v2/sim_trace.hpp):
  Act.u_op  int32[16]  unit ops (OP_* below; unit 0 = farmer), OP_PASS beyond n_cmd
  Act.u_arg int32[16]  crop id (PLANT) or item id (PICKUP / PLACE), else 0
  Act.u_n   int32[16]  quantity (PICKUP / PLACE, >= 1), else 1
  Act.n_cmd int32[]    number of unit commands issued (= units the network sees, min(16, n_units))
  Act.o_op  int32[10]  market orders (M_* below), compacted at the front, M_NONE after n_ord
  Act.o_item int32[10] item id (SELL / BUY_PRODUCT / BUY_ANIMAL: ITEMS index, BUY_SEED: crop id), else 0
  Act.o_n   int32[10]  quantity (HIRE / BUY_LAND: 1)
  Act.n_ord int32[]    number of orders (<= 10)
Items are the engine / features order: WHEAT CARROT TOMATO STRAWBERRY MELON EGG MILK WOOL FERTILIZER GOOSE COW SHEEP.

View = what the rule layer and the quantity clamp read from the game state for the acting seat (one game); the real
backend builds it from the JAX engine state (backend.py), the tests build it from the C++ engine's observation dict.
Python reference semantics (il/features.py, il/live_rollout.py) are reproduced exactly, including sequential effects
(a PICKUP clamps the projected shed at 0 before a later DROP adds to it) and stable tie-breaks (products in engine
order; the observation's shed / prices dicts are in that order).  Verified by test_rules.py on C++ engine games.
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

# ---- engine enums (sim_trace.hpp) -----------------------------------------------------------------------------------
(OP_PASS, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST, OP_PICKUP, OP_DROP, OP_PLACE, OP_PLANT, OP_WATER, OP_HARVEST,
 OP_FERTILIZE, OP_DIG, OP_BUILD_COOP, OP_BUILD_PASTURE, OP_FEED, OP_COLLECT_FERTILIZER, OP_CARE) = range(18)
M_NONE, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL = range(7)
ENGINE_OP_NAMES = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE", "PLANT", "WATER", "HARVEST",
                   "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"]
MARKET_OP_NAMES = ["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = PRODUCTS[:5]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ITEMS = PRODUCTS + ANIMALS
I_WHEAT, I_FERT, I_SHEEP = 0, 8, 11
NP, NI, NU, NO = 9, 12, 16, 10            # products, items, units the network sees, market slots

# ---- policy vocabulary (il/features.py) -------------------------------------------------------------------------------
SIMPLE = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP",
          "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
OPS = SIMPLE + ["PLANT_" + c for c in CROPS] + ["PICKUP_" + i for i in ITEMS] + ["PLACE_" + i for i in ITEMS]
N_OPS, N_QTY = len(OPS), 12
OP_PLANT0, OP_PICKUP0, OP_PLACE0 = 15, 20, 32
MARKET_HEADS = ([("SELL:" + p, 18) for p in PRODUCTS] + [("BUY_SEED:" + c, 8) for c in CROPS] +
                [("BUY_PRODUCT:WHEAT", 18), ("BUY_PRODUCT:FERTILIZER", 4)] + [("BUY_ANIMAL:" + a, 4) for a in ANIMALS] +
                [("HIRE", 11), ("BUY_LAND", 2)])
N_HEADS = len(MARKET_HEADS)                                   # 21
HEAD_SIZES = [n for _, n in MARKET_HEADS]
HEAD_OFF = np.cumsum([0] + HEAD_SIZES)                        # 249 logits in total
H_SEED0, H_BUYW, H_BUYF, H_ANIMAL0, H_HIRE, H_LAND = 9, 14, 15, 16, 19, 20

# class -> engine op / arg
_CLS_OP = np.zeros(N_OPS, np.int32); _CLS_ARG = np.zeros(N_OPS, np.int32)
for _c, _name in enumerate(OPS):
    if _name.startswith("PLANT_"):
        _CLS_OP[_c], _CLS_ARG[_c] = OP_PLANT, CROPS.index(_name[6:])
    elif _name.startswith("PICKUP_"):
        _CLS_OP[_c], _CLS_ARG[_c] = OP_PICKUP, ITEMS.index(_name[7:])
    elif _name.startswith("PLACE_"):
        _CLS_OP[_c], _CLS_ARG[_c] = OP_PLACE, ITEMS.index(_name[6:])
    else:
        _CLS_OP[_c] = ENGINE_OP_NAMES.index(_name)
CLS_OP, CLS_ARG = jnp.asarray(_CLS_OP), jnp.asarray(_CLS_ARG)

# decode_market candidate list in the leaders' canonical order (sells, hires, land, animals, seeds, product buys)
SELL_ORDER = ["FERTILIZER", "WHEAT", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO", "MELON"]
_cand = []                                                     # (head, op, item, kind) kind: 0 sell 1 hire-slot j ...
for _p in SELL_ORDER:
    _cand.append((PRODUCTS.index(_p), M_SELL, PRODUCTS.index(_p), 0))
for _j in range(10):
    _cand.append((H_HIRE, M_HIRE, 0, 100 + _j))                   # hire slot j active when class > j
_cand.append((H_LAND, M_BUY_LAND, 0, 1))
for _a in range(3):
    _cand.append((H_ANIMAL0 + _a, M_BUY_ANIMAL, 9 + _a, 1))
for _c in range(5):
    _cand.append((H_SEED0 + _c, M_BUY_SEED, _c, 1))
_cand.append((H_BUYW, M_BUY_PRODUCT, I_WHEAT, 2))               # class 17 -> 35
_cand.append((H_BUYF, M_BUY_PRODUCT, I_FERT, 1))
CAND_HEAD = jnp.asarray([c[0] for c in _cand], jnp.int32)
CAND_OP = jnp.asarray([c[1] for c in _cand], jnp.int32)
CAND_ITEM = jnp.asarray([c[2] for c in _cand], jnp.int32)
CAND_KIND = jnp.asarray([c[3] for c in _cand], jnp.int32)
N_CAND = len(_cand)                                             # 30

SHED_TILES = jnp.asarray([(4, 4), (5, 4), (5, 5), (4, 5)], jnp.int32)   # CPython iteration order of features.SHED_ACCESS
SHED_CAP = 100
LAST_STEP = 718                                                 # the engine's last decision step (done after it)


class Act(NamedTuple):
    u_op: jnp.ndarray
    u_arg: jnp.ndarray
    u_n: jnp.ndarray
    n_cmd: jnp.ndarray
    o_op: jnp.ndarray
    o_item: jnp.ndarray
    o_n: jnp.ndarray
    n_ord: jnp.ndarray


class View(NamedTuple):
    step: jnp.ndarray          # int32[]   decision step 0..718
    money: jnp.ndarray         # float32[] own money
    nq: jnp.ndarray            # int32[]   unlocked quadrants incl. NW (1..4)
    n_units: jnp.ndarray       # int32[]   all own units (may exceed 16)
    pos: jnp.ndarray           # int32[16,2] (x, y) of units 0..15
    inv: jnp.ndarray           # int32[16,12] per-unit inventories (units 0..15)
    carried: jnp.ndarray       # int32[]   everything carried by ALL units (incl. units beyond 16)
    shed: jnp.ndarray          # int32[12]
    seeds: jnp.ndarray         # int32[5]
    t_yield: jnp.ndarray       # int32[16] yield_units of the own tile under unit k (0 if not a dict tile)
    t_dict: jnp.ndarray        # bool[16]  tile under unit k is a dict (not None / LOCKED)
    t_animal: jnp.ndarray      # bool[16]  ... holds an animal
    t_fert: jnp.ndarray        # bool[16]  ... has fertilizer_available
    prices: jnp.ndarray        # int32[9]
    sheep_total: jnp.ndarray   # int32[]   sheep placed on tiles + in the shed + carried (SHEEP_CAP rule)


class RuleCfg(NamedTuple):
    """Rule switches (static: part of the jit cache key).  Defaults = live_rollout.py defaults."""
    midnight: int = 1          # MIDNIGHT: the whole rule layer (as live_rollout / BCAgent)
    final_fix: int = 1         # FINALFIX
    homing: int = 1            # HOMING
    se_guard: int = 0          # SE_GUARD (plan rule): buy the 3rd land day 10 h SE_HOUR .. day SE_LAST_DAY
    se_hour: int = 12
    se_last_day: int = 13
    se_reserve: int = 300
    sheep_cap: int = 0         # SHEEP_CAP (plan rule)
    seed_budget: int = 0       # SEED_BUDGET (plan rule)


def empty_act():
    z16 = jnp.zeros(NU, jnp.int32); z10 = jnp.zeros(NO, jnp.int32)
    return Act(z16, z16, jnp.ones(NU, jnp.int32), jnp.int32(0), z10, z10, z10, jnp.int32(0))


# ------------------------------------------------------------------------------------------------ policy -> engine
def unit_actions(op_cls, qty_cls, nu, view):
    """op_cls int[16] (policy classes, anything for k >= nu), qty_cls int[16] (quantity class 0..11) -> unit part of
    Act.  Quantity clamp as live_rollout: qc = max(0, min(qc, have - 1)), have = shed stock (PICKUP) or the unit's own
    inventory (PLACE), both from the observation (not reduced by earlier units)."""
    k = jnp.arange(NU)
    live = k < nu
    op = jnp.where(live, CLS_OP[op_cls], OP_PASS)
    arg = jnp.where(live, CLS_ARG[op_cls], 0)
    pick = live & (op == OP_PICKUP); place = live & (op == OP_PLACE)
    have = jnp.where(pick, view.shed[arg], jnp.where(place, view.inv[k, arg], 0))
    qc = jnp.maximum(0, jnp.minimum(qty_cls, have - 1))
    n = jnp.where(pick | place, qc + 1, 1)
    return op.astype(jnp.int32), arg.astype(jnp.int32), n.astype(jnp.int32)


def _compact(keep, op, item, n):
    """Keep flagged candidates in order, packed into NO slots."""
    dest = jnp.where(keep, jnp.cumsum(keep) - 1, NO)          # NO = dropped (out of bounds)
    z = jnp.zeros(NO, jnp.int32)
    o_op = z.at[dest].set(op, mode="drop")
    o_item = z.at[dest].set(item, mode="drop")
    o_n = z.at[dest].set(n, mode="drop")
    return o_op, o_item, o_n, jnp.minimum(keep.sum(), NO).astype(jnp.int32)


def decode_market(classes):
    """classes int[21] -> (o_op, o_item, o_n, n_ord): features.decode_market (canonical order, SELL_ORDER canonical),
    at most 10 orders: trailing SELLs are dropped first, never a HIRE / purchase."""
    c = classes[CAND_HEAD]
    kind = CAND_KIND
    active = jnp.where(kind >= 100, c > (kind - 100), c > 0)
    n = jnp.where(kind >= 100, 1, jnp.where(kind == 1, jnp.where(CAND_OP == M_BUY_LAND, 1, c), c))
    n = jnp.where((kind == 0) & (c == 17), 1000, n)             # SELL class 17 -> 1000
    n = jnp.where((kind == 2) & (c == 17), 35, n)               # BUY_PRODUCT WHEAT class 17 -> 35
    is_sell = CAND_OP == M_SELL
    other = active & ~is_sell
    other_rank = jnp.cumsum(other) - other
    keep_other = other & (other_rank < NO)
    n_other = jnp.minimum(other.sum(), NO)
    sell = active & is_sell
    sell_rank = jnp.cumsum(sell) - sell
    keep_sell = sell & (sell_rank < NO - n_other)
    return _compact(keep_other | keep_sell, CAND_OP, CAND_ITEM, n.astype(jnp.int32))


# ------------------------------------------------------------------------------------------------ order-list helpers
def _insert(o_op, o_item, o_n, n_ord, at, op, item, n, size):
    """Insert one order at position `at` into lists of static length `size` (entries >= n_ord are empty)."""
    j = jnp.arange(size)
    sh = lambda a: jnp.concatenate([a[:1], a[:-1]])             # a[j-1]
    new = lambda a, v: jnp.where(j < at, a, jnp.where(j == at, v, sh(a)))
    return new(o_op, op), new(o_item, item), new(o_n, n), n_ord + 1


def _pad(a, size):
    return jnp.concatenate([a, jnp.zeros(size - a.shape[0], a.dtype)])


# ------------------------------------------------------------------------------------------------ rule layer
def plan_fix(view, a, cfg: RuleCfg):
    """features._plan_fix (SEED_BUDGET, SHEEP_CAP, SE_GUARD; WASH is not ported).  If the SE guard cannot make room
    (the Python raises IndexError and midnight_fix keeps the original action) the whole plan fix is discarded."""
    orig = a
    step = view.step; day, hour = step // 24, step % 24
    if cfg.seed_budget:
        k = jnp.arange(NU)
        is_pl = (k < a.n_cmd) & (a.u_op == OP_PLANT)
        crop = jnp.where(is_pl, a.u_arg, 0)
        same = is_pl[None, :] & is_pl[:, None] & (crop[None, :] == crop[:, None]) & (k[None, :] < k[:, None])
        rank = same.sum(1)                                      # earlier requesters of the same crop
        drop = is_pl & (rank >= view.seeds[crop])
        a = a._replace(u_op=jnp.where(drop, OP_PASS, a.u_op), u_arg=jnp.where(drop, 0, a.u_arg),
                       u_n=jnp.where(drop, 1, a.u_n))
    if cfg.sheep_cap:
        j = jnp.arange(NO)
        valid = j < a.n_ord
        is_sh = valid & (a.o_op == M_BUY_ANIMAL) & (a.o_item == I_SHEEP)
        fire = (view.nq < 4) & is_sh.any()
        room0 = jnp.maximum(0, cfg.sheep_cap - view.sheep_total)
        req = jnp.where(is_sh, jnp.maximum(0, a.o_n), 0)
        before = jnp.cumsum(req) - req
        q = jnp.where(is_sh, jnp.clip(room0 - before, 0, req), a.o_n)   # sequential room allocation
        q = jnp.where(is_sh, jnp.minimum(q, req), q)
        keep = valid & ~(is_sh & (q <= 0))
        o_op, o_item, o_n, n_ord = _compact(keep, a.o_op, a.o_item, jnp.where(is_sh, q, a.o_n))
        a = jax.tree_util.tree_map(lambda new, old: jnp.where(fire, new, old),
                                   a._replace(o_op=o_op, o_item=o_item, o_n=o_n, n_ord=n_ord), a)
    ok = jnp.bool_(True)
    if cfg.se_guard:
        j = jnp.arange(NO)
        valid = j < a.n_ord
        has_land = (valid & (a.o_op == M_BUY_LAND)).any()
        cond = ((view.nq == 3) & (step < LAST_STEP) & (((day == 10) & (hour >= cfg.se_hour)) | ((day > 10) & (day <= cfg.se_last_day)))
                & ~has_land & (view.money >= 4000 + cfg.se_reserve))
        at = jnp.max(jnp.where(valid & ((a.o_op == M_SELL) | (a.o_op == M_HIRE)), j + 1, 0))
        S = NO + 1
        o_op, o_item, o_n, n_ord = _insert(_pad(a.o_op, S), _pad(a.o_item, S), _pad(a.o_n, S), a.n_ord, at,
                                           M_BUY_LAND, 0, 1, S)
        jj = jnp.arange(S)
        v2 = jj < n_ord
        buy = v2 & (o_op != M_SELL) & (o_op != M_HIRE) & (o_op != M_BUY_LAND)
        sel = v2 & (o_op == M_SELL)
        last_buy = jnp.max(jnp.where(buy, jj, -1)); last_sell = jnp.max(jnp.where(sel, jj, -1))
        cand = jnp.where(last_buy >= 0, last_buy, last_sell)
        over = n_ord > NO
        pop = over & (cand >= 0)
        drop_at = jnp.where(pop, cand, S)
        keep = v2 & (jj != drop_at)
        dest = jnp.where(keep, jnp.cumsum(keep) - 1, NO)
        z = jnp.zeros(NO, jnp.int32)
        new = a._replace(o_op=z.at[dest].set(o_op, mode="drop"), o_item=z.at[dest].set(o_item, mode="drop"),
                         o_n=z.at[dest].set(o_n, mode="drop"), n_ord=jnp.minimum(keep.sum(), NO).astype(jnp.int32))
        ok = ~(cond & over & (cand < 0))                        # IndexError in the Python: keep the original action
        a = jax.tree_util.tree_map(lambda n_, o_: jnp.where(cond, n_, o_), new, a)
    return jax.tree_util.tree_map(lambda n_, o_: jnp.where(ok, n_, o_), a, orig)


def _prod_carrier(view, a):
    k = jnp.arange(NU)
    return (k < a.n_cmd) & (k < view.n_units) & (view.inv[:, :NP] > 0).any(1)


def final_fix(view, a):
    """features._final_fix (the engine's last turn): units at a shed tile carrying products DROP, the market becomes
    SELL of every product in shed + dropped stock, largest stock first (stable in product order)."""
    at_shed = ((view.pos[:, None, :] == SHED_TILES[None]).all(-1)).any(1)
    drop = _prod_carrier(view, a) & at_shed
    stock = view.shed[:NP] + jnp.where(drop[:, None], view.inv[:, :NP], 0).sum(0)
    order = jnp.argsort(-stock, stable=True)
    s = stock[order]
    keep = s > 0
    o_op = jnp.where(keep, M_SELL, M_NONE).astype(jnp.int32)
    o_item = jnp.where(keep, order, 0).astype(jnp.int32)
    o_n = jnp.where(keep, s, 0).astype(jnp.int32)
    z = jnp.zeros(NO - NP, jnp.int32)
    return a._replace(u_op=jnp.where(drop, OP_DROP, a.u_op), u_arg=jnp.where(drop, 0, a.u_arg), u_n=jnp.where(drop, 1, a.u_n),
                      o_op=jnp.concatenate([o_op, z]), o_item=jnp.concatenate([o_item, z]), o_n=jnp.concatenate([o_n, z]),
                      n_ord=keep.sum().astype(jnp.int32))


def homing(view, a):
    """features._homing (steps 690..717): a unit carrying products walks to its nearest shed tile as late as possible
    and DROPs there on the last turn before the final one."""
    left = LAST_STEP - view.step
    carrier = _prod_carrier(view, a)
    d4 = jnp.abs(SHED_TILES[None, :, 0] - view.pos[:, None, 0]) + jnp.abs(SHED_TILES[None, :, 1] - view.pos[:, None, 1])
    t = jnp.argmin(d4, axis=1)                                  # first minimum = CPython set order
    d = d4[jnp.arange(NU), t]
    tx, ty = SHED_TILES[t, 0], SHED_TILES[t, 1]
    x, y = view.pos[:, 0], view.pos[:, 1]
    move = jnp.where(tx > x, OP_EAST, jnp.where(tx < x, OP_WEST, jnp.where(ty > y, OP_SOUTH, OP_NORTH)))
    do_drop = carrier & (d == 0) & (left <= 1)
    do_move = carrier & (d > 0) & (d >= left - 1) & (d <= left)
    op = jnp.where(do_drop, OP_DROP, jnp.where(do_move, move, a.u_op))
    ch = do_drop | do_move
    return a._replace(u_op=op.astype(jnp.int32), u_arg=jnp.where(ch, 0, a.u_arg), u_n=jnp.where(ch, 1, a.u_n))


def midnight_sale(view, a):
    """The midnight overflow rule (hour 23, not the last turn): project shed + carried after this turn's own unit
    actions and orders and sell the excess now from the shed, cheapest product first."""
    k = jnp.arange(NU)
    uv = (k < a.n_cmd) & (k < view.n_units)

    def unit(carry, i):
        stock, gain = carry
        op, v = a.u_op[i], uv[i]
        gain = gain + jnp.where(v & (op == OP_HARVEST) & view.t_dict[i], jnp.maximum(0, view.t_yield[i]), 0)
        gain = gain + (v & (op == OP_COLLECT_FERTILIZER) & view.t_dict[i] & view.t_animal[i] & view.t_fert[i])
        it = a.u_arg[i]
        stock = jnp.where(v & (op == OP_PICKUP), stock.at[it].set(jnp.maximum(0, stock[it] - a.u_n[i])), stock)
        stock = jnp.where(v & (op == OP_DROP), stock + view.inv[i], stock)
        return (stock, gain), None

    (stock, gain), _ = jax.lax.scan(unit, (view.shed, jnp.int32(0)), jnp.arange(NU))

    def order(carry, j):
        sold, bought = carry
        v = j < a.n_ord
        op, it, n = a.o_op[j], a.o_item[j], a.o_n[j]
        q = jnp.minimum(jnp.maximum(0, n), jnp.maximum(0, stock[it] - sold[it]))
        sold = jnp.where(v & (op == M_SELL), sold.at[it].add(q), sold)
        bought = bought + jnp.where(v & ((op == M_BUY_PRODUCT) | (op == M_BUY_ANIMAL)), jnp.maximum(0, n), 0)
        return (sold, bought), None

    (sold, bought), _ = jax.lax.scan(order, (jnp.zeros(NI, jnp.int32), jnp.int32(0)), jnp.arange(NO))
    over0 = view.shed.sum() + view.carried + gain + bought - sold.sum() - SHED_CAP
    left = stock[:NP] - sold[:NP]
    prod_order = jnp.argsort(view.prices, stable=True)          # cheapest first, ties in product order

    def add(carry, r):
        o_op, o_item, o_n, n_ord, over = carry
        it = prod_order[r]
        go = (over > 0) & (left[it] > 0)
        q = jnp.minimum(over, left[it])
        j = jnp.arange(NO)
        valid = j < n_ord
        hits = valid & (o_op == M_SELL) & (o_item == it)
        has_hit = hits.any()
        hit = jnp.argmax(hits)
        # (a) grow the existing SELL
        o_n_a = o_n.at[hit].add(q)
        # (b) insert a new SELL after the last SELL
        at = jnp.max(jnp.where(valid & (o_op == M_SELL), j + 1, 0))
        ins = _insert(o_op, o_item, o_n, n_ord, at, M_SELL, it, q, NO)
        can_ins = n_ord < NO
        do_a = go & has_hit
        do_b = go & ~has_hit & can_ins
        o_op2 = jnp.where(do_b, ins[0], o_op)
        o_item2 = jnp.where(do_b, ins[1], o_item)
        o_n2 = jnp.where(do_a, o_n_a, jnp.where(do_b, ins[2], o_n))
        n_ord2 = jnp.where(do_b, ins[3], n_ord)
        over2 = jnp.where(do_a | do_b, over - q, over)
        return (o_op2, o_item2, o_n2, n_ord2, over2), None

    (o_op, o_item, o_n, n_ord, _), _ = jax.lax.scan(add, (a.o_op, a.o_item, a.o_n, a.n_ord, over0), jnp.arange(NP))
    new = a._replace(o_op=o_op, o_item=o_item, o_n=o_n, n_ord=n_ord.astype(jnp.int32))
    return jax.tree_util.tree_map(lambda n_, o_: jnp.where(over0 > 0, n_, o_), new, a)


def apply_rules(view, a, cfg: RuleCfg):
    """features.midnight_fix(obs, action) for ONE game (static cfg): plan fix, then the last turn (final fix), the last
    day (homing) and the midnight sale."""
    if not cfg.midnight:
        return a
    if cfg.se_guard or cfg.sheep_cap or cfg.seed_budget:
        a = plan_fix(view, a, cfg)
    step = view.step
    out = a
    if cfg.homing:
        h = homing(view, a)
        out = jax.tree_util.tree_map(lambda n_, o_: jnp.where((step >= 690) & (step < LAST_STEP), n_, o_), h, out)
    m = midnight_sale(view, out)
    out = jax.tree_util.tree_map(lambda n_, o_: jnp.where((step % 24 == 23) & (step < LAST_STEP), n_, o_), m, out)
    if cfg.final_fix:
        f = final_fix(view, a)
        out = jax.tree_util.tree_map(lambda n_, o_: jnp.where(step >= LAST_STEP, n_, o_), f, out)
    return out


# ------------------------------------------------------------------------------------------------ numpy converters
def act_to_dict(a):
    """Act of ONE game (numpy / jax arrays) -> the Python action dict the C++ engine and features.py use."""
    a = jax.tree_util.tree_map(np.asarray, a)
    cmds = []
    for i in range(int(a.n_cmd)):
        op, arg, n = int(a.u_op[i]), int(a.u_arg[i]), int(a.u_n[i])
        name = ENGINE_OP_NAMES[op]
        if op == OP_PLANT:
            cmds.append(["PLANT", CROPS[arg]])
        elif op in (OP_PICKUP, OP_PLACE):
            cmds.append([name, ITEMS[arg], n])
        else:
            cmds.append([name])
    mk = []
    for j in range(int(a.n_ord)):
        op, it, n = int(a.o_op[j]), int(a.o_item[j]), int(a.o_n[j])
        if op == M_HIRE:
            mk.append(["HIRE"])
        elif op == M_BUY_LAND:
            mk.append(["BUY_LAND"])
        elif op == M_BUY_SEED:
            mk.append(["BUY_SEED", CROPS[it], n])
        else:
            mk.append([MARKET_OP_NAMES[op], ITEMS[it], n])
    return {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": mk}


def dict_to_act(d, n_cmd=None):
    """Python action dict (policy-representable commands, <= 16 units, <= 10 orders) -> Act (numpy)."""
    cmds = [d.get("farmer") or ["PASS"]] + list(d.get("hands") or [])
    cmds = cmds[:NU]
    u_op = np.zeros(NU, np.int32); u_arg = np.zeros(NU, np.int32); u_n = np.ones(NU, np.int32)
    for i, c in enumerate(cmds):
        u_op[i] = ENGINE_OP_NAMES.index(c[0])
        if c[0] == "PLANT":
            u_arg[i] = CROPS.index(c[1])
        elif c[0] in ("PICKUP", "PLACE"):
            u_arg[i] = ITEMS.index(c[1]); u_n[i] = int(c[2]) if len(c) > 2 else 1
    o_op = np.zeros(NO, np.int32); o_item = np.zeros(NO, np.int32); o_n = np.zeros(NO, np.int32)
    mk = list(d.get("market") or [])
    assert len(mk) <= NO, mk
    for j, o in enumerate(mk):
        o_op[j] = MARKET_OP_NAMES.index(o[0])
        if o[0] in ("HIRE", "BUY_LAND"):
            o_n[j] = 1
        elif o[0] == "BUY_SEED":
            o_item[j] = CROPS.index(o[1]); o_n[j] = int(o[2])
        else:
            o_item[j] = ITEMS.index(o[1]); o_n[j] = int(o[2])
    return Act(u_op, u_arg, u_n, np.int32(len(cmds) if n_cmd is None else n_cmd), o_op, o_item, o_n, np.int32(len(mk)))


def obs_to_view(obs):
    """C++ engine / Kaggle observation dict (acting player) -> View (numpy).  Used by the tests and as the reference
    for the JAX-engine adapter."""
    me = int(obs["player"]); farm = obs["farms"][me]; priv = obs["private"]
    positions = [farm["farmer"]] + list(farm.get("hands") or [])
    invs = priv.get("inventories") or []
    pos = np.zeros((NU, 2), np.int32); inv = np.zeros((NU, NI), np.int32)
    t_yield = np.zeros(NU, np.int32); t_dict = np.zeros(NU, bool); t_animal = np.zeros(NU, bool); t_fert = np.zeros(NU, bool)
    for k, p in enumerate(positions[:NU]):
        pos[k] = (int(p[0]), int(p[1]))
        tl = farm["tiles"][int(p[1])][int(p[0])]
        if isinstance(tl, dict):
            t_dict[k] = True; t_yield[k] = int(tl.get("yield_units", 0) or 0)
            t_animal[k] = bool(tl.get("animal")); t_fert[k] = bool(tl.get("fertilizer_available"))
        if k < len(invs):
            for it, v in invs[k].items():
                inv[k, ITEMS.index(it)] = int(v)
    shed = priv.get("shed") or {}
    sheep = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
    sheep += int(shed.get("SHEEP", 0)) + sum(int((iv or {}).get("SHEEP", 0)) for iv in invs)
    seeds = priv.get("seeds") or {}
    prices = obs["market"]["prices"]
    return View(step=np.int32(obs["step"]), money=np.float32(farm["money"]), nq=np.int32(len(farm.get("unlocked_quadrants") or [])),
                n_units=np.int32(len(positions)), pos=pos, inv=inv,
                carried=np.int32(sum(int(v) for iv in invs for v in iv.values())),
                shed=np.array([int(shed.get(i, 0)) for i in ITEMS], np.int32), seeds=np.array([int(seeds.get(c, 0)) for c in CROPS], np.int32),
                t_yield=t_yield, t_dict=t_dict, t_animal=t_animal, t_fert=t_fert,
                prices=np.array([int(prices.get(p, 0)) for p in PRODUCTS], np.int32), sheep_total=np.int32(sheep))
