"""features6.encode_obs (v5 planes/glob/units + the 81 "if nobody acts" future features) in JAX, bit-exact.

Input: RawObs, a plain array view of ONE seat's observation (what the Kaggle / C++ engine observation dict holds),
built from the observation dict by raw_from_obs (tests, reference) or from the JAX engine state by an adapter
(see adapters at the bottom / INTERFACE.md).  Farm axis: index 0 = the acting seat ("own"), 1 = the rival.
All functions work on ONE game; batch with jax.vmap (encode_batch below).

RawObs fields (NU = unit capacity of the arrays, >= the most units a farm can have; the engine allows 40):
  step        int32 []            game step (day = step // 24, hour = step % 24)
  money       float32 [2]         integer-valued money (own, rival)
  n_units     int32 [2]           farmer + hands
  unit_xy     int32 [2,NU,2]      (x, y) of unit u (u = 0 farmer, 1.. hands); rows >= n_units are ignored
  hires_today int32 [2]
  n_quads     int32 [2]           unlocked quadrants 1..4 (they unlock in the order NW, NE, SW, SE)
  tile_kind   int32 [2,10,10]     0 EMPTY (None) 1 LOCKED 2 WEED 3 PLANT 4 COOP 5 PASTURE      (index [farm, y, x])
  tile_what   int32 [2,10,10]     PLANT: crop 0..4 (WHEAT CARROT TOMATO STRAWBERRY MELON); animal tile: 0..2 (GOOSE COW
                                  SHEEP); else -1
  tile_animal bool  [2,10,10]     COOP / PASTURE holding an animal (the dict has an "animal" key)
  tile_day    int32 [2,10,10]     PLANT planted_day / animal placed_day
  tile_yield  int32 [2,10,10]     yield_units
  tile_dry    int32 [2,10,10]     PLANT consecutive_unwatered / animal consecutive_unfed
  tile_watered bool [2,10,10]     PLANT watered_today
  tile_fed    bool  [2,10,10]     animal fed_today
  tile_cared  bool  [2,10,10]     animal cared_today
  tile_fert   bool  [2,10,10]     animal fertilizer_available
  tile_bonus  int32 [2,10,10]     animal pending_care_bonus
  tile_fert_until int32 [2,10,10] PLANT fertilized_until_day (-1 when absent)
  tile_life   int32 [2,10,10]     PLANT max_lifespan_step (-1 when absent)
  shed        int32 [12]          own shed, ITEMS order (WHEAT..FERTILIZER, GOOSE, COW, SHEEP)
  seeds       int32 [5]           own seeds, CROPS order
  inv         int32 [NU,12]       own per-unit inventories
  mkt_inv     int32 [9]           market inventory, PRODUCTS order
  mkt_price   int32 [9]           market prices
  shops       int32 [8]           instances of each shop in town.unlocked_shops (SHOPS order, a shop can appear twice)
Policy state (not in the observation): prev_ops int32 [16], the op class each unit issued last step (-1 = none /
new unit / step 0).

Outputs of encode(raw, prev_ops, tabs): dict(planes uint8 [47,10,10], glob float32 [179], pos int32 [16,2],
uf float32 [16,76], valid bool [16]) -- the reference encode_obs arrays zero-padded to 16 units, as live_rollout
stacks them.  net_inputs() turns them into the network's float inputs (planes / 255).
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from kagjax.policy import consts as K


class RawObs(NamedTuple):
    step: jnp.ndarray
    money: jnp.ndarray
    n_units: jnp.ndarray
    unit_xy: jnp.ndarray
    hires_today: jnp.ndarray
    n_quads: jnp.ndarray
    tile_kind: jnp.ndarray
    tile_what: jnp.ndarray
    tile_animal: jnp.ndarray
    tile_day: jnp.ndarray
    tile_yield: jnp.ndarray
    tile_dry: jnp.ndarray
    tile_watered: jnp.ndarray
    tile_fed: jnp.ndarray
    tile_cared: jnp.ndarray
    tile_fert: jnp.ndarray
    tile_bonus: jnp.ndarray
    tile_fert_until: jnp.ndarray
    tile_life: jnp.ndarray
    shed: jnp.ndarray
    seeds: jnp.ndarray
    inv: jnp.ndarray
    mkt_inv: jnp.ndarray
    mkt_price: jnp.ndarray
    shops: jnp.ndarray


# ---------------------------------------------------------------------------------------------- dict -> RawObs
_KIND = {"WEED": K.K_WEED, "PLANT": K.K_PLANT, "COOP": K.K_COOP, "PASTURE": K.K_PASTURE}


def raw_from_obs(obs, nu=40):
    """Observation dict of the acting seat (C++ engine / Kaggle format) -> RawObs of numpy arrays (one game)."""
    me = int(obs["player"]); step = int(obs["step"]); day = step // 24
    farms = [obs["farms"][me], obs["farms"][1 - me]]
    sh = (2, K.B, K.B)
    kind = np.zeros(sh, np.int32); what = np.full(sh, -1, np.int32); animal = np.zeros(sh, bool)
    tday = np.zeros(sh, np.int32); yld = np.zeros(sh, np.int32); dry = np.zeros(sh, np.int32)
    wat = np.zeros(sh, bool); fed = np.zeros(sh, bool); cared = np.zeros(sh, bool); fert = np.zeros(sh, bool)
    bonus = np.zeros(sh, np.int32); fu = np.full(sh, -1, np.int32); life = np.full(sh, -1, np.int32)
    n_units = np.zeros(2, np.int32); xy = np.zeros((2, nu, 2), np.int32)
    money = np.zeros(2, np.float32); hires = np.zeros(2, np.int32); nq = np.zeros(2, np.int32)
    for f, fm in enumerate(farms):
        money[f] = float(fm.get("money", 0)); hires[f] = int(fm.get("hires_today", 0))
        nq[f] = len(fm.get("unlocked_quadrants") or [])
        units = [fm["farmer"]] + list(fm.get("hands") or [])
        assert len(units) <= nu, (len(units), nu)
        n_units[f] = len(units)
        for u, p in enumerate(units):
            xy[f, u] = (int(p[0]), int(p[1]))
        for y, row in enumerate(fm["tiles"]):
            for x, t in enumerate(row):
                if t is None:
                    continue
                if t == "LOCKED":
                    kind[f, y, x] = K.K_LOCKED; continue
                if not isinstance(t, dict):
                    continue
                kind[f, y, x] = _KIND.get(t.get("kind"), K.K_EMPTY)
                if t.get("kind") == "PLANT":
                    c = t.get("crop")
                    what[f, y, x] = K.CROPS.index(c) if c in K.CROPS else -1
                    tday[f, y, x] = int(t.get("planted_day", day)); yld[f, y, x] = int(t.get("yield_units", 0) or 0)
                    dry[f, y, x] = int(t.get("consecutive_unwatered", 0)); wat[f, y, x] = bool(t.get("watered_today"))
                    fu[f, y, x] = int(t.get("fertilized_until_day", -1)); life[f, y, x] = int(t.get("max_lifespan_step", -1))
                elif "animal" in t:
                    a = t.get("animal")
                    animal[f, y, x] = True
                    what[f, y, x] = K.ANIMALS.index(a) if a in K.ANIMALS else -1
                    tday[f, y, x] = int(t.get("placed_day", day)); yld[f, y, x] = int(t.get("yield_units", 0) or 0)
                    dry[f, y, x] = int(t.get("consecutive_unfed", 0)); fed[f, y, x] = bool(t.get("fed_today"))
                    cared[f, y, x] = bool(t.get("cared_today")); fert[f, y, x] = bool(t.get("fertilizer_available"))
                    bonus[f, y, x] = int(t.get("pending_care_bonus", 0))
    priv = obs["private"]
    shed_d = priv.get("shed", {}) or {}; seeds_d = priv.get("seeds", {}) or {}
    inv = np.zeros((nu, len(K.ITEMS)), np.int32)
    for u, iv in enumerate(priv.get("inventories") or []):
        for it, v in (iv or {}).items():
            inv[u, K.ITEMS.index(it)] = int(v)
    shops = np.zeros(len(K.SHOPS), np.int32)
    for s in obs["town"].get("unlocked_shops", []) or []:
        shops[K.SHOPS.index(s)] += 1
    mi, mp = obs["market"]["inventory"], obs["market"]["prices"]
    return RawObs(step=np.int32(step), money=money, n_units=n_units, unit_xy=xy, hires_today=hires, n_quads=nq,
                  tile_kind=kind, tile_what=what, tile_animal=animal, tile_day=tday, tile_yield=yld, tile_dry=dry,
                  tile_watered=wat, tile_fed=fed, tile_cared=cared, tile_fert=fert, tile_bonus=bonus,
                  tile_fert_until=fu, tile_life=life,
                  shed=np.array([int(shed_d.get(i, 0)) for i in K.ITEMS], np.int32),
                  seeds=np.array([int(seeds_d.get(c, 0)) for c in K.CROPS], np.int32), inv=inv,
                  mkt_inv=np.array([int(mi.get(p, K.I0)) for p in K.PRODUCTS], np.int32),
                  mkt_price=np.array([int(mp.get(p, K.BASE[p])) for p in K.PRODUCTS], np.int32), shops=shops)


def prev_array(prev_ops):
    """Reference prev_ops (list of op classes by unit, or None) -> int32 [16] with -1 = none."""
    a = np.full(K.MAX_UNITS, -1, np.int32)
    if prev_ops is not None:
        for k, v in enumerate(list(prev_ops)[:K.MAX_UNITS]):
            a[k] = int(v)
    return a


def stack(items):
    """List of per-game pytrees (RawObs / arrays) -> batched pytree (numpy)."""
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs), *items)


# ---------------------------------------------------------------------------------------------- tables
_TABS = None


def tables_jnp():
    global _TABS
    if _TABS is None:
        t = K.tables()
        _TABS = {k: jnp.asarray(v) for k, v in t.items()}
    return _TABS


# ---------------------------------------------------------------------------------------------- JAX encoder
_CROP_FIRST = jnp.asarray(K.CROP_FIRST); _CROP_MAXDAY = jnp.asarray(K.CROP_MAXDAY)
_CROP_ONGOING = jnp.asarray(K.CROP_ONGOING); _ANIMAL_FIRST = jnp.asarray(K.ANIMAL_FIRST)
_ANIMAL_PROD = jnp.asarray(K.ANIMAL_PROD); _BASE = jnp.asarray(K.BASE_ARR)
_SHOP_RATE = jnp.asarray(K.SHOP_RATE); _TOWN_RATE = jnp.asarray(K.TOWN_RATE)
_SHED_MASK = np.zeros((K.B, K.B), bool)
for (_x, _y) in K.SHED_ACCESS:
    _SHED_MASK[_y, _x] = True
_SPAWN_MASK = np.zeros((K.B, K.B), bool); _SPAWN_MASK[K.SPAWN[1], K.SPAWN[0]] = True
_SHED_MASK_J = jnp.asarray(_SHED_MASK); _SPAWN_MASK_J = jnp.asarray(_SPAWN_MASK)
F32 = jnp.float32


def _q(tabs, name, n):
    """Exact integer / constant quotient (reference float64 division stored as float32) via consts.QUOT tables."""
    dv, lo, hi = K.QUOT[name]
    return tabs[name][jnp.clip(n, lo, hi) - lo]


def _money_q(m, tabs):
    """money / 1e5 (integer-valued float32 money)."""
    mi = jnp.clip(m, 0, K.LOG_N - 1).astype(jnp.int32)
    return jnp.where((m >= 0) & (m < K.LOG_N), tabs["money_q"][mi], m / F32(1e5))


def _u8(b):
    return jnp.where(b, jnp.uint8(255), jnp.uint8(0))


def _unit_counts(xy, n):
    """xy int [NU,2], n int -> units per cell [10,10] (int32)."""
    nu = xy.shape[0]
    valid = jnp.arange(nu) < n
    idx = jnp.clip(xy[:, 1], 0, 9) * K.B + jnp.clip(xy[:, 0], 0, 9)
    cnt = jnp.zeros(K.B * K.B, jnp.int32).at[idx].add(valid.astype(jnp.int32))
    return cnt.reshape(K.B, K.B)


def _farm_masks(raw, f):
    kind = raw.tile_kind[f]; what = raw.tile_what[f]
    is_plant = (kind == K.K_PLANT) & (what >= 0) & (what < len(K.CROPS))
    is_animal = raw.tile_animal[f] & (kind != K.K_PLANT) & (what >= 0) & (what < len(K.ANIMALS))
    return kind, what, is_plant, is_animal


def encode_planes(raw, tabs):
    step = raw.step; day = step // 24
    ch = []
    # ---- own farm (31) ----
    kind, what, is_plant, is_animal = _farm_masks(raw, 0)
    for c in range(6):
        ch.append(_u8(kind == c))
    for c in range(5):
        ch.append(_u8(is_plant & (what == c)))
    for a in range(3):
        ch.append(_u8(is_animal & (what == a)))
    w = jnp.clip(what, 0, 4)
    wa = jnp.clip(what, 0, 2)
    yu = raw.tile_yield[0]
    age = day - raw.tile_day[0]
    watered = raw.tile_watered[0]; dry = raw.tile_dry[0]
    ch.append(_u8(is_plant & watered))                                                          # 14
    ch.append(_u8(is_plant & ~watered & (dry >= 1)))                                            # 15
    yi = jnp.clip(yu, 0, K.YMAX)
    y16 = jnp.where(is_plant, tabs["lut_yield"][w, yi], jnp.where(is_animal, tabs["lut_yield"][5 + wa, yi], jnp.uint8(0)))
    ch.append(y16)                                                                              # 16
    ai = jnp.clip(age - K.AGE_LO, 0, K.AGE_HI - K.AGE_LO)
    ch.append(jnp.where(is_plant, tabs["lut_age"][w, ai], jnp.uint8(0)))                       # 17
    ready_plant = is_plant & (yu > 0) & (age >= _CROP_FIRST[w])
    ready_animal = is_animal & (yu > 0)
    ch.append(_u8(ready_plant | ready_animal))                                                  # 18
    ch.append(_u8(is_plant & (raw.tile_fert_until[0] >= day)))                                  # 19
    my = _CROP_MAXDAY[w]
    ch.append(_u8(is_plant & ~_CROP_ONGOING[w] & ((my + 1) // 2 <= age) & (age <= my)))         # 20
    life = raw.tile_life[0]
    ch.append(_u8(is_plant & (life >= 0) & (step >= life - 12)))                                # 21
    fed = raw.tile_fed[0]
    ch.append(_u8(is_animal & fed))                                                             # 22
    ch.append(_u8(is_animal & raw.tile_cared[0]))                                               # 23
    ch.append(_u8(is_animal & raw.tile_fert[0]))                                                # 24
    ch.append(_u8(is_animal & ~fed & (dry >= 1)))                                               # 25
    ch.append(_u8(is_animal & (raw.tile_bonus[0] > 0)))                                         # 26
    ch.append(_u8(is_animal & (day - raw.tile_day[0] >= _ANIMAL_FIRST[wa] - 1)))                # 27
    cnt = _unit_counts(raw.unit_xy[0], raw.n_units[0])
    ch.append(tabs["lut_units"][jnp.clip(cnt, 0, 41)])                                          # 28
    ch.append(_u8(_SHED_MASK_J))                                                                # 29
    ch.append(_u8(_SPAWN_MASK_J))                                                               # 30
    # ---- rival farm (16) ----
    kind, what, is_plant, is_animal = _farm_masks(raw, 1)
    for c in range(6):
        ch.append(_u8(kind == c))
    for c in range(5):
        ch.append(_u8(is_plant & (what == c)))
    for a in range(3):
        ch.append(_u8(is_animal & (what == a)))
    ch.append(_u8((is_plant | is_animal) & (raw.tile_yield[1] > 0)))                            # 45
    cnt = _unit_counts(raw.unit_xy[1], raw.n_units[1])
    ch.append(tabs["lut_units"][jnp.clip(cnt, 0, 41)])                                          # 46
    return jnp.stack(ch)                                                                        # [47,10,10] uint8


def _log_money(m, tabs):
    mi = jnp.clip(m, 0, K.LOG_N - 1).astype(jnp.int32)
    fallback = jnp.log1p(jnp.maximum(m, 0.0)) / F32(12)
    return jnp.where(m < K.LOG_N, tabs["log_money"][mi], fallback)


def encode_glob5(raw, tabs):
    step = raw.step; day = step // 24; hour = step % 24
    g = [_q(tabs, "q720", step)[None], _q(tabs, "q29", day)[None], jax.nn.one_hot(hour, 24, dtype=F32)]
    mo, mr = raw.money[0], raw.money[1]
    d = jnp.clip(mo - mr, -K.MDIFF, K.MDIFF).astype(jnp.int32)
    g += [_money_q(mo, tabs)[None], _money_q(mr, tabs)[None], jnp.clip(tabs["mdiff_q"][d + K.MDIFF], -3, 3)[None],
          _log_money(mo, tabs)[None]]
    g.append(jnp.minimum(F32(2), _q(tabs, "q10", raw.seeds)))
    g.append(jnp.minimum(F32(2), _q(tabs, "q50", raw.shed)))
    g.append(_q(tabs, "q100", raw.shed.sum())[None])
    pi = jnp.clip(raw.mkt_price, 0, K.OBS_PRICE_N - 1)
    g.append(jnp.where(raw.mkt_price < K.OBS_PRICE_N, tabs["obs_price_q"][jnp.arange(9), pi],
                       raw.mkt_price.astype(F32) / _BASE.astype(F32)))
    g.append(jnp.clip(_q(tabs, "qinv", raw.mkt_inv - K.I0), -3, 3))
    g.append(_q(tabs, "q2", raw.shops))
    g.append(_q(tabs, "q8", raw.shops.sum())[None])
    g.append(_q(tabs, "q15", jnp.stack([raw.hires_today[0], raw.n_units[0] - 1, raw.hires_today[1], raw.n_units[1] - 1])))
    q = jnp.arange(2, 5)
    g.append((raw.n_quads[0] >= q).astype(F32)); g.append((raw.n_quads[1] >= q).astype(F32))
    g.append(_q(tabs, "q72", (3 - day % 3) * 24 - hour)[None])
    nu = raw.inv.shape[0]
    valid = (jnp.arange(nu) < raw.n_units[0])[:, None]
    g.append(jnp.minimum(F32(2), _q(tabs, "q30", jnp.where(valid, raw.inv, 0).sum(0))))
    return jnp.concatenate(g)


def future_features(raw, tabs):
    step = raw.step; day = step // 24
    rate = raw.shops @ _SHOP_RATE                                            # [9] per 4-turn tick
    out = []
    for k in K.HORIZONS:
        n4 = (step + k) // 4 - step // 4
        n24 = (step + k) // 24 - step // 24
        x = raw.mkt_inv - (n4 * rate + n24 * _TOWN_RATE)
        out.append(tabs["price_ratio"][jnp.arange(9), jnp.clip(x - K.PRICE_LO, 0, K.PRICE_N - 1)])
    x = raw.mkt_inv + raw.shed[:9] // 2
    out.append(tabs["price_ratio"][jnp.arange(9), jnp.clip(x - K.PRICE_LO, 0, K.PRICE_N - 1)])
    for f in (0, 1):
        kind, what, is_plant, is_animal = _farm_masks(raw, f)
        w = jnp.clip(what, 0, 4); wa = jnp.clip(what, 0, 2)
        yu = raw.tile_yield[f]
        age = day - raw.tile_day[f]
        first = _CROP_FIRST[w]
        p_now = jnp.where(is_plant & (age >= first), yu, 0)
        p_soon = jnp.where(is_plant & (age < first) & (age + 1 >= first), jnp.maximum(1, yu), 0)
        crop_oh = jax.nn.one_hot(w, 9, dtype=jnp.int32)                     # crop index == product index
        prod_oh = jax.nn.one_hot(_ANIMAL_PROD[wa], 9, dtype=jnp.int32)
        now = (p_now[..., None] * crop_oh).sum((0, 1)) + (jnp.where(is_animal, yu, 0)[..., None] * prod_oh).sum((0, 1))
        fert_now = (is_animal & raw.tile_fert[f]).sum()
        fert_soon = (is_animal & ~raw.tile_fert[f] & raw.tile_fed[f]).sum()
        now = now.at[8].add(fert_now)
        soon = (p_soon[..., None] * crop_oh).sum((0, 1)).at[8].add(fert_soon)
        out.append(jnp.minimum(F32(3), _q(tabs, "q20", now)))
        out.append(jnp.minimum(F32(3), _q(tabs, "q20", soon)))
    return jnp.concatenate(out)                                              # [81]


def encode_units(raw, prev_ops, tabs):
    M = K.MAX_UNITS
    n = jnp.minimum(M, raw.n_units[0])
    k = jnp.arange(M)
    valid = k < n
    xy = raw.unit_xy[0, :M]
    xy = jnp.where(valid[:, None], xy, 0)
    inv = jnp.where(valid[:, None], raw.inv[:M], 0)
    at_shed = _SHED_MASK_J[jnp.clip(xy[:, 1], 0, 9), jnp.clip(xy[:, 0], 0, 9)] & valid
    idx_oh = jnp.eye(M, dtype=F32) * valid[:, None]
    xyf = jnp.where(valid[:, None], _q(tabs, "q9", xy), F32(0))
    invf = jnp.minimum(F32(2), _q(tabs, "q10", inv))
    carrying = ((inv > 0).any(1) & valid).astype(F32)[:, None]
    prev = jnp.where(valid, prev_ops, -1)
    prev_oh = jax.nn.one_hot(prev, K.N_OPS, dtype=F32)                       # -1 -> zeros
    uf = jnp.concatenate([idx_oh, xyf, invf, carrying, at_shed.astype(F32)[:, None], prev_oh], 1)
    return xy.astype(jnp.int32), uf, valid


def encode(raw, prev_ops, tabs=None):
    """ONE game -> dict(planes uint8 [47,10,10], glob f32 [179], pos int32 [16,2], uf f32 [16,76], valid bool [16])."""
    tabs = tabs if tabs is not None else tables_jnp()
    planes = encode_planes(raw, tabs)
    glob = jnp.concatenate([encode_glob5(raw, tabs), future_features(raw, tabs)])
    pos, uf, valid = encode_units(raw, prev_ops, tabs)
    return {"planes": planes, "glob": glob, "pos": pos, "uf": uf, "valid": valid}


def net_inputs(enc, tabs=None):
    """Encoder output (batched) -> (planes float [N,47,10,10] = uint8 / 255 exactly as the reference, glob, pos, uin)."""
    tabs = tabs if tabs is not None else tables_jnp()
    return tabs["q255"][enc["planes"].astype(jnp.int32)], enc["glob"], enc["pos"], enc["uf"]


encode_batch = jax.jit(jax.vmap(encode, in_axes=(0, 0, None)))
