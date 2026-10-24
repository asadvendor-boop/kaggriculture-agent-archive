"""Engine state (kagjax/state.py GameState, spec.md) -> policy RawObs (encode.py), and policy Act -> engine Actions.

Differences bridged here (engine ids are the C++ ones, the policy uses il/features.py codes):
  tile kind   engine 0 EMPTY 1 LOCKED 2 WEED 3 COOP 4 PASTURE 5 PLANT -> features 0 EMPTY 1 LOCKED 2 WEED 3 PLANT 4 COOP 5 PASTURE
  tile what   engine: PLANT crop id 0-4, animal item id 9-11           -> crop 0-4 / animal 0-2 / -1
  shops       engine: unlock list (ids, -1 empty) + n_shops           -> instance count per shop id
  prices      engine: table lookup P[item, minv] (engine.prices)      -> passed in (int32 [9])
  units       engine: 40 per farm, stale beyond n_units               -> same arrays (the encoder masks by n_units)
Everything is per game (no batch dim); use raw_from_state / actions_from_policy for batches.
"""
import numpy as np
import jax
import jax.numpy as jnp
from kagjax import consts as EC
from kagjax.policy import consts as K
from kagjax.policy import encode as E

_KIND_MAP = np.zeros(6, np.int32)
_KIND_MAP[EC.K_EMPTY], _KIND_MAP[EC.K_LOCKED], _KIND_MAP[EC.K_WEED] = K.K_EMPTY, K.K_LOCKED, K.K_WEED
_KIND_MAP[EC.K_COOP], _KIND_MAP[EC.K_PASTURE], _KIND_MAP[EC.K_PLANT] = K.K_COOP, K.K_PASTURE, K.K_PLANT
_KIND_MAP_J = jnp.asarray(_KIND_MAP)
assert EC.ITEM_NAMES == K.ITEMS and EC.SHOP_NAMES == K.SHOPS and EC.MARKET_OPS == list(K.MOP) and EC.UNIT_OPS == list(K.UOP)


def raw_from_game(g, t, seat, prices):
    """ONE game: GameState fields without the batch dim, t int step, seat 0/1 (may be traced), prices int32 [9]
    (engine.prices of this game) -> RawObs from `seat`'s point of view (farm 0 = seat, farm 1 = the rival)."""
    order = jnp.stack([seat, 1 - seat]).astype(jnp.int32)
    tiles = g.tiles[order]                                                   # [2,100,13]
    ke = tiles[..., EC.F_KIND]
    is_plant = ke == EC.K_PLANT
    animal = (tiles[..., EC.F_ANIMAL] == 1) & ((ke == EC.K_COOP) | (ke == EC.K_PASTURE))
    what = jnp.where(is_plant, tiles[..., EC.F_WHAT], jnp.where(animal, tiles[..., EC.F_WHAT] - EC.GOOSE, -1))
    r = lambda a: a.reshape(2, K.B, K.B)
    col = lambda c: r(tiles[..., c])
    shops = (jax.nn.one_hot(g.shops, len(K.SHOPS), dtype=jnp.int32) *
             (jnp.arange(g.shops.shape[0]) < g.n_shops)[:, None]).sum(0)
    return E.RawObs(
        step=jnp.asarray(t, jnp.int32), money=g.money[order].astype(jnp.float32), n_units=g.n_units[order],
        unit_xy=jnp.stack([g.ux[order], g.uy[order]], -1), hires_today=g.hires[order], n_quads=g.n_quads[order],
        tile_kind=r(_KIND_MAP_J[ke]), tile_what=r(what), tile_animal=r(animal), tile_day=col(EC.F_DAY),
        tile_yield=col(EC.F_YIELD), tile_dry=col(EC.F_DRY), tile_watered=col(EC.F_WATERED) != 0,
        tile_fed=col(EC.F_FED) != 0, tile_cared=col(EC.F_CARED) != 0, tile_fert=col(EC.F_FERTAV) != 0,
        tile_bonus=col(EC.F_CARE), tile_fert_until=col(EC.F_FERTU), tile_life=col(EC.F_MLS),
        shed=g.shed[seat], seeds=g.seeds[seat], inv=g.inv[seat], mkt_inv=g.minv, mkt_price=prices, shops=shops)


def raw_from_state(g, t, seat, prices):
    """Batched GameState g [B,...], scalar t, seat (int or int [B]), prices [B,9] -> RawObs [B,...]."""
    seat = jnp.broadcast_to(jnp.asarray(seat, jnp.int32), g.money.shape[:1])
    return jax.vmap(raw_from_game, in_axes=(0, None, 0, 0))(g, t, seat, prices)


def raw_both_seats(g, t, prices):
    """Batched GameState [B,...] -> RawObs [2B,...]: rows 0..B-1 = seat 0, rows B..2B-1 = seat 1 (one policy forward
    for both players of every game)."""
    b = g.money.shape[0]
    both = jax.tree_util.tree_map(lambda a: jnp.concatenate([a, a]), g)
    seats = jnp.concatenate([jnp.zeros(b, jnp.int32), jnp.ones(b, jnp.int32)])
    return raw_from_state(both, t, seats, jnp.concatenate([prices, prices]))


def actions_from_policy(act, u_total=EC.U):
    """policy Act [B,...] (16 unit commands, 10 orders) -> engine state.Actions [B,...] (40 unit slots; units beyond
    the 16 the network sees PASS, as the Python agent sends no command for them)."""
    from kagjax.state import Actions
    b = act.u_op.shape[0]
    pad = lambda a, v: jnp.concatenate([a, jnp.full((b, u_total - a.shape[1]), v, a.dtype)], 1)
    return Actions(uop=pad(act.u_op, EC.OP_PASS), uarg=pad(act.u_arg, 0), un=pad(act.u_n, 1),
                   mtype=act.o_op, mitem=act.o_item, mn=act.o_n)
