"""Pure-JAX, batched Kaggriculture engine (rules 1.32.7), faithful to engine/src_v2/sim_trace.hpp.

API (all arrays batched on the leading dim B; the step index t is shared by the whole batch):
    state = init(B, seed=0)                          # EnvState at step 0 (keys for training randomness)
    state = step(state, a0, a1, words=None)          # one environment step for every game
    money = rewards(state)                           # int32[B,2] current banks (== final reward when done)
    done(state)                                      # python bool: the episode is over (t > 718)

`words` (uint32[B,624]) is the randomness of the end-of-day generator: the outputs of CPython's
random.Random((seed*1000003) ^ day), i.e. rng.getrandbits(32) x 624 (see rng.py). Only read on day-boundary
steps ((t+1) % 24 == 0). Pass the exact words for bit-exact parity with the reference engine; pass None to draw
i.i.d. uniform words from the per-game PRNG key in the state (same distribution: weeds p=0.005 per empty tile,
uniform shop draws), which is what training uses.
"""
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

try:
    from . import consts as C
    from .state import GameState, EnvState, Actions, initial_game, stack_games
except ImportError:
    import consts as C
    from state import GameState, EnvState, Actions, initial_game, stack_games

i32 = jnp.int32


class Tabs(NamedTuple):
    P: jnp.ndarray       # int32[9*NTAB]   price at inventory v:  P[j*NTAB + v-INV_LO]
    CS: jnp.ndarray      # int32[9*NTAB]   exclusive prefix sums (mod 2^32)
    E2: jnp.ndarray      # int32[9*NTAB]   exclusive stride-2 prefix sums (mod 2^32)
    FLOOR: jnp.ndarray   # int32[9]        first inventory with price 1


_TABS = None


def tables():
    global _TABS
    if _TABS is None:
        P, CS, E2, FL = C.build_price_tables()
        _TABS = Tabs(jnp.asarray(P.reshape(-1)), jnp.asarray(CS.reshape(-1)), jnp.asarray(E2.reshape(-1)),
                     jnp.asarray(FL))
    return _TABS


def _look(tab, j, v):
    return tab[j * C.NTAB + jnp.clip(v - C.INV_LO, 0, C.NTAB - 1)]


# small constant tables (become literals inside jit)
CROP_SEED = jnp.asarray(C.CROP_SEED)
CROP_FYD = jnp.asarray(C.CROP_FYD)
CROP_MAXYD = jnp.asarray(C.CROP_MAXYD)
CROP_INTERVAL = jnp.asarray(C.CROP_INTERVAL)
CROP_MAXY = jnp.asarray(C.CROP_MAXY)
CROP_ONGOING = jnp.asarray(C.CROP_ONGOING)
ANIMAL_COST = jnp.asarray(C.ANIMAL_COST)
ANIMAL_FYD = jnp.asarray(C.ANIMAL_FYD)
ANIMAL_INTERVAL = jnp.asarray(C.ANIMAL_INTERVAL)
ANIMAL_MAXHELD = jnp.asarray(C.ANIMAL_MAXHELD)
ANIMAL_PRODUCT = jnp.asarray(C.ANIMAL_PRODUCT)
FIB = jnp.asarray(C.FIB)
LAND_PRICES = jnp.asarray(C.LAND_PRICES)
QUAD_OF_TILE = jnp.asarray(C.QUAD_OF_TILE)
SHOP_DRAIN = jnp.asarray(C.SHOP_DRAIN)
CENTER_DRAIN = jnp.asarray(C.CENTER_DRAIN)
AX = jnp.asarray([p[0] for p in C.SHED_ACCESS], i32)
AY = jnp.asarray([p[1] for p in C.SHED_ACCESS], i32)
FRESH = jnp.asarray(C.FRESH_TILE)


def _fresh(kind):
    return FRESH.at[C.F_KIND].set(kind)


# ============================================================================================ unit actions
def _unit_phase(tiles, ux, uy, shed, seeds, inv, ins, uop, uarg, un, n_units, day, t, umax, unroll=1):
    """Sequential per-unit actions of ONE farm (C++ apply_unit_actions). Shapes: tiles[100,NF], ux/uy[U],
    shed[12], seeds[5], inv/ins[U,12], uop/uarg/un[U]. `umax` (unbatched) bounds the loop."""
    cr = jnp.arange(C.N_CROPS)
    items = jnp.arange(C.N_ITEMS)
    # Atomic PLANT validation over every submitted unit action (even for units that do not exist).
    is_pl = (uop == C.OP_PLANT) & (uarg >= 0) & (uarg < C.N_CROPS)
    demand = jnp.sum((is_pl[:, None] & (uarg[:, None] == cr[None, :])).astype(i32), axis=0)
    blocked = demand > seeds
    uop = jnp.where(is_pl & blocked[jnp.clip(uarg, 0, C.N_CROPS - 1)], C.OP_PASS, uop)

    def body(i, c):
        tiles, ux, uy, shed, seeds, inv, ins = c
        op = jnp.where(i < n_units, uop[i], C.OP_PASS)
        arg = uarg[i]
        n = un[i]
        x = ux[i]
        y = uy[i]
        # ---------------- movement (locked tiles are passable; off-board is a no-op)
        dx = (op == C.OP_EAST).astype(i32) - (op == C.OP_WEST).astype(i32)
        dy = (op == C.OP_SOUTH).astype(i32) - (op == C.OP_NORTH).astype(i32)
        nx = x + dx
        ny = y + dy
        mv = (op >= C.OP_NORTH) & (op <= C.OP_WEST) & (nx >= 0) & (nx < C.BOARD) & (ny >= 0) & (ny < C.BOARD)
        ux = ux.at[i].set(jnp.where(mv, nx, x))
        uy = uy.at[i].set(jnp.where(mv, ny, y))
        # ---------------- the tile under the unit (pre-move position; moves leave it unchanged)
        pos = y * C.BOARD + x
        tile = tiles[pos]
        kind = tile[C.F_KIND]
        what = tile[C.F_WHAT]
        animal = tile[C.F_ANIMAL] == 1
        watered = tile[C.F_WATERED] == 1
        fed = tile[C.F_FED] == 1
        cared = tile[C.F_CARED] == 1
        fertav = tile[C.F_FERTAV] == 1
        yld = tile[C.F_YIELD]
        pday = tile[C.F_DAY]
        fertu = tile[C.F_FERTU]
        adj = ((x == 4) | (x == 5)) & ((y == 4) | (y == 5))
        irow = inv[i]
        srow = ins[i]
        room = jnp.maximum(0, C.SHED_CAP - jnp.sum(shed))
        arg_ok = (arg >= 0) & (arg < C.N_ITEMS)
        ac = jnp.clip(arg, 0, C.N_ITEMS - 1)
        unlocked = kind != C.K_LOCKED

        # DROP: whole inventory into the shed in insertion order; overflow is discarded
        is_drop = (op == C.OP_DROP) & adj
        before = jnp.sum(jnp.where(srow[None, :] < srow[:, None], irow[None, :], 0), axis=1)
        dtake = jnp.minimum(irow, jnp.maximum(0, room - before))
        # PICKUP
        ptake = jnp.minimum(n, shed[ac])
        is_pick = (op == C.OP_PICKUP) & adj & (n > 0) & arg_ok & (ptake > 0)
        # PLACE: animal onto a matching empty structure (returns even if the unit has none), else into the shed
        is_place = (op == C.OP_PLACE) & arg_ok
        is_an_item = (arg >= C.GOOSE) & (arg <= C.SHEEP)
        want = jnp.where(arg == C.GOOSE, C.K_COOP, C.K_PASTURE)
        an_branch = is_place & is_an_item & (kind == want) & ~animal
        an_ok = an_branch & (irow[ac] >= 1)
        n2 = jnp.minimum(jnp.minimum(n, irow[ac]), room)
        pl_shed = is_place & ~an_branch & adj & (n2 > 0)
        # PLANT (blocked crops were turned into PASS above)
        acr = jnp.clip(arg, 0, C.N_CROPS - 1)
        is_plant = ((op == C.OP_PLANT) & unlocked & (arg >= 0) & (arg < C.N_CROPS) & (kind == C.K_EMPTY)
                    & (seeds[acr] > 0))
        # WATER
        wc = jnp.clip(what, 0, C.N_CROPS - 1)
        ongo = CROP_ONGOING[wc] == 1
        is_water = (op == C.OP_WATER) & unlocked & (kind == C.K_PLANT) & ~watered
        age = day - pday
        maxyd = CROP_MAXYD[wc]
        inwin = ~ongo & (age >= (maxyd + 1) // 2) & (age <= maxyd)
        wyld = jnp.where(inwin, jnp.minimum(CROP_MAXY[wc], yld + jnp.where(fertu >= day, 2, 1)), yld)
        # HARVEST
        hv = (op == C.OP_HARVEST) & unlocked & (kind != C.K_EMPTY) & (kind != C.K_WEED) & (yld > 0)
        hv_plant = hv & (kind == C.K_PLANT) & (day - pday >= CROP_FYD[wc])
        hv_an = hv & (kind != C.K_PLANT) & animal
        hv_item = jnp.where(kind == C.K_PLANT, what, ANIMAL_PRODUCT[jnp.clip(what - C.GOOSE, 0, 2)])
        # FERTILIZE / DIG / BUILD / FEED / COLLECT / CARE
        is_fert = (op == C.OP_FERTILIZE) & unlocked & (kind == C.K_PLANT) & (irow[C.FERTILIZER] >= 1)
        is_dig = (op == C.OP_DIG) & unlocked & (kind != C.K_EMPTY) & ~animal
        is_coop = (op == C.OP_BUILD_COOP) & unlocked & (kind == C.K_EMPTY)
        is_past = (op == C.OP_BUILD_PASTURE) & unlocked & (kind == C.K_EMPTY)
        is_feed = (op == C.OP_FEED) & unlocked & animal & ~fed & (irow[C.WHEAT] >= 1)
        is_coll = (op == C.OP_COLLECT_FERTILIZER) & unlocked & animal & fertav
        is_care = (op == C.OP_CARE) & unlocked & animal & ~cared

        # ---------------- new tile row
        animal_row = (_fresh(want).at[C.F_WHAT].set(arg).at[C.F_ANIMAL].set(1).at[C.F_DAY].set(day))
        p_ongo = CROP_ONGOING[acr] == 1
        plant_row = (_fresh(C.K_PLANT).at[C.F_WHAT].set(acr).at[C.F_DAY].set(day).at[C.F_DRY].set(1)
                     .at[C.F_YIELD].set(jnp.where(p_ongo, 0, 1))
                     .at[C.F_MLS].set(jnp.where(p_ongo, -1, (day + CROP_MAXYD[acr] + 1) * C.TURNS_PER_DAY)))
        new = tile
        new = jnp.where(an_ok, animal_row, new)
        new = jnp.where(is_plant, plant_row, new)
        new = jnp.where(is_water, tile.at[C.F_WATERED].set(1).at[C.F_YIELD].set(wyld), new)
        new = jnp.where(hv_plant | hv_an,
                        jnp.where(hv_plant & ~ongo, _fresh(C.K_EMPTY), tile.at[C.F_YIELD].set(0)), new)
        new = jnp.where(is_fert, tile.at[C.F_FERTU].set(jnp.maximum(fertu, day + 2)), new)
        new = jnp.where(is_dig, _fresh(C.K_EMPTY), new)
        new = jnp.where(is_coop, _fresh(C.K_COOP), new)
        new = jnp.where(is_past, _fresh(C.K_PASTURE), new)
        new = jnp.where(is_feed, tile.at[C.F_FED].set(1), new)
        new = jnp.where(is_coll, tile.at[C.F_FERTAV].set(0), new)
        new = jnp.where(is_care, tile.at[C.F_CARED].set(1), new)
        tiles = tiles.at[pos].set(new)

        # ---------------- unit inventory (one add or one take or a full drop per action)
        add_item = jnp.where(is_pick, ac, jnp.where(is_coll, C.FERTILIZER, hv_item))
        add_amt = jnp.where(is_pick, ptake, jnp.where(is_coll, 1, jnp.where(hv_plant | hv_an, yld, 0)))
        take_item = jnp.where(is_fert, C.FERTILIZER, jnp.where(is_feed, C.WHEAT, ac))
        take_amt = jnp.where(is_fert | is_feed | an_ok, 1, jnp.where(pl_shed, n2, 0))
        irow2 = jnp.where(is_drop, 0, irow)
        irow2 = irow2 - jnp.where(items == take_item, take_amt, 0)
        was0 = irow2[jnp.clip(add_item, 0, C.N_ITEMS - 1)] == 0
        irow2 = irow2 + jnp.where(items == add_item, add_amt, 0)
        srow2 = jnp.where((items == add_item) & (add_amt > 0) & was0, t, srow)
        inv = inv.at[i].set(irow2)
        ins = ins.at[i].set(srow2)
        # ---------------- shed / seeds
        shed = (shed + jnp.where(is_drop, dtake, 0)
                + jnp.where(items == ac, jnp.where(pl_shed, n2, 0) - jnp.where(is_pick, ptake, 0), 0))
        seeds = seeds - jnp.where((cr == acr) & is_plant, 1, 0)
        return tiles, ux, uy, shed, seeds, inv, ins

    carry = (tiles, ux, uy, shed, seeds, inv, ins)
    if isinstance(umax, int):                          # static trip count (optionally unrolled)
        return lax.fori_loop(0, umax, body, carry, unroll=unroll)
    return lax.fori_loop(0, umax, body, carry)


# ------------------------------------------------------------------------ parallel (round-based) unit phase
# Exactly equivalent to the sequential loop above, with far fewer sequential iterations:
#  * a unit acts once per turn and only ever touches (a) its own position, (b) its own inventory, (c) the tile it
#    stands on (tile ops), (d) the shared shed (PICKUP / DROP / PLACE into the shed), (e) seeds (PLANT).
#  * moves are independent of everything -> all in parallel.
#  * tile ops on DIFFERENT tiles commute; on the SAME tile they must run in unit order -> "rounds": round r runs,
#    in parallel, the r-th tile-op unit of every tile (usually 1-2 rounds).
#  * PLANT's seed check cannot fail for a crop that passed the atomic check (seeds >= demand >= plants), so seed
#    consumption is order-free.
#  * shed ops only interact with other shed ops (and the unit's own inventory, untouched by others) -> one
#    sequential pass over the shed-op units only, in unit order. PLACE of an animal is a tile op when the unit
#    stands on a matching empty structure at its turn, otherwise it falls through to the shed path (C++ order).
_TILE_OPS = (C.OP_PLANT, C.OP_WATER, C.OP_HARVEST, C.OP_FERTILIZE, C.OP_DIG, C.OP_BUILD_COOP, C.OP_BUILD_PASTURE,
             C.OP_FEED, C.OP_COLLECT_FERTILIZER, C.OP_CARE)


def _unit_prep(tiles, ux, uy, seeds, uop, uarg, un, n_units):
    """Shared classification of ONE farm's unit actions (vectorised over units)."""
    uidx = jnp.arange(C.U)
    cr = jnp.arange(C.N_CROPS)
    is_pl = (uop == C.OP_PLANT) & (uarg >= 0) & (uarg < C.N_CROPS)
    demand = jnp.sum((is_pl[:, None] & (uarg[:, None] == cr[None, :])).astype(i32), axis=0)
    blocked = demand > seeds
    op = jnp.where(is_pl & blocked[jnp.clip(uarg, 0, C.N_CROPS - 1)], C.OP_PASS, uop)
    op = jnp.where(uidx < n_units, op, C.OP_PASS)
    pos = uy * C.BOARD + ux
    kind0 = tiles[pos, C.F_KIND]                        # tile kinds never change LOCKED<->other in this phase
    unlocked = kind0 != C.K_LOCKED
    adj = ((ux == 4) | (ux == 5)) & ((uy == 4) | (uy == 5))
    arg_ok = (uarg >= 0) & (uarg < C.N_ITEMS)
    is_an_item = (uarg >= C.GOOSE) & (uarg <= C.SHEEP)
    tileop = jnp.zeros(C.U, bool)
    for o in _TILE_OPS:
        tileop = tileop | (op == o)
    tileop = tileop & ~((op == C.OP_PLANT) & ~((uarg >= 0) & (uarg < C.N_CROPS)))
    tileop = (tileop | ((op == C.OP_PLACE) & is_an_item)) & unlocked
    same = (pos[:, None] == pos[None, :]) & tileop[None, :] & (uidx[None, :] < uidx[:, None])
    rank = jnp.where(tileop, jnp.sum(same.astype(i32), axis=1), -1)
    # shed-op candidates (a PLACE of an animal becomes one only if it did not take the animal branch)
    shedcand = adj & ((op == C.OP_DROP) | ((op == C.OP_PICKUP) & (un > 0) & arg_ok) | ((op == C.OP_PLACE) & arg_ok))
    return op, pos, adj, arg_ok, tileop, rank, shedcand


def unit_bounds(g, acts):
    """Batch-level loop bounds for the parallel unit phase: (rounds, shed ops) maxima (unbatched scalars)."""
    f = jax.vmap(jax.vmap(_unit_prep))
    op, pos, adj, arg_ok, tileop, rank, shedcand = f(g.tiles, g.ux, g.uy, g.seeds, acts.uop, acts.uarg, acts.un,
                                                     g.n_units)
    return jnp.max(rank) + 1, jnp.max(jnp.sum(shedcand.astype(i32), axis=-1))


def _unit_phase_par(tiles, ux, uy, shed, seeds, inv, ins, uop, uarg, un, n_units, day, t, rmax, kmax):
    """Round-based unit phase of ONE farm; identical results to _unit_phase. rmax / kmax: unbatched bounds from
    unit_bounds (or python ints)."""
    op, pos, adj, arg_ok, tileop, rank, shedcand = _unit_prep(tiles, ux, uy, seeds, uop, uarg, un, n_units)
    items = jnp.arange(C.N_ITEMS)
    cr = jnp.arange(C.N_CROPS)
    arg = uarg
    ac = jnp.clip(arg, 0, C.N_ITEMS - 1)
    acr = jnp.clip(arg, 0, C.N_CROPS - 1)
    # ---------------- moves (independent of everything else)
    dx = (op == C.OP_EAST).astype(i32) - (op == C.OP_WEST).astype(i32)
    dy = (op == C.OP_SOUTH).astype(i32) - (op == C.OP_NORTH).astype(i32)
    nx = ux + dx
    ny = uy + dy
    mv = (op >= C.OP_NORTH) & (op <= C.OP_WEST) & (nx >= 0) & (nx < C.BOARD) & (ny >= 0) & (ny < C.BOARD)
    ux2 = jnp.where(mv, nx, ux)
    uy2 = jnp.where(mv, ny, uy)

    # ---------------- tile-op rounds
    def rnd(r, c):
        tiles, inv, ins, planted, anb = c
        sel = tileop & (rank == r)
        tile = tiles[pos]                                   # [U,NF] current rows under every unit
        kind = tile[:, C.F_KIND]
        what = tile[:, C.F_WHAT]
        animal = tile[:, C.F_ANIMAL] == 1
        watered = tile[:, C.F_WATERED] == 1
        fed = tile[:, C.F_FED] == 1
        cared = tile[:, C.F_CARED] == 1
        fertav = tile[:, C.F_FERTAV] == 1
        yld = tile[:, C.F_YIELD]
        pday = tile[:, C.F_DAY]
        fertu = tile[:, C.F_FERTU]
        irow = inv
        want = jnp.where(arg == C.GOOSE, C.K_COOP, C.K_PASTURE)
        an_branch = sel & (op == C.OP_PLACE) & (kind == want) & ~animal
        an_ok = an_branch & (jnp.take_along_axis(irow, ac[:, None], 1)[:, 0] >= 1)
        is_plant = sel & (op == C.OP_PLANT) & (kind == C.K_EMPTY)
        wc = jnp.clip(what, 0, C.N_CROPS - 1)
        ongo = CROP_ONGOING[wc] == 1
        is_water = sel & (op == C.OP_WATER) & (kind == C.K_PLANT) & ~watered
        age = day - pday
        maxyd = CROP_MAXYD[wc]
        inwin = ~ongo & (age >= (maxyd + 1) // 2) & (age <= maxyd)
        wyld = jnp.where(inwin, jnp.minimum(CROP_MAXY[wc], yld + jnp.where(fertu >= day, 2, 1)), yld)
        hv = sel & (op == C.OP_HARVEST) & (kind != C.K_EMPTY) & (kind != C.K_WEED) & (yld > 0)
        hv_plant = hv & (kind == C.K_PLANT) & (day - pday >= CROP_FYD[wc])
        hv_an = hv & (kind != C.K_PLANT) & animal
        hv_item = jnp.where(kind == C.K_PLANT, what, ANIMAL_PRODUCT[jnp.clip(what - C.GOOSE, 0, 2)])
        is_fert = sel & (op == C.OP_FERTILIZE) & (kind == C.K_PLANT) & (irow[:, C.FERTILIZER] >= 1)
        is_dig = sel & (op == C.OP_DIG) & (kind != C.K_EMPTY) & ~animal
        is_coop = sel & (op == C.OP_BUILD_COOP) & (kind == C.K_EMPTY)
        is_past = sel & (op == C.OP_BUILD_PASTURE) & (kind == C.K_EMPTY)
        is_feed = sel & (op == C.OP_FEED) & animal & ~fed & (irow[:, C.WHEAT] >= 1)
        is_coll = sel & (op == C.OP_COLLECT_FERTILIZER) & animal & fertav
        is_care = sel & (op == C.OP_CARE) & animal & ~cared
        fresh = jnp.broadcast_to(FRESH, tile.shape)
        colk = jnp.arange(C.NF)[None, :]
        setc = lambda base, col, val: jnp.where(colk == col, val[:, None] if jnp.ndim(val) else val, base)
        animal_row = setc(setc(setc(setc(fresh, C.F_KIND, want), C.F_WHAT, arg), C.F_ANIMAL, 1), C.F_DAY, day)
        p_ongo = CROP_ONGOING[acr] == 1
        plant_row = setc(setc(setc(setc(setc(setc(fresh, C.F_KIND, jnp.full(C.U, C.K_PLANT)), C.F_WHAT, acr),
                                             C.F_DAY, day), C.F_DRY, 1), C.F_YIELD, jnp.where(p_ongo, 0, 1)),
                         C.F_MLS, jnp.where(p_ongo, -1, (day + CROP_MAXYD[acr] + 1) * C.TURNS_PER_DAY))
        empty_row = setc(fresh, C.F_KIND, C.K_EMPTY)
        new = tile
        new = jnp.where(an_ok[:, None], animal_row, new)
        new = jnp.where(is_plant[:, None], plant_row, new)
        new = jnp.where(is_water[:, None], setc(setc(tile, C.F_WATERED, 1), C.F_YIELD, wyld), new)
        new = jnp.where((hv_plant | hv_an)[:, None],
                        jnp.where((hv_plant & ~ongo)[:, None], empty_row, setc(tile, C.F_YIELD, 0)), new)
        new = jnp.where(is_fert[:, None], setc(tile, C.F_FERTU, jnp.maximum(fertu, day + 2)), new)
        new = jnp.where(is_dig[:, None], empty_row, new)
        new = jnp.where(is_coop[:, None], setc(fresh, C.F_KIND, C.K_COOP), new)
        new = jnp.where(is_past[:, None], setc(fresh, C.F_KIND, C.K_PASTURE), new)
        new = jnp.where(is_feed[:, None], setc(tile, C.F_FED, 1), new)
        new = jnp.where(is_coll[:, None], setc(tile, C.F_FERTAV, 0), new)
        new = jnp.where(is_care[:, None], setc(tile, C.F_CARED, 1), new)
        tiles = tiles.at[jnp.where(sel, pos, C.NT)].set(new, mode="drop")   # distinct tiles within a round
        add_item = jnp.where(is_coll, C.FERTILIZER, hv_item)
        add_amt = jnp.where(is_coll, 1, jnp.where(hv_plant | hv_an, yld, 0))
        take_item = jnp.where(is_fert, C.FERTILIZER, jnp.where(is_feed, C.WHEAT, ac))
        take_amt = jnp.where(is_fert | is_feed | an_ok, 1, 0)
        irow2 = irow - jnp.where(items[None, :] == take_item[:, None], take_amt[:, None], 0)
        was0 = jnp.take_along_axis(irow2, jnp.clip(add_item, 0, C.N_ITEMS - 1)[:, None], 1)[:, 0] == 0
        irow2 = irow2 + jnp.where(items[None, :] == add_item[:, None], add_amt[:, None], 0)
        ins = jnp.where((items[None, :] == add_item[:, None]) & ((add_amt > 0) & was0)[:, None], t, ins)
        planted = planted + jnp.sum((is_plant[:, None] & (acr[:, None] == cr[None, :])).astype(i32), axis=0)
        anb = anb | an_branch
        return tiles, irow2, ins, planted, anb

    carry = (tiles, inv, ins, jnp.zeros(C.N_CROPS, i32), jnp.zeros(C.U, bool))
    tiles, inv, ins, planted, anb = lax.fori_loop(0, rmax, rnd, carry)
    seeds = seeds - planted

    # ---------------- shed pass: shed ops in unit order
    shedop = shedcand & ~anb
    srank = jnp.cumsum(shedop.astype(i32)) - 1
    order = jnp.full(C.U, C.U, i32).at[jnp.where(shedop, srank, C.U)].set(jnp.arange(C.U, dtype=i32), mode="drop")

    def sbody(k, c):
        shed, inv, ins = c
        u = order[k]
        valid = u < C.U
        uc = jnp.clip(u, 0, C.U - 1)
        o = op[uc]
        a = ac[uc]
        n = un[uc]
        irow = inv[uc]
        srow = ins[uc]
        room = jnp.maximum(0, C.SHED_CAP - jnp.sum(shed))
        is_drop = valid & (o == C.OP_DROP)
        before = jnp.sum(jnp.where(srow[None, :] < srow[:, None], irow[None, :], 0), axis=1)
        dtake = jnp.minimum(irow, jnp.maximum(0, room - before))
        ptake = jnp.minimum(n, shed[a])
        is_pick = valid & (o == C.OP_PICKUP) & (ptake > 0)
        n2 = jnp.minimum(jnp.minimum(n, irow[a]), room)
        pl_shed = valid & (o == C.OP_PLACE) & (n2 > 0)
        irow2 = jnp.where(is_drop, 0, irow)
        irow2 = irow2 - jnp.where((items == a) & pl_shed, n2, 0)
        was0 = irow2[a] == 0
        irow2 = irow2 + jnp.where((items == a) & is_pick, ptake, 0)
        srow2 = jnp.where((items == a) & is_pick & was0, t, srow)
        shed = (shed + jnp.where(is_drop, dtake, 0)
                + jnp.where(items == a, jnp.where(pl_shed, n2, 0) - jnp.where(is_pick, ptake, 0), 0))
        inv = inv.at[uc].set(irow2)
        ins = ins.at[uc].set(srow2)
        return shed, inv, ins

    shed, inv, ins = lax.fori_loop(0, kmax, sbody, (shed, inv, ins))
    return tiles, ux2, uy2, shed, seeds, inv, ins


# ============================================================================================ market
def _bsearch(cap, ok):
    """max q in [0, cap] (cap <= 127) with ok(q), for ok monotone (true on a prefix, ok(0) true).
    Binary search: 7 dependent steps (cheap on CPU)."""
    lo = jnp.zeros_like(cap)
    for bit in (64, 32, 16, 8, 4, 2, 1):
        cand = lo + bit
        lo = jnp.where((cand <= cap) & ok(cand), cand, lo)
    return lo


def _pcount(cap, ok):
    """Same result as _bsearch by counting: ok is true exactly on a prefix, so max q = #{1<=q<=cap: ok(q)}.
    One wide gather + reduction instead of 7 dependent steps (fewer kernels on GPU). cap: [...]; ok(q) must
    broadcast q of shape [..., 127]."""
    q = jnp.arange(1, 128, dtype=i32)
    q = jnp.broadcast_to(q, cap.shape + (127,))
    return jnp.sum(((q <= cap[..., None]) & ok(q)).astype(i32), axis=-1)


def _market_phase(tb, money, shed, seeds, minv, hires, n_units, ux, uy, n_quads, mt, mi, mn, smax, unroll=1,
                  count_search=False):
    """C++ process_market for ONE game: both players' order lists slot by slot (per-unit lockstep), in closed form.
    money/hires/n_units/n_quads[2], shed[2,12], seeds[2,5], minv[9], ux/uy[2,U], mt/mi/mn[2,10]."""
    items = jnp.arange(C.N_ITEMS)
    uidx = jnp.arange(C.U)
    pidx = jnp.arange(2)
    search = _pcount if count_search else _bsearch

    def bc(x, q):                     # broadcast a per-player value against q ([2] for _bsearch, [2,127] else)
        return x[:, None] if (count_search and jnp.ndim(x) == 1) else x

    def slot(s, c):
        money, shed, seeds, minv, hires, n_units, ux, uy, n_quads = c
        typ = mt[:, s]
        item = mi[:, s]
        cnt = mn[:, s]
        # ---------------- atomic orders (players are independent here)
        cost = FIB[jnp.clip(hires, 0, C.U + 1)]
        ok_h = (typ == C.M_HIRE) & (money >= cost) & (n_units < C.U)
        money = money - jnp.where(ok_h, cost, 0)
        hires = hires + ok_h.astype(i32)
        um = uidx[None, :] < n_units[:, None]
        occ = jnp.sum((um[:, :, None] & (ux[:, :, None] == AX[None, None, :])
                       & (uy[:, :, None] == AY[None, None, :])).astype(i32), axis=1)      # [2,4]
        best = jnp.argmin(occ, axis=1)                                                    # first minimum
        newu = ok_h[:, None] & (uidx[None, :] == n_units[:, None])
        ux = jnp.where(newu, AX[best][:, None], ux)
        uy = jnp.where(newu, AY[best][:, None], uy)
        n_units = n_units + ok_h.astype(i32)
        extra = n_quads - 1
        lp = LAND_PRICES[jnp.clip(extra, 0, 2)]
        ok_l = (typ == C.M_BUY_LAND) & (extra < 3) & (money >= lp)
        money = money - jnp.where(ok_l, lp, 0)
        n_quads = n_quads + ok_l.astype(i32)          # tiles are unlocked after the market phase

        # ---------------- unit-priced orders
        live = (typ >= C.M_BUY_SEED) & (cnt > 0)
        room = jnp.maximum(0, C.SHED_CAP - jnp.sum(shed, axis=1))
        # BUY_SEED / BUY_ANIMAL: fixed price, no market inventory -> independent of the other player
        seed_ok = live & (typ == C.M_BUY_SEED) & (item >= 0) & (item < C.N_CROPS)
        ic = jnp.clip(item, 0, C.N_CROPS - 1)
        sp = CROP_SEED[ic]
        k_seed = jnp.where(seed_ok, jnp.minimum(cnt, money // sp), 0)
        money = money - k_seed * sp
        seeds = seeds + jnp.where(jnp.arange(C.N_CROPS)[None, :] == ic[:, None], k_seed[:, None], 0)
        an_ok = live & (typ == C.M_BUY_ANIMAL) & (item >= C.GOOSE) & (item <= C.SHEEP)
        ap = ANIMAL_COST[jnp.clip(item - C.GOOSE, 0, 2)]
        k_an = jnp.where(an_ok, jnp.minimum(jnp.minimum(cnt, money // ap), room), 0)
        money = money - k_an * ap
        shed = shed + jnp.where(items[None, :] == item[:, None], k_an[:, None], 0)

        # SELL / BUY_PRODUCT: price depends on the shared inventory; players interact only on the same item
        sell = live & (typ == C.M_SELL) & (item >= 0) & (item < C.N_PRODUCTS)
        buy = live & (typ == C.M_BUY_PRODUCT) & ((item == C.WHEAT) | (item == C.FERTILIZER))
        mk = sell | buy
        j = jnp.clip(item, 0, C.N_PRODUCTS - 1)
        I = minv[j]
        shed_j = shed[pidx, j]
        ksell = jnp.where(sell, jnp.minimum(cnt, shed_j), 0)       # a SELL fails only when the shed runs out
        capb = jnp.where(buy, jnp.minimum(cnt, room), 0)            # a BUY also needs cash (searched below)
        inter = mk[0] & mk[1] & (j[0] == j[1])
        J = j[0]
        Iv = I[0]
        F = tb.FLOOR[J]
        # -- both SELL: same quote to both, inventory +2 per unit pair while price > 1
        ss = inter & sell[0] & sell[1]
        m_ss = jnp.minimum(ksell[0], ksell[1])
        tstar = jnp.where(Iv < F, (F - Iv + 1) // 2, 0)
        a_ss = jnp.minimum(m_ss, tstar)
        rev_ss = (_look(tb.E2, J, Iv + 2 * a_ss) - _look(tb.E2, J, Iv)) + (m_ss - a_ss)
        # -- both BUY_PRODUCT: both pay P(inv-1), inventory -2 per pair
        bb = inter & buy[0] & buy[1]
        base2 = _look(tb.E2, J, Iv + 1)
        J2 = search(capb, lambda q: base2 - _look(tb.E2, J, Iv + 1 - 2 * q) <= bc(money, q))
        m_bb = jnp.minimum(J2[0], J2[1])
        cost_bb = base2 - _look(tb.E2, J, Iv + 1 - 2 * m_bb)
        # -- one SELL + one BUY_PRODUCT: inventory stays put while the sale price > 1
        mx = inter & (sell[0] != sell[1])
        s0 = sell[0]
        ks = jnp.where(s0, ksell[0], ksell[1])
        cb = jnp.where(s0, capb[1], capb[0])
        mb = jnp.where(s0, money[1], money[0])
        floor_case = Iv >= F
        L = Iv - F + 1                                 # $1-sale units until the buyer pulls inventory under F
        C1L = _look(tb.CS, J, Iv) - _look(tb.CS, J, Iv - L)
        phA = floor_case & (ks >= L) & (cb >= L) & (C1L <= mb)
        par = mx & floor_case & ~phA                   # descent never completes: equivalent to two solo orders
        S = jnp.where(phA, F - 1, Iv)
        pre = jnp.where(phA, L, 0)
        pre_cost = jnp.where(phA, C1L, 0)
        ks2 = ks - pre
        cb2 = cb - pre
        mb2 = mb - pre_cost
        ps = _look(tb.P, J, S)
        pb = _look(tb.P, J, S - 1)
        Jb = jnp.minimum(cb2, mb2 // pb)
        m_mx = jnp.minimum(ks2, Jb)
        mxe = mx & ~par
        inter_eff = inter & ~par
        seller_p = mxe & sell
        buyer_p = mxe & buy
        jrev = jnp.where(ss, rev_ss, 0) + jnp.where(seller_p, pre + m_mx * ps, 0)
        jcost = jnp.where(bb, cost_bb, 0) + jnp.where(buyer_p, pre_cost + m_mx * pb, 0)
        jsold = jnp.where(ss, m_ss, 0) + jnp.where(seller_p, pre + m_mx, 0)
        jbought = jnp.where(bb, m_bb, 0) + jnp.where(buyer_p, pre + m_mx, 0)
        jdelta = jnp.where(ss, 2 * a_ss, 0) + jnp.where(bb, -2 * m_bb, 0) + jnp.where(mxe, S - Iv, 0)
        stopped_b = (bb & (J2 == m_bb)) | (buyer_p & (Jb == m_mx))
        # ---------------- solo phase (after the joint phase, or the whole order when not interacting)
        start = jnp.where(inter_eff, Iv + jdelta, I)
        srem = ksell - jsold
        Fp = tb.FLOOR[j]
        k1 = jnp.clip(Fp - start, 0, srem)                 # units sold at a price > 1
        srev = (_look(tb.CS, j, start + k1) - _look(tb.CS, j, start)) + (srem - k1)
        bcap = jnp.where(stopped_b, 0, capb - jbought)
        bmoney = money - jcost
        base1 = _look(tb.CS, j, start)
        qb = search(bcap, lambda q: bc(base1, q) - _look(tb.CS, bc(j, q), bc(start, q) - q) <= bc(bmoney, q))
        bcost = base1 - _look(tb.CS, j, start - qb)
        sdelta = jnp.where(sell, k1, 0) - jnp.where(buy, qb, 0)
        money = money + jrev - jcost + jnp.where(sell, srev, 0) - jnp.where(buy, bcost, 0)
        dshed = jbought - jsold - jnp.where(sell, srem, 0) + jnp.where(buy, qb, 0)
        shed = shed + jnp.where((items[None, :] == j[:, None]) & mk[:, None], dshed[:, None], 0)
        prod = jnp.arange(C.N_PRODUCTS)
        minv = (minv + jnp.where((prod == J) & inter_eff, jdelta, 0)
                + jnp.sum(jnp.where((prod[None, :] == j[:, None]) & mk[:, None], sdelta[:, None], 0), axis=0))
        return money, shed, seeds, minv, hires, n_units, ux, uy, n_quads

    carry = (money, shed, seeds, minv, hires, n_units, ux, uy, n_quads)
    if isinstance(smax, int):
        return lax.fori_loop(0, smax, slot, carry, unroll=unroll)
    return lax.fori_loop(0, smax, slot, carry)


# ============================================================================================ town / decay / day
def _town(minv, shops, t):
    cnt = jnp.sum((shops[:, None] == jnp.arange(C.MAX_SHOPS)[None, :]).astype(i32), axis=0)   # -1 never matches
    drain = cnt @ SHOP_DRAIN
    minv = minv - jnp.where(t % C.SHOP_SELL_INTERVAL == 0, drain, 0)
    minv = minv - jnp.where(t % C.CENTER_SELL_INTERVAL == 0, CENTER_DRAIN, 0)
    return minv


def _decay(tiles, t):
    kind = tiles[..., C.F_KIND]
    mls = tiles[..., C.F_MLS]
    hit = (kind == C.K_PLANT) & (mls >= 0) & (t >= mls) & (((t - mls) % 2) == 0)
    y2 = tiles[..., C.F_YIELD] - hit.astype(i32)
    tiles = tiles.at[..., C.F_YIELD].set(y2)
    die = hit & (y2 <= 0)
    return jnp.where(die[..., None], _fresh(C.K_WEED), tiles)


def _weed_bits(words):
    a = words[0:400:2] >> 5
    b = words[1:400:2] >> 6
    return (a < C.WEED_A) | ((a == C.WEED_A) & (b <= C.WEED_B))       # [200] draw k -> random() < 0.005


def _end_of_day(tiles, shed, inv, ins, ux, uy, n_units, hires, shops, n_shops, words, day):
    next_day = day + 1
    kind = tiles[..., C.F_KIND]
    what = tiles[..., C.F_WHAT]
    animal = tiles[..., C.F_ANIMAL] == 1
    watered = tiles[..., C.F_WATERED] == 1
    fed = tiles[..., C.F_FED] == 1
    cared = tiles[..., C.F_CARED] == 1
    dry = tiles[..., C.F_DRY]
    yld = tiles[..., C.F_YIELD]
    care = tiles[..., C.F_CARE]
    pday = tiles[..., C.F_DAY]
    mls = tiles[..., C.F_MLS]
    fertu = tiles[..., C.F_FERTU]
    # ---- plants (daily_refresh_plants)
    isP = kind == C.K_PLANT
    dryP = jnp.where(watered, 0, dry + 1)
    dies = isP & (dryP >= 2)
    wc = jnp.clip(what, 0, C.N_CROPS - 1)
    since = next_day - pday - CROP_FYD[wc]
    itv = jnp.maximum(CROP_INTERVAL[wc], 1)
    count = since // itv + 1
    prodP = (isP & ~dies & (CROP_ONGOING[wc] == 1) & (since >= 0) & (since % itv == 0)
             & (count <= CROP_MAXY[wc]))
    fert = watered & (fertu >= day)
    yldP = jnp.where(prodP, jnp.minimum(CROP_MAXY[wc], yld + jnp.where(fert, 2, 1)), yld)
    mlsP = jnp.where(prodP & (count == CROP_MAXY[wc]), (next_day + 1) * C.TURNS_PER_DAY, mls)
    plant_rows = (tiles.at[..., C.F_DRY].set(dryP).at[..., C.F_WATERED].set(0).at[..., C.F_YIELD].set(yldP)
                  .at[..., C.F_MLS].set(mlsP))
    # ---- animals (daily_refresh_animals)
    dryA = jnp.where(fed, 0, dry + 1)
    esc = animal & (dryA >= 2)
    ai = jnp.clip(what - C.GOOSE, 0, 2)
    sinceA = next_day - pday - ANIMAL_FYD[ai]
    prodA = animal & ~esc & (sinceA >= 0) & (sinceA % ANIMAL_INTERVAL[ai] == 0)
    yldA = jnp.where(prodA, jnp.minimum(ANIMAL_MAXHELD[ai], yld + 1 + jnp.where(fed, care, 0)), yld)
    careA = jnp.where(prodA, 0, care) + (cared & fed).astype(i32)
    animal_rows = (tiles.at[..., C.F_DRY].set(dryA).at[..., C.F_YIELD].set(yldA).at[..., C.F_CARE].set(careA)
                   .at[..., C.F_FERTAV].set(1).at[..., C.F_FED].set(0).at[..., C.F_CARED].set(0))
    escaped_rows = jnp.broadcast_to(FRESH, tiles.shape).at[..., C.F_KIND].set(kind)
    new = tiles
    new = jnp.where((isP & ~dies)[..., None], plant_rows, new)
    new = jnp.where(dies[..., None], _fresh(C.K_WEED), new)
    new = jnp.where((animal & ~esc)[..., None], animal_rows, new)
    new = jnp.where(esc[..., None], escaped_rows, new)
    # ---- weeds: farm 0 rolls its empty tiles (row-major) first, then farm 1 continues the same stream
    empty = new[..., C.F_KIND] == C.K_EMPTY                                     # [2,100]
    E = jnp.sum(empty.astype(i32), axis=1)
    rank = jnp.cumsum(empty.astype(i32), axis=1) - empty.astype(i32)
    draw = rank + jnp.where(jnp.arange(2) == 1, E[0], 0)[:, None]
    wb = _weed_bits(words)
    spawn = empty & wb[jnp.clip(draw, 0, 199)]
    new = jnp.where(spawn[..., None], _fresh(C.K_WEED), new)
    # ---- end-of-day drop of every unit inventory (unit order, then insertion order), capacity-limited
    tot = jnp.sum(inv, axis=2)                                                   # [2,U]
    before_u = jnp.cumsum(tot, axis=1) - tot
    R0 = jnp.maximum(0, C.SHED_CAP - jnp.sum(shed, axis=1))
    Ru = jnp.maximum(0, R0[:, None] - before_u)
    S = jnp.sum(jnp.where(ins[:, :, None, :] < ins[:, :, :, None], inv[:, :, None, :], 0), axis=3)  # [2,U,12]
    take = jnp.minimum(inv, jnp.maximum(0, Ru[:, :, None] - S))
    shed = shed + jnp.sum(take, axis=1)
    inv = jnp.zeros_like(inv)
    n_units = jnp.ones_like(n_units)
    ux = ux.at[:, 0].set(C.SPAWN_X)
    uy = uy.at[:, 0].set(C.SPAWN_Y)
    hires = jnp.zeros_like(hires)
    # ---- shop unlock: rng.choice(8) = first later word whose top 4 bits are < 8
    do_shop = (next_day % C.SHOP_UNLOCK_INTERVAL == 0) & (n_shops < C.MAX_SHOPS)
    widx = jnp.arange(C.NW_WORDS)
    top = (words >> 28).astype(i32)
    cand = (widx >= 2 * (E[0] + E[1])) & (top < 8)
    shop = top[jnp.argmax(cand)]
    ns = jnp.clip(n_shops, 0, C.MAX_SHOPS - 1)
    shops = shops.at[ns].set(jnp.where(do_shop, shop, shops[ns]))
    n_shops = n_shops + do_shop.astype(i32)
    info = (jnp.sum(spawn.astype(i32), axis=1), E, jnp.where(do_shop, shop, -1))   # diagnostics only
    return new, shed, inv, ins, ux, uy, n_units, hires, shops, n_shops, info


# ============================================================================================ one game step
def _step_game(tb, g, acts, t, words, umax, smax, cfg):
    """g: GameState of one game (no batch dim); acts: Actions with a leading player axis [2,...];
    t: step index (unbatched scalar); words: uint32[624] or None (then drawn from g.key at a day end);
    umax / smax: loop bounds (python ints = static loops, or unbatched traced scalars); cfg: LoopCfg."""
    day = t // C.TURNS_PER_DAY
    if cfg.unit_par:
        rmax, kmax = umax
        unit = jax.vmap(partial(_unit_phase_par, rmax=rmax, kmax=kmax),
                        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None))
    else:
        unit = jax.vmap(partial(_unit_phase, umax=umax, unroll=cfg.unit_unroll),
                        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None))
    tiles, ux, uy, shed, seeds, inv, ins = unit(g.tiles, g.ux, g.uy, g.shed, g.seeds, g.inv, g.ins,
                                                acts.uop, acts.uarg, acts.un, g.n_units, day, t)
    money, shed, seeds, minv, hires, n_units, ux, uy, n_quads = _market_phase(
        tb, g.money, shed, seeds, g.minv, g.hires, g.n_units, ux, uy, g.n_quads,
        acts.mtype, acts.mitem, acts.mn, smax, cfg.slot_unroll, cfg.count_search)
    # BUY_LAND effects on tiles: every LOCKED tile of an owned quadrant becomes EMPTY
    kind = tiles[..., C.F_KIND]
    unlock = (kind == C.K_LOCKED) & (QUAD_OF_TILE[None, :] < n_quads[:, None])
    tiles = tiles.at[..., C.F_KIND].set(jnp.where(unlock, C.K_EMPTY, kind))
    minv = _town(minv, g.shops, t)
    if cfg.no_cond:
        tiles = _decay(tiles, t)                        # the formula itself is a no-op on odd steps
    else:
        tiles = lax.cond(t % 2 == 0, _decay, lambda tl, _t: tl, tiles, t)   # odd steps provably never decay
    key = g.key
    shops, n_shops = g.shops, g.n_shops

    no_info = (jnp.zeros(2, i32), jnp.zeros(2, i32), jnp.asarray(-1, i32))

    def eod(args):
        tiles, shed, inv, ins, ux, uy, n_units, hires, shops, n_shops, key, _ = args
        if words is None:
            key, sub = jax.random.split(key)
            w = jax.random.bits(sub, (C.NW_WORDS,), jnp.uint32)
        else:
            w = words
        out = _end_of_day(tiles, shed, inv, ins, ux, uy, n_units, hires, shops, n_shops, w, day)
        return out[:10] + (key, out[10])

    args = (tiles, shed, inv, ins, ux, uy, n_units, hires, shops, n_shops, key, no_info)
    is_eod = (t + 1) % C.TURNS_PER_DAY == 0
    if cfg.no_cond:                                     # compute every step, keep on day boundaries (no host sync)
        new_args = eod(args)
        args = jax.tree_util.tree_map(lambda n, o: jnp.where(is_eod, n, o), new_args, args)
    else:
        args = lax.cond(is_eod, eod, lambda a: a, args)
    tiles, shed, inv, ins, ux, uy, n_units, hires, shops, n_shops, key, info = args
    oob = g.oob | jnp.any(minv < C.INV_LO + 300) | jnp.any(minv > C.INV_HI - 300)
    return GameState(money=money, tiles=tiles, ux=ux, uy=uy, n_units=n_units, n_quads=n_quads, hires=hires,
                     shed=shed, seeds=seeds, inv=inv, ins=ins, minv=minv, shops=shops, n_shops=n_shops,
                     oob=oob, key=key), info


def _stack_players(a0, a1):
    return Actions(*[jnp.stack([jnp.asarray(x0, i32), jnp.asarray(x1, i32)], axis=1) for x0, x1 in zip(a0, a1)])


class LoopCfg(NamedTuple):
    """Loop structure only (results are identical for every setting; tested in test_loopcfg.py)."""
    unit_static: bool = False   # True: loop over all 40 unit slots; False: to max(n_units) over the batch
    slot_static: bool = False   # True: all 10 market slots; False: to the last non-empty slot over the batch
    unit_unroll: int = 1        # unroll factor for static loops
    slot_unroll: int = 1
    unit_par: bool = False      # round-based parallel unit phase (_unit_phase_par) instead of the sequential loop
    count_search: bool = False  # market BUY searches as one wide prefix count instead of a 7-step binary search
    no_cond: bool = False       # no lax.cond (decay always, end of day computed + selected, game-over by select):
                                # together with static unrolled loops a step has no device->host sync at all


def _step_impl(tb, state, a0, a1, words, cfg=LoopCfg(), with_info=False):
    g, t = state.g, state.t
    if words is not None and words.ndim == 3:                                  # [B,days,624]: pick today's row
        words = lax.dynamic_index_in_dim(words, jnp.minimum(t // C.TURNS_PER_DAY, words.shape[1] - 1), axis=1,
                                         keepdims=False)
    acts = _stack_players(a0, a1)                                              # [B,2,...]
    if cfg.unit_par:                                                           # loop bounds shared by the batch
        umax = (C.U, C.U) if cfg.unit_static else unit_bounds(g, acts)
    else:
        umax = C.U if cfg.unit_static else jnp.max(g.n_units)
    smax = C.NSLOT if cfg.slot_static else jnp.max(jnp.where(acts.mtype != C.M_NONE,
                                                             jnp.arange(1, C.NSLOT + 1), 0))

    def do(gg):
        run = jax.vmap(lambda g1, a1_, w1: _step_game(tb, g1, a1_, t, w1, umax, smax, cfg),
                       in_axes=(0, 0, None if words is None else 0))
        return run(gg, acts, words)

    B = g.money.shape[0]
    idle = (g, (jnp.zeros((B, 2), i32), jnp.zeros((B, 2), i32), jnp.full((B,), -1, i32)))
    live = t <= C.LAST_STEP
    if cfg.no_cond:
        g2, info = jax.tree_util.tree_map(lambda n, o: jnp.where(live, n, o), do(g), idle)
    else:
        g2, info = lax.cond(live, do, lambda gg: idle, g)
    out = EnvState(t=jnp.where(live, t + 1, t), g=g2)
    if with_info:          # per game: (weeds spawned [B,2], empty tiles rolled [B,2], shop drawn or -1 [B])
        return out, info
    return out


def make_step(cfg=LoopCfg(), with_info=False):
    """A jitted step(tb, state, a0, a1, words) with the given loop structure. with_info=True also returns
    (weeds_spawned[B,2], empty_tiles_rolled[B,2], shop_drawn[B]) of this step (zeros / -1 off day boundaries)."""
    return jax.jit(partial(_step_impl, cfg=cfg, with_info=with_info))


_step_jit = make_step()          # replaced below by the backend's best (bit-identical) loop structure


_STEP_CACHE = {}


def step(state, a0, a1, words=None, cfg=None):
    """Advance every game one step. a0 / a1: Actions (batch B) of player 0 / 1.
    words: None (training randomness from the state keys), uint32[B,624] (this day's exact words) or
    uint32[B,days,624] (every day's words, e.g. np.moveaxis(rng.batch_words(seeds), 0, 1); row t//24 is used).
    cfg: LoopCfg (loop structure only; every setting gives identical results). Default: default_cfg()
    (CFG_SEQ on CPU, CFG_GPU on GPU)."""
    if cfg is None:
        return _step_jit(tables(), state, a0, a1, words)
    if cfg not in _STEP_CACHE:
        _STEP_CACHE[cfg] = make_step(cfg)
    return _STEP_CACHE[cfg](tables(), state, a0, a1, words)


CFG_SEQ = LoopCfg()                  # sequential unit loop (fastest on 1 CPU core)
CFG_PAR = LoopCfg(unit_par=True, count_search=True)   # round-based units + one-shot searches
CFG_GPU = LoopCfg(unit_static=True, slot_static=True, count_search=True, no_cond=True)
# ^ static trip counts (40 units, 10 slots) and no conds: no device->host sync inside a step. Measured on the A100
#   (shared with another job): 10.6 ms/step at B=4096 = 388k game-steps/s, vs 110 ms/step for CFG_SEQ.
CFG_UNROLLED = LoopCfg(unit_static=True, slot_static=True, unit_unroll=C.U, slot_unroll=C.NSLOT, count_search=True,
                       no_cond=True)  # fully unrolled straight-line step (minutes to compile on GPU; not faster)


def default_cfg():
    """CFG_SEQ on CPU, CFG_GPU on accelerators (results are identical; only speed differs)."""
    return CFG_SEQ if jax.default_backend() == "cpu" else CFG_GPU


_step_jit = make_step(default_cfg())


# ============================================================================================ init / helpers
def init(B, seed=0):
    """Initial EnvState for B games; per-game PRNG keys from `seed` (training randomness only)."""
    g1 = initial_game()
    g = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (B,) + np.shape(x)), g1)
    g = g._replace(key=jax.random.split(jax.random.PRNGKey(seed), B))
    return EnvState(t=jnp.asarray(0, i32), g=g)


def from_games(g_np, t):
    """EnvState from a batched numpy GameState (e.g. state.stack_games of obs_to_game results) at step t."""
    return EnvState(t=jnp.asarray(t, i32), g=jax.tree_util.tree_map(jnp.asarray, g_np))


def rewards(state):
    return state.g.money


def done(state):
    return int(state.t) > C.LAST_STEP


def prices(g):
    """int32[B,9] current prices (what observations show), from the market inventory."""
    tb = tables()
    return tb.P[jnp.arange(C.N_PRODUCTS)[None, :] * C.NTAB + jnp.clip(g.minv - C.INV_LO, 0, C.NTAB - 1)]
