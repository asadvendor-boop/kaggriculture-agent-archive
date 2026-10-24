# kagjax: batched JAX Kaggriculture engine: state, action and event layout

Rules: Kaggriculture 1.32.7, as implemented by the C++ reference `engine/src_v2/sim_trace.hpp` (bit-exact with
Kaggle). Every array below is `int32` unless stated otherwise. `B` = batch (number of games). Sizes: `U = 40` units
per farm, `NT = 100` tiles per farm, `NF = 13` tile fields, `NSLOT = 10` market slots. All ids and tables live in
`consts.py` (import it; do not hard-code numbers).

## 1. Ids

| id space | values |
|---|---|
| item (`ITEM_NAMES`) | 0 WHEAT, 1 CARROT, 2 TOMATO, 3 STRAWBERRY, 4 MELON, 5 EGG, 6 MILK, 7 WOOL, 8 FERTILIZER, 9 GOOSE, 10 COW, 11 SHEEP. Crops are 0-4, products (market) 0-8, animals 9-11. 255 = invalid name (from the dict parser). |
| unit op (`UNIT_OPS`) | 0 PASS, 1 NORTH, 2 SOUTH, 3 EAST, 4 WEST, 5 PICKUP, 6 DROP, 7 PLACE, 8 PLANT, 9 WATER, 10 HARVEST, 11 FERTILIZE, 12 DIG, 13 BUILD_COOP, 14 BUILD_PASTURE, 15 FEED, 16 COLLECT_FERTILIZER, 17 CARE; 18 = INVALID (any unknown value behaves as PASS) |
| market op (`MARKET_OPS`) | 0 NONE (empty or malformed "dead" slot), 1 HIRE, 2 BUY_LAND, 3 BUY_SEED, 4 BUY_PRODUCT, 5 BUY_ANIMAL, 6 SELL |
| shop (`SHOP_NAMES`, sorted order) | 0 BAKERY, 1 BRUNCH_SPOT, 2 FARMERS_MARKET, 3 ICE_CREAM_SHOP, 4 PET_CAFE, 5 PIZZA_SHOP, 6 SMOOTHIE_SHOP, 7 YARN_STORE |
| tile kind | 0 EMPTY, 1 LOCKED, 2 WEED, 3 COOP, 4 PASTURE, 5 PLANT |
| quadrant | 0 NW, 1 NE, 2 SW, 3 SE; `consts.QUAD_OF_TILE[100]`; land unlocks NE, SW, SE in that order for 1000/2000/4000 |

Coordinates: tile index `i = y*10 + x`, `x` = column (EAST +1), `y` = row (SOUTH +1). Shed-access tiles (x,y) =
(4,4), (5,4), (4,5), (5,5) (NWSE order; hires spawn on the least occupied one, ties to the first). The farmer
respawns at (4,4) every morning.

## 2. State: `EnvState(t, g)`

`t`: int32 **scalar shared by the whole batch** = index of the next step to execute = observation `step`
(0..719). `day = t // 24`, `hour = t % 24`. Steps 0..718 are executed; after step 718 the episode is done
(`t == 719`, further `step` calls are no-ops) and the reward of player p is `g.money[:, p]` (the C++ reward).
All games in a batch advance in lockstep (synchronous episodes), which lets day/decay/town conditionals be real
branches instead of batched selects.

`g: GameState` (NamedTuple; every field has the leading batch dim B; player axis p = seat 0/1):

| field | shape | meaning |
|---|---|---|
| `money` | [B,2] | bank (C++ double, always integral; >= 0) |
| `tiles` | [B,2,100,13] | tile table, columns below |
| `ux`, `uy` | [B,2,40] | unit positions; unit 0 = main farmer, 1.. = hands in hire order; only `u < n_units` meaningful (stale otherwise) |
| `n_units` | [B,2] | 1 + hands hired today (reset to 1 at midnight; cap 40) |
| `n_quads` | [B,2] | owned quadrants 1..4 (`QUAD_NAMES[:n_quads]`) |
| `hires` | [B,2] | hires_today; next HIRE costs `consts.FIB[hires]` (1,1,2,3,5,...) |
| `shed` | [B,2,12] | shed contents per item (sum <= 100 = `SHED_CAP`); animals bought land here |
| `seeds` | [B,2,5] | seed stock per crop (no cap) |
| `inv` | [B,2,40,12] | per-unit carried inventory (cleared at midnight after the capped drop into the shed) |
| `ins` | [B,2,40,12] | insertion stamp of `inv[...,item]`: the unit's dict order is ascending stamp among items with `inv > 0` (stale when `inv == 0`). Order decides what survives a capacity-limited DROP / midnight drop |
| `minv` | [B,9] | shared market inventory per product (starts 10000). Price = `P[item, minv]`, see `engine.prices(g)` |
| `shops` | [B,8] | unlocked shop ids in unlock order, -1 = empty slot (duplicates allowed) |
| `n_shops` | [B] | number of unlocked shops (one every 3 days, max 8) |
| `oob` | bool[B] | diagnostic: a market inventory came within 300 of the exact price-table range [-10000, 50000] (never seen in real play) |
| `key` | uint32[B,2] | per-game PRNG key, consumed only by training-mode randomness |

Tile columns (`consts.F_*`, `tiles[..., c]`):

| c | name | meaning (and where it is meaningful) |
|---|---|---|
| 0 | `F_KIND` | tile kind id |
| 1 | `F_WHAT` | PLANT: crop id 0-4; COOP/PASTURE with animal: animal item id 9-11 |
| 2 | `F_ANIMAL` | 1 if a COOP/PASTURE holds an animal |
| 3 | `F_WATERED` | PLANT: watered_today |
| 4 | `F_FED` | animal: fed_today |
| 5 | `F_CARED` | animal: cared_today |
| 6 | `F_FERTAV` | animal: fertilizer_available (COLLECT_FERTILIZER) |
| 7 | `F_DRY` | PLANT: consecutive_unwatered (plant turns WEED at 2); animal: consecutive_unfed (escapes at 2) |
| 8 | `F_YIELD` | yield_units (harvestable now; plants and animals) |
| 9 | `F_CARE` | animal: pending_care_bonus |
| 10 | `F_DAY` | PLANT: planted_day; animal: placed_day |
| 11 | `F_MLS` | PLANT: max_lifespan_step (-1 none); after it, yield decays by 1 every 2 steps, then WEED |
| 12 | `F_FERTU` | PLANT: fertilized_until_day (-1 none) |

Fields not meaningful for a tile's kind hold the canonical defaults `consts.FRESH_TILE` (0 except MLS = FERTU = -1);
the engine resets them on every kind change, exactly like the C++ `Tile{}`, so the observation is the full state.

Observation equivalence (what `observe(seat)` shows, rebuilt by `state.game_to_obs`): both farms' `money`, `tiles`,
`farmer = (ux[0],uy[0])`, `hands = (ux[1:n],uy[1:n])`, `unlocked_quadrants`, `hires_today`; shared `market`
(inventory `minv`, prices from the table), `town.unlocked_shops`, `day`, `hour`, `step = t`; private (own seat only):
`shed`, `seeds`, `inventories` (per unit, dict in `ins` order). A policy must only read the opponent's public part
(its tiles/units/money/quadrants/hires), not its `shed`, `seeds`, `inv`.

## 3. Actions: `Actions` (one per player; batch leading)

| field | shape | meaning |
|---|---|---|
| `uop` | [B,40] | unit op for unit u (0 = farmer). Units `>= n_units` never act (their PLANT still counts toward the atomic seed check, as in the C++) |
| `uarg` | [B,40] | item id for PICKUP / PLACE / PLANT (else ignored). 255 = invalid (no-op) |
| `un` | [B,40] | count for PICKUP / PLACE (default 1; `<= 0` no-op) |
| `mtype` | [B,10] | market op per slot (0 = nothing) |
| `mitem` | [B,10] | item id (ignored for HIRE / BUY_LAND) |
| `mn` | [B,10] | count (orders with `mn <= 0` are dead) |

Semantics are exactly the C++ ones; any integer values are safe (out-of-range = no-op). Highlights:
- Atomic PLANT: if more units submit PLANT for crop c than `seeds[c]`, every PLANT of c this turn is dropped.
- Units act sequentially (farmer, hand 1, ...), sharing shed, seeds and tiles; moves onto LOCKED tiles are legal,
  tile work there is not. PICKUP / DROP / shed-PLACE need a shed-access tile. PLACE of an animal on its matching
  empty structure consumes 1 carried animal.
- Market slots are processed in order; in each slot both players' orders run in per-unit lockstep: both quote from
  the same pre-commit inventory, player 0 commits then player 1, one unit at a time. SELL any product (price at
  `minv`; a $1 sale adds no supply), BUY_PRODUCT only WHEAT / FERTILIZER (price at `minv-1`, lands in the shed,
  needs shed room), BUY_SEED fixed price, BUY_ANIMAL fixed price (needs shed room), HIRE (Fibonacci cost),
  BUY_LAND. The JAX engine computes every slot in closed form (prefix sums of the exact price table) - identical
  results to the unit-by-unit loop (test_market.py).
- `state.encode_action(dict)` converts an agent's dict exactly like `kagsim_trace.cpp conv_action` (first 10 raw
  orders, malformed orders keep their slot as dead, first 39 hands, count coercion `int(x)` clamped).

## 4. Step order (= sim_trace.hpp `Sim::step`)

1. unit actions, player 0 then 1 (independent farms), units in order; 2. market slots; 3. town demand
(`t % 4 == 0`: every unlocked shop instance drains its products, single-product shops 2x; `t % 24 == 0`: the town
center drains 1 of every product but fertilizer); 4. plant decay (even steps); 5. if `(t+1) % 24 == 0`: end of day
for farm 0 then farm 1 (plant refresh: unwatered twice -> WEED, ongoing-crop production; animal refresh: unfed twice
-> escape, production with care bonus, fertilizer available; weeds on EMPTY tiles; capped drop of every unit
inventory into the shed; units reset), then a shop unlock when `(day+1) % 3 == 0`.

## 5. Events / randomness

The only randomness is the end-of-day generator `random.Random((seed*1000003) ^ day)`: one `random() < 0.005`
per EMPTY tile (farm 0 row-major, then farm 1 continuing the same stream), then `choice(8)` for a shop unlock.
The engine consumes the raw MT19937 outputs of that generator: `words: uint32[B,624]` passed to `step` (only read
on day-boundary steps). Two modes:
- exact: `rng.day_words(seed, day)` (CPython's own generator, exact; see `rng.py`) -> bit-exact weeds and shops;
  `parity.py` uses this. The 624-word window covers every case except > 224 consecutive shop-draw rejections
  (probability 2^-224).
- training: `words=None` -> i.i.d. uniform words from `g.key` (same distribution: weed p = 0.005 per empty tile,
  uniform shop draws).

## 6. API (engine.py)

```python
import engine as E, state as S
st = E.init(B, seed=0)                      # EnvState at t=0
st = E.step(st, a0, a1, words=None, cfg=None)   # a0/a1: Actions (numpy or jnp); words: None | [B,624] | [B,30,624]
E.rewards(st)                               # int32[B,2] banks
E.done(st)                                  # python bool
E.prices(st.g)                              # int32[B,9]
E.from_games(S.stack_games([S.obs_to_game(obs0, obs1)[0], ...]), t)  # start from real observations
S.game_to_obs(jax.tree_util.tree_map(np.asarray, st.g), b, seat, int(st.t))  # C++-identical observe() dict
```
Inside a larger jitted program (e.g. an RL rollout `lax.scan`) call the pure function
`E.make_step(cfg)(E.tables(), st, a0, a1, words)` (build it once; pass the tables in as an argument, do not close
over them).

### Loop structure (`E.LoopCfg`; every setting gives bit-identical results, see test_loopcfg.py)
| config | structure | use |
|---|---|---|
| `E.CFG_SEQ` | sequential unit loop to the batch max of `n_units`; market loop to the last non-empty slot; `lax.cond` for decay / end of day / game over | CPU (fastest there; default on CPU) |
| `E.CFG_GPU` | static trip counts (40 units, 10 slots, not unrolled), no `cond`: no device->host sync inside a step | GPU (default on GPU): 10.6 ms/step at B=4096, 22.4 ms at B=16384 on a shared A100 |
| `E.CFG_PAR` | round-based parallel unit phase (moves parallel, tile ops in rounds of distinct tiles, shed ops sequential) + one-shot prefix-count market searches | experimental (dynamic loops: slow on GPU) |
| `E.CFG_UNROLLED` | CFG_GPU fully unrolled | not recommended (minutes to compile) |

`E.step(..., cfg=None)` and `E._step_jit` use `E.default_cfg()` (CFG_SEQ on CPU, CFG_GPU otherwise). In your own
jitted rollout use `E.make_step(E.default_cfg())` (or CFG_GPU explicitly on GPU).

With `no_cond=True` the end of day is computed every step and selected on day boundaries, and `step` past the end
of the episode is a select (the state is unchanged, as with the conds).

## 7. Diagnostics and verification tools

- `E.make_step(cfg, with_info=True)` returns `(state, (weeds_spawned[B,2], empty_tiles_rolled[B,2], shop_drawn[B]))`
  for the step (zeros / -1 off day boundaries); the plain step is unchanged.
- `digest.digest(g)` / `digest.digest_np(g)`: uint32[B,2] hash of the complete canonical game state (identical for
  JAX on any device and for the C++ reference through `digest.ref_digest(games)`); used to compare millions of
  game-steps cheaply.
- `parity.py` (recorded games, full observe() dicts every step), `gpu_verify.py` (build datasets with per-step C++
  digests from tapes or bot fuzz; replay on any device; batch independence), `fuzz.py`, `test_market.py`,
  `test_units.py`, `test_loopcfg.py`, `test_obs_roundtrip.py`, `rng_stats.py` (training randomness = exact
  randomness in distribution). Results are in `results/` (Mac) and `logs_server/` / `results/server/` (A100).
