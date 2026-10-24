"""planner/planner.py -- the second-half planner: runs the whole farm from hour 0 of day D (default 12) to the end.

    from planner.planner import Planner, hybrid_agent
    agent = hybrid_agent(tape_agent, D=12)              # Kaggle agent: tape before step 24*D, planner after
    # or, with the handover module directly:
    agent = handover.make_hybrid(tape_agent, Planner.factory(tape=tape_agent), 12)

Standard library only.  Imports its siblings relatively (model, executor, handover; optional allocate, sales) so the
package can be inlined into a single Kaggle file under any package name (tools/planner/test_planner.py build).

ONE STEP (Planner.__call__(obs) -> engine action dict)
  1. the tape agent (kept alive by default) is called first on every observation, so it stays in sync and its action
     is ready as the fallback for this step, and its SELL orders are passed to the sales policy as context;
  2. extract_state(obs); at hour 0 (or on the first call) the DAY PLAN is built:
       * the built-in plan (planner.allocate.plan_day only with use_allocate=True, measured worse): a harvested
         WHEAT / CARROT / MELON tile is replanted with the same crop until day 24 ("mirror"), other free tiles (up to
         the handover's field size + fill_cap) get the short crop with the best MARGIN value (own revenue gain +
         denial x rival revenue loss - seed - fertilizer) from a market forecast (model.Forecast: observed book, exact
         town drain, rival_sales_schedule, our own projected output), greedy with commitment;
       * the tape's day-18 LAND + TOMATO program: with >= 3 tomato shops (PIZZA_SHOP / FARMERS_MARKET) visible, the
         next quadrant is bought at hour 0 and 20 tomatoes are planted on its tiles nearest the shed (_expansion);
       * herd kept; animal ops valued exactly (model.animal_op_values: own revenue - wheat; a FEED of an animal that
         escapes tonight while a production <= day 28 remains is a must, floor 400);
       * leave-one-out dollar values of today's ops on every plant (day-level engine simulation);
  3. Task dicts: commitment_tasks (escapes / dry deaths / last full-value harvests), plant ops, animal ops, PLANT (+DIG)
     for the plan, DROP of carried goods in the drop windows and from hour 15; on a tile with a must-water / must-feed
     every other task is due by eod - crit_gap (the executor acts FERTILIZE < WATER and HARVEST < COLLECT < FEED, so a
     unit arriving at hour 23 would otherwise fertilize and let the plant die); an early must deadline (must_hour)
     falls back to the end of the day once passed (static tasks keep their dl2);
  4. Executor.act (hires from suggest_hires at hour 0, at most 12 hands);
  5. market orders: BUY_LAND of the expansion, seeds for today's plan, feed wheat (tomorrow's at hour 23, today's
     shortfall at once), fertilizer shortfall, SELLs from planner.sales.SalesPolicy (forecast timing, the tape's SELLs
     for WHEAT / FERTILIZER and against clones; the built-in SalesStub if the module is missing or fails), midnight
     room top-up; the 10-order cap is respected (executor HIREs limited so buys fit; the policy's SELLs go first).
  Any exception inside the planner falls back to the tape's action for that step; after `max_consec_errors`
  consecutive failures the tape plays the rest of the game (if the tape is dead too: a PASS action for every unit).

DEV MEASUREMENTS (tools/planner/test_planner.py eval, the first 40 games of the interleaved handover set = 20 V94 +
20 leader tapes; paired margin differences, 95% bootstrap CI; runs/planner/eval_<label>.jsonl)
  where the loss is: with the tape's units and this planner's sales (units="tape") the margin is -1.3k (v94) / -0.6k
  (leader) vs LiveRepair on 20 games, with the planner's units about -8k whatever the sales policy: the unit program
  (what gets done each day) is the gap, not routing efficiency (47% of unit-steps are tile actions for both).
  decisions (each vs the configuration before it):
    animal-op value floors off (feed 60 / care 25 / harvest 20 / collect 10)  +2,024 [+1,235, +2,838]
    plant floors off (water 15 / harvest 25)                                  +276 and +483 (two runs, noise-level)
    crit_gap 2 (vs 0)                                                         +434 [-265, +1,142]
    harvest_eps 0.15 (vs 0)                                                   +2,706 [+1,449, +4,017]; 0.3 -259, 0.6 +120
    cargo "skip" (vs deposit-the-surplus: destroyed 45 -> 11 units/game)      +668 [+111, +1,265]
    tape running, its SELLs as context (vs no tape == tape without context)   +684 [+61, +1,320]
    follow the tape's WHEAT + FERTILIZER SELLs (vs WHEAT only / none)         +1,054 / +560
    the tape's SELLs against clones / near-clones (sticky, sim >= 0.85)       +380 [+19, +775]
    land + 20 tomatoes on day 18 (4 of the 40 games expand)                   +2,934 [+712, +5,873]; 10 tomatoes +2,068
  no gain / worse: max_hands 11 (-1,033 [-1,888, -179]), 14 (worse), must_hour 12 / 16 (-5k .. -7k: the executor
  rushes and misses more), max_new_per_day 4 / 8, fill_cap 0, no safety watering, fertilizer ratio 1.5 / 3 and no
  fertilizer purchases (+148 / -104 / -290), care floor 25 alone (+203), drop window from hour 5 (-387), low-priority
  cargo deposits (+29 .. +156), SalesPolicy RECOMMENDED / following the tape for STRAWBERRY (+36 / -411).
  result on these 40 games: margin vs LiveRepair full -4,020 (v94) / -2,238 (leader); vs the tape's freeze(12)
  +6,699 / +5,270 (the session started at -7,948 / -6,897 on 20 of them).

OFFICIAL (agents/cand/HYB12.py = LiveRepair + this package inlined, 2026-09-23; outputs runs/planner/hyb12_*.out)
  handover_study.py eval HYB12.py 12 --warmup, 120 games: W-L 60-60 (LiveRepair 92-28), own bank -1,894 vs full.
    margin vs full: v94 set -4,036 [-4,903, -3,160] (W 24 vs 54), leader set -2,458 [-3,233, -1,623] (W 36 vs 38),
    leader both-dp -1,787; holdout (the 80 games not used for tuning): v94 -4,044, leader -2,568.
    vs freeze(12): +7,527 / +4,690 -- the planner's own new activity is worth less than the tape's (+11.6k / +7.1k).
  evalcand.py HYB12 --fresh 40 --top 120: V94 fresh W-L 10-30 (base 40-0), margin -5,826 [-7,241, -4,522];
    V48 fresh 15-25 (39-1), margin -6,504; 120 leader tapes: margin -2,643 [-3,368, -1,971], own bank -1,116,
    W 56 vs 58, dp W 16/62 (base 18/61), both-dp margin -2,236.
  => the hybrid does NOT beat LiveRepair's own days 12-29; do not submit it.
  failures per game after the switch (planner | LiveRepair), v94 / leader sets: escapes 4.3|0.1 / 3.1|0.1 (harmful,
    i.e. with a production <= day 28 left: 0.32 / 0.13 -- the rest are animals left unfed on purpose), plants dead dry
    12.2|1.3 / 14.1|1.2, goods destroyed 45.8|1.9 / 26.3|1.0 units ($1,876 / $1,121; ~85% wheat cargo at midnight),
    refused orders 69|36 / 23|32 (dead SELLs of the tape's clone-race layer, SHED_EMPTY), orders over the cap 0.
  timing (whole agent: tape + planner): mean 5.0 ms/step after the switch, max 381 ms (hour-12 re-plan), hour-0 plan
  <= 25 ms; whole game max 492 ms (evalcand); 0 planner errors, 0 fallback steps, 0 agent errors; identical results
  with Kaggle Struct observations every 7th step.

INTEGRATION (2026-09-23, allocate.py 5c23185b + sales.py ead978e2; tools/planner/diag_leader.py: the 120 leader tapes
of the evalcand --top ruler (jobs 0-119) with per-product / per-day ledgers, paired with LiveRepair; rows
runs/planner/diag_<label>.jsonl, reports runs/planner/diag_cmp_*.txt)
  HYB12  (built-in plan, agents/cand/HYB12.py)   vs LiveRepair: margin -2,643 [-3,324, -1,989], wins 56 vs 58
  HYB12b (use_allocate, agents/cand/HYB12b.py)   vs LiveRepair: margin -4,114 [-5,105, -3,126], wins 43 vs 58;
         vs HYB12 -1,471 [-2,451, -573]: allocate plants 37 fewer wheat and 7 fewer carrots per game than
         LiveRepair (wheat -$5.5k, carrot -$1.5k net margin) and buys animals (+$0.7k); its tomato / melon /
         strawberry (+$6.2k) and animal products do not make up for it.
  where HYB12 loses vs LiveRepair (net of the product's own buys, ours - leader's, $/game): FERTILIZER -2,167 (it
  buys $1.8k more; the tape's own FERTILIZER SELLs are followed while the planner's FERTILIZE tasks buy the
  shortfall back), hires / seeds / land -1,032, WOOL -794 (missed feeds / cares: ~0.7 feed + ~0.5 care more per day
  left undone than LiveRepair, each losing a banked care unit), STRAWBERRY -202, EGG -118, MILK -95; gains CARROT
  +902, TOMATO +750, WHEAT +112 (both farms buy ~$10k and sell ~$11k of wheat; the churn nets out).
  fixes tried (paired vs HYB12, same 120 tapes) -- none helped, all stay off:
    iteration 1  plan_merge (built-in plan + allocate's TOMATO / MELON / STRAWBERRY + animals)  -464 [-1,394, +196]
                 no fertilizer purchases (fert_buy_last_day 11)                                -644 [-1,147, -191]
                 both                                                                          -678 [-1,561, +61]
    iteration 2  clamp_inputs (the tape's WHEAT / FERTILIZER SELLs never below the reserve)   -969 [-1,317, -628]
                 (the held inputs crowd the shed: +15 units destroyed at midnight, 64 fertilizer / 47 wheat fewer
                 sold at the early high prices); + allocate's animals (merge_crops ())         -1,407 [-2,555, -516]
  => the fertilizer "loss" is the cost side of a fertilizing program that pays (without purchases the crops lose
     more than the purchases cost), and the churn of the tape's input SELLs is value-positive; the remaining gap is
     the unit program itself (missed feeds / cares, 12 plants dead dry vs 1.2, 31 units destroyed vs 1).

INTEGRATION ROUND 2 (2026-09-23, execution leaks; all on the 120 leader jobs of diag_leader.py, paired, 95% CI)
  diagnosis (HYB12 on real games, scratch tracing of tasks / routes / unit actions):
    * dry deaths: mostly WHEAT / CARROT at age 2 (not watered at age 1: the leave-one-out value of that watering is ~0
      because the valuation assumes a perfect tomorrow) and strawberries on their must day, in the far corners; the
      executor's routes are distance-ordered (sweep / NN / 2-opt from the shed), so when a day overruns the tail jobs
      are lost whatever their value; units then PASS at the shed: end-of-day PASS 27 unit-steps/day vs LiveRepair's
      10, and in 20% of those steps a valid must-water / must-feed was reachable (it sat in another unit's route);
    * escapes: 2.9 / game but harmful only 0.1 (animals left unfed on purpose after their last production);
    * feeding / care: with MILK / WOOL at the $1 floor the exact animal values are ~0 (CARE 1.0, FEED < 0), so cows /
      sheep are fed only every other day (escape prevention) -- still, forcing care (animal_uv_frac 0.35) was worse;
    * midnight overflow: units carry 100-139 units at hour 23 (harvested wheat, collected fertilizer) while the shed
      holds 0-23; the overflow is destroyed (23 wheat / game).
  iteration 1 (each vs HYB12): floor_water 30 -1,081 [-1,511, -685] (dry deaths 12.1 -> 8.0 but 91 more waterings
    displaced 33 harvests); TriageExecutor work stealing -140 [-515, +202] (no effect on deaths, max step 984 ms);
    no 10-14 deposit window + timing-valued deposits -654 [-1,057, -274]; fert_single_ratio 99 -30 [-411, +355].
    12-job dev checks (not kept): triage by a pessimistic clock, must_hour 20, animal_uv_frac 0.35 were all worse.
  iteration 2: min_task_value 20 + fert_single_ratio 99 = HYB12c: +839 [+401, +1,245] vs HYB12 (own +1,852,
    fertilizer purchases -1,267, destroyed $1,277 -> $880); vs LiveRepair -1,804 [-2,450, -1,144], wins 57 vs 58,
    dp wins 16/67 vs 18/61.  min_task_value 45: -7,590 vs 20.
  => the unit program is labour-bound; leak fixes that add work lose, cutting low-value work gains.  The hybrid still
     does NOT beat LiveRepair's own days 12-29.  New opt-in code: TriageExecutor (params "triage"), drop_goods_mode,
     animal_uv_frac, plant_last_hour (all off by default).

Measured behaviour, variants and failure accounting: tools/planner/test_planner.py (unit / eval / build).
"""
import math
import time

from . import model as M
from . import executor as X
from . import handover as H

try:                                    # optional sibling modules built by other agents
    from . import allocate as _ALLOC
except Exception:                       # noqa: BLE001 - absent or broken: the built-in plan is used
    _ALLOC = None
try:
    from . import sales as _SALES
except Exception:                       # noqa: BLE001
    _SALES = None

TPD = 24
LAST_STEP = 718
LAST_DAY = 29
LAST_REFRESH_DAY = 28
GOODS = ("MILK", "WOOL", "STRAWBERRY", "EGG", "TOMATO", "CARROT", "MELON")
TOOLS = ("WHEAT", "FERTILIZER")
SELL_ORDER = ("STRAWBERRY", "MILK", "WOOL", "MELON", "TOMATO", "EGG", "CARROT", "FERTILIZER", "WHEAT")
ANIMAL_KINDS = ("GOOSE", "COW", "SHEEP")

DEFAULTS = {
    # tape
    "run_tape": True,          # keep the tape agent running after the switch (fallback + SELL context)
    "tape_sells": False,       # True: the tape's SELL orders are the sale plan (clamped to our stock)
    "tape_market": "off",      # "off" | "sells" | "all": follow the tape's SELL (and BUY_PRODUCT) orders as they come,
                               # clamped to our stock / shed room; the built-in policy only clears leftovers
    "max_consec_errors": 2,    # consecutive planner failures before the tape takes over for good
    "units": "planner",        # "planner" | "tape" (diagnostic: the tape's unit actions and non-SELL orders, our SELLs)
    # day plan
    "use_allocate": False,     # use planner.allocate.plan_day when it exists (measured worse: HYB12b, see INTEGRATION)
    "plan_merge": False,       # with use_allocate: built-in plan + allocate's merge_crops on non-mirror tiles (_merge_plans)
    "merge_crops": ("TOMATO", "MELON", "STRAWBERRY"),
    "merge_fertilize": False,  # keep allocate's FERTILIZE list of existing plants in merge mode
    "plan_mode": "mirror",     # built-in plan: "mirror" (replant a harvested WHEAT/CARROT/MELON tile with the same crop,
                               # other free tiles get the best short crop) or "model" (margin-greedy over new_crops)
    "new_crops": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"),
    "short_crops": ("WHEAT", "CARROT"),
    "fill_empty": False,       # plant every empty / weed tile (the tape leaves big empty areas empty: labour-bound)
    "fill_cap": 6,             # else: keep the crop field at (plants at the handover + min(free tiles then, fill_cap)),
                               # filling empty / weed tiles nearest the shed first
    "last_plant_day": {"WHEAT": 24, "CARROT": 27, "TOMATO": 20, "STRAWBERRY": 17, "MELON": 17},
    "mirror_until": 24,        # after this day a mirrored tile gets the best short crop instead of the same crop
    "fert_new_single": True,   # allow fertilizer on new single-harvest plantings (WHEAT / CARROT / MELON)
    "denial": 1.0,             # weight of the rival's revenue loss in crop values (margin objective)
    "long_margin": 30.0,       # a long crop must beat the short-crop opportunity cost by this much
    "min_net": 5.0,            # a crop worth less than this is not planted
    "max_new_per_day": 40,
    # land + tomato expansion (the tape's own day-18 program, handover records: with >= 3 tomato shops (PIZZA_SHOP /
    # FARMERS_MARKET) among the visible shops it bought the next quadrant and 10-20 tomato seeds at hour 1 of day 18
    # in 18 of 20 games (0 of 100 otherwise) and sold $6.8k-38k of tomatoes; a day-18 tomato gets exactly the 4
    # productions that end by day 28)
    "land_expand": True,
    "land_day": 18,
    "land_min_shops": 3,
    "land_n": 20,              # tomato tiles planted on the new quadrant (nearest the shed first)
    "land_min_cash": 12000.0,  # money left after land + seeds
    "land_value": 300.0,       # executor value of each of those plantings
    "fert_single_ratio": 99.0, # fertilize a single-harvest crop only if its extra units are worth this x fertilizer
                               # (99 = never; HYB12 used 1.0 -- see min_task_value)
    "plant_hour_free": 2,      # planting hour assumed when valuing a crop on an already free tile
    "plant_hour_after": 12,    # ... on a tile freed by today's harvest
    "sell_lag": 4,
    "reval_hour": 12,          # re-plan the free tiles at this hour (tiles freed mid-day); None = never
    # values
    "harvest_eps": 0.15,       # extra value of harvesting today = eps x units x price (sale timing)
    "min_task_value": 20.0,    # tasks worth less than this are not generated (commitments are raised to it).  HYB12
                               # used 0.5; 20 + fert_single_ratio 99 (HYB12c) measured +$839/game [+401, +1,245] vs HYB12
                               # on the 120 leader jobs (the unit program is labour-bound: $1 CAREs of floor-priced
                               # products, small harvests / collects and wheat-carrot fertilizing displaced better
                               # work); 45 measured -$7,590 vs 20 (it drops real work)
    # value floors.  Animal ops are valued exactly (model.animal_op_values: own revenue - wheat); floors on them
    # (full daily care like the tape: feed 60 / care 25 / harvest 20 / collect 10) measured -$2,024/game
    # [-2,838, -1,235] (40 paired dev games, eval d1_floors vs d1_base), so they are off.
    "floor_feed": 0.0,         # FEED while a production at an end-of-day <= 28 remains
    "floor_must_feed": 400.0,  # FEED of an animal that escapes tonight (production left)
    "floor_care": 0.0,         # CARE while care_today()
    "floor_animal_harvest": 0.0,
    "floor_collect": 0.0,
    "floor_must_water": 200.0, # WATER of a plant that dies tonight (production / harvest still ahead)
    "floor_water": 0.0,        # other baseline waterings (15 measured -$380/game, e_pfloor + f_pfloor)
    "floor_plant_harvest": 0.0,
    "crit_gap": 2,             # other tasks on a tile with a must-water / must-feed are due by eod - crit_gap
    "must_hour": None,         # must-water / must-feed tasks get this hour as deadline (urgent for the dispatcher)
                               # until it passes, then the end of the day
    # hires
    "max_hands": 12,
    "hire_margin": 1.5,
    # market
    "sales": "module",         # "stub" or "module" (planner.sales.SalesPolicy when present)
    "sales_params": {"clone_sim": 0.85, "clone_sticky": True, "clone_mode": "tape"},
                               # SalesPolicy params: forecast timing, the tape's SELLs for WHEAT / FERTILIZER, and the
                               # tape's SELLs for everything against clones / near-clones (sticky) -- +$380/game
                               # [+19, +775] vs the policy defaults (40 paired dev games, h_clonetape vs g_base);
                               # following the tape for STRAWBERRY too: -$411; selling at once vs clones: -$550
    "clamp_inputs": False,     # WHEAT / FERTILIZER SELLs never below the planner's reserve (_clamp_inputs)
    "tape_sells_context": True,   # pass the tape's SELLs to the SalesPolicy as context (it follows them for its
                                  # follow_tape items)
    "lot_k": 1.0,              # lot = lot_k x per-tick drain of the visible shops
    "min_lot": 1,
    "sell_phase": 1,           # sell at steps with step % 4 == sell_phase (1 = right after the drain tick)
    "strawberry_defer": False, # hold strawberries to hour 22 (non-clone rivals)
    "melon_reservation": 0.8,
    "fert_reservation": 0.0,   # FERTILIZER has no town drain: its price only falls, so surplus is sold at once
    "fert_reserve_days": 2,
    "fert_buffer": 3,
    "fert_buy_last_day": 26,   # no fertilizer purchases after this day
    "feed_buffer": 2,
    "wheat_keep_days": 2,      # wheat above this many days of feed (+buffer) is sold
    "dump_day": 28,            # wheat/fertilizer reserves are released from this day on
    "drop_windows": ((10, 14),),   # hours in which carrying units deposit goods at the shed (same-day sales)
    "goods_hour": 15,          # from this hour carried goods are deposited (deadline: hour 22) when the midnight
                               # load (shed + everything carried) would exceed the shed, or always if goods_always
    "goods_always": True,
    "drop_min_value": 60.0,
    "drop_value_k": 1.0,       # a deposit task is worth this x the carried goods' value (they cannot be sold otherwise)
    "cargo_hour": 15,          # from this hour WHEAT / FERTILIZER cargo is deposited when the midnight load is too big
    "cargo_value": 20.0,       # $ per carried wheat / fertilizer unit for that deposit
    "cargo_mode": "skip",      # "skip": no deposit while any FEED / FERTILIZE needing it is pending; "surplus": deposit
                               # carried tools beyond those needs (+cargo_keep), worth the overflow it prevents -- cuts
                               # the destroyed units 45 -> 11 / game but measured -$668/game [-1,265, -111] (40 paired
                               # dev games, e_base vs e_cskip): the deposit trips cost more work than the wheat is worth
    "cargo_keep": 2,
    "cargo_price_k": 1.0,      # surplus mode: weight of the price of the units the midnight overflow would destroy
    "drop_deadline_hour": 22,
    "final_drop_hour": 10,     # day 29: everything carried is deposited from this hour (sold by step 718)
    "room_margin": 4,          # midnight: keep room for carried cargo + this many units
    "max_hires_per_step": 7,
    "exec_params": None,       # extra executor.DEFAULT_PARAMS overrides
    "harvest_rate": False,     # harvest WHEAT / CARROT at the age with the best units per day (fertilized: earlier)
    "safety_water": True,      # also water WHEAT / CARROT / MELON the day before their window (age ws-1) so a missed
                               # window watering is not fatal
    "static_tasks": True,      # build the day's maintenance + planting tasks once (hour 0 / re-plan) and keep them
                               # (stable values and deadlines -> stable routes); deposits / seedlings stay dynamic
    # --- integrator round 2 (2026-09-23): execution-leak fixes, all opt-in (None / 0 = the measured HYB12 behaviour)
    "triage": None,            # dict for TriageExecutor: under a pessimistic clock (start delayed by slack_frac x the
                               # steps left) a route that drops a job worth >= min_protect sheds its jobs worth
                               # < ratio x that value, cheapest first, while that raises the route's value
    "drop_goods_mode": "full", # "full": a goods deposit is worth drop_value_k x the carried goods (HYB12);
                               # "timing": that only when the midnight load would overflow, else drop_timing_k x it
                               # (the goods drop into the shed at midnight anyway and sell tomorrow)
    "drop_timing_k": 0.25,
    "animal_uv_frac": 0.0,     # animal products valued at max(price, frac x base) for FEED / CARE / HARVEST / COLLECT
                               # (a product at the $1 floor now is sold days later, after the town drain)
    "plant_last_hour": None,   # PLANT tasks due by this hour (None: eod - 2)
}


# =====================================================================================================================
# small helpers
# =====================================================================================================================
def _g(o, k, d=None):
    if isinstance(o, dict):
        return o.get(k, d)
    return getattr(o, k, d)


def feed_today(species, placed_day, day):
    """Feed on `day` only while a production at an end-of-day <= 28 is still to come (leaders' stop rule)."""
    if day > LAST_REFRESH_DAY:
        return False
    return any(e >= day for e in M.animal_production_days(species, placed_day, LAST_REFRESH_DAY))


def care_today(species, placed_day, day):
    """Care banks +1 for the next production strictly after today (if fed then); useful while one remains <= 28."""
    if day >= LAST_REFRESH_DAY:
        return False
    return any(e > day for e in M.animal_production_days(species, placed_day, LAST_REFRESH_DAY))


def drain4(shops):
    d = {P: 0 for P in M.PRODUCTS}
    for s in shops:
        prods = M.SHOPS.get(s, ())
        m = 2 if len(prods) == 1 else 1
        for P in prods:
            d[P] += m
    return d


def revenues(fc, P, extra=None):
    """(our revenue, rival revenue) of product P over fc's horizon, with our flows + `extra` ({step: units}).
    Mirrors model.Forecast._run; the rival's revenue is its fractional sales x the price it meets (buys: - cost)."""
    inv = fc.inv0[P]
    ours = fc.ours[P]
    rival = fc.rival[P]
    drain = fc.drain[P]
    s0 = fc.step0
    own = 0.0
    riv = 0.0
    mp = M.market_price
    for i in range(fc.n):
        s = s0 + i
        r = rival.get(s)
        if r:
            if r > 0:
                p = mp(P, inv)
                riv += r * p
                if p > 1:
                    inv += r
            else:
                inv += r
                riv += r * mp(P, inv)
        o = ours.get(s, 0)
        if extra:
            o += extra.get(s, 0)
        if o:
            if o > 0:
                for _ in range(int(o)):
                    p = mp(P, inv)
                    own += p
                    if p > 1:
                        inv += 1
            else:
                for _ in range(int(-o)):
                    inv -= 1
                    own -= mp(P, inv)
        inv -= drain[i]
    return own, riv


# =====================================================================================================================
# plant simulation from an observed tile (day level) and leave-one-out op values
# =====================================================================================================================
HARVEST_RATE = [False]      # module switch (Planner param "harvest_rate"): harvest short crops at the best units/day


def best_harvest_age(crop, planted_day, yield_units, fert_until, day, watered_today=False):
    """Harvest age of a single-harvest crop watered on every window day from `day` on: the earliest age >= first_yield_day
    reaching its best reachable yield (cap, or the end of the window); with HARVEST_RATE on (and a replanting cycle still
    ahead) the age with the best units per day of tile occupancy instead (fertilized WHEAT: 5 units at age 3 beat 6 at
    age 4; fertilized CARROT: 3 at age 2 beat 4 at age 3)."""
    cd = M.CROPS[crop]
    ws, we = M.window(crop)
    y = yield_units
    ys = {}
    for d in range(day, planted_day + we + 1):
        age = d - planted_day
        if ws <= age <= we and not (d == day and watered_today):
            y = min(cd["max_yield"], y + (2 if fert_until >= d else 1))
        if age >= cd["first_yield_day"]:
            ys[age] = y
            if not HARVEST_RATE[0] and (y >= cd["max_yield"] or age >= we):
                return age
    if not ys:
        return max(cd["first_yield_day"], day - planted_day)
    if HARVEST_RATE[0] and planted_day + we + 3 <= LAST_DAY:
        return max(ys, key=lambda a: (round(ys[a] / float(a), 6), a))
    best = max(ys.values())
    return min(a for a, v in ys.items() if v >= best)


SAFETY_WATER = [False]      # module switch (Planner param "safety_water"): single-harvest crops also watered at age 1


def plant_baseline_ops(t, d, fert_policy, ha, last_day=LAST_DAY):
    """Ops the baseline care policy does on day d for plant tile t (state at the start of day d, or now)."""
    crop = t["crop"]
    cd = M.CROPS[crop]
    pd = t["planted_day"]
    age = d - pd
    ops = []
    y = t["yield_units"]
    wt = t["watered_today"]
    cu = t["consecutive_unwatered"]
    fu = t.get("fertilized_until_day", -1)
    if not cd["ongoing"]:
        ws, we = M.window(crop)
        harvest_now = age >= cd["first_yield_day"] and y > 0 and (age >= ha or y >= cd["max_yield"] or d >= last_day)
        in_win = ws <= age <= we and y < cd["max_yield"]
        if fert_policy and age == ws and fu < d and in_win and not wt:
            ops.append("FERTILIZE")
        safety = SAFETY_WATER[0] and age == ws - 1 and not harvest_now
        if not wt and (in_win or safety or (cu >= 1 and not harvest_now and d <= LAST_REFRESH_DAY)):
            ops.append("WATER")
        if harvest_now:
            ops.append("HARVEST")
        return ops
    R = [r for r in M.production_refresh_days(crop, pd) if r <= LAST_REFRESH_DAY]
    last_r = R[-1] if R else None
    if y > 0 and age >= cd["first_yield_day"]:
        ops.append("HARVEST")
    if last_r is not None and d <= last_r and d <= LAST_REFRESH_DAY:
        prod = d in R
        fert_on = fu >= d
        if prod and fert_policy and not fert_on:
            ops.insert(0, "FERTILIZE")
            fert_on = True
        if not wt and ((prod and fert_on) or cu >= 1):
            ops.insert(1 if ops and ops[0] == "FERTILIZE" else 0, "WATER")
    return ops


def sim_plant(tile, day, today_ops, fert_policy, ha, hour=0, last_day=LAST_DAY):
    """Day-level exact simulation of an observed plant tile from `day` (state now, at `hour`): today's ops
    `today_ops` (in FERTILIZE, WATER, HARVEST order), then the baseline policy.  Harvests on a decaying single-harvest
    crop lose the decay ticks that happened before the hour of the visit (assumed `hour` today, 1 on later days).
    Returns (units harvested, fertilizers used)."""
    t = dict(tile)
    units = 0
    ferts = 0
    for d in range(day, last_day + 1):
        ops = today_ops if d == day else plant_baseline_ops(t, d, fert_policy, ha, last_day)
        h = hour if d == day else 1
        for op in ("FERTILIZE", "WATER", "HARVEST"):
            if op not in ops:
                continue
            if op == "FERTILIZE":
                if t.get("fertilized_until_day", -1) < d + 2:
                    M.plant_fertilize(t, d)
                    ferts += 1
            elif op == "WATER":
                M.plant_water(t, d)
            else:
                mls = t.get("max_lifespan_step", -1)
                y = t["yield_units"]
                if mls >= 0:
                    s = d * TPD + h
                    if s > mls:
                        y -= (s - mls + 1) // 2           # decay ticks at mls, mls+2, ... before this step
                if y > 0 and M.can_harvest(t, d):
                    units += y
                    t["yield_units"] = 0
                    if not M.CROPS[t["crop"]]["ongoing"]:
                        return units, ferts
                elif y <= 0 and t["yield_units"] > 0:
                    return units, ferts                    # decayed to a weed before the visit
        # decay within the day (single-harvest crops past their lifespan, ongoing crops after the 4th production)
        mls = t.get("max_lifespan_step", -1)
        if mls >= 0 and d * TPD + TPD - 1 >= mls:
            lo = max(mls, d * TPD)
            ticks = sum(1 for s in range(lo, d * TPD + TPD) if (s - mls) % 2 == 0)
            t["yield_units"] -= ticks
            if t["yield_units"] <= 0:
                return units, ferts
        if d > LAST_REFRESH_DAY:
            continue
        t = M.plant_refresh(t, d)
        if not M.is_plant(t):
            return units, ferts
    return units, ferts


def plant_op_values(tile, day, unit_value, fert_value, fert_policy, hour=0, eps=0.0):
    """Dollars lost if each of today's baseline ops on this plant is skipped today (leave-one-out, exact day-level
    simulation of the rest of the season under the baseline policy).  Returns ({op: value}, baseline ops, ha)."""
    crop = tile["crop"]
    cd = M.CROPS[crop]
    ha = None
    if not cd["ongoing"]:
        ha = best_harvest_age(crop, tile["planted_day"], tile["yield_units"], tile.get("fertilized_until_day", -1),
                              day, tile["watered_today"])
    ops = plant_baseline_ops(tile, day, fert_policy, ha)
    if not ops:
        return {}, ops, ha
    u0, f0 = sim_plant(tile, day, ops, fert_policy, ha, hour)
    base = u0 * unit_value - f0 * fert_value
    out = {}
    for op in ops:
        rest = [o for o in ops if o != op]
        u1, f1 = sim_plant(tile, day, rest, fert_policy, ha, hour)
        v = base - (u1 * unit_value - f1 * fert_value)
        if op == "HARVEST":
            v = max(v, 0.0) + eps * tile["yield_units"] * unit_value
        out[op] = v
    return out, ops, ha


def ongoing_finished(tile, day):
    """An ongoing crop with no production left at an end-of-day <= 28 (from `day` on)."""
    crop = tile["crop"]
    R = [r for r in M.production_refresh_days(crop, tile["planted_day"]) if day <= r <= LAST_REFRESH_DAY]
    return not R


# =====================================================================================================================
# built-in sales policy
# =====================================================================================================================
class SalesStub:
    """Pacing policy from the leaders' habits (targets report): perishable goods (milk, wool, strawberry, egg, tomato,
    carrot) are sold from the shed in lots of ~lot_k x the per-tick drain of the visible shops right after each drain
    tick (step % 4 == 1), enlarged so the stock clears by the end of the day (holding across days fails: the midnight
    cargo fills the shed); melon and fertilizer wait for their reservation prices (0.8x / 0.75x base) except under
    room pressure and at the end; wheat and fertilizer keep a feed / fertilizing reserve, the rest is sold; everything
    is dumped on the last day (step <= 718)."""

    def __init__(self, params):
        self.p = params

    def orders(self, obs, st, reserve, extra_room=0, tape_sells=None):
        p = self.p
        step = st["step"]
        hour = st["hour"]
        day = st["day"]
        shed = st["shed"]
        prices = st["market"]["prices"]
        d4 = drain4(st["market"]["shops"])
        last = step >= LAST_STEP
        # ticks at which we still sell today (incl. now)
        end = min(day * TPD + TPD - 1, LAST_STEP)
        ticks_left = sum(1 for s in range(step, end + 1) if s % M.SHOP_TICK == p["sell_phase"])
        is_tick = step % M.SHOP_TICK == p["sell_phase"]
        final_hour = step == end
        out = []
        sold = {}
        if tape_sells:
            for it, n in tape_sells:
                if it not in shed:
                    continue
                avail = shed.get(it, 0) - reserve.get(it, 0) - sold.get(it, 0)
                q = min(int(n), avail)
                if q > 0:
                    out.append(["SELL", it, q])
                    sold[it] = sold.get(it, 0) + q
        tape_mode = tape_sells is not None
        for P in SELL_ORDER:
            avail = shed.get(P, 0) - reserve.get(P, 0) - sold.get(P, 0)
            if avail <= 0:
                continue
            q = 0
            if last or (day >= LAST_DAY and final_hour):
                q = avail
            elif tape_mode:
                if final_hour and P in GOODS and P != "MELON":
                    q = avail                      # the tape's layers did not clear it: no holding across days
            elif P == "MELON":
                ok = prices.get(P, 0) >= p["melon_reservation"] * M.MARKET_PARAMS[P]["base"] or day >= p["dump_day"]
                if ok and (is_tick or final_hour):
                    q = avail if (final_hour or day >= LAST_DAY) else max(1, -(-avail // max(1, ticks_left)))
            elif P == "FERTILIZER":
                ok = prices.get(P, 0) >= p["fert_reservation"] * M.MARKET_PARAMS[P]["base"] or day >= p["dump_day"]
                if ok and (hour <= 1 or is_tick or final_hour):
                    q = avail
            elif P == "WHEAT":
                if is_tick or final_hour:
                    q = max(1, -(-avail // max(1, ticks_left)))
                    q = max(q, min(avail, int(round(p["lot_k"] * max(1, d4.get(P, 0))))))
                    if final_hour:
                        q = avail
            else:
                if p["strawberry_defer"] and P == "STRAWBERRY" and hour < 22 and day < LAST_DAY:
                    continue
                if is_tick or final_hour or (P == "STRAWBERRY" and p["strawberry_defer"] and hour >= 22):
                    lot = max(p["min_lot"], int(round(p["lot_k"] * max(1, d4.get(P, 0)))))
                    q = max(lot, -(-avail // max(1, ticks_left)))
                    if final_hour:
                        q = avail
            q = min(q, avail)
            if q > 0:
                out.append(["SELL", P, q])
                sold[P] = sold.get(P, 0) + q
        # midnight room: everything carried drops into the shed after hour 23 (overflow destroyed)
        if extra_room > 0:
            need = extra_room
            for P in ("FERTILIZER", "WHEAT") + GOODS:
                if need <= 0:
                    break
                avail = shed.get(P, 0) - sold.get(P, 0)
                if P in reserve and P in TOOLS:
                    avail -= 0 if need > 8 else reserve.get(P, 0)
                q = min(avail, need)
                if q > 0:
                    out.append(["SELL", P, q])
                    sold[P] = sold.get(P, 0) + q
                    need -= q
        # merge duplicate products into one order each (fewer slots)
        merged = {}
        for o in out:
            merged[o[1]] = merged.get(o[1], 0) + o[2]
        return [["SELL", P, n] for P, n in merged.items() if n > 0]


# =====================================================================================================================
# the planner
# =====================================================================================================================
TRIAGE_DEFAULTS = {"slack_frac": 0.0, "min_protect": 150.0, "ratio": 0.5, "ms": 25.0,
                   "steal": True, "steal_idle": 3, "steal_k": 12, "steal_min": 50.0, "steal_ms": 30.0}


class TriageExecutor(X.Executor):
    """Executor with value triage under overload.

    The route constructions are distance-driven (sweep / nearest neighbour / 2-opt from the shed), so far tiles sit at
    the tail of the routes; the route simulation is optimistic (new tasks appear during the day, deposits, pickups),
    and when a day overruns, the tail jobs are the ones lost whatever their value -- measured on the leader tapes:
    must-waters of far-corner wheat / carrot (200 each) missed while low-value FERTILIZE / COLLECT / CARE jobs and
    goods deposits were done, 12 plants dead dry per game vs LiveRepair's 1.2.
    After the base improve / balance passes, each route is re-simulated with its start delayed by slack_frac x the
    steps left today (a pessimistic clock); if that drops a job worth >= min_protect, the route's jobs worth less than
    ratio x the most valuable dropped one are removed, cheapest first, as long as each removal raises the route's
    (pessimistic) value.  Removed jobs are unassigned: the next step's insertion puts them back where they fit."""

    def __init__(self, params=None, triage=None, **kw):
        super().__init__(params, **kw)
        self.tri = dict(TRIAGE_DEFAULTS)
        if isinstance(triage, dict):
            self.tri.update(triage)
        self.tri_stats = {"removed": 0, "steps": 0}

    def _balance(self, S, jd, routes, t_start):
        routes = super()._balance(S, jd, routes, t_start)
        try:
            if self.tri["steal"]:
                routes = self._steal(S, jd, routes)
            if self.tri["slack_frac"] > 0:
                routes = self._triage(S, jd, routes)
        except Exception:                  # noqa: BLE001 - triage is an optimisation, never fatal
            pass
        return routes

    def _steal(self, S, jd, routes):
        """Work stealing: a unit whose route ends before eod - steal_idle (typically empty after a shed errand) takes
        the most valuable tile jobs of the other routes, one at a time, wherever the joint objective improves
        (the job's owner reaches it later than the idle unit would, or not at all).  Measured before this pass:
        in 20% of the end-of-day PASS steps an idle unit had a valid must-water / must-feed within reach."""
        p = self.tri
        U = len(S.units)
        if U < 2 or S.eod - S.step <= 1:
            return routes
        t0 = time.perf_counter()
        budget = p["steal_ms"] / 1000.0
        obj, res = self._objective(S, jd, routes)
        idle = [u for u in range(U) if res[u][1] <= S.eod - p["steal_idle"]]
        if not idle:
            return routes
        for u in idle:
            cands = []
            for v in range(U):
                if v == u:
                    continue
                for k in routes[v]:
                    j = jd[k]
                    if j.shed or j.value < p["steal_min"] or (j.only is not None and j.only != u):
                        continue
                    d = abs(S.units[u][0] - j.x) + abs(S.units[u][1] - j.y)
                    if S.step + d + 1 > S.eod:
                        continue
                    cands.append((-j.value, d, v, k))
            cands.sort()
            for _, _, v, k in cands[:p["steal_k"]]:
                if time.perf_counter() - t0 > budget:
                    return routes
                if k not in routes[v]:
                    continue
                trial = list(routes)
                trial[v] = [x for x in routes[v] if x != k]
                best = None
                r_u = routes[u]
                for pos in sorted({0, len(r_u)}):
                    tr = list(trial)
                    tr[u] = r_u[:pos] + [k] + r_u[pos:]
                    o2 = self._objective(S, jd, tr)[0]
                    if best is None or o2 > best[0]:
                        best = (o2, tr)
                if best is not None and best[0] > obj + 1e-9:
                    obj, routes = best
                    self.tri_stats["removed"] += 1
        return routes

    def _triage(self, S, jd, routes):
        p = self.tri
        left = S.eod - S.step
        if left <= 1:
            return routes
        slack = int(math.ceil(p["slack_frac"] * left))
        if slack <= 0:
            return routes
        U = len(S.units)
        saved = S.start
        base = [(saved[u] if saved is not None else S.step) for u in range(U)]
        S.start = [b + slack for b in base]
        t0 = time.perf_counter()
        budget = p["ms"] / 1000.0
        tools = X.TOOL_ITEMS
        self.tri_stats["steps"] += 1
        try:
            for u in range(U):
                r = routes[u]
                if len(r) < 2:
                    continue
                res = self._sim_route(S, u, r, jd, {it: S.shed.get(it, 0) for it in tools}, dict(S.seeds))
                val, dropped = res[0], res[4]
                if not dropped:
                    continue
                hi = max(jd[k].value for k in dropped)
                if hi < p["min_protect"]:
                    continue
                cands = sorted((jd[k].value, i, k) for i, k in enumerate(r)
                               if not jd[k].shed and jd[k].value < p["ratio"] * hi)
                for v, _, k in cands:
                    if time.perf_counter() - t0 > budget:
                        break
                    r2 = [x for x in r if x != k]
                    res2 = self._sim_route(S, u, r2, jd, {it: S.shed.get(it, 0) for it in tools}, dict(S.seeds))
                    if res2[0] > val + 1e-9:
                        r, val, dropped = r2, res2[0], res2[4]
                        self.tri_stats["removed"] += 1
                        if not dropped or max(jd[k2].value for k2 in dropped) < p["min_protect"]:
                            break
                routes[u] = r
                if time.perf_counter() - t0 > budget:
                    break
        finally:
            S.start = saved
        return routes


class Planner:
    """Runs the farm from the handover step on.  Build with Planner(st, obs, tape=..., params=...) or
    Planner.factory(tape=..., params=...) (the planner_factory signature of handover.make_hybrid)."""

    def __init__(self, st=None, obs=None, tape=None, params=None, sales_policy=None):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        SAFETY_WATER[0] = bool(self.p["safety_water"])
        HARVEST_RATE[0] = bool(self.p["harvest_rate"])
        self.tape = tape if self.p["run_tape"] else None
        self.tape_alive = self.tape is not None
        self.mode = "planner"
        ep = {"max_hires_per_step": self.p["max_hires_per_step"]}
        ep.update(self.p["exec_params"] or {})
        self.ex = TriageExecutor(ep, triage=self.p["triage"]) if self.p["triage"] else X.Executor(ep)
        self.tracker = M.FlowTracker()
        self.day = -1
        self.plan = {}             # (x, y) -> {"crop", "fert", "value", "dig"}
        self.alloc = None          # extras of planner.allocate's day plan (see _plan_from_module)
        self.alloc_bought = False
        self.fert_policy = {}      # (x, y, planted_day) -> bool
        self.values = {}           # (x, y) -> {op: value} (hour-0 leave-one-out values)
        self.hires_today = None
        self.last_orders = []
        self.last_shed = {}
        self.prev_step = None
        self.consec = 0
        self._module_ok = False
        self._others = []
        self.sales = SalesStub(self.p)
        self.sales_module = sales_policy
        if self.sales_module is None and self.p["sales"] == "module":
            self.sales_module = make_sales_policy(self.p)
        self.stats = {"steps": 0, "ms_total": 0.0, "ms_max": 0.0, "ms_max_step": None, "errors": 0,
                      "tape_errors": 0, "fallback_steps": 0, "last_error": None, "plan_source": {},
                      "tape_takeover_step": None, "ms_by_hour0": []}

    @classmethod
    def factory(cls, tape=None, params=None, sales_policy=None):
        def make(st, obs):
            return cls(st, obs, tape=tape, params=params, sales_policy=sales_policy)
        return make

    # -----------------------------------------------------------------------------------------------------------------
    def __call__(self, obs):
        t0 = time.perf_counter()
        tape_act = None
        if self.tape_alive:
            try:
                tape_act = self.tape(obs)
                if not isinstance(tape_act, dict):
                    tape_act = None
            except Exception as ex:        # noqa: BLE001
                self.tape_alive = False
                self.stats["tape_errors"] += 1
                self.stats["last_error"] = f"tape: {type(ex).__name__}: {ex}"[:200]
        if self.mode == "tape" and tape_act is not None:
            act = tape_act
            self.stats["fallback_steps"] += 1
        else:
            try:
                act = self._step(obs, tape_act)
                self.consec = 0
            except Exception as ex:        # noqa: BLE001 - never let the planner kill the farm
                self.stats["errors"] += 1
                self.consec += 1
                self.stats["last_error"] = f"step {_g(obs, 'step')}: {type(ex).__name__}: {ex}"[:300]
                if self.consec >= self.p["max_consec_errors"] and tape_act is not None:
                    self.mode = "tape"
                    self.stats["tape_takeover_step"] = int(_g(obs, "step", 0))
                self.stats["fallback_steps"] += 1
                act = tape_act if tape_act is not None else self.safe_action(obs)
                self.last_orders = list(act.get("market", []) or [])
        dt = (time.perf_counter() - t0) * 1000.0
        st = self.stats
        st["steps"] += 1
        st["ms_total"] += dt
        if dt > st["ms_max"]:
            st["ms_max"] = dt
            st["ms_max_step"] = int(_g(obs, "step", 0))
        return act

    @staticmethod
    def safe_action(obs):
        farms = _g(obs, "farms") or []
        me = farms[int(_g(obs, "player", 0))] if farms else {}
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (_g(me, "hands") or [])], "market": []}

    # -----------------------------------------------------------------------------------------------------------------
    def _step(self, obs, tape_act):
        step = int(obs["step"])
        self._track(obs)
        st = H.extract_state(obs)
        if self.p["units"] == "tape" and tape_act is not None:
            if st["day"] != self.day:
                self.day = st["day"]
                self.values = {}
                self._compute_values(obs, st)
                self.plan = {}
            tm = [list(o) for o in (tape_act.get("market") or []) if isinstance(o, (list, tuple))]
            market = None
            if self.sales_module is not None and hasattr(self.sales_module, "market"):
                try:
                    stock = self._after_units(obs, st, tape_act)["shed"]
                    market = self.sales_module.market(obs, st, dict(stock), {"market": tm, "reserve": self._reserve(st)})
                except Exception as ex:    # noqa: BLE001
                    self.stats["sales_errors"] = self.stats.get("sales_errors", 0) + 1
                    self.stats["last_error"] = f"sales.market: {type(ex).__name__}: {ex}"[:200]
                    market = None
            if market is None:
                sells = self._sell_orders(obs, st, None, {})
                keep = [o for o in tm if not (o and o[0] == "SELL")]
                market = keep + sells
            act = {"farmer": tape_act.get("farmer", ["PASS"]), "hands": tape_act.get("hands", []),
                   "market": list(market)[:X_MAX_ORDERS]}
            self.last_orders = act["market"]
            self.last_shed = dict(st["shed"])
            self.prev_step = step
            return act
        day, hour = st["day"], st["hour"]
        if day != self.day:
            self._day_start(obs, st)
        elif self.p["reval_hour"] is not None and hour == self.p["reval_hour"]:
            self._replan_free(obs, st)
        tasks = self._tasks(obs, st)
        pre = self._priority_orders(obs, st, tasks)
        hw = None
        if hour == 0 or self.hires_today is None:
            inc = {"seeds": {}}
            for o in pre:
                if o[0] == "BUY_SEED":
                    inc["seeds"][o[1]] = inc["seeds"].get(o[1], 0) + o[2]
            try:
                hw = self.ex.suggest_hires(obs, tasks, max_hands=self.p["max_hands"], margin=self.p["hire_margin"],
                                           incoming=inc)
            except Exception:              # noqa: BLE001
                hw = None
            self.hires_today = hw
        self.ex.p["max_hires_per_step"] = max(0, min(self.p["max_hires_per_step"], X_MAX_ORDERS - len(pre)))
        out = self.ex.act(obs, tasks, hw)
        room_left = X_MAX_ORDERS - len(out.get("market_prefix", [])) - len(pre)
        st_s = self._after_units(obs, st, out)
        self._others = list(out.get("market_prefix", [])) + pre
        sells = self._sell_orders(obs, st_s, tape_act, out)[:max(0, room_left)]
        if self.sales_module is not None and self._module_ok:
            market = sells + list(out.get("market_prefix", [])) + pre      # the policy's SELLs go in front
        else:
            market = list(out.get("market_prefix", [])) + pre + sells
        act = {"farmer": out["farmer"], "hands": out["hands"], "market": market[:X_MAX_ORDERS]}
        self.last_orders = act["market"]
        self.last_shed = dict(st["shed"])
        self.prev_step = step
        return act

    @staticmethod
    def _after_units(obs, st, out):
        """State copy with the shed / carried cargo after this step's unit actions (deposits and pickups at the shed,
        in unit order, as the engine applies them) -- the market runs after the units."""
        acts = [out.get("farmer")] + list(out.get("hands") or [])
        shed = dict(st["shed"])
        room = 100 - sum(shed.values())
        units = st["units"]
        carried = dict(st["carried"])
        changed = False
        for i, a in enumerate(acts):
            if i >= len(units) or not isinstance(a, list) or not a:
                continue
            pos = units[i]["pos"]
            if pos not in H.SHED_ACCESS:
                continue
            inv = dict(units[i]["inv"])
            if a[0] == "PLACE" and len(a) >= 3 and a[1] in inv and a[1] not in ANIMAL_KINDS:
                q = max(0, min(int(a[2]), inv.get(a[1], 0), room))
                if q:
                    shed[a[1]] = shed.get(a[1], 0) + q
                    carried[a[1]] = carried.get(a[1], 0) - q
                    room -= q
                    changed = True
            elif a[0] == "DROP":
                for it, c in inv.items():
                    q = max(0, min(c, room))
                    if q:
                        shed[it] = shed.get(it, 0) + q
                        carried[it] = carried.get(it, 0) - q
                        room -= q
                        changed = True
            elif a[0] == "PICKUP" and len(a) >= 3:
                q = max(0, min(int(a[2]), shed.get(a[1], 0)))
                if q:
                    shed[a[1]] -= q
                    carried[a[1]] = carried.get(a[1], 0) + q
                    room += q
                    changed = True
        if not changed:
            return st
        st2 = dict(st)
        st2["shed"] = shed
        st2["shed_total"] = sum(shed.values())
        st2["shed_room"] = max(0, 100 - st2["shed_total"])
        st2["carried"] = {k: v for k, v in carried.items() if v > 0}
        return st2

    # -----------------------------------------------------------------------------------------------------------------
    def _track(self, obs):
        """FlowTracker update with our estimated executed net flow of the previous step (sells clipped by stock)."""
        net = {}
        if self.prev_step is not None and self.prev_step == int(obs["step"]) - 1:
            stock = dict(self.last_shed)
            for o in self.last_orders:
                if not isinstance(o, (list, tuple)) or len(o) < 3:
                    continue
                if o[0] == "SELL":
                    q = min(int(o[2]), stock.get(o[1], 0))
                    stock[o[1]] = stock.get(o[1], 0) - q
                    net[o[1]] = net.get(o[1], 0) + q
                elif o[0] == "BUY_PRODUCT":
                    net[o[1]] = net.get(o[1], 0) - int(o[2])
        self.tracker.update(obs, net)

    # -----------------------------------------------------------------------------------------------------------------
    # day plan
    # -----------------------------------------------------------------------------------------------------------------
    def _day_start(self, obs, st):
        t0 = time.perf_counter()
        self.day = st["day"]
        self.hires_today = None
        self.plan = {}
        self.alloc = None
        self.values = {}
        self._compute_values(obs, st)
        plan = None
        if self.p["use_allocate"] and _ALLOC is not None and hasattr(_ALLOC, "plan_day"):
            try:
                plan = self._plan_from_module(obs, st)
                if plan is not None:
                    self.stats["plan_source"]["allocate"] = self.stats["plan_source"].get("allocate", 0) + 1
            except Exception as ex:        # noqa: BLE001
                self.stats["last_error"] = f"allocate: {type(ex).__name__}: {ex}"[:200]
                plan = None
        if plan is not None and self.p["plan_merge"]:
            plan = self._merge_plans(obs, st, plan)
        if plan is None:
            plan = self._plan_builtin(obs, st)
            self.stats["plan_source"]["builtin"] = self.stats["plan_source"].get("builtin", 0) + 1
        self.plan = plan
        self.expand = None
        if self.p["land_expand"] and st["day"] == self.p["land_day"]:
            ex = self._expansion(st)
            if ex is not None:
                self.expand = ex
                for xy in ex["tiles"]:
                    self.plan[xy] = {"crop": ex["crop"], "fert": True, "value": float(self.p["land_value"]), "dig": False}
                self.stats["expansion"] = {"day": st["day"], "quadrant": ex["quadrant"], "n": len(ex["tiles"]),
                                           "shops": ex["shops"]}
        self.stats["ms_by_hour0"].append(round((time.perf_counter() - t0) * 1000.0, 1))

    QUADRANT_TILES = {"NE": (range(5, 10), range(0, 5)), "SW": (range(0, 5), range(5, 10)),
                      "SE": (range(5, 10), range(5, 10))}

    def _expansion(self, st):
        """The tape's day-18 land + tomato program (see DEFAULTS land_*): {"quadrant", "price", "tiles", "crop"} or None."""
        nl = M.next_land(st["me"]["unlocked"])
        if nl is None or nl[0] not in self.QUADRANT_TILES:
            return None
        q, price = nl
        shops = list(st["market"]["shops"])
        k = sum(1 for sh in shops if sh in ("PIZZA_SHOP", "FARMERS_MARKET"))
        if k < self.p["land_min_shops"]:
            return None
        n = int(self.p["land_n"])
        cost = price + n * M.CROPS["TOMATO"]["seed"]
        if st["money"] - cost < self.p["land_min_cash"]:
            return None
        xs, ys = self.QUADRANT_TILES[q]
        tiles = sorted(((x, y) for x in xs for y in ys), key=lambda t: (abs(t[0] - 4.5) + abs(t[1] - 4.5), t[1], t[0]))
        return {"quadrant": q, "price": price, "tiles": tiles[:n], "crop": "TOMATO", "shops": k, "bought": False}

    def _plan_from_module(self, obs, st):
        """planner.allocate.plan_day(obs, tracker=...) -> our plan dict, or None if the result is unusable.
        Accepted shapes: res["plant"] = [{"x", "y", "crop", "fertilize", "value", "dig"}, ...] or [(x, y, crop, hour)];
        extras kept in self.alloc: fertilize [(x, y)], animals {sp: n}, place [(x, y, sp)], build [(x, y, kind)],
        land (bool), buy {item: n}, wheat_reserve."""
        try:
            res = _ALLOC.plan_day(obs, tracker=self.tracker if self.tracker.history else None)
        except TypeError:
            res = _ALLOC.plan_day(obs)
        if not isinstance(res, dict):
            return None
        items = res.get("plant")
        if not isinstance(items, (list, tuple)):
            return None
        tiles = obs["farms"][int(obs["player"])]["tiles"]
        plan = {}
        for it in items:
            if isinstance(it, dict):
                x, y, c = it.get("x"), it.get("y"), it.get("crop")
                fert, val, dig = bool(it.get("fertilize", False)), float(it.get("value", 50.0) or 0.0), bool(it.get("dig", False))
            elif isinstance(it, (list, tuple)) and len(it) >= 3:
                x, y, c = it[0], it[1], it[2]
                fert, val, dig = False, 50.0, False
            else:
                continue
            if c not in M.CROPS or not isinstance(x, int) or not isinstance(y, int) or not (0 <= x < 10 and 0 <= y < 10):
                continue
            t = tiles[y][x]
            if t == "LOCKED" or M.is_animal(t):
                continue
            need_dig = isinstance(t, dict) and (t.get("kind") in ("COOP", "PASTURE") or (M.is_plant(t) and dig))
            plan[(x, y)] = {"crop": c, "fert": fert, "value": max(10.0, val), "dig": need_dig}
        ex = {"fertilize": set(), "animals": {}, "place": [], "build": [], "land": False, "buy": {}, "wheat_reserve": None}
        for xy in res.get("fertilize") or []:
            try:
                ex["fertilize"].add((int(xy[0]), int(xy[1])))
            except Exception:              # noqa: BLE001
                pass
        if isinstance(res.get("animals"), dict):
            ex["animals"] = {k: int(v) for k, v in res["animals"].items() if k in M.ANIMALS and int(v) > 0}
        for e in res.get("place") or []:
            if isinstance(e, (list, tuple)) and len(e) >= 3 and e[2] in M.ANIMALS:
                ex["place"].append((int(e[0]), int(e[1]), e[2]))
        for e in res.get("build") or []:
            if isinstance(e, (list, tuple)) and len(e) >= 3 and e[2] in ("COOP", "PASTURE"):
                ex["build"].append((int(e[0]), int(e[1]), e[2]))
        ex["land"] = bool(res.get("land"))
        if isinstance(res.get("buy"), dict):
            ex["buy"] = {k: int(v) for k, v in res["buy"].items() if k in TOOLS and int(v) > 0}
        if res.get("wheat_reserve") is not None:
            try:
                ex["wheat_reserve"] = int(res["wheat_reserve"])
            except Exception:              # noqa: BLE001
                pass
        self.alloc = ex
        self.alloc_bought = False
        return plan

    def _merge_plans(self, obs, st, alloc_plan):
        """plan_merge: the built-in plan (mirror: harvested WHEAT / CARROT / MELON tiles keep their crop, other free
        tiles up to the field target get the best short crop) plus planner.allocate's plantings of the crops in
        merge_crops (default the long crops TOMATO / MELON / STRAWBERRY) on every tile that is not a mirror tile;
        allocate's extras (animals, BUILD / PLACE, land, wheat reserve) are kept, its FERTILIZE list only with
        merge_fertilize.  Why: on the 120 leader tapes allocate's plan alone planted 37 fewer wheat and 7 fewer
        carrots per game than LiveRepair (-$5.5k wheat, -$1.5k carrot margin) while its long crops and animals
        gained (+$6.2k tomato / melon / strawberry, +$1.1k animal products) -- runs/planner/diag_HYB12b.jsonl."""
        base = self._plan_builtin(obs, st)
        keep = set(self.p["merge_crops"])
        out = dict(base)
        n = 0
        for xy, pp in alloc_plan.items():
            if pp["crop"] not in keep:
                continue
            cur = base.get(xy)
            if cur is not None and cur.get("mirror"):
                continue
            out[xy] = pp
            n += 1
        if self.alloc is not None and not self.p["merge_fertilize"]:
            self.alloc["fertilize"] = set()
        self.stats["merge_long"] = self.stats.get("merge_long", 0) + n
        return out

    def _free_tiles(self, st):
        """Tiles a crop can be started on today: {(x, y): {"hour": ready hour estimate, "dig": needs an explicit DIG}}."""
        me = st["me"]
        day = st["day"]
        out = {}
        spots = [((x, y), 0) for (x, y) in me["empty"]] + [((x, y), 1) for (x, y) in me["weeds"]]
        if not self.p["fill_empty"]:
            if getattr(self, "field_target", None) is None:
                self.field_target = len(me["plants"]) + min(len(spots), int(self.p["fill_cap"] or 0))
            room = self.field_target - len(me["plants"])
            spots.sort(key=lambda e: (abs(e[0][0] - 4.5) + abs(e[0][1] - 4.5), e[0]))
            spots = spots[:max(0, room)]
        for (xy, w) in spots:
            out[xy] = {"hour": self.p["plant_hour_free"] + w, "dig": False}   # the executor DIGs a weed itself
        for pl in me["plants"]:
            key = (pl["x"], pl["y"])
            v = self.values.get(key, {})
            if not pl["ongoing"]:
                if "HARVEST" in v.get("_ops", ()):
                    out[key] = {"hour": self.p["plant_hour_after"], "dig": False, "prev": pl["crop"]}
            else:
                t = {"crop": pl["crop"], "planted_day": pl["planted_day"]}
                if ongoing_finished(t, day) and (pl["yield"] == 0 or "HARVEST" in v.get("_ops", ())):
                    out[key] = {"hour": self.p["plant_hour_after"], "dig": True}
        return out

    def _forecast(self, obs, st):
        """Market forecast: observed book, exact drain, rival schedule, plus our own projected output."""
        try:
            rs = M.rival_sales_schedule(obs, self.tracker if self.tracker.history else None)
        except Exception:                  # noqa: BLE001
            rs = None
        fc = M.forecast_from_obs(obs, rival_schedule=rs)
        day = st["day"]
        farm = obs["farms"][int(obs["player"])]
        proj = M.project_farm_output(farm, day)
        feeds = []
        for d in range(day, LAST_DAY + 1):
            feeds.append(sum(1 for a in st["me"]["animals"] if feed_today(a["species"], a["placed_day"], d)))
        hours = (5, 9, 13, 17, 21)
        for P, row in proj.items():
            if P == "FERTILIZER":
                continue                     # mostly used / sold in small lots: left out of the book forecast
            for i, u in enumerate(row):
                d = day + i
                if P == "WHEAT":
                    u = u - feeds[i]
                n = int(round(u))
                if n == 0:
                    continue
                sgn = 1 if n > 0 else -1
                n = abs(n)
                for k in range(n):
                    h = hours[k % len(hours)]
                    s = min(LAST_STEP, d * TPD + h)
                    if s >= fc.step0:
                        fc.add_ours(P, s, sgn)
        return fc

    def _crop_option(self, crop, day, hour, fc, fert_value, cache):
        """Best (fert or not) option of planting `crop` on (day, hour) given forecast fc: dict or None."""
        best = None
        ferts = (False, True) if (M.CROPS[crop]["ongoing"] or self.p["fert_new_single"]) else (False,)
        for fert in ferts:
            key = (crop, day, hour, fert)
            res = cache.get(key)
            if res is None:
                res = M.crop_outcome(crop, day, hour, fert, 6, LAST_STEP, self.p["sell_lag"])
                cache[key] = res
            if not res["plan"].get("feasible") or res["units"] <= 0:
                continue
            extra = {}
            for s, u in res["harvests"]:
                ss = min(LAST_STEP, s + self.p["sell_lag"])
                extra[ss] = extra.get(ss, 0) + u
            own0, riv0 = revenues(fc, crop)
            own1, riv1 = revenues(fc, crop, extra)
            net = (own1 - own0) + self.p["denial"] * (riv0 - riv1) - M.CROPS[crop]["seed"] - res["ferts"] * fert_value
            tf = res["plan"].get("tile_free_day")
            occ = max(1, (tf if tf is not None else LAST_DAY + 1) - day)
            r = {"crop": crop, "fert": fert, "net": net, "occ": occ, "extra": extra, "units": res["units"]}
            if best is None or r["net"] > best["net"]:
                best = r
        return best

    def _plan_builtin(self, obs, st, only=None, fc=None):
        """Greedy margin allocation of today's free tiles (see module docstring)."""
        day = st["day"]
        free = self._free_tiles(st)
        if only is not None:
            free = {k: v for k, v in free.items() if k in only}
        if not free or day > LAST_DAY - 2:
            return {}
        if fc is None:
            fc = self._forecast(obs, st)
        fert_value = float(st["market"]["prices"].get("FERTILIZER", 70))
        crops = [c for c in self.p["new_crops"] if c in M.CROPS]
        cache = {}
        held = dict(st["seeds"])
        plan = {}
        by_hour = {}
        for key, f in free.items():
            by_hour.setdefault(f["hour"], []).append(key)
        n_new = 0
        for hour in sorted(by_hour):
            keys = sorted(by_hour[hour], key=lambda k: (abs(k[0] - 4.5) + abs(k[1] - 4.5)))
            opts = {c: self._crop_option(c, day, hour, fc, fert_value, cache) for c in crops}
            for key in keys:
                if n_new >= self.p["max_new_per_day"]:
                    break
                valid = {c: o for c, o in opts.items() if o is not None}
                if self.p["plan_mode"] == "mirror":
                    prev = free[key].get("prev")
                    allow = (prev,) if (prev in M.CROPS and day <= self.p["mirror_until"]) else tuple(self.p["short_crops"])
                    valid = {c: o for c, o in valid.items() if c in allow}
                lpd = self.p["last_plant_day"] or {}
                valid = {c: o for c, o in valid.items() if day <= lpd.get(c, LAST_DAY)}
                if not valid:
                    continue
                # seeds already bought are sunk
                for c, o in valid.items():
                    o["score_net"] = o["net"] + (M.CROPS[c]["seed"] if held.get(c, 0) > 0 else 0)
                short = [o for c, o in valid.items() if c in self.p["short_crops"]]
                lam = max([max(0.0, o["score_net"] / o["occ"]) for o in short] or [0.0])
                days_left = LAST_DAY + 1 - day
                best = None
                for c, o in valid.items():
                    if o["score_net"] < self.p["min_net"]:
                        continue
                    if c in self.p["short_crops"]:
                        sc = o["score_net"] - lam * min(o["occ"], days_left)
                    else:
                        sc = o["score_net"] - lam * min(o["occ"], days_left) - self.p["long_margin"]
                    if best is None or sc > best[0] + 1e-9:
                        best = (sc, c, o)
                if best is None:
                    break
                _, c, o = best
                plan[key] = {"crop": c, "fert": o["fert"], "value": max(10.0, o["score_net"]), "dig": free[key]["dig"],
                             "mirror": bool(free[key].get("prev"))}
                n_new += 1
                if held.get(c, 0) > 0:
                    held[c] -= 1
                for s, u in o["extra"].items():
                    fc.add_ours(c, s, u)
                opts[c] = self._crop_option(c, day, hour, fc, fert_value, cache)
        return plan

    def _replan_free(self, obs, st):
        """Mid-day: tiles that became free without a plan (e.g. a crop died, a harvest came early) get one."""
        if st["day"] > LAST_DAY - 2:
            return
        free = self._free_tiles(st)
        missing = {k for k in free if k not in self.plan}
        if not missing:
            return
        try:
            extra = self._plan_builtin(obs, st, only=missing)
        except Exception:                  # noqa: BLE001
            extra = {}
        self.plan.update(extra)

    # -----------------------------------------------------------------------------------------------------------------
    # values and tasks
    # -----------------------------------------------------------------------------------------------------------------
    def _unit_value(self, st, P):
        return float(st["market"]["prices"].get(P, M.MARKET_PARAMS[P]["base"]))

    def _fert_policy_for(self, st, pl):
        key = (pl["x"], pl["y"], pl["planted_day"])
        if key in self.fert_policy:
            return self.fert_policy[key]
        crop = pl["crop"]
        uv = self._unit_value(st, crop)
        fv = self._unit_value(st, "FERTILIZER")
        if M.CROPS[crop]["ongoing"]:
            pol = 2 * uv > fv
        else:
            pol = 2 * uv > self.p["fert_single_ratio"] * fv
        self.fert_policy[key] = pol
        return pol

    def _compute_values(self, obs, st):
        day, hour = st["day"], st["hour"]
        fv = self._unit_value(st, "FERTILIZER")
        wv = self._unit_value(st, "WHEAT")
        farm = obs["farms"][int(obs["player"])]
        tiles = farm["tiles"]
        eps = self.p["harvest_eps"]
        for pl in st["me"]["plants"]:
            x, y = pl["x"], pl["y"]
            tile = dict(tiles[y][x])
            uv = self._unit_value(st, pl["crop"])
            pol = self._fert_policy_for(st, pl)
            vals, ops, ha = plant_op_values(tile, day, uv, fv, pol, max(1, hour), eps)
            vals["_ops"] = tuple(ops)
            vals["_pd"] = pl["planted_day"]
            self.values[(x, y)] = vals
        for an in st["me"]["animals"]:
            x, y = an["x"], an["y"]
            tile = dict(tiles[y][x])
            uv = self._unit_value(st, an["product"])
            if self.p["animal_uv_frac"]:
                uv = max(uv, self.p["animal_uv_frac"] * M.MARKET_PARAMS[an["product"]]["base"])
            vals = M.animal_op_values(tile, day, uv, fv, wv, LAST_REFRESH_DAY)
            if tile["yield_units"] > 0:
                vals["HARVEST"] = max(0.0, vals.get("HARVEST", 0.0)) + eps * tile["yield_units"] * uv
            if not care_today(an["species"], an["placed_day"], day):
                vals["CARE"] = min(vals.get("CARE", 0.0), 0.0)
            self.values[(x, y)] = vals

    def _must_deadline(self, step, day, eod):
        h = self.p["must_hour"]
        if h is None:
            return eod
        s = day * TPD + h
        return s if step < s else eod

    def _tasks(self, obs, st):
        if not self.p["static_tasks"]:
            return self._static_tasks(obs, st) + self._dynamic_tasks(obs, st)
        if getattr(self, "_task_day", None) != (st["day"], len(self.plan)):
            self._day_tasks = self._static_tasks(obs, st)
            self._task_day = (st["day"], len(self.plan))
        tiles = obs["farms"][int(obs["player"])]["tiles"]
        live = []
        for t in self._day_tasks:
            pd = t.get("pd")
            if pd is not None:
                tl = tiles[t["y"]][t["x"]]
                if not (isinstance(tl, dict) and tl.get("kind") == "PLANT" and tl.get("planted_day") == pd):
                    continue                    # that plant is gone (harvested / replanted / dead)
            if t.get("dl2") is not None and st["step"] > t["deadline"]:
                t = dict(t)                     # an early (must) deadline has passed: still due by the end of the day
                t["deadline"] = t["dl2"]
            live.append(t)
        return live + self._dynamic_tasks(obs, st)

    def _static_tasks(self, obs, st):
        day, step = st["day"], st["step"]
        eod = min(day * TPD + TPD - 1, LAST_STEP)
        me = st["me"]
        tasks = []
        minv = self.p["min_task_value"]
        seen = set()
        # inherited obligations (escapes tonight, dry deaths tonight, last full-value harvests)
        feedable = {(a["x"], a["y"]) for a in me["animals"] if feed_today(a["species"], a["placed_day"], day)}
        for t in H.commitment_tasks(st):
            if t["kind"] in ("PLANT", "PLACE") and t["x"] < 0:
                continue
            if t["kind"] == "FEED" and (t["x"], t["y"]) not in feedable:
                continue                           # no production left: an escape costs nothing
            t = dict(t)
            t["value"] = max(t["value"], minv)
            if t["kind"] == "WATER":
                t["value"] = max(t["value"], self.p["floor_must_water"])
                t["dl2"] = t["deadline"]
                t["deadline"] = min(t["deadline"], self._must_deadline(step, day, eod))
            elif t["kind"] == "FEED":
                t["value"] = max(t["value"], self.p["floor_must_feed"])
                t["dl2"] = t["deadline"]
                t["deadline"] = min(t["deadline"], self._must_deadline(step, day, eod))
            tasks.append(t)
            seen.add((t["kind"], t["x"], t["y"]))
        # plants
        P = self.p
        for pl in me["plants"]:
            key = (pl["x"], pl["y"])
            vals = self.values.get(key)
            if vals is None or vals.get("_pd") != pl["planted_day"]:
                vals = {"_ops": ()}             # a plant started after the values were computed
            for op in vals.get("_ops", ()):
                if (op, key[0], key[1]) in seen:
                    continue
                v = vals.get(op, 0.0)
                if op == "WATER":
                    v = max(v, P["floor_must_water"] if pl["must_water_today"] else P["floor_water"])
                elif op == "HARVEST":
                    v = max(v, P["floor_plant_harvest"])
                if v < minv:
                    continue
                dl = eod
                dl2 = None
                if op == "WATER" and pl["must_water_today"]:
                    dl = self._must_deadline(step, day, eod)
                    dl2 = eod
                if op == "HARVEST" and pl["decay_step"] is not None and not pl["ongoing"]:
                    dl = min(eod, max(step, pl["decay_step"] - 1))
                if op == "HARVEST" and day == LAST_DAY:
                    dl = min(dl, LAST_STEP - 2)
                t = {"kind": op, "x": key[0], "y": key[1], "item": pl["crop"] if op == "HARVEST" else None, "n": 1,
                     "earliest": step, "deadline": dl, "value": float(v), "needs": {"FERTILIZER": 1} if op == "FERTILIZE" else {},
                     "pd": pl["planted_day"]}
                if dl2 is not None:
                    t["dl2"] = dl2
                tasks.append(t)
        # animals
        for an in me["animals"]:
            key = (an["x"], an["y"])
            vals = self.values.get(key, {})
            feed_ok = feed_today(an["species"], an["placed_day"], day)
            for op in ("FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"):
                if (op, key[0], key[1]) in seen:
                    continue
                v = vals.get(op, 0.0)
                if op == "FEED" and feed_ok and not an["fed_today"]:
                    v = max(v, P["floor_must_feed"] if an["must_feed_today"] else P["floor_feed"])
                elif op == "CARE" and feed_ok and not an["cared_today"] and care_today(an["species"], an["placed_day"], day):
                    v = max(v, P["floor_care"])
                elif op == "HARVEST" and an["yield"] > 0:
                    v = max(v, P["floor_animal_harvest"])
                elif op == "COLLECT_FERTILIZER" and an["fertilizer_available"]:
                    v = max(v, P["floor_collect"])
                if v < minv:
                    continue
                dl = eod if day < LAST_DAY else LAST_STEP - 2
                dl2 = None
                if op == "FEED" and an["must_feed_today"] and feed_ok:
                    dl2 = dl
                    dl = self._must_deadline(step, day, dl)
                t = {"kind": op, "x": key[0], "y": key[1], "item": an["product"] if op == "HARVEST" else None, "n": 1,
                     "earliest": step, "deadline": dl, "value": float(v), "needs": {"WHEAT": 1} if op == "FEED" else {}}
                if dl2 is not None:
                    t["dl2"] = dl2
                tasks.append(t)
        # allocate extras: FERTILIZE of existing plants, BUILD / PLACE for new animals
        if self.alloc:
            for (x, y) in self.alloc["fertilize"]:
                if ("FERTILIZE", x, y) in seen or any(t["kind"] == "FERTILIZE" and (t["x"], t["y"]) == (x, y) for t in tasks):
                    continue
                v = self.values.get((x, y), {}).get("FERTILIZE", 40.0)
                tasks.append({"kind": "FERTILIZE", "x": x, "y": y, "item": None, "n": 1, "earliest": step,
                              "deadline": eod, "value": float(max(v, 20.0)), "needs": {"FERTILIZER": 1}})
            for (x, y, kind) in self.alloc["build"]:
                tasks.append({"kind": "BUILD_" + kind, "x": x, "y": y, "item": None, "n": 1, "earliest": step,
                              "deadline": eod, "value": 200.0, "needs": {}})
            for (x, y, sp) in self.alloc["place"]:
                tasks.append({"kind": "PLACE", "x": x, "y": y, "item": sp, "n": 1, "earliest": step,
                              "deadline": eod, "value": 400.0, "needs": {sp: 1}})
        # plantings of the day plan
        if day <= LAST_DAY - 1:
            pdl = eod - 2
            if self.p["plant_last_hour"] is not None:
                pdl = min(pdl, day * TPD + int(self.p["plant_last_hour"]))
            for (x, y), pp in self.plan.items():
                tile = obs["farms"][int(obs["player"])]["tiles"][y][x]
                if M.is_plant(tile) and tile["planted_day"] == day:
                    continue
                if pp["dig"] and M.is_plant(tile):
                    tasks.append({"kind": "DIG", "x": x, "y": y, "item": None, "n": 1, "earliest": step,
                                  "deadline": pdl, "value": float(pp["value"]), "needs": {}})
                tasks.append({"kind": "PLANT", "x": x, "y": y, "item": pp["crop"], "n": 1, "earliest": step,
                              "deadline": pdl, "value": float(pp["value"]), "needs": {}})
        self._protect_critical(tasks, day, eod)
        return tasks

    def _protect_critical(self, tasks, day, eod):
        """The executor performs a tile's acts in the engine's legal order (FERTILIZE < WATER < HARVEST < COLLECT <
        FEED < CARE): a unit reaching a tile at the last step of the day does the FERTILIZE and the must-WATER is
        lost (the plant dies at midnight), or HARVEST / COLLECT and the must-FEED is lost (the animal escapes).
        On tiles with a critical task (must-water / must-feed), every other task gets deadline <= eod - crit_gap,
        so late in the day only the critical act is left there."""
        gap = self.p["crit_gap"]
        if not gap or day >= LAST_DAY:
            return
        crit = set()
        for t in tasks:
            if (t["kind"] == "WATER" and t["value"] >= self.p["floor_must_water"] - 1e-9) or \
               (t["kind"] == "FEED" and t["value"] >= self.p["floor_must_feed"] - 1e-9):
                crit.add((t["x"], t["y"], t["kind"]))
        tiles = {(x, y) for (x, y, _) in crit}
        for t in tasks:
            key = (t["x"], t["y"])
            if key in tiles and (t["x"], t["y"], t["kind"]) not in crit:
                t["deadline"] = min(t["deadline"], eod - gap)
                if t.get("dl2") is not None:
                    t["dl2"] = min(t["dl2"], eod - gap)

    def _pending_harvest_units(self, obs, st):
        """Units the day's pending HARVEST tasks will still add to the carried cargo."""
        tiles = obs["farms"][int(obs["player"])]["tiles"]
        day = st["day"]
        n = 0
        for t in getattr(self, "_day_tasks", None) or []:
            if t["kind"] != "HARVEST":
                continue
            x, y = t["x"], t["y"]
            if not (0 <= x < 10 and 0 <= y < 10):
                continue
            tl = tiles[y][x]
            if isinstance(tl, dict) and X.action_valid("HARVEST", X.tile_state(tl), day):
                n += int(tl.get("yield_units", 0))
        return n

    def _pending_tool_needs(self, obs, st):
        """Tool units still needed today by pending tasks: {"WHEAT": feeds, "FERTILIZER": fertilizations}."""
        tiles = obs["farms"][int(obs["player"])]["tiles"]
        day = st["day"]
        need = {"WHEAT": 0, "FERTILIZER": 0}
        for t in getattr(self, "_day_tasks", None) or []:
            k = t["kind"]
            if k not in ("FEED", "FERTILIZE"):
                continue
            x, y = t["x"], t["y"]
            if not (0 <= x < 10 and 0 <= y < 10):
                continue
            if X.action_valid(k, X.tile_state(tiles[y][x]), day):
                need["WHEAT" if k == "FEED" else "FERTILIZER"] += 1
        return need

    def _dynamic_tasks(self, obs, st):
        """Tasks that depend on the moment: seedling waterings, deposits of carried goods / cargo."""
        day, step = st["day"], st["step"]
        eod = min(day * TPD + TPD - 1, LAST_STEP)
        tasks = []
        tiles = obs["farms"][int(obs["player"])]["tiles"]
        for (x, y), pp in self.plan.items():
            tile = tiles[y][x]
            if M.is_plant(tile) and tile["planted_day"] == day:
                self.fert_policy[(x, y, day)] = pp["fert"]
                if not tile["watered_today"]:          # a seedling dies tonight unless watered today
                    tasks.append({"kind": "WATER", "x": x, "y": y, "item": None, "n": 1, "earliest": step,
                                  "deadline": eod, "value": float(pp["value"]), "needs": {}})
        # deposits of carried goods (for same-day sales; nothing is sold from a unit's hands) and of wheat / fertilizer
        # cargo when the midnight load (shed + everything carried drops in at midnight, the overflow is destroyed) is
        # too big; deadline hour 22 so what is deposited can still be sold before midnight
        hour = st["hour"]
        last_day = day >= LAST_DAY
        dl = min(eod, day * TPD + self.p["drop_deadline_hour"]) if not last_day else LAST_STEP - 1
        if step > dl:
            return tasks
        load = st["shed_total"] + sum(st["carried"].values()) + self._pending_harvest_units(obs, st)
        tight = load > 100 - self.p["room_margin"]
        in_win = (any(a <= hour <= b for a, b in self.p["drop_windows"])
                  or (hour >= self.p["goods_hour"] and (self.p["goods_always"] or tight))
                  or (last_day and hour >= self.p["final_drop_hour"]))
        if in_win:
            prices = st["market"]["prices"]
            tot = 0.0
            for u in st["units"]:
                for it, c in u["inv"].items():
                    if it in GOODS:
                        tot += c * prices.get(it, 0)
            if tot >= self.p["drop_min_value"]:
                k_val = self.p["drop_value_k"]
                if self.p["drop_goods_mode"] == "timing" and not tight and not last_day:
                    k_val = self.p["drop_timing_k"]      # no overflow risk: only the same-day sale is at stake
                tasks.append({"kind": "DROP", "x": 4, "y": 4, "item": None, "n": 0, "earliest": step, "deadline": dl,
                              "value": k_val * tot, "needs": {}})
        if (hour >= self.p["cargo_hour"] and tight) or (last_day and hour >= self.p["final_drop_hour"]):
            pending = self._pending_tool_needs(obs, st)
            over = load - (100 - self.p["room_margin"])
            prices = st["market"]["prices"]
            for it in TOOLS:
                c = st["carried"].get(it, 0)
                if self.p["cargo_mode"] == "skip":
                    if pending.get(it, 0) > 0 and not (last_day and hour >= 20):
                        continue               # still needed by a pending FEED / FERTILIZE: no deposit/pickup loop
                    n = c
                    v = self.p["cargo_value"] * c
                else:
                    # deposit the surplus over what today's pending FEED / FERTILIZE still need; worth what the
                    # midnight overflow would destroy (plus the plain cargo value)
                    keep = 0 if (last_day and hour >= 20) else pending.get(it, 0) + self.p["cargo_keep"]
                    n = c - keep
                    v = self.p["cargo_value"] * max(0, n) + \
                        self.p["cargo_price_k"] * max(0, min(n, over)) * float(prices.get(it, 30))
                if n > 0 and c > 0:
                    tasks.append({"kind": "DROP", "x": 4, "y": 4, "item": it, "n": n, "earliest": step, "deadline": dl,
                                  "value": v, "needs": {}})
        return tasks

    # -----------------------------------------------------------------------------------------------------------------
    # market
    # -----------------------------------------------------------------------------------------------------------------
    def _wheat_need(self, st, day, include_done=False):
        n = 0
        for a in st["me"]["animals"]:
            if feed_today(a["species"], a["placed_day"], day):
                if day == st["day"] and a["fed_today"] and not include_done:
                    continue
                n += 1
        return n

    def _fert_need(self, st, days):
        """Planned FERTILIZE ops over the next `days` days (today's pending + baseline)."""
        n = 0
        day = st["day"]
        for pl in st["me"]["plants"]:
            v = self.values.get((pl["x"], pl["y"]), {})
            if "FERTILIZE" in v.get("_ops", ()) and pl["fertilized_until_day"] < day:
                n += 1
            if pl["ongoing"] and self._fert_policy_for(st, pl):
                for r in M.production_refresh_days(pl["crop"], pl["planted_day"]):
                    if day < r <= min(day + days - 1, LAST_REFRESH_DAY) and r > pl["fertilized_until_day"]:
                        n += 1
                        break
        for pp in self.plan.values():
            if pp["fert"]:
                n += 1
        return n

    def _reserve(self, st):
        day, hour = st["day"], st["hour"]
        res = {}
        if day < self.p["dump_day"]:
            carried_w = st["carried"].get("WHEAT", 0)
            need = self._wheat_need(st, day) + sum(self._wheat_need(st, d, True) for d in range(day + 1, day + 1 + self.p["wheat_keep_days"]))
            res["WHEAT"] = max(0, need + self.p["feed_buffer"] - carried_w)
            if self.alloc and self.alloc.get("wheat_reserve") is not None:
                res["WHEAT"] = max(res["WHEAT"], self.alloc["wheat_reserve"] - carried_w)
            carried_f = st["carried"].get("FERTILIZER", 0)
            res["FERTILIZER"] = max(0, self._fert_need(st, self.p["fert_reserve_days"]) + self.p["fert_buffer"] - carried_f)
        else:
            res["WHEAT"] = max(0, self._wheat_need(st, day) + (self._wheat_need(st, day + 1, True) if hour >= 20 else 0)
                               - st["carried"].get("WHEAT", 0))
            res["FERTILIZER"] = max(0, self._fert_need(st, 1) - st["carried"].get("FERTILIZER", 0))
        return res

    def _priority_orders(self, obs, st, tasks):
        """Buys: seeds for the plan, feed wheat, fertilizer shortfall.  (Before sells; they must fit in the cap.)"""
        day, hour, step = st["day"], st["hour"], st["step"]
        money = st["money"]
        out = []
        if step >= LAST_STEP:
            return out
        shed = st["shed"]
        ex = getattr(self, "expand", None)
        if ex is not None and not ex["bought"]:
            if ex["quadrant"] in st["me"]["unlocked"]:
                ex["bought"] = True
            elif money >= ex["price"]:
                out.append(["BUY_LAND"])       # re-sent next step if it did not execute (checked on the obs)
                money -= ex["price"]
        # seeds for today's plantings (bought at hour 0 -> usable from hour 1; any time a shortfall appears)
        if hour <= 20 and self.plan:
            want = {}
            farm_tiles = obs["farms"][int(obs["player"])]["tiles"]
            for (x, y), pp in self.plan.items():
                tile = farm_tiles[y][x]
                if M.is_plant(tile) and tile["planted_day"] == day:
                    continue
                want[pp["crop"]] = want.get(pp["crop"], 0) + 1
            for c, n in sorted(want.items()):
                k = n - st["seeds"].get(c, 0)
                cost = k * M.CROPS[c]["seed"]
                if k > 0 and cost <= money - 50:
                    out.append(["BUY_SEED", c, k])
                    money -= cost
        # allocate's purchases (once, at the first step of the day plan)
        if self.alloc and not self.alloc_bought:
            self.alloc_bought = True
            if self.alloc["land"]:
                nl = M.next_land(st["me"]["unlocked"])
                if nl is not None and nl[1] <= money - 500:
                    out.append(["BUY_LAND"])
                    money -= nl[1]
            for sp, n in sorted(self.alloc["animals"].items()):
                cost = n * M.ANIMALS[sp]["cost"]
                if cost <= money - 300:
                    out.append(["BUY_ANIMAL", sp, n])
                    money -= cost
            for it, n in sorted(self.alloc["buy"].items()):
                out.append(["BUY_PRODUCT", it, n])
        # feed wheat
        wheat_have = shed.get("WHEAT", 0) + st["carried"].get("WHEAT", 0)
        need_today = self._wheat_need(st, day)
        if hour < 23 and need_today > wheat_have:
            k = need_today - wheat_have + 1
            out.append(["BUY_PRODUCT", "WHEAT", k])
            wheat_have += k
        elif hour == 23 and day < LAST_REFRESH_DAY + 1:
            need_tom = self._wheat_need(st, day + 1, True) + self.p["feed_buffer"]
            if need_tom > wheat_have:
                k = need_tom - wheat_have
                room = 100 - st["shed_total"] - sum(st["carried"].values())
                k = min(k, max(0, room - 1))
                if k > 0:
                    out.append(["BUY_PRODUCT", "WHEAT", k])
        # fertilizer shortfall for today's FERTILIZE tasks
        if hour < 20 and day <= self.p["fert_buy_last_day"]:
            need_f = sum(1 for t in tasks if t["kind"] == "FERTILIZE")
            have_f = shed.get("FERTILIZER", 0) + st["carried"].get("FERTILIZER", 0)
            if need_f > have_f:
                out.append(["BUY_PRODUCT", "FERTILIZER", need_f - have_f])
        return out

    @staticmethod
    def _clamp_inputs(st, sells, reserve):
        """clamp_inputs: WHEAT / FERTILIZER SELLs (the tape's own, followed verbatim by the sales policy, which ignores
        context['reserve'] for followed items) never take the shed below the planner's reserve (tomorrow's FEED /
        planned FERTILIZE).  Without it the tape sold the fertilizer the planner's FERTILIZE tasks needed and the
        planner bought it back the same day (a buy-sell churn: dropping all fertilizer purchases removed ~50
        re-sold units / game, runs/planner/diag_cmp_I1_nofb_HYB12.txt)."""
        left = {P: max(0, int(st["shed"].get(P, 0)) - int(reserve.get(P, 0))) for P in TOOLS}
        out = []
        for o in sells:
            if o[1] in left:
                q = min(int(o[2]), left[o[1]])
                if q <= 0:
                    continue
                left[o[1]] -= q
                o = [o[0], o[1], q]
            out.append(o)
        return out

    def _room_topup(self, st, sells, reserve):
        """Hour 23 / last step: extra SELLs so that everything carried fits into the shed at midnight."""
        if not (st["hour"] == 23 or st["step"] >= LAST_STEP):
            return sells
        shed = dict(st["shed"])
        for o in sells:
            shed[o[1]] = shed.get(o[1], 0) - int(o[2])
        need = sum(st["carried"].values()) + self.p["room_margin"] - (100 - sum(max(0, v) for v in shed.values()))
        if st["step"] >= LAST_STEP:
            need = 10 ** 6                      # nothing is worth anything after step 718
        if need <= 0:
            return sells
        extra = {}
        for P in ("FERTILIZER",) + GOODS + ("WHEAT",):
            if need <= 0:
                break
            q = min(max(0, shed.get(P, 0)), need)
            if q > 0:
                extra[P] = q
                need -= q
        out = [list(o) for o in sells]
        for o in out:
            if o[1] in extra:
                o[2] = int(o[2]) + extra.pop(o[1])
        for P, q in extra.items():
            out.append(["SELL", P, q])
        return out

    def _tape_market(self, st, tape_act):
        """The tape's SELL (and, in mode "all", BUY_PRODUCT) orders of this step, clamped to our shed stock / room."""
        mode = self.p["tape_market"]
        if mode == "off" or tape_act is None:
            return None
        shed = dict(st["shed"])
        room = 100 - st["shed_total"]
        out = []
        for o in tape_act.get("market") or []:
            if not isinstance(o, (list, tuple)) or len(o) < 3:
                continue
            try:
                n = int(o[2])
            except Exception:              # noqa: BLE001
                continue
            if o[0] == "SELL" and o[1] in shed:
                q = min(n, shed.get(o[1], 0))
                if q > 0:
                    out.append(["SELL", o[1], q])
                    shed[o[1]] -= q
                    room += q
            elif o[0] == "BUY_PRODUCT" and mode == "all" and o[1] in TOOLS:
                q = min(n, room)
                if q > 0:
                    out.append(["BUY_PRODUCT", o[1], q])
                    room -= q
        return out

    def _sell_orders(self, obs, st, tape_act, out):
        step, hour, day = st["step"], st["hour"], st["day"]
        reserve = self._reserve(st)
        # midnight room: at hour 23 everything carried (after this step's unit actions) drops into the shed
        extra_room = 0
        if hour == 23 or step >= LAST_STEP:
            carried = sum(st["carried"].values())
            extra_room = carried + self.p["room_margin"] - (100 - st["shed_total"])
        tape_orders = self._tape_market(st, tape_act)
        tape_sells = None
        if tape_orders is not None:
            tape_sells = []            # already placed below; the stub only clears leftovers / room / the last day
        elif self.p["tape_sells"] and tape_act is not None:
            tape_sells = [(o[1], o[2]) for o in (tape_act.get("market") or [])
                          if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] in GOODS + TOOLS]
        self._module_ok = False
        if self.sales_module is not None:
            try:
                ctx = {"other_orders": list(getattr(self, "_others", [])), "reserve": reserve,
                       "carried": dict(st["carried"]), "max_orders": X_MAX_ORDERS}
                if self.p["tape_sells_context"] and tape_act is not None:
                    ctx["tape_sells"] = [list(o) for o in (tape_act.get("market") or [])
                                         if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL"]
                res = self.sales_module.orders(obs, st, dict(st["shed"]), ctx)
                if isinstance(res, list):
                    self._module_ok = True
                    res = [list(o) for o in res if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL"]
                    if self.p["clamp_inputs"]:
                        res = self._clamp_inputs(st, res, reserve)
                        sm = self.sales_module
                        if hasattr(sm, "_net_of") and hasattr(sm, "pending_net"):
                            try:                   # keep the policy's own-flow bookkeeping (rival inference) exact
                                sm.pending_net = (st["step"], sm._net_of(obs, res + list(ctx["other_orders"]),
                                                                         dict(st["shed"])))
                            except Exception:      # noqa: BLE001
                                pass
                    return self._room_topup(st, res, reserve)
            except Exception as ex:        # noqa: BLE001 - the built-in policy takes over for this step
                self.stats["sales_errors"] = self.stats.get("sales_errors", 0) + 1
                self.stats["last_error"] = f"sales: {type(ex).__name__}: {ex}"[:200]
        if tape_orders is not None:
            st2 = dict(st)
            sh = dict(st["shed"])
            for o in tape_orders:
                if o[0] == "SELL":
                    sh[o[1]] = sh.get(o[1], 0) - o[2]
                else:
                    sh[o[1]] = sh.get(o[1], 0) + o[2]
            st2["shed"] = sh
            st2["shed_total"] = sum(sh.values())
            if extra_room > 0:
                extra_room = sum(st["carried"].values()) + self.p["room_margin"] - (100 - st2["shed_total"])
            rest = self.sales.orders(obs, st2, reserve, max(0, extra_room), tape_sells)
            merged = {}
            order = []
            for o in tape_orders + rest:
                k = (o[0], o[1])
                if k not in merged:
                    order.append(k)
                    merged[k] = 0
                merged[k] += o[2]
            return [[k[0], k[1], merged[k]] for k in order if merged[k] > 0]
        return self.sales.orders(obs, st, reserve, max(0, extra_room), tape_sells)


X_MAX_ORDERS = 10


def make_sales_policy(params):
    """planner.sales.SalesPolicy(params['sales_params']) when the module exists, else None."""
    if _SALES is None or not hasattr(_SALES, "SalesPolicy"):
        return None
    try:
        return _SALES.SalesPolicy(params.get("sales_params") or None)
    except Exception:                      # noqa: BLE001
        return None


def hybrid_agent(tape_agent, D=12, params=None):
    """Kaggle agent: `tape_agent` (obs, configuration) before step 24*D, the Planner from then on (via
    handover.make_hybrid).  The tape keeps running inside the planner (fallback / sale context) and is called with
    the same configuration the Kaggle runner passes."""
    code = getattr(tape_agent, "__code__", None)
    n_args = code.co_argcount if code is not None and hasattr(code, "co_argcount") else 2
    box = {"cfg": None}

    def tape_call(obs):
        return tape_agent(*[obs, box["cfg"]][:n_args])

    p = dict(DEFAULTS)
    p.update(params or {})
    sp = make_sales_policy(p) if p["sales"] == "module" else None
    start = 24 * D
    inner = H.make_hybrid(tape_agent, Planner.factory(tape=tape_call, params=params, sales_policy=sp), D)

    def agent(obs, configuration=None):
        box["cfg"] = configuration
        act = inner(obs, configuration)
        if sp is not None and int(_g(obs, "step", 0)) < start:
            try:                          # keep the policy's rival / clone trackers warm before the switch
                sp.observe(obs, act)
            except Exception:              # noqa: BLE001
                pass
        return act
    agent.handover = inner.handover
    agent.sales_policy = sp
    return agent
