"""Task executor (worker routing) for Kaggriculture.  Standard library only.

    from planner.executor import Executor
    ex = Executor()                                   # keep one instance for the whole game
    k = ex.suggest_hires(obs, tasks)                  # optional, at hour 0: hands worth having today
    out = ex.act(obs, tasks, hires_wanted=k)          # {"farmer": [...], "hands": [[...], ...], "market_prefix": [["HIRE"], ...]}
    action = Executor.merge(out, market_orders)       # engine-ready {"farmer", "hands", "market"}; orders past the
                                                      # engine's 10-per-turn cap come back in action["dropped"]
    ex.last_info                                      # seed_shortfall, need_tools, planned/unassigned value, overflow_risk, ms ...

Task dict (shared interface): kind (WATER HARVEST PLANT FEED CARE COLLECT_FERTILIZER FERTILIZE DIG BUILD_COOP
BUILD_PASTURE PLACE PICKUP DROP), x, y, item, n, earliest, deadline, value, needs.  A task list may be
regenerated every step or kept static for the day: every step the executor re-reads the observation and
keeps only tasks whose action would still change something (tile state, shed, inventories), so done
tasks disappear by themselves.  Deadlines are clamped to the end of the current day (hands vanish).

How it works (default mode "routes"):
  * jobs: the pending tasks of one tile, put into a legal order by projecting the tile state
    (FERTILIZE < WATER < HARVEST < ... < DIG < BUILD < PLACE < PLANT, each only once valid), plus implicit
    acts: a DIG in front of a PLANT/BUILD on a weed, a WATER after every PLANT (seedlings die unwatered);
    a DIG is never applied to today's planting or to a structure a BUILD task asks for;
  * routes: every unit keeps an ordered list of jobs across steps.  Routes are (re)built from two
    constructions -- an angular sweep from the shed (balanced by work, nearest-neighbour + 2-opt) and an
    event-driven dispatch (whichever unit is free first takes the nearest job; early deadlines first) --
    and the better one by objective is kept; between rebuilds they are repaired (done jobs removed, new
    jobs inserted where they gain most), unfinishable / deadline-missing jobs are relocated, and tail jobs
    of late routes are handed to early-finishing units;
  * a joint simulation walks every route act by act with travel, waiting for `earliest`, deadlines,
    carried tools (FEED needs wheat, FERTILIZE fertilizer, PLACE the animal; harvested wheat and collected
    fertilizer are usable later in the same route), shed pickups (at the route start when on a shed
    tile, or as a detour if worth it) and seeds.  Scarce seeds / shed stock are granted in time order
    across units.  The objective is task value, minus small penalties for late completion;
  * this step's action of each unit is the first stop of its simulated route: PICKUP at a shed tile,
    the next act on the job tile, or one move towards the stop (x first when |dx| >= |dy|).
Mode "greedy" (one tile per unit by value per step) is kept only as a benchmark baseline.

Engine facts relied on (all checked in tools/planner/test_executor.py on kagsim_trace and on the official
kaggle_environments interpreter):
  * units act in order farmer, hand 1, hand 2, ... on the live state; the market runs after all units,
    so a HIRE / BUY placed at step t shows at step t+1 (a PICKUP in the same step as the BUY gets nothing);
  * a hired hand spawns on the shed-access tile with the fewest units (NWSE order) and acts from the next
    step; hands vanish after hour 23 with their inventories dropped into the shed (overflow discarded);
    the farmer respawns at (4,4) every morning; the n-th hire of a day costs fib(n) = 1,1,2,3,5,8,...;
    a HIRE at hour 23 is paid for but the hand never acts; at most 10 market orders run per turn;
  * moves never collide, off-board moves are no-ops, units may enter LOCKED tiles; tile actions no-op on
    LOCKED tiles;
  * PICKUP / DROP / shed-PLACE work only on the four shed-access tiles (4,4) (5,4) (4,5) (5,5), even LOCKED
    ones; PICKUP is capped at the stock; carrying is unlimited; DROP discards what does not fit in the
    shed (cap 100) while PLACE <item> <n> on a shed tile deposits only what fits -> the executor never
    emits a DROP that would discard;
  * PLANT consumes a private seed; if the PLANT orders for a crop in one turn exceed the seeds held, ALL
    of them are dropped -> the executor never issues more PLANTs than seeds; PLANT on a WEED is a no-op;
  * a plant not watered on the day it is sown is a WEED at midnight -> no PLANT at hour 23;
  * a second identical action on the same tile in the same step (e.g. two WATERs) is not executed;
  * HARVEST before first_yield_day is a no-op; FERTILIZE then WATER (in the growth window) adds 2 yield,
    WATER then FERTILIZE adds 1.
"""
import heapq
import math
import time

DAY_STEPS = 24
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946]

CROP_FIRST_YIELD = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
CROP_ONGOING = {"WHEAT": False, "CARROT": False, "TOMATO": True, "STRAWBERRY": True, "MELON": False}
ANIMAL_STRUCT = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}

TILE_KINDS = frozenset(("WATER", "HARVEST", "PLANT", "FEED", "CARE", "COLLECT_FERTILIZER", "FERTILIZE",
                        "DIG", "BUILD_COOP", "BUILD_PASTURE", "PLACE"))
SHED_KINDS = frozenset(("PICKUP", "DROP"))
# Kinds that change what the tile *is*; a later action on the same tile may depend on them.
STATE_CHANGING = frozenset(("PLANT", "DIG", "BUILD_COOP", "BUILD_PASTURE", "PLACE", "HARVEST"))
# Order in which same-tile tasks are performed when several are valid at once:
# fertilize before water (the watering bonus reads the fertilizer), water before harvest (in-window
# watering adds yield to wheat/carrot/melon), clear before build/plant, build before place.
PRIORITY = ("FERTILIZE", "WATER", "HARVEST", "COLLECT_FERTILIZER", "FEED", "CARE", "DIG",
            "BUILD_COOP", "BUILD_PASTURE", "PLACE", "PLANT")
TOOL_ITEMS = ("WHEAT", "FERTILIZER", "GOOSE", "COW", "SHEEP")
TOOL_SET = frozenset(TOOL_ITEMS)
BIG = 10 ** 6

DEFAULT_PARAMS = {
    "mode": "routes",       # "routes" (route planner) or "greedy" (one-tile-per-unit baseline)
    # route mode
    "sweep_rot": 8,         # start angles tried by the sweep construction
    "rebuild_every": 4,     # try a fresh construction every k steps; keep it only if it scores better
    "rebuild_margin": 0.0,
    "two_opt_passes": 3,
    "improve_ms": 40,       # time budget for moving unfinishable jobs between routes
    "detour_value": 8.0,    # a mid-route shed detour for tools is taken if the tool acts are worth >= this per extra step
    "urgent_bonus": 8,      # dispatch: an early-deadline job still reachable in time counts as this many steps closer
    "w_time": 0.002,        # objective: penalty per step of job completion time (prefers earlier, balanced routes)
    "balance_moves": 12,    # max tail-job moves per step from late routes to early ones
    "max_hires_per_step": 8,  # HIREs issued per step (the engine executes <= 10 market orders per turn;
                              # the rest of the hires go out next step, leaving room for the caller's orders)
    "hire_persist_until": 6,  # a remembered hire target is pursued only up to this hour
    # shared
    "max_wait": 3,          # a unit may wait on a tile up to this many steps for a task's `earliest`
    # greedy (baseline) mode
    "stick": 0.35,          # score bonus for the tile a unit targeted last step (anti-thrash)
    "urg_window": 4,        # slack (steps) below which a job's score is boosted ...
    "urg_gain": 1.0,        # ... by up to this factor
    "look": 0.5,            # weight of the best follow-up job reachable from a job's tile (clustering)
    "look_k": 6,            # number of nearest jobs examined for the follow-up estimate
    "free_detour": 2,       # greedy: pickup detours of at most this many extra steps are always taken
    # shared
    "pick_cap": 12,         # max units of one tool item picked up in one PICKUP
    "pick_extra": 1,        # extra tool units picked above the unit's fair share
    "fresh_water_value": 30.0,  # value of watering a plant sown today when no task covers it
}


def fib_cost(n_already_today, mult=1):
    return mult * (FIB[n_already_today] if n_already_today < len(FIB) else 10 ** 9)


def shed_tiles(n):
    h = n // 2
    return ((h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h))


# ----------------------------------------------------------------------------------------------
# Tile state projection (just enough of the engine to know whether an action would have an effect)
# ----------------------------------------------------------------------------------------------
def tile_state(tile):
    """Compact mutable copy: [k, ...]. k in E(mpty) L(ocked) W(eed) P(lant) COOP PASTURE ?"""
    if tile is None:
        return {"k": "E"}
    if tile == "LOCKED":
        return {"k": "L"}
    k = tile.get("kind")
    if k == "PLANT":
        return {"k": "P", "crop": tile["crop"], "pd": tile["planted_day"], "wt": tile["watered_today"],
                "y": tile["yield_units"], "fu": tile.get("fertilized_until_day", -1)}
    if k == "WEED":
        return {"k": "W"}
    if k in ("COOP", "PASTURE"):
        if "animal" in tile:
            return {"k": k, "an": tile["animal"], "fed": tile["fed_today"], "car": tile["cared_today"],
                    "fa": tile["fertilizer_available"], "y": tile["yield_units"]}
        return {"k": k, "an": None}
    return {"k": "?"}


def action_valid(kind, st, day, item=None):
    """True iff performing `kind` on a tile in state `st` would change something."""
    k = st["k"]
    if kind == "WATER":
        return k == "P" and not st["wt"]
    if kind == "HARVEST":
        if k == "P":
            return st["y"] > 0 and day - st["pd"] >= CROP_FIRST_YIELD[st["crop"]]
        return k in ("COOP", "PASTURE") and st.get("an") is not None and st["y"] > 0
    if kind == "PLANT":
        return k == "E" and item in CROP_FIRST_YIELD
    if kind == "FEED":
        return st.get("an") is not None and not st["fed"]
    if kind == "CARE":
        return st.get("an") is not None and not st["car"]
    if kind == "COLLECT_FERTILIZER":
        return st.get("an") is not None and st["fa"]
    if kind == "FERTILIZE":
        return k == "P" and st["fu"] < day + 2
    if kind == "DIG":
        return k in ("W", "P") or (k in ("COOP", "PASTURE") and st.get("an") is None)
    if kind in ("BUILD_COOP", "BUILD_PASTURE"):
        return k == "E"
    if kind == "PLACE":
        return k == ANIMAL_STRUCT.get(item) and st.get("an") is None
    return False


def apply_action(kind, st, day, item=None):
    k = st["k"]
    if kind == "WATER":
        st["wt"] = True
    elif kind == "HARVEST":
        if k == "P" and not CROP_ONGOING[st["crop"]]:
            st.clear(); st["k"] = "E"
        else:
            st["y"] = 0
    elif kind == "PLANT":
        st.clear(); st.update(k="P", crop=item, pd=day, wt=False, y=0 if CROP_ONGOING[item] else 1, fu=-1)
    elif kind == "FEED":
        st["fed"] = True
    elif kind == "CARE":
        st["car"] = True
    elif kind == "COLLECT_FERTILIZER":
        st["fa"] = False
    elif kind == "FERTILIZE":
        st["fu"] = max(st["fu"], day + 2)
    elif kind == "DIG":
        st.clear(); st["k"] = "E"
    elif kind == "BUILD_COOP":
        st.clear(); st.update(k="COOP", an=None)
    elif kind == "BUILD_PASTURE":
        st.clear(); st.update(k="PASTURE", an=None)
    elif kind == "PLACE":
        st.update(an=item, fed=False, car=False, fa=False, y=0)


class Job:
    """All pending actions on one tile (or one unit's shed errand), in executable order."""
    __slots__ = ("key", "x", "y", "acts", "shed", "only", "value", "need_tot", "uses_seed", "tasks", "dl_min")

    def __init__(self, key, x, y, shed=False, only=None):
        self.key = key; self.x = x; self.y = y; self.acts = []; self.shed = shed; self.only = only
        self.value = 0.0; self.need_tot = {}; self.uses_seed = False; self.tasks = []; self.dl_min = 10 ** 9


class _State:
    pass


class _Unlimited(dict):
    def get(self, k, d=None):
        return BIG

    def __missing__(self, k):
        return BIG


class Executor:
    def __init__(self, params=None, farm_hand_cost_mult=1, shed_capacity=100):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.hire_mult = farm_hand_cost_mult
        self.cap = shed_capacity
        self.day = -1
        self.prev = {}            # unit index -> job key targeted last step
        self.last_info = {}
        self.stats = {"steps": 0, "ms_total": 0.0, "ms_max": 0.0}
        self.routes = {}          # route mode: unit index -> ordered job keys
        self.built_units = -1
        self.last_build = -1
        self._plan_info = {}
        self._share = None
        self.prev_day_units = 0
        self.max_units_today = 0
        self.hire_target = None

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def merge(out, market=None, max_orders=10):
        """Engine-ready action: executor HIREs first, then the caller's market orders.  The engine silently
        drops every order past the 10th in a turn, so anything that does not fit is returned in
        action["dropped"] (the engine ignores that key) for the caller to resubmit next step."""
        m = list(out.get("market_prefix", [])) + list(market or [])
        act = {"farmer": out["farmer"], "hands": out["hands"], "market": m[:max_orders]}
        if len(m) > max_orders:
            act["dropped"] = m[max_orders:]
        return act

    # ------------------------------------------------------------------------------------------
    def _build_jobs(self, tasks, tiles, n, day, now, eod, invs, units):
        by_tile = {}
        shed_tasks = []
        for t in tasks:
            kind = t.get("kind")
            if kind in TILE_KINDS:
                x = int(t["x"]); y = int(t["y"])
                if not (0 <= x < n and 0 <= y < n):
                    continue
                by_tile.setdefault((x, y), []).append(t)
            elif kind in SHED_KINDS:
                shed_tasks.append(t)
        # safety net: a plant sown today that is not yet watered dies at midnight -> implicit WATER
        fresh_v = self.p["fresh_water_value"]
        for y, row in enumerate(tiles):
            for x, tl in enumerate(row):
                # isinstance, not type() is dict: a real Kaggle agent receives kaggle_environments Struct
                # objects (a dict subclass) for every tile, and an exact-type test silently skips them all
                if (isinstance(tl, dict) and tl.get("kind") == "PLANT" and tl["planted_day"] == day
                        and not tl["watered_today"]):
                    lst = by_tile.setdefault((x, y), [])
                    if not any(tt["kind"] == "WATER" for tt in lst):
                        v = max([float(tt.get("value", 0) or 0) for tt in lst if tt["kind"] == "PLANT"] or [fresh_v])
                        lst.append({"kind": "WATER", "x": x, "y": y, "value": v, "implicit": True})
        jobs = []
        for (x, y), lst in by_tile.items():
            st = tile_state(tiles[y][x])
            if st["k"] == "L":
                continue
            # normalise and dedupe by (kind, item)
            pend = {}
            for t in lst:
                kind = t["kind"]; item = t.get("item")
                e = int(t.get("earliest", now) if t.get("earliest") is not None else now)
                dl = int(t.get("deadline", eod) if t.get("deadline") is not None else eod)
                dl = min(dl, eod)
                if dl < now or e > eod:
                    continue
                key = (kind, item if kind in ("PLANT", "PLACE") else None)
                if kind == "PLANT":
                    dl = min(dl, eod - 1)      # a seedling must be watered the same day: keep a step for it
                    if dl < now:
                        continue
                v = float(t.get("value", 1.0) or 0.0)
                old = pend.get(key)
                if old is None or v > old[2]:
                    pend[key] = [kind, item, v, max(e, now), dl, t]
            if not pend:
                continue
            job = Job((x, y), x, y)
            rem = list(pend.values())
            # DIG only clears the way: never on a plant sown today, never after a PLANT / BUILD of this job,
            # never on a structure that a BUILD task of this job asks for (a static task list keeps its DIG
            # after the tile has been cleared and replanted)
            builds = {r[0][6:] for r in rem if r[0] in ("BUILD_COOP", "BUILD_PASTURE")}
            no_dig = (st["k"] == "P" and st["pd"] == day) or (st["k"] in builds)
            guard = 0
            while rem and guard < 16:
                guard += 1
                chosen = None
                for pk in PRIORITY:
                    if pk == "DIG" and no_dig:
                        continue
                    for r in rem:
                        if r[0] == pk and action_valid(pk, st, day, r[1]):
                            chosen = r
                            break
                    if chosen:
                        break
                if chosen is None:
                    # implicit DIG: a weed (or empty structure) blocks a requested PLANT / BUILD
                    if st["k"] == "W" and any(r[0] in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") for r in rem):
                        self._add_act(job, "DIG", None, 0.0, now, eod, None, None, True)
                        apply_action("DIG", st, day)
                        continue
                    break
                rem.remove(chosen)
                kind, item, v, e, dl, t = chosen
                # the engine fixes what FEED / FERTILIZE / PLACE consume; a task's `needs` cannot override it
                if kind == "FEED":
                    ni, nn = "WHEAT", 1
                elif kind == "FERTILIZE":
                    ni, nn = "FERTILIZER", 1
                elif kind == "PLACE":
                    ni, nn = item, 1
                else:
                    needs = t.get("needs")
                    if needs:
                        ni, nn = next(iter(needs.items()))
                        nn = max(0, int(nn or 0)) or None
                        if nn is None:
                            ni = None
                    else:
                        ni, nn = None, None
                prod = None
                if kind == "HARVEST":
                    if st["k"] == "P":
                        prod = (st["crop"], st["y"])
                    elif st.get("an"):
                        prod = (ANIMAL_PRODUCT[st["an"]], st["y"])
                self._add_act(job, kind, item, v, e, dl, ni, nn, False, t, prod)
                apply_action(kind, st, day, item)
                if kind in ("PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE"):
                    no_dig = True
                if kind == "PLANT":
                    job.uses_seed = True
                    if not any(r[0] == "WATER" for r in rem):
                        # a fresh plant dies tonight unless watered today: carry its value
                        self._add_act(job, "WATER", None, v, e, eod, None, None, True)
                        apply_action("WATER", st, day)
            if job.acts and job.value > 0:
                jobs.append(job)
        # shed errands: DROP (per carrying unit) and PICKUP (any unit)
        for t in shed_tasks:
            kind = t["kind"]; item = t.get("item")
            e = max(now, int(t.get("earliest", now) or now)); dl = min(eod, int(t.get("deadline", eod) or eod))
            if dl < now or e > eod:
                continue
            v = float(t.get("value", 1.0) or 0.0)
            if kind == "DROP":
                carried = [(u, (inv.get(item, 0) if item else sum(c for i2, c in inv.items() if i2 not in TOOL_ITEMS)))
                           for u, inv in enumerate(invs[:len(units)])]
                tot = sum(c for _, c in carried)
                if tot <= 0:
                    continue
                want = int(t.get("n") or tot)
                for u, c in carried:
                    if c <= 0:
                        continue
                    job = Job(("DROP", item, u), -1, -1, shed=True, only=u)
                    self._add_act(job, "DROP", item, v * min(1.0, c / max(1, want)), e, dl, None, None, False, t)
                    jobs.append(job)
            else:  # PICKUP: pending while the units together carry fewer than n
                want = int(t.get("n") or 1)
                have = sum(inv.get(item, 0) for inv in invs[:len(units)])
                if have >= want:
                    continue
                job = Job(("PICKUP", item), -1, -1, shed=True)
                self._add_act(job, "PICKUP", item, v, e, dl, None, want - have, False, t)
                jobs.append(job)
        return jobs

    @staticmethod
    def _add_act(job, kind, item, v, e, dl, ni, nn, implicit, task=None, prod=None):
        job.acts.append((kind, item, v, e, dl, ni, nn or 0, kind in STATE_CHANGING, implicit, prod))
        job.value += v
        if v > 0 and dl < job.dl_min:
            job.dl_min = dl
        if ni is not None:
            job.need_tot[ni] = job.need_tot.get(ni, 0) + (nn or 0)
        if task is not None:
            job.tasks.append(task)

    # ------------------------------------------------------------------------------------------
    def _visit(self, acts, t, have, seeds_av, now, shed_av=None):
        """Simulate doing `acts` in order from arrival time t.
        Returns (value, t_end, n, min_slack, plants, first_act_performed)."""
        value = 0.0; n = 0; slack = 99; plants = None; maxwait = self.p["max_wait"]; first = None
        hv = dict(have) if have else {}
        for a in acts:
            kind, item, v, e, dl, ni, nn, sc, imp, prod = a
            if ni is not None and hv.get(ni, 0) < nn:
                # a missing state-changing act blocks the rest; a missing FERTILIZE blocks the WATER
                # behind it only while fertilizer can still be fetched (else water unfertilized)
                if sc or (kind == "FERTILIZE" and shed_av and shed_av.get(ni, 0) > 0):
                    break
                continue
            if kind == "PLANT":
                used = plants.get(item, 0) if plants else 0
                if seeds_av.get(item, 0) - used <= 0:
                    break
            if t < e:
                if n == 0 and e - t <= maxwait:
                    t = e
                else:
                    break
            if t > dl:
                if sc:
                    break
                continue
            if ni is not None:
                hv[ni] -= nn
            if kind == "PLANT":
                if plants is None:
                    plants = {}
                plants[item] = plants.get(item, 0) + 1
            value += v; n += 1
            if first is None:
                first = a
            if dl - t < slack:
                slack = dl - t
            t += 1
        return value, t, n, slack, plants, first

    def _evaluate(self, u, pos, inv, job, now, shed_av, seeds_av, share, sheds):
        """Best way for unit u to do job. Returns (score, plan) or None.
        plan = (job, target_xy, pickups[(item, q)], via_tile|None, value, cost_steps, slack, plants).
        A pickup detour (extra travel + pickup steps) is amortised over the q tool units picked, since one
        pickup serves q FEED / FERTILIZE / PLACE tasks."""
        ux, uy = pos
        if job.shed:
            if job.only is not None and job.only != u:
                return None
            best = None
            for s in sheds:
                d = abs(ux - s[0]) + abs(uy - s[1])
                if best is None or d < best[0]:
                    best = (d, s)
            d, s = best
            kind, item, v, e, dl, ni, nn, sc, imp, prod = job.acts[0]
            if kind == "PICKUP" and shed_av.get(item, 0) <= 0:
                return None
            t = now + d
            if t < e:
                t = e
            if t > dl:
                return None
            return v / (t + 1 - now), (job, s, [], None, v, t + 1 - now, dl - t, None, job.acts[0])
        jx, jy = job.x, job.y
        d = abs(ux - jx) + abs(uy - jy)
        have = None
        miss = None
        for it, c in job.need_tot.items():
            h = inv.get(it, 0)
            if have is None:
                have = {}
            have[it] = h
            if h < c:
                if miss is None:
                    miss = {}
                miss[it] = c - h
        val, t_end, nact, slack, plants, first = self._visit(job.acts, now + d, have, seeds_av, now, shed_av)
        best = None
        if nact > 0 and val > 0:
            best = (val / (t_end - now), (job, (jx, jy), [], None, val, t_end - now, slack, plants, first))
        if miss:
            picks = []
            for it, c in miss.items():
                a = shed_av.get(it, 0)
                if a > 0:
                    q = min(a, max(c, share.get(it, c)), max(c, self.p["pick_cap"]))
                    picks.append((it, q))
            if picks:
                bs = None
                for s in sheds:
                    dd = abs(ux - s[0]) + abs(uy - s[1]) + abs(s[0] - jx) + abs(s[1] - jy)
                    if bs is None or dd < bs[0]:
                        bs = (dd, s)
                dd, s = bs
                h2 = dict(have)
                for it, q in picks:
                    h2[it] = h2.get(it, 0) + q
                val2, t2, n2, sl2, pl2, f2 = self._visit(job.acts, now + dd + len(picks), h2, seeds_av, now)
                if n2 > 0 and val2 > 0:
                    extra = dd - d + len(picks)
                    qmin = min(q for _, q in picks)
                    cost2 = (t2 - now) - extra + extra / max(1, qmin)
                    sc2 = val2 / cost2
                    # a cheap detour (unit already at / next to the shed) is always taken: skipping the
                    # tool now means a full round trip to the shed later
                    if best is None or sc2 > best[0] or dd - d <= self.p["free_detour"]:
                        best = (sc2, (job, (jx, jy), picks, s, val2, cost2, sl2, pl2, f2))
        return best

    # ------------------------------------------------------------------------------------------
    def act(self, obs, tasks, hires_wanted=None):
        t0 = time.perf_counter()
        S = self._parse(obs)
        if S.day != self.day:
            self.day = S.day
            self.prev = {}
            self.routes = {}
            self.built_units = -1
            self.prev_day_units = self.max_units_today
            self.max_units_today = 0
        self.max_units_today = max(self.max_units_today, len(S.units))
        jobs = self._build_jobs(tasks or [], S.tiles, S.n, S.day, S.step, S.eod, S.invs, S.units)

        # ---- hires first: they change how the tools are shared out ----
        prefix = []
        # hires_wanted is today's target number of hands; it persists (early in the day) until changed
        if hires_wanted is not None:
            self.hire_target = (S.day, int(hires_wanted))
        elif (self.hire_target is not None and self.hire_target[0] == S.day
              and S.hour <= self.p["hire_persist_until"]):
            hires_wanted = self.hire_target[1]
        if hires_wanted is not None and S.hour < DAY_STEPS - 1:
            missing = int(hires_wanted) - (len(S.units) - 1)
            money = float(S.farm.get("money", 0))
            k = int(S.farm.get("hires_today", 0))
            while missing > 0 and len(prefix) < self.p["max_hires_per_step"]:
                c = fib_cost(k, self.hire_mult)
                if c > money:
                    break
                prefix.append(["HIRE"]); money -= c; k += 1; missing -= 1
        S.expected_units = len(S.units) + len(prefix)
        if hires_wanted is not None:
            S.expected_units = max(S.expected_units, 1 + int(hires_wanted))
        elif S.hour <= 1:
            S.expected_units = max(S.expected_units, self.prev_day_units)

        err = None
        if self.p["mode"] == "greedy":
            stops = self._plan_greedy(S, jobs)
        else:
            try:
                stops = self._plan_routes(S, jobs)
            except Exception as e:                      # never let routing kill the agent: fall back
                err = f"{type(e).__name__}: {e}"[:200]
                self.routes = {}
                self.built_units = -1
                stops = self._plan_greedy(S, jobs)
        acts_out = self._emit(S, stops)

        dt = (time.perf_counter() - t0) * 1000.0
        st = self.stats
        st["steps"] += 1; st["ms_total"] += dt; st["ms_max"] = max(st["ms_max"], dt)
        need_all = {}
        shortfall = {}
        for j in jobs:
            for it, c in j.need_tot.items():
                need_all[it] = need_all.get(it, 0) + c
            if j.uses_seed:
                for a in j.acts:
                    if a[0] == "PLANT":
                        shortfall[a[1]] = shortfall.get(a[1], 0) + 1
        for crop in list(shortfall):
            shortfall[crop] -= S.seeds.get(crop, 0)
            if shortfall[crop] <= 0:
                del shortfall[crop]
        carried = sum(c for inv in S.invs[:len(S.units)] for c in inv.values())
        room = max(0, self.cap - sum(S.shed.values()))
        info = {"n_jobs": len(jobs), "ms": dt, "seed_shortfall": shortfall, "need_tools": need_all,
                "carried": carried, "shed_room": room, "overflow_risk": carried > room,
                "assigned": {u: (s[3].key if s and s[3] is not None else None) for u, s in enumerate(stops)}}
        info.update(self._plan_info)
        if err:
            info["route_error"] = err
            self.stats["route_errors"] = self.stats.get("route_errors", 0) + 1
        self.last_info = info
        return {"farmer": acts_out[0], "hands": acts_out[1:], "market_prefix": prefix}

    # ------------------------------------------------------------------------------------------
    def _parse(self, obs):
        S = _State()
        S.me = obs["player"]
        S.farm = obs["farms"][S.me]
        S.priv = obs["private"]
        S.tiles = S.farm["tiles"]
        S.n = len(S.tiles)
        S.step = int(obs.get("step", obs.get("day", 0) * DAY_STEPS + obs.get("hour", 0)))
        S.day = S.step // DAY_STEPS
        S.hour = S.step % DAY_STEPS
        S.eod = S.day * DAY_STEPS + DAY_STEPS - 1
        S.sheds = shed_tiles(S.n)
        S.shed_set = set(S.sheds)
        S.units = [tuple(S.farm["farmer"])] + [tuple(h) for h in S.farm.get("hands", [])]
        invs = list(S.priv.get("inventories") or [{}])
        while len(invs) < len(S.units):
            invs.append({})
        S.invs = invs
        S.shed = S.priv.get("shed", {})
        S.seeds = dict(S.priv.get("seeds", {}))
        S.expected_units = len(S.units)
        S.start = None
        return S

    # ------------------------------------------------------------------------------------------
    # emission (shared by both modes)
    # stop = (kind, target_xy, picks, job, first_act); kind in PICK / JOB / SHED
    # ------------------------------------------------------------------------------------------
    def _emit(self, S, stops):
        out = []
        shed_live = dict(S.shed)          # the shed as it evolves within this step (units act in order)
        room = max(0, self.cap - sum(shed_live.values()))
        seeds_left = dict(S.seeds)
        new_prev = {}
        for u, pos in enumerate(S.units):
            st = stops[u] if u < len(stops) else None
            if st is None:
                out.append(["PASS"])
                continue
            kind, target, picks, job, fa = st
            if job is not None:
                new_prev[u] = job.key
            a = None
            if kind == "PICK":
                if pos in S.shed_set:
                    for it, q in picks:
                        q = min(q, shed_live.get(it, 0))
                        if q > 0:
                            a = ["PICKUP", it, q]
                            shed_live[it] -= q
                            room += q
                            break
                    if a is None and job is not None:
                        kind = "SHED" if job.shed else "JOB"
                        target = target if job.shed else (job.x, job.y)
                else:
                    a = self._move(pos, target)
            if a is None and job is not None:
                if kind == "SHED":
                    if pos in S.shed_set:
                        a, room = self._shed_act(job, S.invs[u], shed_live, room)
                    else:
                        a = self._move(pos, target)
                elif pos == target and fa is not None:
                    k, item, v, e, dl, ni, nn, sc, imp, prod = fa
                    if S.step < e:
                        a = ["PASS"]
                    elif k == "PLANT":
                        if seeds_left.get(item, 0) > 0:
                            seeds_left[item] -= 1
                            a = ["PLANT", item]
                        else:
                            a = ["PASS"]
                    elif k == "PLACE":
                        a = ["PLACE", item, 1]
                    else:
                        a = [k]
                elif pos != target:
                    a = self._move(pos, target)
            out.append(a or ["PASS"])
        self.prev = new_prev
        return out

    # ------------------------------------------------------------------------------------------
    # greedy mode (baseline): one tile per unit, best value per step
    # ------------------------------------------------------------------------------------------
    def _plan_greedy(self, S, jobs):
        units, invs, step = S.units, S.invs, S.step
        shed_av = {it: S.shed.get(it, 0) for it in TOOL_ITEMS}
        seeds_av = dict(S.seeds)
        share = self._shares(S, jobs)
        follow = self._follow_values(jobs, S.sheds) if self.p["look"] > 0 and len(jobs) > 1 else None
        stick = self.p["stick"]; uw = self.p["urg_window"]; ug = self.p["urg_gain"]
        heap = []
        version = 0
        for u, pos in enumerate(units):
            pk = self.prev.get(u)
            for ji, job in enumerate(jobs):
                r = self._evaluate(u, pos, invs[u], job, step, shed_av, seeds_av, share, S.sheds)
                if r is None:
                    continue
                heap.append((-self._adjust(r, job, pk, stick, uw, ug, follow, ji, step), u, ji, version, len(heap), r[1]))
        heapq.heapify(heap)
        taken_u = {}
        taken_j = set()
        tie = 0
        while heap and len(taken_u) < len(units):
            negsc, u, ji, ver, _, plan = heapq.heappop(heap)
            if u in taken_u or ji in taken_j:
                continue
            job = jobs[ji]
            shared = bool(plan[2]) or job.uses_seed or (job.shed and job.acts[0][0] == "PICKUP")
            if shared and ver != version:
                r = self._evaluate(u, units[u], invs[u], job, step, shed_av, seeds_av, share, S.sheds)
                if r is not None:
                    sc = self._adjust(r, job, self.prev.get(u), stick, uw, ug, follow, ji, step)
                    tie += 1
                    heapq.heappush(heap, (-sc, u, ji, version, 10 ** 6 + tie, r[1]))
                continue
            taken_u[u] = plan
            taken_j.add(ji)
            if shared:
                for it, q in plan[2]:
                    shed_av[it] = shed_av.get(it, 0) - q
                if not job.shed and plan[7]:
                    for crop, c in plan[7].items():
                        seeds_av[crop] = seeds_av.get(crop, 0) - c
                if job.shed and job.acts[0][0] == "PICKUP":
                    it = job.acts[0][1]
                    shed_av[it] = shed_av.get(it, 0) - min(shed_av.get(it, 0), job.acts[0][6])
                version += 1
        stops = []
        for u in range(len(units)):
            plan = taken_u.get(u)
            if plan is None:
                stops.append(None)
                continue
            job, target, picks, via = plan[0], plan[1], plan[2], plan[3]
            if via is not None and picks:
                stops.append(("PICK", via, picks, job, plan[8]))
            elif job.shed:
                stops.append(("SHED", target, [], job, job.acts[0]))
            else:
                stops.append(("JOB", target, [], job, plan[8]))
        self._plan_info = {"unassigned_value": sum(j.value for ji, j in enumerate(jobs) if ji not in taken_j)}
        return stops

    def _shares(self, S, jobs):
        need_all = {}
        for j in jobs:
            for it, c in j.need_tot.items():
                need_all[it] = need_all.get(it, 0) + c
        share = {}
        nu = max(1, S.expected_units)
        for it, c in need_all.items():
            carried = sum(inv.get(it, 0) for inv in S.invs[:len(S.units)])
            rest = c - carried
            if rest > 0:
                share[it] = min(self.p["pick_cap"], -(-rest // nu) + self.p["pick_extra"])
        return share

    # ------------------------------------------------------------------------------------------
    # route mode: every unit follows a route (ordered list of job keys) kept across steps
    # ------------------------------------------------------------------------------------------
    def _plan_routes(self, S, jobs):
        jd = {j.key: j for j in jobs}
        for j in jobs:
            if j.shed:
                j.x, j.y = S.sheds[0]
        U = len(S.units)
        # tool shares only while hires ordered this step have not arrived yet (hour 0 farmer)
        self._share = self._shares(S, jobs) if S.expected_units > U else None
        t_b = time.perf_counter()
        rebuilt = False
        if self.built_units != U or not self.routes:
            cands = [self._construct(S, jd), self._construct_dispatch(S, jd)]
            routes = max(cands, key=lambda r: self._objective(S, jd, r)[0])
            rebuilt = True
            self.built_units = U
            self.last_build = S.step
        else:
            routes, unassigned = self._repair(S, jd)
            routes = self._insert(S, jd, routes, unassigned)
            if S.step - self.last_build >= self.p["rebuild_every"]:
                self.last_build = S.step
                o_old = self._objective(S, jd, routes)[0]
                for cand in (self._construct_dispatch(S, jd), self._construct(S, jd)):
                    o_new = self._objective(S, jd, cand)[0]
                    if o_new > o_old + self.p["rebuild_margin"]:
                        routes = cand
                        o_old = o_new
                        rebuilt = True
        routes = self._improve(S, jd, routes, t_b)
        routes = self._balance(S, jd, routes, t_b)
        self.routes = {u: r for u, r in enumerate(routes)}
        # final simulation in unit order -> this step's stop for each unit
        stops = []
        tot_v = 0.0
        done_keys = set()
        ends = []
        for u, (v, t_end, first, done, dropped, ts) in enumerate(self._sim_all(S, jd, routes)):
            tot_v += v
            done_keys.update(done)
            ends.append(t_end)
            stops.append(first)
        self._plan_info = {"planned_value": tot_v,
                           "unassigned_value": sum(j.value for k, j in jd.items() if k not in done_keys),
                           "route_len": [len(r) for r in routes], "route_end_hour": [e - S.day * DAY_STEPS for e in ends],
                           "rebuilt": rebuilt}
        return stops

    def _sim_route(self, S, u, route, jd, stock, seeds, rec=None):
        """Walk unit u's route from its current state; stock / seeds are reserved (mutated).
        rec: optional list collecting (time, unit, 'T'|'S', item, qty) shed-tool / seed demands.
        Returns (value, t_end, first_stop, done_keys, dropped_keys, sum_of_completion_steps)."""
        x, y = S.units[u]
        t = S.start[u] if S.start is not None else S.step
        inv = {}
        uinv = S.invs[u]
        for it in TOOL_ITEMS:
            c = uinv.get(it, 0)
            if c:
                inv[it] = c
        need_left = {}
        for k in route:
            for it, c in jd[k].need_tot.items():
                need_left[it] = need_left.get(it, 0) + c
        cap = self.p["pick_cap"]
        share = self._share
        value = 0.0
        first = None
        done = []
        dropped = []
        tsum = 0
        shed_set = S.shed_set
        if need_left and (x, y) in shed_set:
            picks = []
            for it, c in need_left.items():
                q = min(c - inv.get(it, 0), stock.get(it, 0), cap)
                if share is not None and it in share:
                    q = min(q, share[it])
                if q > 0:
                    picks.append((it, q))
                    inv[it] = inv.get(it, 0) + q
                    stock[it] -= q
                    if rec is not None:
                        rec.append((t, u, "T", it, q))
            if picks:
                first = ["PICK", (x, y), picks, None, None]
                t += len(picks)
        lam = self.p["detour_value"]
        for k in route:
            j = jd[k]
            if j.shed:
                best = None
                for s in S.sheds:
                    dd = abs(x - s[0]) + abs(y - s[1])
                    if best is None or dd < best[0]:
                        best = (dd, s)
                kind, item, v, e, dl = j.acts[0][:5]
                tt = t + best[0]
                if tt < e:
                    tt = e
                if tt > dl or (kind == "PICKUP" and stock.get(item, 0) <= 0):
                    dropped.append(k)
                    continue
                if kind == "PICKUP":
                    q = min(j.acts[0][6] or 1, stock.get(item, 0))
                    stock[item] -= q
                    inv[item] = inv.get(item, 0) + q
                    if rec is not None:
                        rec.append((tt, u, "T", item, q))
                if first is None:
                    first = ["SHED", best[1], [], j, j.acts[0]]
                elif first[3] is None:
                    first[3] = j; first[4] = j.acts[0]
                value += v
                t = tt + 1
                tsum += t - S.step
                x, y = best[1]
                done.append(k)
                continue
            jx, jy = j.x, j.y
            d = abs(x - jx) + abs(y - jy)
            picks = None
            via = None
            arr = t + d
            hv = inv
            lack_val = 0.0
            for it, c in j.need_tot.items():
                if inv.get(it, 0) < c and stock.get(it, 0) > 0:
                    if picks is None:
                        picks = []
                    q = min(max(c, need_left.get(it, 0)) - inv.get(it, 0), stock[it], max(c, cap))
                    picks.append((it, q))
            if picks:
                for a in j.acts:
                    if a[5] is not None and inv.get(a[5], 0) < a[6]:
                        lack_val += a[2]
                best = None
                for s in S.sheds:
                    dd = abs(x - s[0]) + abs(y - s[1]) + abs(s[0] - jx) + abs(s[1] - jy)
                    if best is None or dd < best[0]:
                        best = (dd, s)
                extra = best[0] - d + len(picks)
                if lack_val >= lam * extra:
                    via = best[1]
                    hv = dict(inv)
                    for it, q in picks:
                        hv[it] = hv.get(it, 0) + q
                    arr = t + best[0] + len(picks)
                else:
                    picks = None
            val, t2, nact, fa, hv2, used = self._visit_inv(j.acts, arr, hv, seeds, stock, False)
            if nact == 0:
                dropped.append(k)
                continue
            if val < j.value - 1e-9 and j.dl_min < S.eod - 1:
                dropped.append(k)          # an early-deadline act of this job is missed: worth relocating
            if picks:
                for it, q in picks:
                    stock[it] -= q
                    if rec is not None:
                        rec.append((arr - 1, u, "T", it, q))
            if used:
                for crop, c in used.items():
                    seeds[crop] = seeds.get(crop, 0) - c
                    if rec is not None:
                        rec.append((arr, u, "S", crop, c))
            inv = hv2
            for it, c in j.need_tot.items():
                need_left[it] = need_left.get(it, 0) - c
            if first is None:
                if via is not None:
                    first = ["PICK", via, picks, j, fa]
                else:
                    first = ["JOB", (jx, jy), [], j, fa]
            elif first[3] is None:
                first[3] = j; first[4] = fa
            value += val
            t = t2
            tsum += t - S.step
            x, y = jx, jy
            done.append(k)
        if first is not None and first[3] is None and first[0] == "PICK":
            pass  # pickups only (tools for later jobs); emission handles it
        return value, t, (tuple(first) if first else None), done, dropped, tsum

    def _visit_inv(self, acts, t, inv, seeds, stock, fert_wait=True):
        value = 0.0; n = 0; first = None; used = None; maxwait = self.p["max_wait"]
        hv = dict(inv)
        for a in acts:
            kind, item, v, e, dl, ni, nn, sc, imp, prod = a
            if ni is not None and hv.get(ni, 0) < nn:
                if sc or (kind == "FERTILIZE" and stock.get(ni, 0) > 0 and fert_wait):
                    break
                continue
            if kind == "PLANT":
                u0 = used.get(item, 0) if used else 0
                if seeds.get(item, 0) - u0 <= 0:
                    break
            if t < e:
                if n == 0 and e - t <= maxwait:
                    t = e
                else:
                    break
            if t > dl:
                if sc:
                    break
                continue
            if ni is not None:
                hv[ni] -= nn
            if kind == "COLLECT_FERTILIZER":
                hv["FERTILIZER"] = hv.get("FERTILIZER", 0) + 1
            elif kind == "HARVEST":
                if prod is not None and prod[0] in TOOL_SET and prod[1] > 0:
                    hv[prod[0]] = hv.get(prod[0], 0) + prod[1]
            elif kind == "PLANT":
                if used is None:
                    used = {}
                used[item] = used.get(item, 0) + 1
            value += v
            n += 1
            if first is None:
                first = a
            t += 1
        return value, t, n, first, hv, used

    def _sim_all(self, S, jd, routes):
        """Joint simulation. Pass 1 runs every route with unlimited seeds / shed tools and records when
        each demand occurs; demands are then granted in time order (earliest first) from the real stock;
        pass 2 re-runs each route with only its grants.  Returns per-unit (value, t_end, first, done, dropped)."""
        U = len(S.units)
        rec = []
        for u in range(U):
            self._sim_route(S, u, routes[u], jd, dict.fromkeys(TOOL_ITEMS, BIG), _Unlimited(), rec)
        pool_t = {it: S.shed.get(it, 0) for it in TOOL_ITEMS}
        pool_s = dict(S.seeds)
        grants = [({}, {}) for _ in range(U)]
        rec.sort(key=lambda r: (r[0], r[1]))
        for t, u, typ, item, q in rec:
            pool = pool_t if typ == "T" else pool_s
            g = min(q, pool.get(item, 0))
            if g > 0:
                pool[item] -= g
                gd = grants[u][0] if typ == "T" else grants[u][1]
                gd[item] = gd.get(item, 0) + g
        return [self._sim_route(S, u, routes[u], jd, grants[u][0], grants[u][1]) for u in range(U)]

    def _objective(self, S, jd, routes):
        res = self._sim_all(S, jd, routes)
        tot = sum(r[0] for r in res)
        mx = max((r[1] for r in res), default=0)
        ts = sum(r[5] for r in res)
        return tot - 0.01 * mx - self.p["w_time"] * ts, [(r[0], r[1], r[3], r[4]) for r in res]

    def _construct(self, S, jd):
        U = len(S.units)
        cx = cy = (S.n - 1) / 2.0
        items = []
        tot = 0.0
        shed_jobs = []
        for k, j in jd.items():
            if j.shed:
                shed_jobs.append(k)
                continue
            w = len(j.acts) + 1.0
            items.append((math.atan2(j.y - cy, j.x - cx), k, w))
            tot += w
        items.sort(key=lambda it: (it[0], it[1]))
        m = len(items)
        R = max(1, min(self.p["sweep_rot"], m))
        best = None
        for r in range(R):
            off = (r * m) // R
            seq = items[off:] + items[:off]
            groups = []
            cur = []
            acc = 0.0
            target = tot / U
            for a, k, w in seq:
                cur.append(k)
                acc += w
                if len(groups) < U - 1 and acc >= target * (len(groups) + 1):
                    groups.append(cur)
                    cur = []
            groups.append(cur)
            while len(groups) < U:
                groups.append([])
            routes = self._match_and_route(S, jd, groups)
            for k in shed_jobs:
                j = jd[k]
                uu = j.only if j.only is not None else 0
                if uu < U:
                    routes[uu].insert(0, k)
            obj = self._objective(S, jd, routes)[0]
            if best is None or obj > best[0]:
                best = (obj, routes)
            if m == 0:
                break
        return best[1]

    def _match_and_route(self, S, jd, groups):
        U = len(S.units)
        pairs = []
        for g, grp in enumerate(groups):
            for u, (ux, uy) in enumerate(S.units):
                if grp:
                    dmin = min(abs(ux - jd[k].x) + abs(uy - jd[k].y) for k in grp)
                else:
                    dmin = 99
                pairs.append((dmin, g, u))
        pairs.sort()
        routes = [[] for _ in range(U)]
        gu = set()
        uu = set()
        for dmin, g, u in pairs:
            if g in gu or u in uu:
                continue
            gu.add(g)
            uu.add(u)
            routes[u] = self._two_opt(S.units[u], self._nn(S.units[u], groups[g], jd), jd)
        return routes

    @staticmethod
    def _nn(start, keys, jd):
        rem = list(keys)
        x, y = start
        out = []
        while rem:
            bi = 0
            bd = 999
            for i, k in enumerate(rem):
                j = jd[k]
                d = abs(x - j.x) + abs(y - j.y)
                if d < bd:
                    bd = d
                    bi = i
            k = rem.pop(bi)
            out.append(k)
            x, y = jd[k].x, jd[k].y
        return out

    def _two_opt(self, start, route, jd):
        if len(route) < 3:
            return route
        pts = [start] + [(jd[k].x, jd[k].y) for k in route]
        keys = [None] + list(route)
        n = len(pts)
        for _ in range(self.p["two_opt_passes"]):
            improved = False
            for i in range(1, n - 1):
                a = pts[i - 1]
                b = pts[i]
                dab = abs(a[0] - b[0]) + abs(a[1] - b[1])
                for j in range(i + 1, n):
                    c = pts[j]
                    dac = abs(a[0] - c[0]) + abs(a[1] - c[1])
                    if j + 1 < n:
                        d = pts[j + 1]
                        delta = dac + abs(b[0] - d[0]) + abs(b[1] - d[1]) - dab - abs(c[0] - d[0]) - abs(c[1] - d[1])
                    else:
                        delta = dac - dab
                    if delta < 0:
                        pts[i:j + 1] = pts[i:j + 1][::-1]
                        keys[i:j + 1] = keys[i:j + 1][::-1]
                        b = pts[i]
                        dab = abs(a[0] - b[0]) + abs(a[1] - b[1])
                        improved = True
            if not improved:
                break
        return keys[1:]

    def _repair(self, S, jd):
        U = len(S.units)
        routes = []
        seen = set()
        for u in range(U):
            r = []
            for k in self.routes.get(u, []):
                if k in jd and k not in seen:
                    j = jd[k]
                    if j.only is not None and j.only != u:
                        continue
                    r.append(k)
                    seen.add(k)
            routes.append(r)
        unassigned = [k for k in jd if k not in seen]
        return routes, unassigned

    def _insert(self, S, jd, routes, keys, exclude=None):
        """Cheapest insertion of `keys` (by value), verified by route simulation."""
        if not keys:
            return routes
        keys = sorted(keys, key=lambda k: -jd[k].value)
        U = len(S.units)
        for k in keys:
            j = jd[k]
            cands = []
            for u in range(U):
                if (j.only is not None and j.only != u) or u == exclude:
                    continue
                r = routes[u]
                px, py = S.units[u]
                for p in range(len(r) + 1):
                    if p < len(r):
                        nj = jd[r[p]]
                        d0 = abs(px - nj.x) + abs(py - nj.y)
                        d1 = abs(px - j.x) + abs(py - j.y) + abs(j.x - nj.x) + abs(j.y - nj.y)
                    else:
                        d0 = 0
                        d1 = abs(px - j.x) + abs(py - j.y)
                    cands.append((d1 - d0, u, p))
                    if p < len(r):
                        px, py = jd[r[p]].x, jd[r[p]].y
            cands.sort()
            trial_c = cands[:4]
            if j.dl_min < S.eod - 1:
                fronts = sorted((abs(S.units[u][0] - j.x) + abs(S.units[u][1] - j.y), u, 0) for u in range(U)
                                if u != exclude and (j.only is None or j.only == u))
                trial_c = trial_c + [c for c in fronts[:3] if c not in trial_c]
            best = None
            old_v = {}
            wt = self.p["w_time"]
            for delta, u, p in trial_c:
                if u not in old_v:
                    r0 = self._sim_route(S, u, routes[u], jd, {it: S.shed.get(it, 0) for it in TOOL_ITEMS}, dict(S.seeds))
                    old_v[u] = (r0[0], r0[0] - wt * r0[5])
                nr = routes[u][:p] + [k] + routes[u][p:]
                r1 = self._sim_route(S, u, nr, jd, {it: S.shed.get(it, 0) for it in TOOL_ITEMS}, dict(S.seeds))
                if r1[0] <= old_v[u][0] + 1e-9:
                    continue
                gain = (r1[0] - wt * r1[5]) - old_v[u][1]
                if best is None or gain > best[0] + 1e-9:
                    best = (gain, u, nr)
            if best is not None:
                routes[best[1]] = best[2]
        return routes

    def _improve(self, S, jd, routes, t_start):
        """Move jobs that a route cannot finish (dropped by the simulation) into routes with room."""
        budget = self.p["improve_ms"] / 1000.0
        obj, res = self._objective(S, jd, routes)
        dropped = []
        for u, (v, t_end, done, dr) in enumerate(res):
            for k in dr:
                dropped.append((jd[k].value, k, u))
        if not dropped:
            return routes
        dropped.sort(key=lambda e: -e[0])       # keys mix tile tuples and shed-errand tuples: sort by value only
        for val, k, u in dropped:
            if time.perf_counter() - t_start > budget:
                break
            r0 = [x for x in routes[u] if x != k]
            trial = list(routes)
            trial[u] = r0
            trial = self._insert(S, jd, trial, [k])
            o2 = self._objective(S, jd, trial)[0]
            if o2 > obj + 1e-9:
                routes = trial
                obj = o2
        return routes

    def _balance(self, S, jd, routes, t_start):
        """Hand the tail jobs of the latest-finishing routes to routes that finish early."""
        budget = self.p["improve_ms"] / 1000.0
        obj, res = self._objective(S, jd, routes)
        tried = set()
        for _ in range(self.p["balance_moves"]):
            if time.perf_counter() - t_start > budget:
                break
            ends = sorted(((r[1], u) for u, r in enumerate(res) if routes[u]), reverse=True)
            if not ends:
                break
            lo = min(r[1] for r in res)
            moved = False
            for end, u in ends[:3]:
                if end - lo < 3:
                    break
                k = routes[u][-1]
                if (k, u) in tried:
                    continue
                tried.add((k, u))
                trial = list(routes)
                trial[u] = routes[u][:-1]
                trial = self._insert(S, jd, trial, [k], exclude=u)
                o2, r2 = self._objective(S, jd, trial)
                if o2 > obj + 1e-9:
                    routes, obj, res = trial, o2, r2
                    moved = True
                    break
            if not moved:
                break
        return routes

    def _construct_dispatch(self, S, jd):
        """Event-driven dispatch: whichever unit is free first takes the nearest remaining job."""
        U = len(S.units)
        eod = S.eod
        tfree = [(S.start[u] if S.start is not None else S.step) for u in range(U)]
        pos = list(S.units)
        rem = [k for k, j in jd.items() if not j.shed]
        routes = [[] for _ in range(U)]
        urg = self.p["urgent_bonus"]
        # early-deadline jobs first, most constrained first (fewest units that can still make the deadline,
        # counting the acts that precede the deadline act), each to the unit that can finish it earliest
        urgent = [k for k in rem if jd[k].dl_min < eod - 1]
        if urgent:
            def lead(j):
                for i, a in enumerate(j.acts):
                    if a[4] == j.dl_min:
                        return i
                return 0
            def feas(j, u):
                t_act = tfree[u] + abs(pos[u][0] - j.x) + abs(pos[u][1] - j.y) + lead(j)
                return t_act if t_act <= j.dl_min else None
            left = set(urgent)
            while left:
                best = None
                for k in left:
                    j = jd[k]
                    opts = [(feas(j, u), u) for u in range(U)]
                    opts = [o for o in opts if o[0] is not None]
                    if not opts:
                        continue
                    t_act, u = min(opts)
                    key = (len(opts), j.dl_min, -j.value)
                    if best is None or key < best[0]:
                        best = (key, k, u)
                if best is None:
                    break
                _, k, u = best
                j = jd[k]
                left.discard(k)
                rem.remove(k)
                routes[u].append(k)
                tfree[u] += abs(pos[u][0] - j.x) + abs(pos[u][1] - j.y) + len(j.acts)
                pos[u] = (j.x, j.y)
        heap = [(tfree[u], u) for u in range(U)]
        heapq.heapify(heap)
        while rem and heap:
            t, u = heapq.heappop(heap)
            x, y = pos[u]
            bi = -1
            bk = None
            bd = 0
            for i, k in enumerate(rem):
                j = jd[k]
                d = abs(x - j.x) + abs(y - j.y)
                if j.dl_min < eod - 1 and t + d <= j.dl_min:
                    key = (d - urg, -j.value)       # an early deadline still reachable: take it first
                else:
                    key = (d, -j.value)
                if bk is None or key < bk:
                    bk = key
                    bi = i
                    bd = d
            k = rem[bi]
            j = jd[k]
            t2 = t + bd + len(j.acts)
            if t2 - 1 > eod:
                continue            # nothing reachable for this unit any more
            rem.pop(bi)
            routes[u].append(k)
            pos[u] = (j.x, j.y)
            heapq.heappush(heap, (t2, u))
        for u in range(U):
            r = routes[u]
            n_urg = 0
            while n_urg < len(r) and jd[r[n_urg]].dl_min < eod - 1:
                n_urg += 1
            if n_urg:
                last = jd[r[n_urg - 1]]
                routes[u] = r[:n_urg] + self._two_opt((last.x, last.y), r[n_urg:], jd)
            else:
                routes[u] = self._two_opt(S.units[u], r, jd)
        for k, j in jd.items():
            if j.shed:
                uu = j.only if j.only is not None else 0
                if uu < U:
                    routes[uu].insert(0, k)
        return routes

    # ------------------------------------------------------------------------------------------
    def suggest_hires(self, obs, tasks, max_hands=16, margin=1.0, incoming=None):
        """How many hands to have today for this task list.  Simulates the day's routes (dispatch
        construction + joint simulation) with k hands spawned now (acting from the next step, as the
        engine does) and keeps adding a hand while the task value it adds exceeds margin x its fib cost.
        incoming = {"seeds": {crop: n}, "shed": {item: n}}: what this step's market orders will add (it
        lands after the unit phase, so it is otherwise invisible).  Returns the total number of hands
        wanted (pass it as `hires_wanted`); self.last_hire_eval lists (hands, planned value, hire cost).
        Takes ~1-3 ms per hand evaluated."""
        S = self._parse(obs)
        if incoming:
            for crop, c in (incoming.get("seeds") or {}).items():
                S.seeds[crop] = S.seeds.get(crop, 0) + c
            if incoming.get("shed"):
                S.shed = dict(S.shed)
                for it, c in incoming["shed"].items():
                    S.shed[it] = S.shed.get(it, 0) + c
        have = len(S.units) - 1
        if S.hour >= DAY_STEPS - 1:
            return have
        jobs = self._build_jobs(tasks or [], S.tiles, S.n, S.day, S.step, S.eod, S.invs, S.units)
        if not jobs:
            return have
        jd = {j.key: j for j in jobs}
        for j in jobs:
            if j.shed:
                j.x, j.y = S.sheds[0]
        base_units = list(S.units)
        base_invs = list(S.invs[:len(S.units)])
        hires_today = int(S.farm.get("hires_today", 0))
        money = float(S.farm.get("money", 0))
        saved_share = self._share
        self._share = None

        def value_with(k):
            units = list(base_units)
            occ = {}
            for p in units:
                occ[p] = occ.get(p, 0) + 1
            for _ in range(k):
                sp = min(S.sheds, key=lambda tl: (occ.get(tl, 0), S.sheds.index(tl)))
                occ[sp] = occ.get(sp, 0) + 1
                units.append(sp)
            S.units = units
            S.invs = base_invs + [{} for _ in range(k)]
            S.start = [S.step] * len(base_units) + [S.step + 1] * k
            routes = self._construct_dispatch(S, jd)
            return sum(r[0] for r in self._sim_all(S, jd, routes))

        best = have
        v_prev = value_with(0)
        self.last_hire_eval = [(have, round(v_prev, 1), 0)]
        cost = 0.0
        misses = 0
        for k in range(1, max_hands - have + 1):
            c = fib_cost(hires_today + k - 1, self.hire_mult)
            if cost + c > money:
                break
            v = value_with(k)
            self.last_hire_eval.append((have + k, round(v, 1), c))
            if v - v_prev > margin * c:
                best = have + k
                misses = 0
            else:
                misses += 1
                if misses >= 2:
                    break
            cost += c
            v_prev = v
        S.units = base_units
        S.invs = base_invs
        S.start = None
        self._share = saved_share
        return best

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _adjust(r, job, prev_key, stick, uw, ug, follow, ji, now):
        plan = r[1]
        if follow is not None:
            sc = (plan[4] + follow[ji][0]) / (plan[5] + follow[ji][1])
        else:
            sc = r[0]
        slack = plan[6]
        if slack < uw:
            sc *= 1.0 + ug * (uw - max(0, slack)) / uw
        if prev_key is not None and job.key == prev_key:
            sc *= 1.0 + stick
        return sc

    def _follow_values(self, jobs, sheds):
        """For each job: (value, steps) of the best nearby next job (discounted by `look`)."""
        look = self.p["look"]; k = self.p["look_k"]
        pts = []
        for j in jobs:
            if j.shed:
                pts.append(sheds[0])
            else:
                pts.append((j.x, j.y))
        out = []
        nj = len(jobs)
        for i in range(nj):
            xi, yi = pts[i]
            best = (0.0, 0.0, -1.0)
            # nearest-k scan (jobs are few: <= ~150)
            cand = []
            for j2 in range(nj):
                if j2 == i:
                    continue
                x2, y2 = pts[j2]
                cand.append((abs(xi - x2) + abs(yi - y2), j2))
            if len(cand) > k:
                cand = heapq.nsmallest(k, cand)
            for d, j2 in cand:
                jb = jobs[j2]
                cost = d + len(jb.acts)
                dens = jb.value / cost
                if dens > best[2]:
                    best = (jb.value, cost, dens)
            out.append((look * best[0], look * best[1]))
        return out

    @staticmethod
    def _move(pos, target):
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        if dx == 0 and dy == 0:
            return ["PASS"]
        if abs(dx) >= abs(dy) and dx != 0:
            return ["EAST"] if dx > 0 else ["WEST"]
        return ["SOUTH"] if dy > 0 else ["NORTH"]

    @staticmethod
    def _shed_act(job, inv, shed_live, room):
        kind, item, v, e, dl, ni, nn, sc, imp, prod = job.acts[0]
        if kind == "PICKUP":
            q = min(nn or 1, shed_live.get(item, 0))
            if q <= 0:
                return ["PASS"], room
            shed_live[item] -= q
            return ["PICKUP", item, q], room + q
        # DROP: deposit without ever discarding (DROP only when everything fits)
        if item:
            c = inv.get(item, 0)
            q = min(c, room)
            if q <= 0:
                return ["PASS"], room
            shed_live[item] = shed_live.get(item, 0) + q
            return ["PLACE", item, q], room - q
        goods = [(i2, c) for i2, c in inv.items() if c > 0 and i2 not in TOOL_ITEMS]
        tot = sum(c for _, c in goods)
        if tot <= 0 or room <= 0:
            return ["PASS"], room
        if tot <= room and len(goods) == sum(1 for c in inv.values() if c > 0):
            for i2, c in goods:
                shed_live[i2] = shed_live.get(i2, 0) + c
            return ["DROP"], room - tot
        i2, c = max(goods, key=lambda ic: ic[1])
        q = min(c, room)
        shed_live[i2] = shed_live.get(i2, 0) + q
        return ["PLACE", i2, q], room - q
