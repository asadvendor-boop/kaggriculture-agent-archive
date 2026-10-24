"""What the top teams do in the second half (days 8-29), as a function of the visible shop world.

planner/targets.json holds decision rules fitted on 1,214 recorded top-10 games (Sep 19-20 archive; top-10 seats
only) by tools/planner/fit_targets.py, validated out of sample on today's top-10 tapes.  This module loads it and
answers queries.  Standard library only; every query is O(1)-ish (a few dict lookups), safe to call every step.

Engine facts used here (checked against the engine by tools/planner/test_targets.py):
  - SHOPS: products each shop drains; a single-product shop drains 2 units per tick, others 1 per product.
  - A drain tick happens on every step with step % 4 == 0; the town centre drains 1 of every product except
    FERTILIZER on every step with step % 24 == 0.
  - A new shop (uniform draw with replacement from the 8 kinds) unlocks at the end of each day d with (d+1) % 3 == 0,
    so at hour 0 of day D the first min(8, D // 3) shops of the final list are visible.
  - The last processed step is 718 (day 29 hour 22).

Usage:
    from planner.targets import Targets
    T = Targets.load()                       # reads targets.json next to this file (or pass a dict/path)
    T.predict("tomato_starts_8_13", shops)   # -> target number, keyed on the shops visible at the rule's day
    T.predict("sheep_day15", shops_now, expect_unseen=True)   # called BEFORE the rule's vis_day: average over the
                                             # shops not drawn yet (the default treats them as draining nothing -> low)
    T.herd_target("GOOSE", shops, day)       # animals on farm the leaders hold at hour 0 of `day`
                                             # (pass expect_unseen=True when `day` is later than today)
    T.field_target("STRAWBERRY", shops, day) # plants on the field the leaders hold at hour 0 of `day`
    T.feed_today(species, placed_day, day)   # stop-feeding rule
    T.sale(product, day)                     # lot size / pacing / reservation price
    T.hires(day, workload)                   # hands per day
"""
import json
import os

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
PRODUCT_SOURCE = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160,
        "WOOL": 200, "FERTILIZER": 100}
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
SHOP_TICK_STEPS = 4          # shops drain on steps with step % 4 == 0
CENTER_TICK_STEPS = 24       # town centre drains on steps with step % 24 == 0
SHOP_UNLOCK_DAYS = 3         # a shop unlocks at the end of every day d with (d + 1) % 3 == 0
MAX_SHOPS = 8
LAST_STEP = 718              # last processed step
TURNS_PER_DAY = 24


def visible_shops(shops, day):
    """Shops visible at hour 0 of `day`, given the (partial or final) unlock-ordered shop list."""
    return list(shops[:min(MAX_SHOPS, day // SHOP_UNLOCK_DAYS)])


def drain4(shops):
    """Units drained per shop tick (every 4 steps) by these shop instances, per product."""
    out = {}
    for s in shops:
        prods = SHOPS.get(s, ())
        m = 2 if len(prods) == 1 else 1
        for p in prods:
            out[p] = out.get(p, 0) + m
    return out


def drain_per_day(shops):
    """Units per day drained by these shops plus the town centre (FERTILIZER has no town-centre drain)."""
    d4 = drain4(shops)
    return {p: d4.get(p, 0) * (TURNS_PER_DAY // SHOP_TICK_STEPS) + (0 if p == "FERTILIZER" else 1) for p in PRODUCTS}


def _ticks_between(step_from, step_to, every):
    """Number of steps s in [step_from, step_to] with s % every == 0 (step_from >= 0)."""
    if step_to < step_from:
        return 0
    return step_to // every - (step_from - 1) // every


def _mean_random_shop_drain4():
    d = {}
    for s in SHOPS:
        for p, n in drain4([s]).items():
            d[p] = d.get(p, 0.0) + n / len(SHOPS)
    return d


RANDOM_SHOP_DRAIN4 = _mean_random_shop_drain4()


def _unseen_drain_dists():
    """{product: [dist for m = 0..8 unseen shops]}, dist = {extra drain4: probability} for m independent uniform draws."""
    out = {}
    for p in PRODUCTS:
        one = {}
        for s in SHOPS:
            v = drain4([s]).get(p, 0)
            one[v] = one.get(v, 0.0) + 1.0 / len(SHOPS)
        dists = [{0: 1.0}]
        for _ in range(MAX_SHOPS):
            nd = {}
            for a, pa in dists[-1].items():
                for b, pb in one.items():
                    nd[a + b] = nd.get(a + b, 0.0) + pa * pb
            dists.append(nd)
        out[p] = dists
    return out


UNSEEN_DRAIN_DIST = _unseen_drain_dists()


def missing_shops(shops, day):
    """Shops that will be visible at hour 0 of `day` but are not in the caller's list (not drawn yet)."""
    return max(0, min(MAX_SHOPS, day // SHOP_UNLOCK_DAYS) - len(shops))


def expected_remaining_drain(shops_visible, step):
    """Expected units the town drains per product from `step` through the last step: the visible shops, the town
    centre, and the expected contribution of shops not yet unlocked (uniform draw over the 8 kinds)."""
    n_vis = len(shops_visible)
    ticks = _ticks_between(step, LAST_STEP, SHOP_TICK_STEPS)
    center = _ticks_between(step, LAST_STEP, CENTER_TICK_STEPS)
    d4 = drain4(shops_visible)
    out = {}
    for p in PRODUCTS:
        v = d4.get(p, 0) * ticks + (0 if p == "FERTILIZER" else center)
        for k in range(n_vis, MAX_SHOPS):
            start = (k + 1) * SHOP_UNLOCK_DAYS * TURNS_PER_DAY
            v += RANDOM_SHOP_DRAIN4.get(p, 0.0) * _ticks_between(max(step, start), LAST_STEP, SHOP_TICK_STEPS)
        out[p] = v
    return out


# ------------------------------------------------------------------------------------------------ animals
# (first_yield_day, interval): production at the end of day e when (e + 1 - placed_day - first_yield_day) >= 0 and
# divisible by interval.  The end of day 29 is never processed (the game stops after step 718), so the last
# production that can ever happen is at the end of day 28 (its yield is harvestable on day 29).
ANIMAL_SCHEDULE = {"GOOSE": (4, 1), "COW": (8, 2), "SHEEP": (6, 3)}
LAST_PRODUCTION_DAY = 28


def next_production_day(species, placed_day, day):
    """Smallest end-of-day e >= day at which this animal produces."""
    fyd, iv = ANIMAL_SCHEDULE[species]
    k = day + 1 - placed_day - fyd
    if k < 0:
        return placed_day + fyd - 1
    return day + (-k) % iv


def feed_today(species, placed_day, day, fed_yesterday=True, economy=False):
    """Stop-feeding rule of the leaders: feed on day d only while a production at an end-of-day <= 28 is still to
    come.  An animal whose last production is already done may then escape at the end of day 28 (2 unfed days):
    that costs nothing (engine-tested: same units harvested as feeding every day, fewer feeds).  economy=True also skips off-days when that is safe (fed yesterday, today is not a production day): the
    animal does not escape (escape needs 2 unfed days in a row) and the base unit still comes; only the care bonus
    that today's care would have banked is lost."""
    if day > LAST_PRODUCTION_DAY:
        return False
    e = next_production_day(species, placed_day, day)
    if e > LAST_PRODUCTION_DAY:
        return False
    if economy and e != day and fed_yesterday:
        return False
    return True


def care_today(species, placed_day, day):
    """Care banks +1 for the next production strictly after today (only if also fed today); worthwhile only while
    such a production at an end-of-day <= 28 remains."""
    return day < LAST_PRODUCTION_DAY and next_production_day(species, placed_day, day + 1) <= LAST_PRODUCTION_DAY


# ------------------------------------------------------------------------------------------------ rules
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DEFAULT_PATH = os.path.join(_HERE, "targets.json")


def hire_cost(n_already_today):
    """Cost of the next hand when n hands were already hired today: fib(n) with fib(0)=fib(1)=1 (engine _hire_cost)."""
    a, b = 1, 1
    for _ in range(n_already_today):
        a, b = b, a + b
    return a


def _clamp(y, lo, hi):
    if lo is not None and y < lo:
        y = lo
    if hi is not None and y > hi:
        y = hi
    return y


class Targets:
    """Query object over targets.json.  `shops` arguments are the unlock-ordered shop list the caller can see
    (obs["town"]["unlocked_shops"]); rules keyed on a later day than the caller's use what is visible now.
    flavor="archive" returns the archive fit; flavor="leaders" adds the mean residual of today's top-10 seats
    (validation set) where the rule has one (e.g. today's leaders plant ~3 more tomatoes on days 8-13);
    flavor="team:<name>" (headline rules only) uses that archive team's own fit (teams with >= 15 seats)."""

    def __init__(self, data):
        self.data = data
        self.rules = data.get("rules", {})

    @classmethod
    def load(cls, src=None):
        """src: None (targets.json next to this file), a path, or an already-parsed dict (for single-file agents)."""
        if isinstance(src, dict):
            return cls(src)
        with open(src or DEFAULT_PATH) as fh:
            return cls(json.load(fh))

    # ---- headline rules
    def rule(self, name):
        return self.rules[name]

    def covariate(self, name, shops):
        r = self.rules[name]
        return drain4(visible_shops(shops, r["vis_day"])).get(r["product"], 0)

    def predict(self, name, shops, flavor="archive", use_levels=True, min_level_n=30, expect_unseen=False):
        """Target value of a headline rule.  Uses the level-table mean when that drain4 level has >= min_level_n
        seats, else the linear fit a + b * drain4.
        expect_unseen=True: when `shops` holds fewer shops than will be visible at the rule's vis_day (a planner
        taking over before vis_day), average the prediction over the uniform draws of the missing shops instead of
        treating them as draining nothing (the default, which is biased low by ~ b * E[drain4 of one shop] per
        missing shop; reviewer test R9 in tools/planner/test_targets_review.py)."""
        r = self.rules[name]
        k = self.covariate(name, shops)
        m = missing_shops(shops, r["vis_day"]) if expect_unseen else 0
        if m:
            return sum(pr * self._predict_k(r, k + dk, flavor, use_levels, min_level_n)
                       for dk, pr in UNSEEN_DRAIN_DIST[r["product"]][m].items())
        return self._predict_k(r, k, flavor, use_levels, min_level_n)

    @staticmethod
    def _predict_k(r, k, flavor, use_levels, min_level_n):
        lv = r.get("levels", {}).get(str(k))
        tm = r.get("by_team", {}).get(flavor[5:]) if flavor.startswith("team:") else None
        if tm:                                   # one team's own fit, e.g. flavor="team:DSM"
            y = tm["a"] + tm["b"] * k
        elif use_levels and lv and lv["n"] >= min_level_n:
            y = lv["mean"]
        else:
            y = r["a"] + r["b"] * k
        if flavor == "leaders":
            y += r.get("leaders_offset") or 0.0
        lo, hi = (r.get("clamp") or [0, None])[:2]
        return _clamp(y, lo, hi)

    # ---- per-day tables
    def _table(self, section, key, shops, day, product, expect_unseen=False):
        day = max(0, min(29, int(day)))
        e = self.data[section][key][str(day)]
        k = drain4(visible_shops(shops, day)).get(product, 0)
        m = missing_shops(shops, day) if expect_unseen else 0
        if m:
            return sum(pr * max(0.0, e["a"] + e["b"] * (k + dk)) for dk, pr in UNSEEN_DRAIN_DIST[product][m].items())
        return max(0.0, e["a"] + e["b"] * k)

    def stock_target(self, key, shops, day, flavor="archive", expect_unseen=False):
        """Plants of a crop / animals of a species the leaders hold at hour 0 of `day`, keyed on drain4 visible that day.
        expect_unseen=True averages over the shops that unlock between now and `day` (see predict)."""
        prod = ANIMAL_PRODUCT.get(key, key)
        y = self._table("stock", key, shops, day, prod, expect_unseen)
        if flavor == "leaders":
            y = max(0.0, y + (self.data["stock"][key][str(max(0, min(29, int(day))))].get("val_bias") or 0.0))
        return y

    herd_target = stock_target
    field_target = stock_target

    def starts_today(self, crop, shops, day, expect_unseen=False):
        """Expected plants of `crop` the leaders start on `day` (mean policy, keyed on drain4 visible that day)."""
        return self._table("starts", crop, shops, day, crop, expect_unseen)

    def animal_buys_today(self, species, shops, day, expect_unseen=False):
        return self._table("animal_buys", species, shops, day, ANIMAL_PRODUCT[species], expect_unseen)

    # ---- feeding (engine-schedule rule, see module functions)
    @staticmethod
    def next_production_day(species, placed_day, day):
        return next_production_day(species, placed_day, day)

    @staticmethod
    def feed_today(species, placed_day, day, fed_yesterday=True, economy=False):
        return feed_today(species, placed_day, day, fed_yesterday, economy)

    @staticmethod
    def care_today(species, placed_day, day):
        return care_today(species, placed_day, day)

    def feeding_stats(self, species):
        return self.data["feeding"]["species"][species]

    # ---- sales
    def sale(self, product, day):
        """Leaders' selling habits for `product` around `day`: phase stats plus cadence / lot / reservation info."""
        s = self.data["sales"][product]
        ph = None
        for k, v in s["phases"].items():
            a, b = (int(x) for x in k.split("-"))
            if a <= day <= b:
                ph = v
        return dict(phase=ph, tick_phase_share=s["tick_phase_share"], lot_per_drain4=s["lot_per_drain4_median"],
                    gap_steps=s["gap_steps_median"], reservation_ratio=s["reservation_ratio"],
                    endgame_share=s["endgame_share"])

    def sale_lot(self, product, shops, day, stock):
        """Lot the leaders would sell now: ~lot_per_drain4 x the per-tick drain of the visible shops (selling into
        the drain, preferably at step % 4 == 1, right after the tick); products with no shop drain (MELON,
        FERTILIZER) use the phase median lot.  Never more than `stock`."""
        if stock <= 0:
            return 0
        s = self.data["sales"][product]
        d4 = drain4(visible_shops(shops, day)).get(product, 0)
        if d4 and s.get("lot_per_drain4_median"):
            lot = int(round(s["lot_per_drain4_median"] * d4))
        else:
            ph = self.sale(product, day)["phase"]
            lot = int(ph["lot_median"]) if ph else stock
        return max(1, min(stock, lot))

    # ---- hands, land, layout, cash
    def hires(self, day, stat="median"):
        return self.data["hires"]["by_day"][str(max(0, min(29, int(day))))][stat]

    def land(self, quadrant):
        return self.data["land"][quadrant]

    def layout(self, key):
        return self.data["layout"][key]

    def unit_service(self, kind="hand"):
        return self.data["layout"]["unit_service"][kind]

    def cash(self, day, stat="median"):
        return self.data["cash"][stat][max(0, min(29, int(day)))]

    def fertilizer(self):
        return self.data["fertilizer"]
