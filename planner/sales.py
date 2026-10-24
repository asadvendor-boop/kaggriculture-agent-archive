"""planner/sales.py -- margin-aware selling for the post-handover game (day 12 on).  Standard library only.

    from planner.sales import SalesPolicy, stock_after_units
    sp = SalesPolicy()                         # one instance per game and seat (sp.reset() at step 0)
    sp.observe(obs, action, stock)             # every step BEFORE the handover (keeps the rival/clone trackers warm)
    orders = sp.orders(obs, state, stock, context)   # every step after it: [["SELL", item, n], ...] for THIS step
    market = sp.market(obs, state, stock, {"market": tape_market})   # hybrid: the full merged market list

  stock    our shed after this step's unit actions (PICKUP / DROP / shed-PLACE), before the market runs:
           stock_after_units(obs, action) computes it exactly as the engine does (farmer, then hands in order).
  state    unused (accepted for the planner interface; everything is read from obs).
  context  optional dict:
    other_orders  this step's non-SELL market orders (HIRE / BUY_*), in engine order.  The SELLs returned here are
                  meant to go IN FRONT of them (money and shed room from the sales are then available to the buys);
                  they are counted for the 10-orders cap, the cash need and the rival-flow bookkeeping.
    tape_sells    the tape's own SELL orders for this step (hybrid: the tape keeps running in parallel).  Items in
                  params['follow_tape'] (default WHEAT, FERTILIZER: the tape's feed / fertilizer inputs; paced goods
                  may be added, then they keep the tape's timing except against a clone) are sold exactly as the tape
                  says (params['dump_inputs'] adds a paced dump of every unit left on day 29).
    reserve       {item: units to keep in the shed} for the planner's own later use (without tape_sells, only
                  stock - reserve of WHEAT / FERTILIZER is sold).
    cash_need     dollars needed by this step's purchases (default: priced from other_orders).
    carried       units the workers will still be carrying at the end of the day (midnight auto-drop; default: what
                  they carry now).
    max_orders    engine cap on market orders per turn (10).

Measured facts this policy is built on (runs/planner/handover_report.txt, planner/targets.json, the engine):
  * both farms sell into one book.  A unit sold at step s raises the inventory for every later sale by anyone
    (the town drain is fixed; inventory never mean-reverts), so the objective is the MARGIN own - rival.
    Moving one of our units from step b to an earlier step a changes only the sales in (a, b):
        dMargin = p(a) - p(b) - sum(our sales in (a,b)) * slope + sum(rival sales in (a,b)) * slope.
    With every paced unit sold by the end of the same day (holding across days fails: the shed fills at midnight
    with wheat/fertilizer cargo and overflow is destroyed), the timing problem is closed inside the day.
  * the town drains every step with step % 4 == 0 AFTER the market; a sale at step % 4 == 1 is the first one into
    the freshly drained book.  Leaders sell milk/wool/strawberries in lots of ~1 x the per-tick drain, just after the
    tick (tick_phase_share peaks at step % 4 == 1).
  * FERTILIZER has no drain at all and MELON only the town centre's 1/day: those books are pure races (a unit held
    is a unit the rival sells first at the higher price).  Their late prices are far below any reservation price
    (fertilizer sells at ~0.2-0.4 x base after day 20), so the reservation price only applies while some real
    drain can lift the price (MELON before day 28).
  * the last applied step is 718 (day 29, hour 22); a good still in the shed after it is worth nothing.

MODES (params['mode']) for the paced goods (STRAWBERRY MILK WOOL EGG TOMATO CARROT):
  forecast  (default) each step, allocate the stock over the remaining steps of the day by margin marginal revenue:
            own price + effect on our later planned units today + denial of the rival's forecast sales in between.
            The rival's intraday flow per hour is the mean of its exact observed flow (FlowTracker identity) at the
            same hour of the last params['rival_days'] days, optionally blended (params['rival_flat']) with its flat
            per-step rate over those days.  Units allocated to the current step are sold now.
  pace      leaders' rule: lot = round(params['lot_per_tick'] x per-tick drain) at step % 4 == 1, catch-up so that
            the stock is gone by the last hour; params['defer'] {item: hour} holds an item until that hour.
  asap      everything now.
  tape      the tape's own SELL orders, untouched (identity; for checks and as a fallback).
Against a clone (the rival's hands and farmer stand on our tiles on >= 4 of the last 6 steps with hands, and the
tile layouts match >= 0.95 as in LiveRepair's _r37_similarity; optionally also a near-clone: similarity >=
params['clone_sim'] on 4 of the last 6 observations -- V48 games that diverged from our route sit at ~0.91 with
units on our tiles only ~23% of the time, leader tapes at 0.08-0.36) the paced goods are sold at once (first-come race), or,
with params['clone_mode'] = 'tape' and the tape running, the tape's own SELLs are kept (LiveRepair's race layers
pre-empt a clone by up to 24 steps and align lockstep slots with dead orders; measured: replacing them with 'sell at
once' cost $1,460/game of margin and 8 of 40 wins against V94, which runs the same public route tape).
Always: every paced unit is sold by params['last_hour'] (day 29: step 718), never leaves the shed above
params['hold_cap'] units after the market (nor shed + workers' cargo above params['hold_cap_total']), and sells
enough (best prices first) to cover this step's cash need.

MEASURED (2026-09-23; LiveRepair + this policy from step 288, SALES layer in tools/planner/test_sales.py; paired with
LiveRepair on the same seeds / tapes; margin = own - rival bank; runs/planner/sales_official.out):
  The layer feeds LiveRepair a VIRTUAL view (its own paced SELLs executed: paced shed minus the units we hold beyond its
  plan, cash plus the revenue difference).  Without it the tape's production reacts to our holding (its CAPHARV layer
  skips overflow-saving harvests at shed + cargo >= 90, among others): dev forecast run lost 2.8 eggs, 2.8 milk, 3.4
  fertilizer per game; a cargo cap instead forced bad sales (-$291/game).  With the view the losses vanish.
  Dev (leader jobs 150-209, all paced goods by forecast): margin +451 [+174,+742], both-dp +44 -- wins on milk and wool,
  losses on strawberry (-135) and carrot (-115) -> those two keep the tape's timing in the official variants.
  Official ruler, evalcand --fresh 40 --top 120 (leader jobs 0-119, disjoint from dev):
                   leader tapes all 120          both-dp (n=61)              W (dp)     fresh V94        fresh V48
    SALES12        +568 [+345,+787]  64 vs 58    +472 [+181,+763]  22 vs 18  L->W 5/1   -1,460 (32-8)    -502 (40-0)
                   (base: LiveRepair 40-0 vs V94, 39-1 vs V48; leader dp W 18/61)
    SALES12b       +355 [+80,+630]   60 vs 58    +149 [-196,+472]  19 vs 18  L->W 4/3   0 (identical)   -262 (39-1)
    SALES12c       +505 [+289,+717]  64 vs 58    +349 [+79,+619]   22 vs 18  L->W 5/1   0 (identical)   0 (identical)
    SALES12  = forecast MILK/WOOL/EGG/TOMATO, tape timing STRAWBERRY/CARROT/WHEAT/FERTILIZER, sell-at-once vs clones;
    SALES12b = + strawberries held to hour 22 (-$323/game vs SALES12 on both-dp: the leader gains $329) + tape vs clones;
    SALES12c = SALES12 + the tape's SELLs vs clones/near-clones (sticky, similarity >= 0.85): identical to LiveRepair vs
               V94/V48; on 5/120 leader tapes (teams sharing our early route) the day-8 flag also fires and gives back
               SALES12's gain there (-$124/game on both-dp vs SALES12) -- a flag judged after the handover (e.g. rolling
               similarity over the last day) should recover it (not measured).
  SALES12c per product (receipts, both-dp): margin MILK +173, WOOL +225, EGG +13, STRAWBERRY -53, WHEAT -47 (tape's
  own wheat trading shifts), rest ~0.  Cost: ~0.15 ms per step (max < 1 ms).
"""
try:
    M  # bound by an inlining builder
except NameError:
    from planner import model as M

PRODUCTS = tuple(M.PRODUCTS)
PACED = ("STRAWBERRY", "MILK", "WOOL", "EGG", "TOMATO", "CARROT")
INPUTS = ("WHEAT", "FERTILIZER")
RACE = ("MELON",)
ANIMAL_ITEMS = ("GOOSE", "COW", "SHEEP")
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
SEED_COST = {c: M.CROPS[c]["seed"] for c in M.CROPS}
ANIMAL_COST = {a: M.ANIMALS[a]["cost"] for a in M.ANIMALS}
SHED_CAP = M.SHED_CAP
LAST_STEP = M.LAST_STEP
TPD = M.TPD

DEFAULT_PARAMS = {
    "mode": "forecast",
    "start_step": 288,          # first step this policy sells for (informational; the caller decides)
    "last_hour": 22,            # every paced unit is sold by this hour (hour 23 sells what is still left)
    "hold_cap": 60,             # shed units allowed after the market while paced goods are held back
    "hold_cap_total": None,     # ... and shed + workers' cargo (None: no cargo limit)
    "rival_days": 3,            # days of rival intraday flow history used by the forecast
    "rival_flat": 0.0,          # weight of the flat per-step rate (last rival_days days) blended into that profile
    "denial_weight": 1.0,       # weight of the rival's revenue loss in the margin objective
    "own_weight": 1.0,          # weight of the depression of our own later units
    "lot_per_tick": 1.0,        # pace mode
    "defer": {},                # pace/forecast: {item: hour} hold until this hour against non-clones
    "clone_asap": True,         # vs a clone: every paced unit now (first-come race) ...
    "clone_mode": "asap",       # ... or "tape": keep the tape's own SELLs (its race layers) when tape_sells are given
    "clone_sticky": False,      # once a clone is seen, keep the flag for the rest of the game
    "clone_sim": None,          # near-clone: tile similarity >= this on 4 of the last 6 observations (None: off)
    "cash_floor": 300.0,        # keep this much after the step's purchases
    "follow_tape": ("WHEAT", "FERTILIZER"),   # items whose tape SELLs are kept verbatim (paced ones too, except vs clones)
    "asap_items": (),           # paced items sold as soon as they are in the shed (tape-less stand-in for follow_tape)
    "dump_day": 29,             # from this day every input unit left is sold (paced over the day's steps) ...
    "dump_inputs": False,       # ... also when the tape's own SELLs are given (the tape already clears its shed)
    "order": "exposure",        # SELL order: "exposure" (rival-batch exposure, LiveRepair r37) or "value"
    "melon_reservation": 0.8,   # x base, until day 28
    "max_orders": 10,
}

# The measured configuration (agents/cand/SALES12c.py; tools/planner/test_sales.py CONFIGS) for a hybrid in which the
# tape keeps running and its own SELLs are passed as context: forecast timing for MILK / WOOL / EGG / TOMATO, the tape's
# timing for STRAWBERRY / CARROT / WHEAT / FERTILIZER, the tape's SELLs against clones and near-clones (sticky).
RECOMMENDED = {"mode": "forecast", "hold_cap_total": 95,
               "follow_tape": ("WHEAT", "FERTILIZER", "STRAWBERRY", "CARROT"),
               "clone_mode": "tape", "clone_sticky": True, "clone_sim": 0.85}
# Without a tape (planner-only; NOT measured): strawberries and carrots sold on arrival (what the tape mostly does),
# inputs as stock - context['reserve'].  Against clones this falls back to selling at once, which lost $1,460/game vs
# V94 in SALES12 -- keep the tape's SELLs available against clones if at all possible.
RECOMMENDED_NO_TAPE = {"mode": "forecast", "hold_cap_total": 95, "asap_items": ("STRAWBERRY", "CARROT"),
                       "clone_sticky": True, "clone_sim": 0.85}


def _g(o, k, d=None):
    try:
        v = o[k]
    except (KeyError, IndexError, TypeError):
        return d
    return d if v is None else v


def _valid(o):
    """An order the engine would act on (anything else is a dead slot that only holds a lockstep position)."""
    if not isinstance(o, list) or not o:
        return False
    if o[0] in ("HIRE", "BUY_LAND"):
        return True
    if o[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(o) >= 3:
        try:
            return int(o[2]) > 0
        except (TypeError, ValueError):
            return False
    return False


def _live_sell(o):
    return _valid(o) and o[0] == "SELL" and o[1] in PRODUCTS


def _pos(p):
    return (int(p[0]), int(p[1]))


def stock_after_units(obs, action):
    """Our shed after this step's unit actions, before the market (engine order: farmer, then hands; PICKUP capped at
    the stock, DROP of everything carried with overflow discarded, shed-PLACE of min(n, carried, room)); only units on
    the four shed-access tiles touch the shed.  Returns {item: units} over products and animals."""
    private = _g(obs, "private", {}) or {}
    shed = {k: max(0, int(v)) for k, v in dict(_g(private, "shed", {}) or {}).items()}
    invs = [dict(i or {}) for i in (_g(private, "inventories", []) or [])]
    farm = obs["farms"][int(obs["player"])]
    positions = [_g(farm, "farmer")] + list(_g(farm, "hands", []) or [])
    acts = [(_g(action, "farmer") or ["PASS"])] + list(_g(action, "hands", []) or []) if isinstance(action, dict) else []
    total = sum(shed.values())
    for i, act in enumerate(acts):
        if i >= len(positions) or positions[i] is None or not isinstance(act, list) or not act:
            continue
        if _pos(positions[i]) not in SHED_TILES:
            continue
        op = act[0]
        inv = invs[i] if i < len(invs) else {}
        if op == "PICKUP" and len(act) >= 2:
            try:
                n = int(act[2]) if len(act) >= 3 else 1
            except (TypeError, ValueError):
                n = 1
            n = min(max(0, n), shed.get(act[1], 0))
            if n > 0:
                shed[act[1]] -= n
                total -= n
                inv[act[1]] = inv.get(act[1], 0) + n
        elif op == "DROP":
            for item, n in list(inv.items()):
                n = max(0, int(n))
                take = min(n, max(0, SHED_CAP - total))
                if take > 0:
                    shed[item] = shed.get(item, 0) + take
                    total += take
                inv[item] = 0
        elif op == "PLACE" and len(act) >= 2 and act[1] not in ANIMAL_ITEMS:
            try:
                n = int(act[2]) if len(act) >= 3 else 1
            except (TypeError, ValueError):
                n = 1
            n = min(max(0, n), max(0, int(inv.get(act[1], 0))), max(0, SHED_CAP - total))
            if n > 0:
                shed[act[1]] = shed.get(act[1], 0) + n
                total += n
                inv[act[1]] -= n
    return shed


def similarity(obs):
    """Share of occupied tiles (crop or animal on either farm) holding the same crop/animal on both farms; 0 when the
    unlocked quadrants differ or fewer than 8 tiles are occupied (LiveRepair's _r37_similarity)."""
    farms = obs["farms"]
    me = int(obs["player"])
    own, rival = farms[me], farms[1 - me]
    if list(_g(own, "unlocked_quadrants", [])) != list(_g(rival, "unlocked_quadrants", [])):
        return 0.0
    matches = total = 0
    for ra, rb in zip(own["tiles"], rival["tiles"]):
        for a, b in zip(ra, rb):
            sa = (_g(a, "crop"), _g(a, "animal")) if isinstance(a, dict) else (None, None)
            sb = (_g(b, "crop"), _g(b, "animal")) if isinstance(b, dict) else (None, None)
            if sa != (None, None) or sb != (None, None):
                total += 1
                matches += sa == sb
    return matches / total if total >= 8 else 0.0


def order_cost(order, obs, hires_before=0):
    """Dollars an order would spend at current prices (0 for SELL / unknown)."""
    if not isinstance(order, list) or not order:
        return 0.0
    op = order[0]
    try:
        n = max(0, int(order[2])) if len(order) >= 3 else 1
    except (TypeError, ValueError):
        n = 0
    if op == "HIRE":
        return float(M.hire_cost(hires_before))
    if op == "BUY_LAND":
        farm = obs["farms"][int(obs["player"])]
        nl = M.next_land(list(_g(farm, "unlocked_quadrants", [])))
        return float(nl[1]) if nl else 0.0
    if op == "BUY_SEED" and len(order) >= 2:
        return float(SEED_COST.get(order[1], 0) * n)
    if op == "BUY_ANIMAL" and len(order) >= 2:
        return float(ANIMAL_COST.get(order[1], 0) * n)
    if op == "BUY_PRODUCT" and len(order) >= 2 and order[1] in PRODUCTS:
        inv = int(obs["market"]["inventory"][order[1]])
        return float(sum(M.buy_quotes(order[1], inv, n)[0]))
    return 0.0


def exposure(obs, item, qty):
    """Revenue of selling `qty` now minus selling it after a rival batch lands first (LiveRepair's _r37_quote_priority:
    batch = the rival's visible standing yield of the item, clamped to 8..24).  Used to order our SELLs: the lockstep
    market runs slot by slot, so a product in an earlier slot than the rival's same product sells first."""
    if qty <= 0 or item not in PRODUCTS:
        return 0.0
    inv = int(obs["market"]["inventory"][item])
    rival = obs["farms"][1 - int(obs["player"])]
    crop = item if item in M.CROPS else None
    animal = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}.get(item)
    standing = 0
    for row in rival["tiles"]:
        for t in row:
            if isinstance(t, dict) and ((crop is not None and _g(t, "crop") == crop) or
                                        (animal is not None and _g(t, "animal") == animal)):
                standing += max(0, int(_g(t, "yield_units", 0)))
    batch = min(24, max(8, standing))
    now = sum(M.price(item, inv + j) for j in range(qty))
    later = sum(M.price(item, inv + batch + j) for j in range(qty))
    return float(now - later)


class SalesPolicy:
    def __init__(self, params=None):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.tracker = M.FlowTracker()
        self.flows = {}             # step -> {P: rival net units}  (last ~4 days)
        self.pending_net = None     # (step, {P: our expected executed net units})
        self.last_step = -1
        self.clone_hist = []
        self.sim_hist = []
        self.clone = False
        self.info = {}

    def _net_of(self, obs, market, stock):
        """Our expected executed net flow for a market list: sells capped by the stock (in order), buys as ordered
        (WHEAT/FERTILIZER only move the market)."""
        left = dict(stock or {})
        net = {}
        for o in market or []:
            if not isinstance(o, list) or len(o) < 3:
                continue
            try:
                n = int(o[2])
            except (TypeError, ValueError):
                continue
            if n <= 0 or o[1] not in PRODUCTS:
                continue
            if o[0] == "SELL":
                k = min(n, max(0, int(left.get(o[1], 0))))
                cur = int(obs["market"]["inventory"][o[1]]) + net.get(o[1], 0)
                for _ in range(k):            # a sale at the $1 floor adds no supply
                    if M.price(o[1], cur) > 1:
                        cur += 1
                        net[o[1]] = net.get(o[1], 0) + 1
                left[o[1]] = left.get(o[1], 0) - k
            elif o[0] == "BUY_PRODUCT" and o[1] in ("WHEAT", "FERTILIZER"):
                net[o[1]] = net.get(o[1], 0) - n
        return net

    def _update(self, obs):
        s = int(obs["step"])
        if s == self.last_step:          # already updated for this step (observe() then orders())
            return
        if s < self.last_step:           # a new game in the same process
            self.reset()
        our = self.pending_net[1] if (self.pending_net and self.pending_net[0] == s - 1) else {}
        res = self.tracker.update(obs, our)
        if res is not None:
            self.flows[s - 1] = res
            for k in [k for k in self.flows if k < s - 24 * 5]:
                del self.flows[k]
        self._clone_update(obs)
        self.last_step = s

    def _clone_update(self, obs):
        farms = obs["farms"]
        me = int(obs["player"])
        own, riv = farms[me], farms[1 - me]
        hands = list(_g(own, "hands", []) or [])
        if self.clone and self.p.get("clone_sticky"):
            return
        if hands:
            eq = [list(h) for h in hands] == [list(h) for h in (_g(riv, "hands", []) or [])] and \
                list(_g(own, "farmer", [])) == list(_g(riv, "farmer", []))
            self.clone_hist.append(bool(eq))
            if len(self.clone_hist) > 6:
                self.clone_hist.pop(0)
        sim = similarity(obs)
        self.sim_hist.append(sim)
        if len(self.sim_hist) > 6:
            self.sim_hist.pop(0)
        strict = len(self.clone_hist) >= 4 and sum(self.clone_hist) >= 4 and sim >= 0.95
        cs = self.p.get("clone_sim")
        near = cs is not None and len(self.sim_hist) >= 4 and sum(x >= cs for x in self.sim_hist) >= 4
        self.clone = strict or near

    def observe(self, obs, action=None, stock=None):
        """Call once per step when this policy does NOT choose the sales (before the handover): tracks the rival's flow
        and the clone signal.  `action` is our final action of this step, `stock` = stock_after_units(obs, action)."""
        self._update(obs)
        if stock is None and isinstance(action, dict):
            stock = stock_after_units(obs, action)
        market = (_g(action, "market", []) or []) if isinstance(action, dict) else []
        self.pending_net = (int(obs["step"]), self._net_of(obs, market[:self.p["max_orders"]], stock or {}))

    # ------------------------------------------------------------------ forecast pieces
    def rival_profile(self, P, step, T):
        """Expected rival net units of P at each step step..T: mean of its observed flow at the same hour over the
        last rival_days days (steps not observed are skipped)."""
        K = int(self.p["rival_days"])
        a = float(self.p.get("rival_flat", 0.0))
        out = []
        for s in range(step, T + 1):
            vals = []
            for k in range(1, K + 1):
                f = self.flows.get(s - 24 * k)
                if f is not None:
                    vals.append(f.get(P, 0))
            out.append(sum(vals) / len(vals) if vals else 0.0)
        if a > 0:
            # blend with the flat rate of the last K days (sporadic sellers are badly timed by same-hour means)
            tot, n = 0.0, 0
            for k in range(1, 24 * K + 1):
                f = self.flows.get(step - k)
                if f is not None:
                    tot += f.get(P, 0)
                    n += 1
            flat = tot / n if n else 0.0
            out = [(1 - a) * v + a * flat for v in out]
        return out

    def plan_window(self, P, inv0, step, T, units, shops, rival=None, first_ok=True):
        """Greedy margin allocation of `units` (all available now) over steps step..T.  Returns {step: units}."""
        n = T - step + 1
        if n <= 0 or units <= 0:
            return {step: units} if units > 0 else {}
        drain = [M.drain_at_step(s, shops)[P] for s in range(step, T + 1)]
        r = rival if rival is not None else self.rival_profile(P, step, T)
        q = [0] * n
        wd = float(self.p["denial_weight"])
        wo = float(self.p["own_weight"])
        price = M.market_price
        defer_h = (self.p.get("defer") or {}).get(P)
        allowed = []
        for i in range(n):
            s = step + i
            ok = (i > 0 or first_ok)
            if defer_h is not None and not self.clone and (s % TPD) < defer_h and s < T:
                ok = False
            allowed.append(ok)
        if not any(allowed):
            allowed[-1] = True
        for _ in range(int(units)):
            # forward pass: inventory entering each step
            starts = [0.0] * n
            inv = float(inv0)
            d_own = [0.0] * n
            d_riv = [0.0] * n
            for i in range(n):
                starts[i] = inv
                ri = r[i]
                base = inv + ri
                k_eff = q[i]
                if q[i]:
                    d_own[i] = price(P, base + q[i]) - price(P, base)
                    while k_eff > 0 and price(P, base + k_eff - 1) <= 1:   # $1 sales add no supply
                        k_eff -= 1
                if ri > 0:
                    mid = inv + ri / 2.0
                    d_riv[i] = ri * (price(P, mid + 1) - price(P, mid))
                inv = base + k_eff - drain[i]
            best, best_v = None, None
            suf_own = 0.0
            suf_riv = 0.0
            for i in range(n - 1, -1, -1):
                if allowed[i]:
                    ri = r[i]
                    base = starts[i] + ri
                    pv = price(P, base + q[i])
                    v = pv + wo * suf_own - wd * suf_riv if pv > 1 else 1.0   # a $1 unit moves nothing
                    if best_v is None or v >= best_v - 1e-9:     # ties -> the earliest step
                        best, best_v = i, v
                suf_own += d_own[i]
                suf_riv += d_riv[i]
            q[best] += 1
        return {step + i: q[i] for i in range(n) if q[i]}

    # ------------------------------------------------------------------ main
    def orders(self, obs, state=None, stock=None, context=None):
        """SELL orders for this step (planner interface): our re-timed SELLs, most exposed first, then -- when
        context['tape_sells'] is given -- the tape's own SELLs of the items it keeps (params['follow_tape']).
        They are meant to go in front of context['other_orders'].  At most max_orders - len(other_orders)."""
        ctx = context or {}
        step = int(obs["step"])
        others = [o for o in (ctx.get("other_orders") or [])]
        slots = max(0, int(ctx.get("max_orders", self.p["max_orders"])) - len(others))
        front, follow = self._decide(obs, stock, ctx)
        kept = [list(o) for o in (ctx.get("tape_sells") or []) if _live_sell(o) and o[1] in follow]
        out = (front + kept)[:slots]
        if stock is None:
            stock = {k: max(0, int(v)) for k, v in dict(obs["private"]["shed"]).items()}
        self.pending_net = (step, self._net_of(obs, out + others, stock))
        return out

    def market(self, obs, state=None, stock=None, context=None):
        """Full market list for this step in the hybrid (tape running in parallel): context['market'] = the tape's own
        list.  The tape's SELLs of items this policy re-times are removed; every other tape order (buys, hires, dead
        slots, SELLs of the items it keeps) stays in its relative order, behind our SELLs.  Dead slots are dropped only
        when our SELLs need the room.  mode 'tape' returns the tape's list unchanged."""
        ctx = dict(context or {})
        cap = int(ctx.get("max_orders", self.p["max_orders"]))
        tm = [o for o in (ctx.get("market") or [])][:cap]
        ctx["tape_sells"] = [o for o in tm if _live_sell(o)]
        ctx["other_orders"] = [o for o in tm if not _live_sell(o)]
        step = int(obs["step"])
        if stock is None:
            stock = {k: max(0, int(v)) for k, v in dict(obs["private"]["shed"]).items()}
        front, follow = self._decide(obs, stock, ctx)
        rest = [o for o in tm if not (_live_sell(o) and o[1] not in follow)]
        while len(front) > cap - len(rest):
            dead = [j for j, o in enumerate(rest) if not _valid(o)]
            if not dead:
                break
            del rest[dead[-1]]
        final = front[:max(0, cap - len(rest))] + rest
        self.pending_net = (step, self._net_of(obs, final, stock))
        return final

    def _decide(self, obs, stock, ctx):
        """-> (front, follow): our SELL orders for the items this policy times (sorted), and the set of items whose
        tape SELLs are kept verbatim."""
        self._update(obs)
        step = int(obs["step"])
        day, hour = step // TPD, step % TPD
        me = int(obs["player"])
        farm = obs["farms"][me]
        prices = {P: int(obs["market"]["prices"][P]) for P in PRODUCTS}
        inv = {P: int(obs["market"]["inventory"][P]) for P in PRODUCTS}
        shops = list(obs["town"]["unlocked_shops"])
        if stock is None:
            stock = {k: max(0, int(v)) for k, v in dict(obs["private"]["shed"]).items()}
        stock = {k: max(0, int(v)) for k, v in stock.items()}
        others = [o for o in (ctx.get("other_orders") or [])]
        mode = self.p["mode"]
        self.info = dict(step=step, clone=self.clone, plan={})
        has_tape = "tape_sells" in ctx
        if mode == "tape" and has_tape:      # identity: the tape's own SELL orders, untouched (checks / fallback)
            return [], set(PRODUCTS)
        if self.clone and has_tape and self.p.get("clone_mode") == "tape":
            return [], set(PRODUCTS)         # vs a clone the tape's own race logic (pre-emption, lockstep slots) sells
        T_last = min(LAST_STEP, day * TPD + int(self.p["last_hour"]))
        last_chance = step >= T_last
        final_day = day >= int(self.p["dump_day"]) or step + TPD > LAST_STEP
        want = {}
        follow = set()

        # ---- inputs (WHEAT / FERTILIZER)
        tape = {}
        for o in ctx.get("tape_sells") or []:
            if _live_sell(o):
                tape[o[1]] = tape.get(o[1], 0) + int(o[2])
        reserve = ctx.get("reserve") or {}
        for P in INPUTS:
            have = stock.get(P, 0)
            if final_day and (not has_tape or self.p["dump_inputs"]):
                # everything left is worthless after step 718: spread it over the remaining post-tick steps
                if have > 0:
                    left_steps = [s for s in range(step, LAST_STEP + 1) if s == step or s % 4 == 1]
                    k = have if len(left_steps) <= 1 else max(tape.get(P, 0), -(-have // len(left_steps)))
                    want[P] = min(have, k)
            elif has_tape and P in tuple(self.p["follow_tape"]):
                follow.add(P)
            elif have > 0:
                want[P] = max(0, have - int(reserve.get(P, 0)))

        # ---- race goods (MELON)
        for P in RACE:
            have = stock.get(P, 0)
            if have <= 0:
                continue
            base = M.MARKET_PARAMS[P]["base"]
            if day >= 28 or final_day or prices[P] >= self.p["melon_reservation"] * base or self.clone:
                want[P] = have

        # ---- paced goods
        plan_info = {}
        for P in PACED:
            if has_tape and P in tuple(self.p["follow_tape"]) and not (self.clone and self.p["clone_asap"]):
                follow.add(P)               # the tape's own timing for this product
                continue
            have = stock.get(P, 0)
            if have <= 0:
                continue
            if mode == "tape":               # no tape given: nothing to follow, sell as it comes
                want[P] = have
                continue
            if last_chance or mode == "asap" or (self.clone and self.p["clone_asap"]) or P in self.p["asap_items"]:
                want[P] = have
                continue
            if mode == "pace":
                d4 = M.shop_tick_drain(shops)[P]
                defer_h = (self.p.get("defer") or {}).get(P)
                if d4 <= 0:
                    want[P] = have
                    continue
                if defer_h is not None and hour < defer_h:
                    continue
                lot = max(1, int(round(float(self.p["lot_per_tick"]) * d4)))
                ticks_left = len([s for s in range(step, T_last + 1) if s % 4 == 1])
                if step % 4 == 1:
                    k = max(lot, have - lot * (ticks_left - 1))
                else:
                    k = max(0, have - lot * ticks_left)
                if k > 0:
                    want[P] = min(have, k)
                continue
            # forecast
            alloc = self.plan_window(P, inv[P], step, T_last, have, shops)
            plan_info[P] = alloc
            if alloc.get(step):
                want[P] = alloc[step]

        # ---- safety: shed room while holding, and cash
        def held_total():
            kept = sum(min(stock.get(P, 0), tape.get(P, 0)) for P in follow)
            return sum(stock.values()) - sum(want.values()) - kept

        cap = int(self.p["hold_cap"])
        cap_total = self.p.get("hold_cap_total")
        if cap_total is not None:
            # the workers' cargo lands in the shed at midnight, and LiveRepair's layers read shed + cargo (its CAPHARV
            # layer skips overflow-saving harvests at >= 90): hold back only while shed + cargo stays under cap_total
            private = obs["private"]
            shed_now = sum(max(0, int(v)) for v in dict(private["shed"]).values())
            carried_now = sum(max(0, int(v)) for inv in (private["inventories"] or []) for v in dict(inv).values())
            carried_after = max(0, carried_now + shed_now - sum(stock.values()))
            cap = min(cap, max(0, int(cap_total) - carried_after))
        if held_total() > cap:
            # release held paced goods, highest current price first (they lose least by going now)
            for P in sorted((x for x in PACED if x not in follow), key=lambda x: -prices[x]):
                extra = min(stock.get(P, 0) - want.get(P, 0), held_total() - cap)
                if extra > 0:
                    want[P] = want.get(P, 0) + extra
                if held_total() <= cap:
                    break
        need = ctx.get("cash_need")
        if need is None:
            need = 0.0
            hires = int(_g(farm, "hires_today", 0) or 0)
            for o in others:
                need += order_cost(o, obs, hires)
                if isinstance(o, list) and o and o[0] == "HIRE":
                    hires += 1
        money = float(_g(farm, "money", 0.0) or 0.0)
        gain = sum(sum(M.sell_quotes(P, inv[P], k)[0]) for P, k in want.items() if k > 0)
        gain += sum(sum(M.sell_quotes(P, inv[P], min(stock.get(P, 0), tape.get(P, 0)))[0]) for P in follow)
        short = need + float(self.p["cash_floor"]) - money - gain
        if short > 0 and need > 0:
            for P in sorted((x for x in PACED + RACE if x not in follow), key=lambda x: -prices[x]):
                while short > 0 and stock.get(P, 0) - want.get(P, 0) > 0:
                    k0 = want.get(P, 0)
                    short -= M.price(P, inv[P] + k0)
                    want[P] = k0 + 1
                if short <= 0:
                    break

        # ---- our SELLs: most exposed revenue first (lockstep slot race)
        front = []
        for P, k in want.items():
            k = min(int(k), stock.get(P, 0))
            if k > 0 and P not in follow:
                front.append(["SELL", P, k])
        if self.p["order"] == "exposure":
            front.sort(key=lambda o: -exposure(obs, o[1], o[2]))
        else:
            front.sort(key=lambda o: -(prices[o[1]] * o[2]))
        self.info["plan"] = plan_info
        return front, follow
