# Planning forensics: how the top teams schedule, and what we lack

Written Sep 28 2026, ~02:30 PKT, for the operator's question: "why can we not look ahead / schedule while our opponents
have done it successfully?"

Method: exact C++ replays (`kagsim_trace.Game(seed, 720, True)`, receipts and unit receipts) of the leaders' recorded
games plus our own live games, on the Mac, 1 core at nice 10. About 1,100 cheap replays in total (0.04-0.1 s each).
No training and no network use. Scripts are in the session scratchpad (list in the appendix).

Data:
- Leaders: the 50 newest games per team from `data/tapes_top12` (Sep 23-25), for DSM, DECEM, Unknown Mother-Goose (UMG),
  Vadim Vasilenko, M & M & P & Q (MMPQ), Boey, Fourth Quadrant (FQ) and Majkel1337. The determinism test used 40
  game pairs per team.
- Specialists: 60 games each of SpaTaro and Planned Economy from `data/tapes_pull`.
- Us: all 97 live games of NEURAL_20MW5_SE (68-29) in `data/tapes_n20mw5`, both seats. The 20MW8_SE tapes are on
  the server only. The 20MW8_SE package has the same decoder and rule layer (checked in `package/NEURAL_20MW8_SE.tar.gz`
  main.py), so the structural findings about our agent apply to it too.

---

## 0. Bottom line

1. **The leaders do not look ahead at run time. Their look-ahead is compiled in.**
   - DSM, DECEM and Vadim are deterministic. In 0 of 40 game pairs did they act differently on an identical observation
     history.
   - They play a fixed calendar:
     - DSM's farm is identical in 98% of games at the start of day 6.
     - Land is bought on day 6 at hour 5 in 50 of 50 games, then on days 9 and 10.
     - Melons: 6 tiles on day 0 and 4 on day 1, always on the same 10 tiles.
     - Hires follow a fixed daily schedule (SD 0.0-0.5 per day).
   - After day 6 their decisions are lookup tables keyed on the shops they can see: sheep numbers are explained by
     the yarn-store count with R^2 = 0.96, strawberries by berry-shop demand with R^2 = 0.94.
   - The opponent's visible farm adds at most 0.10 R^2 to any decision except geese.
   - They show no speculative holding: the chance of selling does not rise with the price.
   - The "planning" is steady-state arithmetic on a deterministic model, done offline by the designers: herd size =
     shop drain / output per animal, calendars fitted to growth times, culls when the product price floors.
   - Boey is the exception: its policy is stochastic (14 of 40 pairs act differently on identical observations) and it
     emits a fixed 10-slot market list with NOOP tokens. That is a learned RL policy, so its look-ahead lives in the
     trained weights.
2. **We already copy most of the macro schedule.** NEURAL_20MW5_SE plants the melons on DSM's exact 10 tiles in 97
   of 97 games, buys land on days 6/9/10 in 94% of games, follows the same hire calendar and uses the same shop-keyed
   herd table. The macro plan is not what we lack.
3. **We lack the exact details that imitation does not learn from averages.** These cost measurable money:
   - **SELL slot order (a probable bug).** Our decoder's "canonical" order puts FERTILIZER first and MELON last. That
     is the reverse of DSM, which puts MELON first in 100% of turns and FERTILIZER last.
     - In same-turn sale races the rival's slot comes first 21 times per game and ours 5 times, a net -$422 per game.
     - A pure permutation of our own SELL orders would give **+$559 margin per game** (se 51; 91% of games
       improved). This is measured with an exact lockstep market port validated on 18,566 turns with 0 mismatches.
   - **No-escape invariant.** We lose 1.9 animals per game while their product is still valuable; DSM loses 0.4.
     - On the day an animal escapes, the shed had wheat for 17 of 24 hours, and we even sold 13 wheat that day.
     - This is a unit-routing failure, not a feed shortage.
   - **Melon race.** MMPQ, FQ and Boey harvest at hours 3-4 and sell at hours 6-7 on day 10; we and DSM sell at
     hour 9.
     - In the same games, our melon price is $12.9 per unit below the rival's (se 2.2), -$768 per game.
     - Selling our day-10/11 lots 3 turns earlier would give +$786 margin per game (exact melon-market model).
   - **Late demand.** Worlds with 2 or more yarn stores and at least one revealed on day 12 or later are 24% of our
     games, and we win only 61% of them.
     - Our two biggest losses (-32.2k vs SpaTaro, -20.5k vs Planned Economy) came after yarn reveals on days 12/15 and
       18/21. We bought 0 extra sheep; the opponents bought 13-16.
   - **Horizon.** 3.6 wheat plantings per game from day 27 on are never harvested (DSM 0.7).
4. **The answer to "why we cannot look ahead":**
   - A neural imitation policy reproduces what the leaders usually do. It does not reproduce their hard rules and
     invariants (slot order, "never two unfed days", harvest-rush timing, payback checks), and it has no model to
     check a plan against.
   - Run-time search is still not deployable for us (see `LOOKAHEAD_REVIEW.md`), and the leaders do not use it either.
   - The leaders' kind of look-ahead is cheap and exact on this game's deterministic mechanics. It fits in our
     existing rule layer (`il/features.py`, like `midnight_fix`, which gave +$2.2k).
   - Recommended order (section 6): R1 slot order (hours of work), then R2 feed guard, then R3 melon rush (about a day
     each), then R4 late-yarn herd floor (if time allows).

---

## 1. Q1: fixed schedules or adaptation?

### 1.1 Determinism and identity

| Team | Games | Distinct step-0 actions (modal share) | Pairs acting differently on an IDENTICAL observation history (of 40) | Median first action divergence (step) | Farm = modal farm at start of day 6 | Same at day 7 |
|---|---|---|---|---|---|---|
| DSM | 137 | 1 (137/137) | **0** | 22 | **0.98** | 0.10 |
| DECEM | 274 | 1 (274/274) | **0** | 25 | 0.96 | 0.06 |
| Unknown Mother-Goose | 276 | 1 (276/276) | 1 | 22 | 0.76 | 0.08 |
| Vadim Vasilenko | 226 | 1 (226/226) | **0** | 21 | 0.82 | 0.08 |
| M & M & P & Q | 269 | 15 (250/269) | **12** | 1 | 0.42 | 0.04 |
| Boey | 170 | 5 (162/170) | **14** | 1 | 0.20 | 0.02 |
| Fourth Quadrant | 212 | 4 (seat-dependent: 108 / 101) | 1 | 29 | 0.18 | 0.04 |
| Majkel1337 | 152 | 2 (106 / 46: two versions) | 2 | 16 | 0.24 | 0.06 |
| US (20MW5_SE) | 97 | copies DSM's step 0 | deterministic (argmax, see note) | - | 0.79 | 0.05 |

- **A shared family.** DSM, DECEM, UMG, Vadim, mtmr_s1, Smackaveli, Densike, tetsuya & yuanzhe & guoqin, Azat (71%),
  HowardLeeTW (73%), seek inspiration (92%) and Orbital Terraformer all open with the identical step 0
  `[BUY_ANIMAL COW 1, BUY_PRODUCT WHEAT 5]`. That is 30% of all seats in tapes_top12, and 100% for those four
  leaders. They share one scripted ancestor.
- **What makes deterministic agents diverge.** When the DSM family diverges, it is because its own money differs:
  prices differ because the opponents traded differently at step 0. Example: `BUY_SEED WHEAT 7` vs `5` at step 13.
  - Their quantities are budget rules on the current cash, not a plan.
  - Fourth Quadrant diverges on new shops (16 of 40 pairs), the opponent's farm (7), own money (7) and prices (5).
    It is the only leader with any sign of opponent-aware decisions (section 2.6).
- **Note on our agent.** `np_runtime.NumpyAgent` is deterministic: argmax for unit orders, and a market head fires
  when P(non-zero) > tau = 0.5. Its lower "crispness" below is function-approximation noise, not sampling.

### 1.2 Key decisions across games (mean ± SD per game, own seat)

| Team | Win | Land days (share of modal) | Melon tiles | Cows | Sheep | Geese | Hires/day d12-27 | Fertilize applied | Idle tiles d10-27 |
|---|---|---|---|---|---|---|---|---|---|
| DSM | 84% | 6/9/10 (100%) | **10.0±0.0** | 9.5±2.9 | 6.8±4.4 | 7.3±3.3 | **11.8±0.2** | 243±14 | 6.9 |
| DECEM | 76% | 6/9/10 (98%) | 10.0±0.1 | 9.4±3.4 | 6.3±4.7 | 6.2±3.0 | 11.6±0.3 | 235±20 | 7.7 |
| UMG | 72% | 6/9/10 (96%) | 13.8±2.2 | 8.3±2.5 | 6.3±4.5 | 6.8±2.6 | 11.8±0.2 | 208±21 | 6.8 |
| Vadim | 80% | 6/9/10 (98%) | 9.9±0.5 | 9.2±2.7 | 6.2±4.4 | 8.2±3.2 | 11.6±0.3 | 182±21 | 7.3 |
| MMPQ | 78% | **6/8 (76%)** | 13.3±2.5 | 7.7±2.4 | 6.8±4.9 | 8.7±3.4 | 11.3±0.4 | 230±14 | **1.1** |
| Boey | 70% | **6/8 (72%)** | 12.5±1.5 | 7.4±3.1 | 6.8±4.2 | 6.6±2.6 | 11.1±0.3 | 203±14 | **1.9** |
| FQ | 62% | 6/8/10 (98%) | 11.6±1.3 | 10.4±2.7 | 7.2±4.5 | 0 | 11.3±1.3 | 132±37 | 10.0 |
| Majkel | 60% | 6/9 (68%) | 13.0±0.9 | 8.0±2.8 | 7.0±4.2 | 4.0±2.8 | 11.1±0.4 | 202±16 | 5.1 |
| **US** | 70% | 6/9/10 (94%) | 10.3±0.7 | 9.3±2.8 | 6.8±4.8 | 6.0±3.3 | 11.6±0.4 | 216±29 | **9.7** |

What is fixed and what varies:
- Fixed (SD near 0): land days, the first melon wave, the hire calendar and the total herd. The total herd is DSM
  22.9±1.9 animals by day 12, while the individual species vary by ±2.8-4.5.
- Variable: the species mix and the crop mix. Section 1.3 shows these follow the visible shops.

### 1.3 What drives the variable decisions: shops, almost never the opponent

Table: R^2 of each decision on the demand of the shops known at decision time. In brackets, the extra R^2 from
adding the opponent's visible count of the same animal or crop.

| Team | Sheep by day 12 | Cows by day 12 | Geese by day 12 | Carrot plantings by day 20 | Tomato by day 20 | Strawberry by day 14 |
|---|---|---|---|---|---|---|
| DSM | **0.96** (+0.02) | 0.71 (+0.09) | 0.22 (+0.47*) | 0.83 (+0.02) | 0.71 (+0.00) | **0.94** (+0.00) |
| DECEM | 0.93 (+0.03) | 0.81 (+0.07) | 0.26 (+0.05) | 0.68 (+0.02) | 0.74 (+0.00) | 0.95 (+0.00) |
| UMG | 0.93 (+0.05) | 0.61 (+0.10) | 0.25 (+0.17) | 0.55 (+0.07) | 0.67 (+0.01) | 0.88 (+0.00) |
| Vadim | 0.89 (+0.00) | 0.58 (+0.10) | 0.19 (+0.05) | 0.77 (+0.01) | 0.63 (+0.00) | 0.91 (+0.01) |
| MMPQ | 0.95 (+0.02) | 0.80 (+0.03) | 0.38 (+0.10) | 0.51 (+0.21) | 0.01 | 0.74 (+0.00) |
| Boey | 0.90 (+0.06) | 0.80 (+0.10) | 0.21 (+0.38*) | 0.62 (+0.00) | none planted | 0.88 (+0.00) |
| FQ | 0.91 (+0.05) | 0.71 (+0.15) | none bought | 0.83 (+0.00) | none planted | 0.88 (+0.03) |
| Majkel | 0.88 (+0.08) | 0.71 (+0.09) | 0.36 (+0.17) | 0.61 (+0.01) | 0.44 (+0.01) | 0.89 (+0.00) |
| **US** | 0.80 (+0.05) | 0.56 (+0.05) | **0.04** (+0.21) | 0.57 (+0.07) | 0.45 (+0.00) | 0.82 (+0.01) |

\* Geese are not an opponent response. They fill the fixed number of animal slots left after sheep and cows.
- For DSM: geese ≈ 18.4 + 0.52·egg shops − 0.62·sheep − 0.81·cows, with R^2 = 0.90 (Boey 0.83, us 0.31).
- The opponent's geese count only looks predictive because it is a proxy for the same shops.

DSM's herd purchases by day. Each cell is the mean number bought that day, split by how many relevant shops were known
at the start of that day (0 / 1 / 2+).

| Day | DSM cows | US cows | DSM sheep | US sheep | DSM geese | US geese |
|---|---|---|---|---|---|---|
| 0 | 2.0 (always) | 2.0 | 3.0 (always) | 3.0 | - | - |
| 6 (after land 1) | 2.9 / 5.0 / 6.2 | 2.7 / 5.0 / 6.4 | 0 / 3.9 / 6.0 | 0 / 4.7 / 6.0 | 1.0 / 2.7 / 4.0 | 0.8 / 2.3 / 4.0 |
| 8 | 0 / 0.5 / 1.9 | 0 / 0.4 / 0.9 | 0 / 2.8 / 3.0 | 0 / 2.2 / 3.0 | - | - |
| 9 (land 2) | 1.0 / 1.4 / 2.2 | 1.2 / 1.5 / 2.2 | 0 / 1.2 / 2.5 | 0 / 0.7 / 0.7 | 1.6 / 2.5 / 3.0 | 1.7 / 2.5 / 2.8 |
| 10 (land 3) | ≈0 | ≈0 | 0 / 1.5 / 3.5 | 0 / 1.2 / 3.3 | 1.7 / 2.4 / 3.8 | 1.3 / 1.3 / 1.5 |

Totals for DSM:
- Sheep 3.3 / 9.5 / 17.5 with 0 / 1 / 2+ yarn stores known at day 9. This is roughly 3 + 7 per yarn store. Steady
  state would be about 9 sheep per yarn store: one store drains 12 wool a day and a cared sheep makes 1.33 a day.
- Cows 5.7 / 8.7 / 12.1, roughly 5.7 + 3.3 per milk shop.

MMPQ and Boey react earlier. When the first shop (end of day 2) is a yarn store or a milk shop, they already buy
sheep and cows on days 3-4. For MMPQ with a milk shop: cows 1.3 on day 3 and 1.4 on day 4.

**Answer to Q1.** The leaders follow a **fixed calendar plus shop-keyed tables**:
- The calendar covers days 0-6, the land days, the first melon wave and the hires.
- The tables cover the herd species, berries, carrots and tomatoes.
- Quantities are sized to the current cash.
- Adaptation to the opponent is essentially absent. Adaptation to prices is limited to culls and cash rules.

We run the same tables, but less crisply: R^2 0.80 / 0.56 / 0.04 against 0.96 / 0.71 / 0.22 for DSM. Within the same
shop condition our sheep spread is ±2.2 against DSM's ±1.1, and our total herd ±3.6 against ±1.9.

---

## 2. Q2: evidence of look-ahead, and how we compare

### 2.1 Selling: micro-timing, not speculation

Milk, wool and strawberry sales:

| Team | Milk sold at t%4==1 (right after the shop drain) | Milk units per order | Milk hold (stock-turns per unit) | Wool at t%4==1 | Wool hold | Strawberry at t%4==1 |
|---|---|---|---|---|---|---|
| DSM | 71% | 3.3 | 7.4 | 58% | 9.3 | 58% |
| DECEM | 73% | 2.8 | 7.2 | 54% | 7.5 | 53% |
| MMPQ | 58% | 4.1 | 6.6 | 38% | 12.5 | 58% |
| Boey | 54% | 3.6 | **0.5** | 67% | 13.4 | 58% |
| FQ | 57% | 4.5 | **0.8** | 53% | 0.8 | 49% |
| **US** | 68% | 2.7 | 6.7 | 50% | 8.8 | 52% |

- Shops drain at t%4==0, after that turn's market phase. Everyone, including us, sells mostly on the next turn. This is
  an exact, known mechanic.
- **No team holds stock for a better price.** DSM's probability of selling milk when it has stock, by price relative
  to base: 0.16 (below 0.5×), 0.26, 0.21, 0.16, and 0.10 (above 1.2×). There is no reservation price. Boey and FQ
  sell on arrival.
- The DSM family and we drip-sell 2-3 units per order.

### 2.2 Slot order inside a turn: the leaders win it, we lose it (new finding)

The market is a lockstep: slot i of player A is processed with slot i of player B, unit by unit. The one who places a
product's SELL in the earlier slot sells before the other's lot lands.

| | DSM | DECEM | Vadim | UMG | MMPQ | FQ | **US** | Our live opponents |
|---|---|---|---|---|---|---|---|---|
| MELON placed before the other SELLs (pairwise share) | **1.00** | 0.99 | 0.88 | 0.90 | 0.56 | 0.93 | **0.00** | 0.72 |
| FERTILIZER placed before the other SELLs | 0.03 | 0.12 | 0.03 | 0.03 | 0.20 | 0.32 | **0.99** | 0.30 |
| Mean slot of the first MELON SELL | 0.00 | 0.00 | 0.54 | - | 0.49 | 0.34 | **5.03** | 0.86 |
| Mean slot of the first MILK SELL | 0.34 | 0.44 | 0.49 | - | 0.28 | 0.36 | 0.98 | 0.59 |

Measured order of SELLs within a turn (pairwise share of being placed first):
- DSM: MELON (1.00) > TOMATO > CARROT > MILK > STRAWBERRY > EGG > WOOL > WHEAT > FERTILIZER (0.03).
- US: FERTILIZER (0.99) > STRAWBERRY > WHEAT > MILK > WOOL > EGG > CARROT > TOMATO > MELON (0.00).

Our order is `decode_market`'s "canonical" list in `il/features.py` line 254 (`SELL_ORDER = "canonical"`, line 247):
FERTILIZER, WHEAT, STRAWBERRY, MILK, WOOL, EGG, CARROT, TOMATO, MELON. It is live in both active packages. The comment
calls it "leaders' fixed order", but it is the leaders' order reversed.

It also carries standing "phantom" orders, `SELL <product> 3` for products we do not hold. These take up slots and
push our real sells later.

Same-turn, same-product sale races in our 97 live games (per game):

| Product | Races per game | Our slot earlier | Their slot earlier | $ we lost when theirs was earlier | $ we gained when ours was earlier |
|---|---|---|---|---|---|
| MILK | 17.3 | 1.71 | **9.24** | 246 | 56 |
| STRAWBERRY | 17.3 | 4.14 | 3.36 | 158 | 68 |
| WOOL | 14.5 | 1.56 | **6.68** | 151 | 27 |
| MELON | 3.7 | 0.07 | 1.02 | 56 | 7 |
| CARROT | 4.7 | 0.65 | 2.99 | 32 | 8 |
| All products | | | | **678** | **257** → net **−$422 per game** |
| For comparison | | | | DSM net +$208 · FQ +$158 · MMPQ +$152 | |

**Counterfactual.**
- Method: the exact pre-market state of every turn is rebuilt from the trace, and the literal lockstep port
  `kagjax/market_ref.process_market` is run on it.
- Validation: the port equals the engine's `after_market` on 18,566 turns, with 0 mismatches.
- Only our SELL orders are permuted. Same orders and same quantities; the opponent is unchanged; no compounding.

| Our SELL order | Own $ per game | Opponent $ per game | Margin per game | Games improved |
|---|---|---|---|---|
| `base` (existing `SELL_ORDER="base"`: by base price) | +295 | −240 | **+535** (se 51) | 91% |
| DSM's revealed order | +289 | −244 | +534 (se 51) | 90% |
| premium first: MELON, WOOL, MILK, STRAWBERRY, TOMATO, CARROT, EGG, WHEAT, FERTILIZER | +307 | −252 | **+559** (se 51) | 91% |
| premium first, but products we hold in the shed first | +308 | −252 | +560 | 91% |
| same, also counting this turn's own drops (upper bound) | +333 | −271 | +604 | 92% |

A warning about the easy shortcut: an open-loop replay of our tape with the reordered sells gave −$12.6k per game.
That is the known tape-desync artefact. A few dollars of difference make a later purchase fail and the fixed tape stops
matching the state. Only the per-turn port numbers are meaningful.

### 2.3 The melon race on day 10

| Team | First melon harvest (d10) | First melon drop = first sale | Wave-1 price | Paired price gap vs the rival, same game, d10-13 |
|---|---|---|---|---|
| DSM | h5 | h9 | $221 | +2.1 (se 2.5) |
| Vadim | h5 | h9 | $207 | −13.3 (se 2.9) |
| MMPQ | **h3** | **h6** | $232 | **+8.2** (se 1.4) |
| Boey | h4 | h7 | $237 | −5.4 (se 1.4) |
| FQ | h4 | h7 | $241 | **+22.0** (se 1.6) → +$1.2k per game |
| **US** | h5 | h9 | $211 | **−12.9** (se 2.2) → **−$768 per game** |

- Everyone sells in the same turn as the drop (0 lag). The difference is the route.
  - The DSM family and we use the identical 10-tile layout: mean distance 3.60 to a shed-access tile, and the first
    melons are harvested at hour 5.
  - FQ's layout averages 3.23 and MMPQ's 3.47, and both harvest at hours 3-4.
- Who sold first, in our 94 games with melon sales on both sides:

| Case | Games | Our paired price gap per unit | Note |
|---|---|---|---|
| We sold first | 17 | +$11.1 (se 4.1) | |
| Same first turn | 41 | −$17.4 (se 2.6) | Slot order plus the timing of later lots |
| Opponent sold first | 36 | −$19.0 (se 3.3) | Their first sale at hour 6 |

- **Melon-market counterfactual** (`melonshift.py`).
  - The melon price depends only on melon inventory, so this can be simulated exactly. The model reproduces the
    recorded melon revenue of both seats in 97 of 97 games.
  - Moving our day-10/11 melon lots earlier and into slot 0:

| Shift | Margin per game |
|---|---|
| Slot 0 only | +$57 |
| 2 turns earlier | +$613 (se 38) |
| 3 turns earlier | **+$786** (se 46) |
| 4 turns earlier | +$933 (se 51) |

### 2.4 Planting, harvesting and the horizon

- **Replanting.** After a one-time crop is harvested (days 8-26), every team replants the tile within 1 turn in
  83-90% of cases (us 90%). No difference.
- **Idle land.**
  - The two-field teams (MMPQ, Boey) keep only 1.1-1.9 unlocked tiles empty or weeded per day, against DSM's 6.9 and
    our 9.7 with three fields.
  - Their "no idle tiles" comes from having fewer fields and filling them. It is not rotation planning.
- **End of season.** One-time-crop plantings that are never harvested, per game:

| | DSM | DECEM | Vadim | MMPQ | Boey | **US** |
|---|---|---|---|---|---|---|
| All | 4.0 | 3.2 | 3.6 | 2.8 | 1.6 | **10.2** |
| of which wheat planted on day 27 or later | 0.7 | 0.3 | 0.4 | 0.5 | - | **3.6** |
| of which carrots planted on day 27 or later | 0.2 | 0.1 | 0.1 | 1.1 | - | 1.3 |

- **Unsold stock at the end:** us $1,568, DSM $1,172, DECEM $598, Vadim $0.

### 2.5 Labour and land: calendars with cash triggers

- **Hires per day** (mean with SD across games): DSM follows 4.0 (0.0), 6.1 (0.4), 8.0 (0.0) on day 6, 11.5 (0.7) on
  day 10, then 11.6-12.0 (SD 0.1-0.5) on days 12-26 and 11.0 on day 28.
  - This is a calendar, not matched to the day's work. We copy it exactly.
- **Land purchases** follow a fixed day plus a cash trigger within that day:

| Purchase | DSM | MMPQ |
|---|---|---|
| Land 1 | Day 6 hour 5 in 50 of 50 games, whatever the cash ($2.7-3.6k) | Day 6 hours 3-4. Median cash one turn earlier is $227: they sell a big lot in the same turn to fund it |
| Land 2 | Day 9, hours 1-7. In 70% of games cash one turn earlier was below $2,000: buys on the first turn when cash, including that turn's sales, covers the price | - |
| Land 3 | Day 10, hours 12-18. In 97% of games cash one turn earlier was below $4,000 | - |

### 2.6 Invariants and culls

Escapes per game. "Valuable" means the product still sold at 0.6× base or more (an accident). "Floored" means below
0.6× base (a cull).

| Team | Valuable, before day 15 | Valuable, days 15-27 | Floored, days 15-27 (culls) |
|---|---|---|---|
| DSM | 0.02 | 0.36 | 5.38 |
| DECEM | 0.02 | 0.52 | 3.84 |
| UMG | 0.02 | 0.52 | 4.22 |
| Vadim | 0.02 | 1.14 | 4.64 |
| MMPQ | 0.00 | 0.68 | 3.74 |
| Boey | 0.02 | **0.08** | 2.50 |
| FQ | 0.00 | 0.50 | 4.20 |
| Majkel | 0.02 | 0.24 | 3.58 |
| **US** | **0.49** | **1.38** | 3.95 |

- **Culls are deliberate for every team, including us.** Animals are left unfed when their product has floored. The
  median price at a sheep escape is $5-11; at a cow escape $9-22 (ours $33).
- **Our early escapes are accidents** (`escape_diag.py`):
  - 49 escapes in 36 of 97 games: 25 cows, 22 geese, 2 sheep.
  - The product price at the time was about base (1.06×).
  - On the day of the escape we had 17.7 animals, and 4.3 of them were already unfed for 1 day at the day start. We
    made 15 FEED actions.
  - The shed held wheat for 17.3 of the 24 hours (up to 26.6 units), and we sold 13 wheat that day.
  - The leaders keep the invariant "never two unfed days unless culling". Our policy sees the danger (feature plane
    25 marks animals unfed today with `consecutive_unfed >= 1`) but does not reliably act on it.
- Overall feeding and care rates are similar to DSM's: cows fed 78-80% and cared 78-81% of animal-days for us, 77% and
  77% for DSM. Boey is the most diligent at 83-88%.

### 2.7 Look-ahead, where it exists

| Candidate look-ahead behaviour | Found? | Who |
|---|---|---|
| Holding stock for a price peak | No | Nobody: no reservation price |
| Post-drain timing (t%4==1) | Yes (exact mechanic) | Everyone, including us |
| Slot priority for contested premium goods | Yes | DSM family, FQ, MMPQ. **We do the reverse** |
| Harvest-rush timing against the field's fixed schedule | Yes | MMPQ, FQ, Boey (hours 6-7 vs hour 9) |
| Timing plantings to high-price days | No | First melon wave is fixed; the late wave sells at $109-143 |
| Hires matched to future work | No | Calendar |
| Land at the first affordable turn of a fixed day | Yes (cash rule) | Everyone |
| No-escape invariant | Yes | All leaders. **We fail it about 1.9 times per game** |
| Payback-aware late herd | Partly | Leaders buy about 2-6 sheep after late yarn reveals; specialists 13-16. Our response is weaker on days 15+ |
| Opponent-aware production | Weak | Only FQ: late melons correlate −0.32 with the opponent's melon tiles. DSM family ≈ 0 |

---

## 3. Q3: melons, milk, wool and fields over the 30 days

### 3.1 Melons

| Team | Planting (tiles by day) | Total | Melon revenue per game | Late melons (days 14+) |
|---|---|---|---|---|
| DSM, DECEM, Vadim | d0: 6, d1: 4. Same tiles in 50 of 50 games | **10.0** | $12.3-13.2k | none |
| UMG | d0: 6, d1: 4, then +3.3 (mostly d7: 2) | 13.8 | $15.7k | $109 per unit |
| MMPQ | d0: 6, d1: 2, d2: 1, d6: 2.7, days 12-19: about 1 | 13.3 | $15.3k | $125 |
| Boey | d0: 7, d2: 3, d6: 1.3 | 12.5 | $14.8k | $129 |
| FQ | d0: 8, d6: 2.2 | 11.6 | $14.9k | $143 |
| Majkel | d0: 5, d1: 5, d3: 1, d5: 1.2 | 13.0 | $15.3k | $132 |
| **US** | d0: 6, d1: 4 (DSM's tiles, 97 of 97 games) | 10.3 | $12.8k | - |

- Every team gets 6.0 units per tile (a 9.7-9.9-day grow). None fertilizes melons.
- The late wave fills free tiles on the new field. For MMPQ it correlates +0.59 with empty tiles on day 7 and 0 with the
  opponent's melons.
- **The best team (DSM) plants exactly 10**, and our melon gap is −$2.5k per game in our wins as well as our losses.
  Melon volume does not decide our games. The race (section 2.3) is the melon lever.
- UMG's extra +$2.45k of melon revenue over DSM came with −$2.4k on tomatoes and −$3.2k on milk.

### 3.2 Cows and milk

Schedule:
- DSM family: 2 cows on day 0; on day 6 (after land 1) +2.9 / 5.0 / 6.2 with 0 / 1 / 2+ milk shops known; day 9
  +1.0-2.2.
- Nobody adds cows after day 12: 0.0-1.0 per milk-shop reveal, us 0.0-0.4.
- MMPQ, Boey, FQ, Majkel and SpaTaro also buy about 1 cow a day on days 2-5 when a milk shop appears at day 3. Many of
  the opponents who beat us do this too (`them 0:2 2:1 3:1 4:1 5:1`).

Production per cow-day (days 6-27):

| DSM | Boey | MMPQ | Majkel | **US** |
|---|---|---|---|---|
| 1.03 | **1.18** | 1.11 | 1.12 | 1.01 |

The ceiling with daily care is 1.5 per cow-day (3 milk every 2 days).

In our milk-heavy losses (4-5 milk shops), several opponents produced 17-64% more milk with a similar cow count
(Yannik 336 vs 248, Aaweg 391 vs 294, SpaTaro 269 vs 164). Their cows came earlier (days 2-5) and yielded more per
cow-day, for example 1.25 against 1.15.

### 3.3 Sheep and wool

Schedule:
- 3 sheep on day 0 (the family; Boey and FQ 2).
- With a yarn store: +3.9 / 6.0 on day 6, +2.8 / 3.0 on day 8, +1.2-3.5 on days 9-10.
- Totals by yarn stores known at day 9: 3.3 / 9.5 / 17.5 for DSM, 3.2 / 10.2 / 18.7 for us.
- Culls when wool floors (median $5-11).

Sheep bought within 4 days of a yarn-store reveal:

| Reveal day | DSM | DECEM | UMG | MMPQ | Boey | FQ | Majkel | SpaTaro | **US** |
|---|---|---|---|---|---|---|---|---|---|
| d12 | 4.0 (n=2) | 4.3 | 4.3 | 5.5 | 4.0 | 4.5 | 4.5 | 5.8 | 4.0 (12) |
| d15 | 4.2 (4) | 3.8 | 4.0 | 2.3 | 5.7 | 3.3 | 2.7 | 3.1 | **2.2** (11) |
| d18 | 1.4 (10) | 2.2 | 0.5 | 3.1 | 1.8 | 0.8 | 2.1 | 1.9 | 1.7 (14) |
| d21-24 | 0.0 | 0.0 | 0.0 | 1.1 | 0.3 | 0.0 | 1.0 | 0.2 | 0.1 (22) |

Our results by yarn draw (97 games):

| Yarn draw | Games | Win | Margin | Wool net, us minus them |
|---|---|---|---|---|
| No yarn store | 32 | 69% | +8.1k | +0.6k |
| 1 store, revealed early | 14 | 86% | +10.6k | +1.9k |
| 1 store, revealed on day 12 or later | 26 | 69% | +9.9k | −0.3k |
| **2+ stores, one revealed on day 12 or later** | **23** | **61%** | **+3.5k** | **−1.6k** |

The worst cases, where we bought no extra sheep:

| Opponent | Margin | Yarn reveals | Our sheep | Their sheep | Wool, us vs them |
|---|---|---|---|---|---|
| SpaTaro | −32.2k | d12, d15 | 3 | 16 (bought d12-16) | $20k vs $65k |
| Planned Economy | −20.5k | d18, d21 | 3 | 13 (bought d18-23) | $15k vs $36k |
| lingxiaojun | −13.7k | d9, d24 | 4 | 9 | - |

Planned Economy is a weak bot overall (3% wins in its last 60 games). It still took $21k of wool from us with sheep
bought on days 18-23.

**Payback of a late sheep.** This follows exactly from the rules: the care bank gives the first yield = 1 + banked
fed-and-cared days, capped at 6 per unit. A sheep placed on day d then yields 6 wool on day d+6 and 4 every 3 days
after that.

| Bought on day | Wool by the end | Break-even wool price (cost $500 + about $30 per day of feed) |
|---|---|---|
| 12 | 18 | $56 |
| 15 | 14 | $66 |
| 18 | 10 | $83 |
| 20 | 10 | $77 |
| 21 | 6 | $123 |

With 2 yarn stores (24 wool a day of drain) wool stays around $150-200 until about 19 sheep in total are on the market.

### 3.4 Fields

| Group | Fields | Details |
|---|---|---|
| DSM family | 3: days 6/9/10 (96-100%) | Hours: land 1 fixed at 5; land 2 at 1-7; land 3 at 12-18, on the first affordable turn |
| MMPQ | 2: days 6/8 (76%) | A 3rd only in 24% of games, on day 13, with $9-27k of cash |
| Boey | 2: days 6/8 (72%) | |
| SpaTaro | 2: days 6/8 (52%) | |
| FQ | 3: days 6/8/10 (98%) | |
| Majkel | 2: days 6/9 (68%) | |

- Two-field players keep 1-2 idle tiles; we keep 9.7.
- In our 29 losses the opponent had 6/8 in 12 games and 6/9/10 in 9. We out-spend them on land by $2.8k in losses.
- The SE guard (the 3rd field) was measured positive for our agent (+$288-696 per game on the 464-world test, memory
  `next-iteration-plan-sep26`). Keep it. The idle-tile gap is an efficiency issue, not a land-count issue.

### 3.5 Where our losses come from (20MW5_SE, 29 losses vs 68 wins)

Net cash per product per game, us minus them:

| Product | Losses | Wins |
|---|---|---|
| WOOL | **−5,136** | +3,038 |
| MILK | **−3,246** | +86 |
| MELON | −2,573 | −2,490 (same, so not decisive) |
| EGG | −1,376 | +4,473 |
| FERTILIZER | −1,156 | +765 |
| TOMATO | +4,413 | +6,867 |
| STRAWBERRY | +3,123 | +3,980 |
| CARROT | +1,532 | +4,079 |
| Land | −2,759 | −3,147 |
| **Margin** | **−8,003** | **+15,593** |

**Answer to Q3.** Our losses are decided by the animal economy (wool, milk, eggs) and by how we respond to late demand.
Melon volume is not decisive; the melon race timing is worth taking.

---

## 4. Q4: what algorithm produces their play?

| Team | Most plausible algorithm | Evidence |
|---|---|---|
| DSM, DECEM, Vadim (UMG, mtmr_s1 and about 8 others) | **Hand-written rule-based script**, one shared ancestor. Tuned offline | Deterministic (0 of 40). Same step 0. Fixed 6-day calendar (farm identical 96-98% at day 6), fixed land day and hour, fixed melon tiles, fixed hire calendar. Shop-keyed herd and crop tables (R^2 0.9). Cash-sized quantities. Price-floor culls. Ordered market lists (MELON first, FERTILIZER last, `[]` dead-slot pads in 27% of DSM's orders) |
| Boey | **Learned policy (RL)**, sampled at inference | 14 of 40 pairs act differently on identical observations at step 1. 68% of market turns have exactly 10 orders, 17% of all orders are NOOP (a fixed-slot decoder). Wash-trades about 3,700 wheat per game. Known JAX PPO self-play author (memory `forum-rl-intel`) |
| Fourth Quadrant | **Deterministic script** with the most state reactivity | 1 of 40 stochastic. Seat-dependent opening. Wash trading (52% of orders are BUY_PRODUCT; every SELL is 10 units or more). Divergences follow shops and the opponent's farm. Late melons −0.32 vs the opponent's melons |
| M & M & P & Q | **Script with a randomized unit-assignment step** | Step 0 like the DSM family plus 3 sheep. Differences on identical observations are in which unit does which task (12 of 40). Two fields. Faster melon layout |
| Majkel1337 | Two script versions | 2 distinct openings, 2 of 40 stochastic |

- **No top team shows on-line search or simulation.** Their production decisions do not respond to the opponent, they
  never hold for price, and their calendars are rigid.
  - The only overage-time data on the Mac is weak: one older game each for Majkel (39.4 s left) and Vadim (45.6 s) in
    `data/recon/tapes`.
- **Why cheap planning is possible in this game.** Almost everything a plan needs is deterministic and visible:

| Mechanic | Predictability | How the leaders use it |
|---|---|---|
| Crop growth and yield | Exact. Melon planted on day 0 → harvestable on day 10, 6 units with daily water | Calendars |
| Animal output | Exact given feed and care: sheep 6 then 4 per 3 days; cow 6 then 3 per 2 days; goose 4 then 2 a day | Herd sizing, payback |
| Town drain | Exact once shops are known: 6 a day per shop and product (12 for single-product shops), town centre 1 a day, at t%4==0 after the market | Herd = drain / output; sell at t%4==1 |
| Price | Exact function of public inventory (our port: 18,566 turns, 0 mismatches) | Lot sizing, race value |
| Lockstep slots | Exact | Premium goods in slot 0 |
| Escapes | Exact: 2 unfed days | "Never two unfed days" |
| Opponent supply | Visible farm (tiles, yields, animals); only the shed (≤100) is hidden | Race timing (FQ, MMPQ) |
| New shops | **Unpredictable**: every 3 days (end of days 2, 5, ..., 23), hidden seed, RNG shared with weeds | React, never bet |

---

## 5. Why our agent cannot "look ahead" like them

- Imitation learns the conditional average of what the leaders do. It learns the calendar and the shop tables well:
  same layout, land days and herd table. It does not learn:
  1. **Exact orderings that are artefacts of the decoder.** The SELL slot order is fixed by our decoder, not the
     network, and our fixed list is the reverse of the leaders'.
  2. **Hard invariants**, such as never two unfed days. They are rarely violated in the data, so the loss barely sees
     them.
  3. **Race timings that depend on the opponent population.** DSM itself sells melons at hour 9.
  4. **Thresholds in rare situations**, such as late yarn stores or the planting horizon.
- The value head and the RL critic cannot price these either (`LOOKAHEAD_REVIEW.md`: $1 of cash counts as $0.4-0.6).
  Search at run time would inherit that error, and it costs 66-189 ms per 20M forward pass.
- The leaders get these right because a human (or billions of RL steps, for Boey) encoded them as rules. We can do the
  same with the rule layer. It is cheap, exact and testable, and it is what worked before (midnight +$2.2k, final turn
  +$195, homing +$475, SE guard +$288-696).

---

## 6. Q5: recommendations before Sep 30 (bolt-on rule layer, in order)

Each rule goes in `il/features.py`, next to `midnight_fix` or after it. It needs a flag in `live_rollout.py` and a
default in `export.py`. Test each one alone, paired, and keep it only if it passes (memory `judge-top-band`).

| # | Rule | Evidence (this report) | Expected effect | Work / risk | Test and what to check |
|---|---|---|---|---|---|
| **R1** | **SELL slot order.** Put SELL orders for products we hold first, premium first (MELON, WOOL, MILK, STRAWBERRY, TOMATO, CARROT, EGG, WHEAT, FERTILIZER), as the last step after `midnight_fix` / `_final_fix`. Permute SELLs only among SELL slots; never drop a HIRE. One-line variant: make the existing `SELL_ORDER="base"` the default in `features.py` and the export (+$535) | §2.2: our order is the reverse of DSM's; −$422 per game net in same-turn races | **+$300 own, −$250 opponent, +$0.53-0.60k margin per game** (pure per-turn port, se 51, 91% of games better) | Hours. No unit changes. Cash is unchanged because SELLs stay before buys | 464 worlds paired (and the 2400+ subset), leader tests fresh and fresh2, the live-loss replay set. Check: slot-race net goes from −$422 to ≥0 per game (`raceslot2.py`) |
| **R2** | **Feed guard** (no-escape invariant). From about hour 16, every own animal with `consecutive_unfed >= 1` and not fed today gets a FEED: send the nearest unit, `PICKUP WHEAT` at the shed if needed. If the shed has no wheat, add `BUY_PRODUCT WHEAT` for the shortfall. Skip deliberate culls (product price < about 0.3× base) | §2.6: 1.87 accidental escapes per game vs DSM 0.38; the shed had wheat 17 of 24 h; we sold 13 wheat that day | **+$0.5-1.1k per game** (about 1.5 animals: roughly $1.3k per early loss, $0.6k per mid-game loss; upper bound $1.4k) | About a day. Unit override in the style of `_homing` | Same rulers. Accidental escapes should fall to ≤0.4 per game; watering and harvest counts must not drop |
| **R3** | **Melon rush, days 10-11.** From hour 0-1, send the farmer plus 3 hands to the ripe melons nearest the shed (the DSM layout is 2-5 steps away), HARVEST, walk back, and DROP or PLACE by hours 6-7. The SELL goes in slot 0 in the same turn. Hire any extra hands at hour 0 | §2.3: our melons sell at h9, MMPQ at h6, FQ and Boey at h7; paired gap −$12.9 per unit | **+$0.6-0.8k margin per game** (exact melon model, se 38-46, 2-3 turns earlier) | About a day. Unit override on 2 mornings; the policy resumes afterwards | Same rulers. First melon sale by hour 7; the day-10/11 feed and water counts must hold |
| **R4** | **Late-demand herd floor (sheep).** Days 9-20: if yarn stores known ≥1 and own sheep < 3 + 7 × yarn stores (DSM's table), with pasture tiles and cash (reserve ≥$300) available, and the payback check holds (expected wool × 0.5 × current wool price > $500 + $30 × days left; see §3.3), buy the shortfall, at most 4 a day. Also stop if our visible sheep plus the opponent's exceed 0.75 × (12 × yarn stores + 1) | §3.3: late multi-yarn worlds are 24% of games with 61% wins; the −32k and −20.5k losses had 0 extra sheep | Pooled **+$0.2-0.5k per game** (uncertain); several $k in the target worlds. Risk: the wool floors if both sides flood | About a day. Market plus pasture building (the policy must PLACE the sheep: check it) | Live-loss replays first (SpaTaro 113xxxx −32k, Planned Economy −20.5k, lingxiaojun, elmo, Breaking1800), then 464 and the leader tests. Watch the wool floor rate and sheep escapes |
| R5 | **Planting horizon.** No PLANT or BUY_SEED of WHEAT or CARROT once the crop cannot be harvested before step 718 (day ≥27 late or day ≥28) | §2.4: 3.6 + 1.3 never-harvested plantings vs DSM 0.7 + 0.2 | +$50-150 per game | Trivial | 464 only (free) |

Not recommended now:
- **A second melon wave.** DSM, the best team, never replants. Our melon gap is identical in wins and losses. Late
  melons sell at $109-143.
- **Dropping the 3rd field.** The SE guard was measured positive for us.
- **Early cows on days 3-4** as MMPQ and Boey do. It competes with the day-6 land cash that our calendar depends on.
  Untested and risky two days before the deadline.
- **Run-time look-ahead search** (`LOOKAHEAD_REVIEW.md`). The leaders do not use it, and we cannot afford it.

Combined static estimate for R1 and R3 on the 97 live games: +$1.35k margin per game.
- Static flips: only 2 of our 29 losses, because our losses are mostly −$3k to −32k.
- These rules are rating insurance for close top-band games, and they cost nothing in games we win.
- The large losses are addressed by R4 (wool) and by the specialist work already under way.

Suggested order for Sep 28-29:
1. R1: implement and run the 464 and leader tests.
2. R2 and R3 in parallel.
3. R4 only if R1-R3 are done by the evening of Sep 28.
4. Package whatever passes, run parity 12/12 and the official check, and bring the upload decision to the operator.
   An upload retires the older active submission (memory `history-lessons`).

---

## 7. Caveats

- The leader data is Sep 23-25 (tapes_top12). Teams may have updated since. The DSM family's structure is stable
  across 4 teams and 900+ games.
- Our agent's numbers are from 20MW5_SE's 97 live games. 20MW8_SE shares the decoder and rules (checked), but its live
  tapes are on the server.
- R1's +$0.56k is a per-turn, no-compounding estimate. R3's +$0.79k assumes the route can deliver the melons 3 turns
  earlier. R2 and R4 are analytic estimates. All need the closed-loop rulers.
- An open-loop tape replay cannot measure these rules. It desyncs: it gave −$12.6k for R1, an artefact.
- Opponent adaptation: slot order within a turn cannot be observed by the opponent in the same turn. The melon race can
  be answered by opponents who also rush (FQ and MMPQ already do; the ladder's DSM-family majority does not).

---

## Appendix: scripts and reproduction

All scripts are in `/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/`.
Run them with `nice -n 10 /Users/asad.ali/kaggriculture/.venv/bin/python <script>`.

**Records**

| Script | Output |
|---|---|
| `extract.py` | Exact replay per game → `recs_top.pkl` (8 teams × 50 games) and `recs_ours.pkl` (97) |
| `extract_opp.py` | Our opponents' seats → `recs_ours_opp.pkl` |
| `extract_spec.py` | SpaTaro and Planned Economy → `recs_spec.pkl` |

**Identity and determinism**

| Script | Section |
|---|---|
| `step0.py`, `step0fam.py`, `ident.py` | Step-0 actions, family, action identity |
| `determ.py` | Observation vs action divergence |
| `anR.py` | Farm identity by day; crop calendars |

**Schedules**

| Script | Section |
|---|---|
| `anA.py` | Table 1.2 |
| `anB.py`, `anG.py`, `anJ.py` | Melons |
| `anD.py`, `anQ.py`, `anO.py`, `anN.py`, `anCr.py` | Herd tables, slots, geese |
| `anM.py` | R^2 vs shops and the opponent |
| `anV.py`, `anX.py`, `anU.py`, `anY.py` | Late yarn |
| `anW.py` | Milk-heavy worlds |
| `anP.py`, `landcash.py` | Replanting, land timing and cash |
| `anH.py` | Hires, escapes |
| `anF.py`, `anF2.py` | Horizon, end stock |
| `anK.py`, `husb.py` | Animal productivity, feeding and care |

**Markets**

| Script | Section |
|---|---|
| `anC.py` | Revenue by product |
| `anE.py` | Sale timing and holding |
| `anS.py` | Order structure |
| `sellorder.py`, `slotpos.py` | SELL order |
| `raceslot.py`, `raceslot2.py` | Slot races |
| `slotcf.py`, `slotcf2.py` | Per-turn port counterfactual, validated 18,566 / 18,566 |
| `anI.py`, `melonrace.py`, `anRace.py`, `melgap.py` | Melon race |
| `melonshift.py` | Melon-market counterfactual, 97 / 97 exact |
| `flips.py` | Static flips |

**Losses:** `anL.py`, `anT.py`, `escape_diag.py`.

**Invalid method, kept as a warning:** `reorder_eval.py` and `reorder_dbg.py` (open-loop tape desync).

Code referenced:
- `il/features.py`: `SELL_ORDER` at line 247, `decode_market` at lines 250-286, `midnight_fix` at line 418, `_homing` at
  line 296.
- `il/np_runtime.py`: `NumpyAgent` at lines 139-181 (deterministic argmax, tau is a firing threshold).
- `kagjax/market_ref.py`: `process_market`.
- `package/NEURAL_20MW8_SE.tar.gz` and `package/NEURAL_20MW5_SE.tar.gz`: both have `SELL_ORDER = "canonical"` and do
  not embed features8.
