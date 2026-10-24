# Top vs top (A): how the leaders beat each other, and what it means for us

Written Sep 28 2026, 11:00-12:00 PKT. Games: 4 raw Kaggle replays from `~/Downloads/competitoreachotherreplays/`
(read in memory only; nothing written there), all newer than every tape in `data/tapes_top12` (Sep 28 UTC).
Cross-checks: 2,733 exact replays of leader games from `data/tapes_top12` (Sep 25-27), the 3 other raw replays of the
folder (another agent's games, used only for comparison), and our 37 live games of NEURAL_20MW8NPS_SOH
(`data/tapes_nps`). Mac, 1 core, `nice -n 10`, exact C++ engine only; about 3 core-minutes in total.

| Game | Seat 0 | Seat 1 | Final | Winner, margin |
|---|---|---|---|---|
| 114449942 | DSM | Fourth Quadrant (FQ) | 100,645 vs 101,725 | FQ by 1,080 |
| 114493956 | M & M & P & Q (MMPQ) | DSM | 111,271 vs 111,432 | DSM by 161 |
| 114514401 | MMPQ | Vadim Vasilenko | 103,188 vs 105,580 | Vadim by 2,392 |
| 114514402 | Vadim | DSM | 118,530 vs 117,217 | Vadim by 1,313 |

**Replay check.** All 7 raw replays reproduce exactly in `kagsim_trace` (recorded rewards equal to the dollar;
action for engine step t = `steps[t+1][s].action`). The exact lockstep market port (`kagjax/market_ref`) equals the
engine on every race turn used here (0 mismatches), and the single-product market model (`pmkt.py`) reproduces both
seats' cash for all 9 products in all 7 games.

Labels: **[M]** = measured exactly on the replays; **[X]** = cross-checked on hundreds of tapes; **[I]** = inference.

---

## 0. Bottom line

1. **The top is now one calendar plus two outsiders.** [X] Since Sep 26-27 DSM, Vadim, DECEM and UMG play the same
   new plan (no third field, SW on day 8 hour 6, about 3.5 cows on days 1-5, a 4-tile melon wave on NE on day 6,
   3 sheep at step 0, melons sold on day 10 at hours 6/9/10/12). Our live NPS agent plays this calendar almost exactly
   (same step 0, lands on days 6/8 in 36/37 games, same melon hours). FQ (3 fields, 4 cows at step 0, no geese,
   zero-sum wheat churn) and Boey (RL) are the outsiders. MMPQ changed its build today (see 3).
2. **Mirror games are decided by table differences, not by the market.** [M] In the DSM-Vadim mirror (game 4) the two
   farms were identical on days 0-5 (cash within $1) and sold melons at the same hours. The game was decided by the crop
   choice on days 11-18: Vadim planted 7 more strawberry tiles (+43 units, +$8.1k), DSM planted 10 tomato tiles
   later (+$5.7k) and more carrots (+$2.0k). Same-turn sale races were worth $50-300 per game in all four games.
3. **MMPQ's Sep 28 build is a regression.** [M] It went back to the third field (4 of 4 games, day 10 hour 11),
   hires 12-13 hands a day (+$3.1-3.5k of hire cost per game against the family's 10-11), plants tomatoes in worlds
   without a tomato shop ($52 a unit), loses 17-19 plants to drought in 2 of 4 games and sells its first melons at
   hour 8 (its own Sep 27 build: hour 6 in 136 of 151 games). It lost all 4 of these games, and it no longer spends
   overage at hour 0 of each day (first turn only, 17-22 s).
4. **The margin effect of supply is large.** [M][X] In a shared market every unit sold lowers the opponent's later
   prices, so supply that both farms sell is worth far more in margin than in own cash:
   - The third field (SE) of FQ / MMPQ in these games: own cash +$1.2k to +$7.1k, margin +$8.5k to +$18.7k.
   - One sheep less in a flooded 1-yarn-store mirror: own cash +$5.2k, but margin -$2.5k (9 of 10 seat-games lose
     margin).
   - This is why the leaders flood wool and never hold back supply. Supply-withholding rules should stay dead.
5. **Where the family (and our clone) loses: multi-yarn worlds.** [X] With 2+ yarn stores DSM wins only 38% against
   FQ (n=8) and 29% against Boey (n=14). FQ and Boey keep 1-2 more sheep (12.9 / 12.6 against 10.5 / 11.6) and win
   the wool market (-$7.7k / -$1.9k for DSM).
6. **The melon race against the family is a tie for us, and it is locked by geometry.** [X] Being 1 step earlier on
   all melon lots would be worth +$426 per mirror game (days 10-11; 89 Sep 27 games, se $7, 100% of games) and +$660
   with the day-16 lots. But the family's routes are optimal for their tile layout (WATER then HARVEST on day 10). The
   only cheap way to save the step (fertilize the melons on days 6-8, so no WATER is needed on day 10) costs about
   $80 of early fertilizer per tile, so the net is zero or negative. Do not build it.
7. **Ranked for us (section 9):**
   1. The SE_SHOPS=2 candidate: the leader data supports it.
   2. The sheep floor in yarn worlds, with the opponent-based safety net loosened.
   3. A strawberry upgrade on days 10-14 when the strawberry price is at least $150.
   4. Defensive rules we already mostly follow.

   The market micro needs no change: SELL_ORDER=held, selling right after the drain and FINAL_FIX all match the
   leaders.

---

## 1. The four games at a glance [M]

| | FQ beat DSM | DSM beat MMPQ | Vadim beat MMPQ | Vadim beat DSM |
|---|---|---|---|---|
| Margin | 1,080 | 161 | 2,392 | 1,313 |
| First three shops (days 3/6/9) | SMOOTHIE, SMOOTHIE, PET_CAFE | SMOOTHIE, PET_CAFE, SMOOTHIE | ICE_CREAM, YARN, PET_CAFE | YARN, SMOOTHIE, PET_CAFE |
| Later shops (days 12-24) | SMOOTHIE, SMOOTHIE, PET_CAFE, PIZZA, FARMERS | BRUNCH, BRUNCH, SMOOTHIE, PIZZA, FARMERS | SMOOTHIE, PET_CAFE, BRUNCH, YARN, ICE_CREAM | FARMERS, ICE_CREAM, FARMERS, BRUNCH, SMOOTHIE |
| Lands (day, hour) | DSM 6h3 NE, 8h6 SW; FQ 6h2, 8h5, **10h11 SE** | MMPQ 6h4, 8h6, **10h11 SE**; DSM 6h3, 8h6 | MMPQ 6h4, 9h4, **10h11 SE**; Vadim 6h3, 8h6 | both 6h3, 8h6 |
| Hires per day, days 12-27 | DSM 10.6, FQ 11.8 | MMPQ 12.1, DSM 10.7 | MMPQ 12.1, Vadim 10.9 | 10.9 / 11.0 |
| Hire $ | 4,360 vs 7,154 | 7,822 vs 4,360 | 8,000 vs 4,915 | 5,004 vs 5,203 |
| Herd on day 12 (cows/sheep/geese) | DSM 10/3/5, FQ 13/4/0 | MMPQ 11/3/6, DSM 10/3/7 | MMPQ 7/10/5, Vadim 7/9/3 | 6/10/5 both |
| First melon sale on day 10 | DSM h6, FQ h10 | MMPQ h8, DSM h6 | MMPQ h8, Vadim h6 | h6 both |
| Cash at d10h0, then d11h0 | DSM 394 -> 8,134; FQ 263 -> 3,508 | MMPQ 458 -> 4,238; DSM 249 -> 7,951 | MMPQ 434 -> 3,897; Vadim 640 -> 8,607 | 2,346 -> 11,110; 1,785 -> 10,153 |

---

## 2. Why the winner won ($ decomposition) [M]

Every row is exact: 3,000 + revenue - purchases = final bank, checked to the dollar (`a_cash.py`). "Volume" and
"price" split a product's revenue gap as (units_w - units_l) × price_l + units_w × (price_w - price_l).

### 2.1 FQ beat DSM by 1,080 (114449942)

| Item (FQ minus DSM) | $ | Why |
|---|---|---|
| Strawberry | +4,679 | 332 vs 309 units, $122.6 vs $116.6; 4 SMOOTHIE shops |
| Milk | +4,287 | 353 vs 303 units: FQ bought 4 cows at step 0 (24 milk by day 8 vs 12, sold at $211) and 5 more on day 8 (13 cows vs 11) |
| Carrot | +3,732 | 301 vs 219 units: 2 PET_CAFE; FQ planted 45 carrot tiles on SE |
| Wheat, net of wheat bought | +2,645 | $6,981 vs $4,336 (FQ churns 2,003 bought / 2,328 sold, see 4.5) |
| Fertilizer, net of fertilizer bought | +981 | |
| Melon | +703 | 78 vs 66 units, but $191 vs $215 each: FQ sells from hour 10 |
| Wool | +652 | |
| **Egg** | **-7,905** | **FQ has no geese**; DSM 181 eggs from 5 geese (no BAKERY or BRUNCH, so $43.7 each) |
| Land | -4,000 | FQ's SE on day 10 hour 11 |
| Hires | -2,794 | 11.8 vs 10.6 hands a day |
| Seeds, animals | -1,900 | |

- **Why:** FQ's third field and early cows outproduced DSM's geese and cheaper farm.
- SE value to FQ at marginal prices (`d_se.py`): own cash +$3.3k after land, seeds and a labour share. The margin value
  was +$9.7k to +$11.1k, because the SE's 246 wheat, 126 carrots and 32 strawberries cut DSM's revenue by $7.8k.
- **Could DSM have won with market micro?** No. Every exact market lever together is about +$0.84k < $1.08k:
  - moving its SELLs to the front in the races it lost: +$91;
  - all melon lots 2 steps earlier: +$632;
  - fertilizer and carrot lots 1 step earlier where the units were already in the shed: +$107;
  - post-drain timing: +$12.
- It needed production: a third field or early cows (FQ's step-0 cows). Note that SE_SHOPS=2 would not buy SE in this
  world (0 crop shops among the first three).

### 2.2 DSM beat MMPQ by 161 (114493956)

| Item (DSM minus MMPQ) | $ | Why |
|---|---|---|
| Land | +4,000 | MMPQ bought SE |
| Hires | +3,462 | 10.7 vs 12.1 hands a day (the 12th and 13th hire cost $144 and $233 each, every day) |
| Egg | +2,804 | 259 vs 198 eggs (7 geese vs 6; 2 BRUNCH shops) |
| Fertilizer | +1,594 | |
| Seeds | +1,220 | MMPQ bought 8 tomato and 48 more wheat seeds |
| Milk | +453 | $98.3 vs $91.6 a unit |
| Wheat | -5,745 | 327 vs 533 units (191 of MMPQ's from SE) |
| Tomato | -3,496 | MMPQ 57 units, DSM none |
| Strawberry | -1,700 | |
| Carrot | -1,362 | |
| Melon, wool | -575 | |

- MMPQ's SE: own cash +$5.8k, margin value +$14.4k to +$16.4k. Without SE, MMPQ's other three fields produce clearly
  less than DSM's two plus geese. MMPQ also lost 17 plants to drought and kept 5.8 idle tiles a day (DSM: 1 and 0.4).
- **The loser could have won with any one of these (exact):**
  - Put STRAWBERRY first in its list in the 8 strawberry races where DSM's slot came first: +$101 own, -$85 DSM =
    **+$186 > $161**. This is the per-turn lockstep port; cash only rises, so there are no knock-on effects.
  - All its SELLs first in the 23 races where DSM's slot came first: +$282.
  - Its day-10/11 melon lots 2 steps earlier, as its own Sep 27 build sold them (first lot at h6, not h8): **+$346**
    (+$638 with the day-16 lots).

### 2.3 Vadim beat MMPQ by 2,392 (114514401)

| Item (Vadim minus MMPQ) | $ |
|---|---|
| Land (MMPQ bought SE) | +4,000 |
| Hires (10.9 vs 12.1 a day) | +3,085 |
| Strawberry (234 vs 214 units) | +2,561 |
| Milk (202 vs 182 units; MMPQ culled a cow on day 22 at a $59 milk price) | +1,991 |
| Animals (MMPQ: 2 more geese, 1 more sheep) | +1,100 |
| Seeds | +960 |
| Wool, melon | +80 |
| Wheat bought | -833 |
| Tomato (MMPQ 75 units at **$51.7**: no PIZZA or FARMERS in this world) | -3,877 |
| Egg (MMPQ 5 geese vs 3) | -2,190 |
| Wheat, carrot, fertilizer | -4,485 |

- MMPQ's SE: own cash +$3.9k, margin +$8.5k to +$10.2k. The SE paid; the extra hands and the tomatoes did not.
- **Could MMPQ have won with market micro?** No: races +$175, melons 2 steps earlier +$731, feasible 1-step-earlier
  wheat/carrot/egg/fertilizer lots +$228, so about +$1.1k < $2.39k.
- It needed the crop choice. Its 10 tomato tiles earned $295-299 each (net of seed). Its strawberry tiles planted in
  the same days earned $789-817. As strawberries, that is about +$4.9k at average prices, less at the margin [I].
  Alternatively, 11 hires a day instead of 12-13 (-$3.1k of cost, less output).

### 2.4 Vadim beat DSM by 1,313 (114514402): the mirror

- Days 0-9 were identical: the same orders, cash within $1 on days 2-5, the same land hours, the same melon
  tiles and hours. The divergence started on day 11:

| Days | Vadim plantings | DSM plantings |
|---|---|---|
| 11 | W12 | C2 S1 W7 |
| 12 (FARMERS revealed) | C2 **S8** W2 | C3 **S4** W5 |
| 15 | C2 **S4** | C3 S1 W3 |
| 18-19 (second FARMERS) | - | **T9, T1** |

| Item (Vadim minus DSM) | $ |
|---|---|
| Strawberry (256 vs 213 units at $185: SMOOTHIE ×2, ICE_CREAM, FARMERS ×2, BRUNCH) | **+8,078** |
| Wheat, egg, fertilizer | +1,203 |
| Tomato (DSM 73 units at $78.7) | -5,743 |
| Carrot | -1,992 |
| Wool (DSM $125.6 vs $119.3) | -597 |
| Costs | +359 |

- **Realised value per tile** (`d_tileyield.py`, net of seed):

| Crop | Planted | $ per tile |
|---|---|---|
| Strawberry | days 10-14 | $1,380-1,384 (8.0 units per tile) |
| Strawberry | days 15-19 | $1,013 |
| Tomato | days 15-19 | $524 |
| Carrot | any | $104-129 per cycle |

- **Could DSM have won with market micro?** Only with a 2-step melon lead on all lots (+$1,264) plus the races
  (+$115), and a 2-step lead is not reachable with this layout (section 7). With 3 more strawberry tiles on days 12-15
  instead of the day-18 tomatoes, at average prices, it would have won (about +$2.6k) [I].

---

## 3. Build orders and each team's current plan

### 3.1 The four teams in these games [M]

| | DSM (games 1, 2, 4) | Vadim (games 3, 4) | MMPQ, Sep 28 build (games 2, 3) | FQ (game 1) |
|---|---|---|---|---|
| Step 0 | COW 1, WHEAT 5, SHEEP 1 ×3 | same as DSM | COW 1, WHEAT 5, SHEEP 1 | COW 1 ×4, HIRE ×4, SHEEP 1 ×2 |
| Step 1 | SELL WHEAT 1, HIRE ×4, COW 1, HIRE | same | SELL WHEAT 1, HIRE ×4, COW 1, SHEEP 1 | BUY WHEAT 1 ×8 |
| Melon tiles by day | d0 6, d1 2, d6 3-4 (11-12) | same | d0 6, d1 2, d2 2, d6 2 (12) | d0 3, d1 5, d2 1, d6 3, d8 1 (13) |
| Early cows (days 1-5) | 5, or 1 + 1 sheep a day with a yarn store on day 3 | same | 3 (one a day, days 2-4) | 3 (plus 4 at step 0) |
| Lands | d6 h3 NE, d8 h6 SW | same | d6 h4, d8 h6 (d9 h4), d10 h11 SE | d6 h2, d8 h5, d10 h11 SE |
| Geese | 5-7 (days 6-9) | 3-5 | 5-6 | 0 |
| Hires per day, days 0-10 | 5,3,5,5,6,6,9,7-9,10,9-10,11-12 | same | 4,4,6,6,6,6,9,7-8,9,10-11,12 | 4,5,6,5,7,6,8,8,10,10,12 |
| Hires per day, days 12-27 | 10-12 (mean 10.6-11.0) | 10-12 (10.9) | 12-13 (12.1) | 11-13 (11.8) |
| Tomatoes | only with FARMERS (game 4: day 18) | none | 8-11 tiles in any world | none |
| Melon lots | d10 h6:6 h9:18 h10:6 h12:6; d11 h10:6 h13:6; d16 h5:6 h7:6-12 ... | identical | d10 h8-h14, 6 per hour; d11-13 extra lots | d10 h10, h11, h14; d11 h13, h18; d12; d16 h4-h7 |

Herds over time against the shops (cows/sheep/geese at hour 0; `b_calendar.py`):

| Game, seat | Yarn store days | d6 | d9 | d12 | d20 | d25 | d29 |
|---|---|---|---|---|---|---|---|
| 1 DSM / FQ | none | 7/3/0 vs 7/3/0 | 10/3/3 vs 13/4/0 | 10/3/5 vs 13/4/0 | 11/3/5 vs 13/4/0 | 11/2/5 vs 13/2/0 | 11/0/5 vs 13/0/0 |
| 2 MMPQ / DSM | none | 5/3/0 vs 7/3/0 | 9/3/3 vs 8/3/7 | 11/3/6 vs 10/3/7 | same | 11/2/6 vs 10/2/7 | 10/1/6 vs 9/0/7 |
| 3 MMPQ / Vadim | 6, 21 | 5/3/0 vs 7/3/0 | 7/10/0 vs 7/8/0 | 7/10/5 vs 7/9/3 | 7/8/5 vs 7/9/3 | 6/8/5 vs 7/9/3 | 6/1/5 vs 7/2/3 |
| 4 Vadim / DSM | 3 | 3/6/0 both | 6/10/0 both | 6/10/5 both | 7/9/5 vs 7/8/5 | 7/7/5 vs 7/6/5 | 7/1/5 vs 7/0/4 |

- The yarn table: about 3 + 7 sheep per early yarn store, bought on days 3-9 (DSM and Vadim identical in game 4).
- No sheep are added after a late yarn store (game 3: the day-21 store brought 0).
- Flocks are culled from day 18-19, when wool floors.

### 3.2 Plan by team and date (cross-check on `tapes_top12`, `x_agg.py`) [X]

"Sep 28" = the 7 raw replays (n = 1-4 per team; indicative only).

| Team | Date | n | Win | Step 0 (modal) | Lands (SE share) | Lands 1/2/3 (median) | Cows d0 / d1-5 / total | Geese | Melon tiles d0/d1/d2-5/d6-9 | Tomato tiles | Hires per day | First melon sale d10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DSM | 09-25 | 124 | 83% | COW 1, WHEAT 5 | 2.77 (77%) | d6h5 / d9h3 / d10h13 | 2.0 / 0.8 / 8.9 | 7.4 | 6.0 / 3.5 / 0 / 0.9 | 17.3 | 11.6 | h9 |
| DSM | 09-26 | 208 | 78% | COW 1, SHEEP 3, WHEAT 5 | 2.05 (5%) | d6h3 / d8h6 | 2.0 / 3.4 / 8.1 | 7.1 | 6.0 / 2.1 / 0 / 3.7 | 1.7 | 11.1 | **h6** |
| DSM | 09-27 | 144 | 85% | same (144/144) | 2.10 (10%) | d6h3 / d8h6 | 2.0 / 3.5 / 8.0 | 6.9 | 6.0 / 1.9 / 0 / 3.8 | 7.4 | 11.1 | h6 |
| DSM | 09-28 | 4 | 50% | same | 2.00 (0%) | d6h3 / d8h6 | 2.0 / 3.5 / 9.2 | 6.0 | 6.0 / 2.0 / 0 / 3.8 | 2.5 | 10.8 | h6 |
| Vadim | 09-25 | 135 | 49% | COW 1, WHEAT 5 | 2.98 (98%) | d6h5 / d9h3 / d10h14 | 2.0 / 0.0 / 8.8 | 7.9 | 6.0 / 4.0 / 0 / 0 | 19.5 | 11.7 | h9 |
| Vadim | 09-26 | 187 | 59% | mixed | 2.30 (30%) | d6h3 / d8h6 | 2.0 / 2.4 / 7.9 | 7.8 | 6.0 / 2.6 / 0 / 2.7 | 6.3 | 11.2 | h6 |
| Vadim | 09-27 | 143 | 56% | COW 1, SHEEP 3, WHEAT 5 | 2.00 (0%) | d6h3 / d8h6 | 2.0 / 3.7 / 7.6 | 7.8 | 6.0 / 2.0 / 0 / 3.9 | 0.3 | 10.9 | h6 |
| MMPQ | 09-26 | 183 | 58% | COW 1, WHEAT 5 | 2.00 (0%) | d6h3 / d8h6 | 2.0 / 3.2 / 7.6 | 7.0 | 6.0 / 3.0 / 0.7 / 2.3 | 2.6 | 10.9 | h8 |
| MMPQ | 09-27 | 151 | 69% | COW 1, SHEEP 3, WHEAT 5 | 2.03 (3%) | d6h3 / d8h6 | 2.0 / 3.5 / 7.8 | 7.2 | 6.0 / 2.1 / 1.5 / 2.4 | 0.9 | 11.0 | h6 |
| **MMPQ** | **09-28** | 4 | **0%** | **COW 1, SHEEP 1, WHEAT 5** | **3.00 (100%)** | d6h4 / d8h6 / **d10h11** | 2.0 / 3.0 / 8.5 | 6.0 | 6.0 / 1.5 / 3.0 / 2.0 | **10.8** | **12.0** | **h8** |
| FQ | 09-26 | 174 | 66% | COW 4, SHEEP 2, HIRE 4 | 3.00 (100%) | d6h2 / d8h5 / d10h10 | 4.0 / 1.8 / 9.1 | **0** | 3.0 / 5.0 / 1.4 / 2.2 | 0 | 11.5 | h8 |
| FQ | 09-27 | 68 | 63% | same (68/68) | 3.00 (100%) | same | 4.0 / 2.1 / 9.6 | 0 | 3.0 / 5.0 / 1.5 / 3.2 | 0 | 11.5 | h10 |
| DECEM | 09-27 | 143 | 63% | COW 1, SHEEP 3, WHEAT 5 | 2.00 (0%) | d6h3 / d8h6 | 2.0 / 3.7 / 7.8 | 7.1 | 6.0 / 2.0 / 0 / 3.8 | 1.4 | 11.0 | h6 |
| UMG | 09-27 | 102 | 51% | same | 2.00 (0%) | d6h3 / d8h6 | 2.0 / 3.1 / 7.6 | 5.1 | 6.0 / 1.9 / 2.8 / 1.2 | 10.7 | 10.9 | h6 |
| Boey | 09-27 | 79 | 46% | SHEEP 1, WHEAT 3+7, HIRE 5 | 2.00 (0%) | d6h1 / d8h8 | 3.0 / 1.9 / 7.3 | 6.8 | 7.0 / 0.2 / 3.4 / 2.5 | 0 | 10.8 | h7 |
| **US (NPS, live)** | Sep 28 | 37 | 97% | COW 1, SHEEP 3, WHEAT 5 (37/37) | 2.03 | d6 / d8 (36/37) | 2.0 / 3.0 / 7.3 | 7.9 | 6.0 / 2.1 / 0.1 / 3.8 | 2.3 | 11.1 | h6 |

**Did they keep the Sep 26 plan?**
- DSM: yes, unchanged Sep 26-28.
- Vadim (by Sep 27), DECEM (by Sep 27) and UMG (by Sep 27) adopted it: SE 0%, the same step 0, the same land hours
  and the same melon hours.
- MMPQ kept it on Sep 26-27 and **changed on Sep 28**: SE again, 12 hires a day, tomatoes, a day-2 melon pair, first
  melon sale at h8. It also no longer spends overage at hour 0 of each day (17.5-21.7 s on the first turn only; Sep 25:
  13.7 s in-game).
- FQ and Boey are unchanged.
- First-turn overage in these games: DSM 14.7-15.9 s, Vadim 13.2-14.7 s, FQ 11.2 s, Boey 15.4-15.7 s, DECEM 0.

**Head to head, Sep 26-28** (`x_agg.py`; A's revenue minus B's per game):

| Pair | A W-L | A margin | Largest items for A |
|---|---|---|---|
| DSM vs MMPQ | 46-11 | +2,296 | egg +1.8k, strawberry +1.1k, carrot +1.0k; wheat -1.4k |
| DSM vs Vadim | 49-7 | +3,696 | tomato +2.7k, strawberry +0.7k, milk +0.7k; wheat -1.1k |
| DSM vs DECEM | 29-3 | +5,263 | milk +2.9k, strawberry +1.7k, egg +1.5k, land +1.5k, races +0.8k |
| DSM vs FQ | 17-15 | +1,590 | egg +12.1k, land +3.5k, hires +1.9k; strawberry -5.1k, wool -3.2k, milk -3.1k |
| MMPQ vs Vadim | 36-26 | +671 | |
| DSM vs Boey | 37-29 | +887 | |

By world [X]:
- DSM's wins against FQ fall from 67% with no yarn store (n=12) to 38% with 2+ (n=8, -$1.0k), and from 58% with 2+
  egg shops to 33% with none (Sep 25-28).
- Against Boey, DSM wins 20-29% with 2+ yarn stores (n=14, -$2.5k).
- Against Vadim, DSM wins about 88% in every world type. Its tables beat Vadim's on average; game 4 is an exception
  (strawberries).

---

## 4. Market micro

### 4.1 Same-turn sale races (`c_races.py`) [M]

"Moves to front" = the later seat's SELLs of that product moved to slot 0 in that turn only (exact lockstep port,
per turn, no compounding).

| Game | Races | Same slot | Earlier-slot counts | Race $ lost (avg-price gap) | Later seat moves to front: own / opp |
|---|---|---|---|---|---|
| 1 DSM-FQ | 155 | 89 | DSM first 60, FQ 6 | DSM 122, FQ 507 | DSM +69/-22; FQ -2,091/-39 (its wheat/fert sells follow its buys: moving them breaks the churn, not a real lever) |
| 2 MMPQ-DSM | 167 | 132 | MMPQ 12, DSM 23 | 162 / 135 | MMPQ **+158/-124**; DSM +145/-110 |
| 3 MMPQ-Vadim | 194 | 165 | 11 / 18 | 124 / 80 | MMPQ +102/-73; Vadim +77/-46 |
| 4 Vadim-DSM | 247 | 219 | 15 / 13 | 53 / 85 | Vadim +46/-42; DSM +72/-43 |

- **Same-slot lockstep is fair.** Both players quote from the same pre-commit inventory in every round, so equal
  units get equal prices. In the mirror, identical melon lots gave $14,567 vs $14,574. Only the larger lot's excess
  sells alone, lower. There is no seat-0 edge.
- **Races are small at the top.** Every leader puts its main product in slot 0. Races are $50-300 a game, but they
  decided game 2: strawberry races alone were worth +$186 to MMPQ, more than the $161 margin.

### 4.2 Sell timing (`c_micro.py`, `e_timing.py`) [M]

Shops drain at t%4==0 after that step's market, so t%4==1 is the first market after a drain.

| Product | Where the leaders sell it |
|---|---|
| Milk | at t%4==1: DSM 67-83%, Vadim 51-58%, MMPQ 59-77%, FQ 80% |
| Strawberry | at t%4==1: 47-79% (FQ 56%) |
| Wool | at t%4==1: 57-67% in yarn-store worlds (no phase preference without a yarn store); by hour, 45-78% at hours 0-5 (overnight production) |
| Eggs | 78-96% at hours 18-23 (DSM, Vadim) |
| Melons | 73-75% at hours 6-11 (family); FQ spreads them; MMPQ 50% at hours 6-11 |
| Wheat | 57-67% at hours 18-23 (family); FQ at all hours |

Lot sizes: 2.5-4.4 units per sale for milk, strawberry and wool (drip); 6-10 for carrots and wheat; FQ uses
"SELL X 10" every turn (sells whatever it holds).

This timing is locally optimal. Moving all lots of a product 1 step (exact single-product model, both directions):

| Product | Margin change per game |
|---|---|
| Milk | -$35 to -$1,360 |
| Strawberry | -$63 to -$1,092 |
| Wool (yarn-store worlds) | -$198 to -$1,088 |
| Sum over products | -$0.03k to -$1.7k earlier; -$2.9k to -$4.5k later |

- Earlier puts the lot before the drain. Later puts it behind the opponent's post-drain lots.
- Exceptions:
  - Melons 1 step earlier: $0 to +$157 on the 8-50% of lots that were already in the shed.
  - Wheat, carrots, eggs, fertilizer 1 step earlier: small (-$28 to +$148 per product).
- **Our agent already sells this way. No change.**

### 4.3 Slot order of their lists [M]

- Every leader puts its SELLs first: before the first HIRE in 46/46 to 48/48 turns (FQ 51/52), before the first BUY in
  58/58 to 97/97 (FQ 393/592).
- Pairwise SELL precedence:

| Team | Order (share of pairs placed first) |
|---|---|
| DSM | MELON (1.00) > MILK/WOOL/TOMATO (0.81-0.98) > WHEAT/CARROT (0.54-0.93) > STRAWBERRY (0.08-0.40) > EGG (0-0.06) > FERTILIZER (0-0.07) |
| Vadim | WOOL (1.00) > MILK (0.85) > WHEAT (0.65-0.88) > CARROT > STRAWBERRY (0.09-0.26) > EGG (0) = FERTILIZER (0) |
| MMPQ | MELON (1.00) > CARROT/TOMATO/MILK/STRAWBERRY (0.44-0.89) > WHEAT > EGG > WOOL (0.09-0.68) > FERTILIZER (0) |
| FQ | STRAWBERRY (0.89) > MILK (0.84) > MELON (0.78) > WOOL > CARROT > WHEAT > FERTILIZER |

- Our SELL_ORDER=held (held first, then MELON > WOOL > MILK > STRAWBERRY > FERT > TOMATO > EGG > CARROT > WHEAT) ties
  their slot-0 product. It wins the strawberry race whenever the family lists a cheap product (carrot, wheat) before
  strawberry. Keep it.

### 4.4 Dead slots and failed orders [M]

- **No dead or NOOP slots** in any leader list in these games. Lists hold 1-3 orders except hour-0 hire blocks.
- 10-order lists appear only at hour 0 (hires) and in FQ's churn.
- Failures are cash-limited retries (DSM 23 BUY_SEED and 8 BUY_PRODUCT INSUFFICIENT_CASH per game) and 2-7 phantom
  SELLs (SHED_EMPTY).
- FQ has 1,995 partial SELLs per game (its "SELL X 10" style).

### 4.5 Wash trades (`c_wash.py`) [M]

| Seat | Bought and resold per game | Resold within 1 step | Own | Opponent | Margin |
|---|---|---|---|---|---|
| FQ, wheat | 2,003 | 30% | +$23 | +$65 | -$42 |
| FQ, fertilizer | 519 | 71% | | | -$26 |
| Boey, wheat (games 5-6) | 2,346-3,040 | | | | -$63 / +$36 |
| Boey, fertilizer | | | | | -$45 each |

- Removing every matched buy-then-sell pair and recomputing both seats' cash gives the columns above.
- **Zero-sum churn.** A buy at p(m-1) followed by a sell at p(m-1) is exactly free. It only inflates their revenue and
  cost columns (FQ's "wheat revenue" of $68.5k).
- **Do not copy it**; the WASH flag should stay 0.

### 4.6 End of game, floors, stock [M]

- **Last day.**
  - At the start of day 29 they hold 60-100 units, e.g. DSM: wheat 11, carrot 20, strawberry 37, egg 6, milk 9. They
    sell almost all of it during day 29 ($5.1k-8.8k of sales on day 29).
  - What is left after step 718 is worth $0-206.
  - Races in the last 12 steps are worth under $25 each: prices are floored (wheat $20, fertilizer $3-7).
- **Soft floor, no hard floor.**
  - When they hold 3 or more units and the price is below 0.3 × base, the DSM family sells in only 6-12% of turns
    for milk (20-28% at higher prices) and 2-14% for wool (25-41%). They hold stock for the end-game dump.
  - Minimum sale prices: wool $1-18, fertilizer $3-9, milk $19-47.
  - FQ sells in 50-89% of turns regardless of price.
- **Hour 23.** 2-10% of units are sold at hour 23. The peak shed load at hour 23 is 22-40 (capacity 100: no overflow
  risk).

---

## 5. Animal care and field efficiency (`d_care.py`) [M]

| Seat | FEED / CARE per animal-day | Milk / wool / egg per animal-day | Escapes while the product was valuable | Plants lost to drought | Idle tiles, days 10-27 | Wheat units per tile |
|---|---|---|---|---|---|---|
| DSM (4 games) | 0.77-0.85 | 0.96-1.14 / 0.58-1.13 / 1.67-1.77 | 0 (1 goose on d28; 4 sheep on d28 at $141 in game 7) | 0-1 | 0.0-0.4 | 4.4-4.7 |
| Vadim | 0.83-0.86 | 1.07-1.09 / 1.03-1.14 / 1.73-1.74 | 0 | 1 | 0.4-0.6 | 4.6 |
| MMPQ (4 games) | 0.72-0.82 | 0.95-1.05 / 0.63-1.10 / 1.47-1.64 | 0 (5 sheep on d28 at $141) | **3-19** | **5.1-7.5** | 4.5-4.7 |
| FQ | 0.78 / 0.74 | 1.11 / 0.62 / - | 0 | 7 | 4.4 | 4.0 |

- **Culls are deliberate.** Every other escape happened with the product below 0.6 × base (wool $1-55, cows $13-59).
- **They stop caring on day 28 and stop feeding on day 29.** Animals whose next yield would come after the game ends
  are let go on day 28, even at wool $141.
- Melons: exactly 6.00 units per tile for everyone. Strawberries planted by day 14: 7.7-8.0 units per tile, falling to
  5.3-6.0 for days 15-19.
- MMPQ's third field is under-staffed: 17-19 drought losses and 5-7 idle tiles a day, despite 12-13 hires.

---

## 6. Exploitable weaknesses by team

### DSM family (DSM, Vadim, DECEM, UMG; one deterministic calendar since Sep 26-27)

1. **Fixed melon calendar** [X]. In 130-143 of 144 Sep 27 games:
   - day 10: h6 6 units, h9 18, h10 6, h12 6;
   - day 11: h10 6, h13 6;
   - day 16: h5 6, h7 8, h8-9 about 7.

   Selling every lot 1 step before them is worth +$426 per game on days 10-11 and +$660 with day 16 (89 mirror games,
   100% improved). Against MMPQ it is worth +$473, Boey +$217, FQ +$163.

   Their routes are optimal for their layout (section 7), so for a clone this is a tie, not a free win.
2. **Too few sheep in multi-yarn worlds** [X]. The yarn table is about 3 + 7 per early store and adds nothing for
   late stores. Against FQ and Boey (1-2 more sheep) they lose 62-71% of 2+ yarn games. Our clone has the same table.
3. **No third field, ever** (Sep 26+) [X]. In crop-shop worlds that leaves +$4.6-10k of SE output (leader data,
   section 8.2) to anyone who takes it.
4. **Egg dependence** [X]. Their edge over FQ is eggs (+$12k); with no BAKERY or BRUNCH, DSM wins only 33% against
   FQ.
5. **Predictable market habits** [M]:
   - milk, strawberries and wool at t%4==1 in 2-4 unit lots;
   - eggs in the evening; wool in the morning;
   - MELON/MILK/WOOL in slot 0, STRAWBERRY/EGG/FERT last;
   - liquidation spread over day 29.

   All of it is locally optimal. The one usable edge is strawberry before cheap products (held already does this).
6. **Deterministic** [X]. Same history, same action. Any pattern read from their tapes holds for the whole current
   version. Versions change between days (Sep 25 -> 26 -> 27 -> 28).

### MMPQ (Sep 28 build)

- Slower melons: first sale h8 (h6 on Sep 27). This costs it about $0.35-0.76k a game against the family (game 2).
- SE plus 12-13 hires a day (+$3.1-3.5k of hires), tomatoes without a tomato shop ($52 a unit), 3-19 drought losses,
  5-7 idle tiles a day.
- Nothing needs exploiting: the family calendar beats it (DSM 46-11). We already sell melons at h6.

### Fourth Quadrant

- No geese: the egg market is ours.
- Late melons (h8-10): our h6/h9 lots take the high prices (DSM $215 vs FQ $191).
- "SELL X 10" dumps every turn it holds stock.
- Its wheat and fertilizer churn is noise.
- Strong spots: the early milk from 4 step-0 cows (+12 milk by day 8 at $211), SE carrots with PET_CAFE, and
  multi-yarn worlds.

### Boey (for comparison; not in my four games)

- First melon lot at h7; a second lot at d11h1 (7.9 units).
- Wins multi-yarn worlds against the family.
- RL, stochastic.

---

## 7. Why the melon race against the family is locked [M]

Engine rules (`sim_trace.hpp`):
- WATER on a one-time crop adds +1 yield per day (+2 if fertilized) at ages 6-12.
- HARVEST needs age ≥ 10.
- Units respawn at the shed every morning. Hands hired at hour 0 act from hour 1.

What that means for melons:
- A day-0 melon watered on days 6-9 has yield 5 at d10h0. It needs a WATER on day 10 to reach 6.
- The family's tile (3,3) is 2 steps from (4,4). The farmer: h0-h1 walk, h2 WATER, h3 HARVEST, h4-h5 walk back, h6
  DROP and SELL. That is their h6 lot.
- Tiles 3 steps away worked by hands: h1-h3 walk, h4 WATER, h5 HARVEST, h6-h8 walk back, h9 DROP. That is their h9
  bulk.

Both are the minimum. To be earlier:
- **Fertilize the melons on days 6-8.** Waterings then give +2 and the melon reaches 6 by day 9, so the day-10 WATER
  can be skipped. That is -1 step on every lot. But fertilizer sells at $80-88 on days 6-8 (it floors at $11-19 by
  day 25): about $650-1,000 for 8-12 tiles against +$426-660 of margin. **Net ≤ 0; do not build.**
- **Harvest at 5 units without watering.** -1 unit (about $200) per tile. Worse.
- **Plant day-0 melons 1-2 steps from the shed**, e.g. (3,4) or (4,3). The first lot could drop at h4. But these are
  pasture tiles in the family layout, and the network copies that layout. **Too invasive before the deadline.**

---

## 8. What generalises (cross-checks)

### 8.1 Melon edge (`x_melon.py`, Sep 27, exact melon model, one family seat's lots moved)

| Matchup | Games | Days 10-11 lots 1 step earlier | Plus day 16 | 2 steps, all lots |
|---|---|---|---|---|
| Family vs family (mirror) | 89 | +$426 (se 7) | +$660 (se 25) | +$871 |
| Family vs MMPQ | 91 | +$253 | +$473 | +$744 |
| Family vs Boey | 63 | +$140 | +$217 | +$372 |
| Family vs FQ | 20 | +$52 | +$163 | +$361 |

### 8.2 Third field value by world (`x_se.py`)

Method:
- 568 leader games with SE: DSM, Vadim, DECEM and UMG from Sep 25 (the old plan); FQ from Sep 26-27; MMPQ from Sep 25.
- Units harvested on SE are removed from the seat's sales (first sales after each harvest), and the exact market is
  recomputed.
- Land, seeds, animals and a labour share are then subtracted.

| PIZZA or FARMERS among the first 3 shops | Games | SE own cash (marginal) | SE margin value (includes the opponent's lost revenue) |
|---|---|---|---|
| 0 (no PET_CAFE / PET_CAFE) | 131 / 112 | +$2.1k / +$2.6k (DSM family +$0.7-4.1k; FQ +$0.5-1.6k) | +$12.6k / +$9.9k |
| 1 | 179 / 69 | +$3.7k / +$4.3k | +$13.0k / +$11.3k |
| 2 | 57 / 12 | +$5.2k / +$4.6k | +$13.0k / +$13.2k |
| 3 | 8 | +$7.1k | +$13.4k |

Reading:
- The **gradient** (about +$1.5k of own cash per crop shop) agrees with our own 750-world paired measurement on the
  new-plan network: -$3.6k / -$1.1k / +$7.4k / +$21.3k for 0 / 1 / 2 / 3 crop shops.
- The **level** does not agree. This attribution ignores labour and care diverted from the other three fields (MMPQ's
  drought losses and idle tiles). It is an upper bound for the leaders, and our paired number is the one that counts
  for our agent.
- PET_CAFE in the first three shops does not raise the SE value. It is not a reason to widen CROP_SHOPS.

### 8.3 Other cross-checks

- **Strawberry tiles on days 10-14** (`x_straw.py`, 300 Sep 27 family games).

| Strawberry price at d10h0 | Games | Strawberry tiles planted, days 10-14 | Strawberry $/tile | Per 10 tile-days: strawberry / wheat / carrots / tomatoes |
|---|---|---|---|---|
| $150-199 | 259 | 6.2 | $1,004 | $625 / $478 / $391 / $470 |
| < $150 | 41 | 4.6 | $540 | $308 / $421 / $369 / $452 |

  - At $150-199, strawberries lead by about 30% per tile-day. Average-price attribution, so an upper bound.
  - Below $150 they do not lead.
- **Wash trades** are zero-sum for both churners in all 3 games that have them.
- **Herd trims** (`d_herd.py`, yarn-world games; the seat's last-placed sheep removed):
  - 1 sheep less is margin-negative in 9 of 10 seat-games (-$0.35k to -$3.1k). The one exception is Boey's seat in
    game 5.
  - 3 sheep less: -$2.4k to -$12.4k.
  - The opponent gains more from the higher wool prices than we save. This is the zero-sum logic of section 0.4.

---

## 9. Recommendations for our agent, ranked

Our live agent is NEURAL_20MW8NPS_SOH (new-plan imitation, SELL_ORDER=held, SE guard off). It already matches the
family on step 0, lands, early cows, the melon wave and melon hours, hires and geese (section 3.2). Its edges against
the top must come from what the family does not do.

| # | Rule | Evidence | Expected $/game | How to test | Risk |
|---|---|---|---|---|---|
| 1 | **SE_SHOPS=2** (the queued candidate): buy SE from d10 h12 only if at least 2 of the first 3 shops are PIZZA or FARMERS | Leader data: SE's direct value rises about +$1.5k per crop shop (+$4.6-7.1k own at 2-3; 8.2). Our 750-world paired test: +$7.4k (2), +$21.3k (3), negative at 0-1. The family never buys SE, so this is where we can out-produce it. In all 4 games here the rule stays off, which matches DSM's choice | About +$1.0-1.4k averaged (worlds with 2+ crop shops are about 15.6% of draws) | As planned: 464 paired, the 2,400+ top band, fresh and fresh2 leader tests. Report the ≥2-crop-shop subset separately | Low. Optional variant SE_SHOPS=1: leader data +$3.7-4.3k own, but our paired test -$1.1k, so test it, don't ship it blind |
| 2 | **SHEEP_YARN with a looser safety net**: keep 3 + 7 sheep per yarn store up to day 20, but drop or raise the "both flocks ≤ 12 × stores + 1" cap (e.g. 20 × stores). Code: `features._plan_fix`, SHEEP_YARN block, `room = 12 * yarn + 1 - mine - theirs` | The family loses 62-71% of 2+ yarn games to FQ and Boey, who keep 1-2 more sheep (wool -$1.9k to -$7.7k). Trimming a sheep is margin-negative in 9 of 10 cases (-$0.35k to -$3.1k per sheep). The current cap stops us at about 3-6 sheep when a family opponent holds 7-10 | +$0.3-0.6k averaged (2+ yarn stores in about 26% of worlds; +$1-2k there) [I] | Live-loss yarn replays (SpaTaro, Planned Economy, Aaweg) first, then 464 and the leader tests. Track the wool floor day, culls and sheep escapes | Medium: flooding both sides is margin-neutral at best when the opponent has no sheep. Keep the payback floor |
| 3 | **STRAW_UP** on days 10-14: when the strawberry price is at least $150 and 2+ strawberry shops are known (SMOOTHIE, ICE_CREAM, BRUNCH, FARMERS), turn up to 4 of the policy's WHEAT/CARROT PLANT orders into STRAWBERRY, MELON_WAVE-style (same tile, same labour; seeds bought the hour before). Code: a new block next to MELON_WAVE in `features._plan_fix`, with a flag in `live_rollout.py` and a default in `export.py` | Game 4 was decided by +7 strawberry tiles (+$8.1k). Family data: $625 vs $478 (wheat) and $391 (carrots) per 10 tile-days at $150-199; strawberries planted by day 14 give 8 units per tile | +$0.2-0.6k averaged (upper bound from average prices) [I] | 464 and the leader tests. Check strawberry units per tile (about 8), the strawberry price path and dead plants | Medium: strawberry is the steepest price curve above base ($1.92 per unit of inventory) |
| 4 | **Defensive rules to keep or check**: no 12th or 13th daily hire without a third field; no tomatoes without PIZZA or FARMERS demand; WASH=0; never trim flocks or withhold supply | MMPQ's hires cost +$3.1-3.5k; its tomatoes sold at $52; churn is zero-sum; supply is worth more in margin than in cash | Protects existing value | Quick 464 check of our tomato plantings by world | None |
| 5 | **Keep the market layer as it is**: SELL_ORDER=held, sell right after the drain, FINAL_FIX, midnight_fix | Races at the top are $50-300 a game and mostly same-slot ties. The leaders' timing is locally optimal (moving 1 step loses $0.3-1.7k). Final-turn races are under $25 | 0 (no change) | - | - |
| 6 | Optional, tiny: FEED_GUARD end-game exception in `features._feed_guard`. On day 28, don't force feeding animals whose next yield falls after step 718 | The leaders let them go on day 28 | +$50-100 | 464 only | None |
| 7 | **Not recommended**: a melon rush against the family via fertilizer (net ≤ 0, section 7); a day-0 melon layout change (invasive); FQ's wheat and fertilizer churn (zero-sum); 3 fields in 0-1 crop-shop worlds (our paired test is negative) | | | | |

**Suggested order for the remaining time.** Finish the SE_SHOPS=2 and SHEEP_YARN runs that are already queued: rerun
SHEEP_YARN with the looser cap if the first version is neutral. Build STRAW_UP only if both are done by the evening of
Sep 29. Upload decisions stay with the operator (memory `judge-top-band`: judge on the 2,400+ band and the leader
tests first).

---

## 10. Caveats

- Four games. Every per-game dollar figure is exact, but game-specific. The general claims rely on the cross-checks
  (sections 3.2 and 8: 89-568 games each). The "Sep 28" rows have n = 1-4.
- Market counterfactuals are exact per product, with no compounding: moved lots keep their units and nobody reacts.
  Production counterfactuals (SE, herd trims, crop choice) are attributions with first-sale removal. They are upper
  bounds and ignore how labour would be redeployed.
- The deterministic family can change its build overnight (it did on Sep 26-27; MMPQ did on Sep 28). Rules tuned to
  their exact hours (the melon calendar) are fragile. Rules tuned to world features (SE by crop shops, sheep by yarn
  stores, strawberries by price) are not.

## Appendix: scripts and outputs

All in `/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/topvstop_A/`
(`out_*.txt` hold the full outputs). Run each with `nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 <script>`.

**Core modules**

| Script | What |
|---|---|
| `core.py` | Raw replay loader: recorded observations, per-seat actions, exact trace, reward assertion |
| `port.py` | Per-turn exact market port (`kagjax/market_ref`) with validation |
| `pmkt.py` | Exact single-product market model (validated: 9 products × 7 games) |

**The four games**

| Script | Section |
|---|---|
| `a_cash.py` | Cash decomposition, volume/price split, end stock (2) |
| `b_calendar.py` | Day-by-day build order, herds, plantings, harvests, feed/care, escapes (3.1) |
| `c_races.py` | Sale races, exact move-to-front counterfactual (4.1) |
| `c_micro.py` | Sell timing, slot order, list shapes, dead and failed orders, day 29 (4.2-4.4, 4.6) |
| `c_floor.py` | Price floors and holding, hour-23 sales, shed load (4.6) |
| `c_wash.py` | Wash-trade matching and removal (4.5) |
| `d_tiles.py`, `d_se.py` | Quadrant attribution; SE marginal value (2, 8.2) |
| `d_care.py` | Feed/care rates, escapes and culls, drought, idle tiles, units per tile (5) |
| `d_herd.py` | Sheep trim counterfactual (8.3) |
| `d_tileyield.py` | Realised $ per planted tile by crop and day (2.4) |
| `e_levers.py`, `e_timing.py` | Drain timing, slot priority, melon shifts, end stock; ±1-2 step timing shifts (2, 4.2) |

**Cross-checks**

| Script | Section |
|---|---|
| `x_extract.py` -> `xrecs.pkl` (2,733 records) | Leader plan records from tapes_top12 |
| `x_raw.py` -> `xrecs_raw.pkl` | The same records for the 7 raw replays |
| `x_ours.py` -> `xrecs_nps.pkl` | Our 37 NPS live games |
| `x_agg.py` | Plans by team and date, head-to-heads (3.2) |
| `x_se.py` | SE value by world (8.2) |
| `x_melon.py` | Melon edge by matchup (8.1) |
| `x_straw.py` | Strawberry vs other crops (8.3) |
