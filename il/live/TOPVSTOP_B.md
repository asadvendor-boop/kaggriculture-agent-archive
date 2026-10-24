# Top vs top (B): how M&M&P&Q, Boey, DECEM and DSM beat each other

Written Sep 28 2026, ~11:30 PKT, for the operator's question "exactly how do the top teams beat each other, complete
details". Three games (Sep 28 UTC episodes, newer than anything in `data/tapes_top12`):

| Episode | Seat 0 | Seat 1 | Result |
|---|---|---|---|
| 114531598 | M & M & P & Q (MMPQ) 93,876 | Boey 95,472 | **Boey +1,596** |
| 114532100 | DECEM 93,873 | Boey 93,055 | **DECEM +818** |
| 114536000 | DSM 112,811 | M & M & P & Q 110,535 | **DSM +2,276** |

Method: exact C++ replays (`kagsim_trace.Game(seed, 720, True)`, market and unit receipts) on the Mac, 1 core, nice 10.
No networks, no Kaggle/API use, nothing written under `~/Downloads`. Cross-checks: 2,128 leader games of Sep 26-27 from
`data/tapes_top12` (every game with MMPQ, DSM, Boey, DECEM or Vadim), our live tapes (`tapes_nps` 34-42 games of the new
upload NPS_SOH as the poll grew, `tapes_n20mw8` 126-127 games), and the 4 games of the companion study `TOPVSTOP_A.md` (glanced at only).
Scripts: `/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/topvstop_B/`
(appendix). "Measured" = computed from the replays; "inference" is marked.

---

## 0. Bottom line

1. **All three replays reproduce exactly** in our engine: rewards equal, and money plus market inventory match the recorded
   observations at all 720 steps (0 mismatches).
2. **None of the three results came from sale-race slot order or from Boey's wash trading.**
   - Slot races (same product, same turn): moving a seat's late SELLs to the front would have changed its money by
     -$85 to +$218 per game (exact lockstep port).
   - Boey's wheat/fertilizer churn: 77-78% of its wheat buys are same-turn round trips, and they are **exactly neutral**:
     +$5 / +$197 for Boey and -$5 / -$8 for the opponent per game (exact port with a net-order counterfactual).
3. **They were decided by plan choices plus one market habit** (winner minus loser, measured):

| Game | Winner, margin | Decisive items (net cash lines, winner minus loser) |
|---|---|---|
| G1 | Boey +1,596 over MMPQ | MMPQ's 3rd field and 12-13 hires/day: land +4,000, hires +2,067 for Boey. Boey's strawberry hold through the day 19-24 crash: $155 vs $131 a unit, **price effect +5,556**. Boey's carrots after PET_CAFE: +3,255. MMPQ's wheat +9,685, tomatoes +4,033, melon price +1,032 were not enough |
| G2 | DECEM +818 over Boey | Boey grew carrots instead of wheat (101 vs 54 carrot plantings, 139 vs 190 wheat): wheat line +7,003 and carrot line -4,243 for DECEM. DECEM's wool +1,125. Boey won milk +1,183, melons +811, fertilizer +725, hires +735 |
| G3 | DSM +2,276 over MMPQ | MMPQ's 3rd field and 12-13 hires/day: land +4,000, hires +3,370 for DSM. **Goose care**: 267 vs 212 eggs from 7 geese each (geese fed on 92% vs 77% of days), eggs +2,377. Wool +2,241. MMPQ won wheat +5,998, tomatoes +4,178, melons +1,920 |

4. **New on Sep 28: MMPQ runs a 3-field + tomato version.**
   - In all 4 Sep 28 MMPQ games we have (these 2 and 2 in study A), MMPQ bought SE on day 10 hours 9-12, hired 12-13 a
     day from day 10 (DSM: 10-11) and planted 8-14 tomato tiles. It lost all 4 (-$161 to -$2,392).
   - This version first appears late on Sep 27. Across its 9 games it is 3-6: **1-5 in towns with no PIZZA_SHOP /
     FARMERS_MARKET among the first 3 shops, 2-1 in towns with at least one** (tomato revenue $12-20k there).
   - 7 of the 9 were against DSM, where it went 3-4. That is better than MMPQ's usual ~20% against DSM, so the version is
     not simply worse. It pays in crop-shop towns and loses without them.
   - This is the SE_SHOPS=2 logic, observed on the #1 team.
5. **Every script leader now plays DSM's Sep 26 plan, and so does our NPS.**
   - The plan: 2 fields (day 6 hour 3, day 8 hour 6), about 3 early cows on days 2-4, melons 6 on day 0 + 2 on day 1 +
     3.8 on days 6-9, first melon sold at **day 10 hour 6**, about 11 hires a day. DECEM and Vadim switched on Sep 27
     (DECEM's 3rd field: 30% of games → 0%).
   - Our live NPS matches it move for move: land d6h3 / d8h6, 2.85 cows on days 2-4, 3.8 late melons, first melon sale at
     d10h6 in 38 of 39 games.
   - Boey is the exception: an RL policy with its own opening, and it sells its first melon one turn later (h7).
6. **Boey's RL style** (section 5):
   - Best animal care of all teams: geese fed on 96-97% of goose-days; eggs / milk / wool per animal-day 1.77 / 1.08 /
     1.06, against 1.61-1.69 / 1.01-1.04 / 0.97-1.01 for the scripts.
   - Wheat is kept in the market instead of the shed: 25-29 units in the shed at day end, against 31-41 for the scripts.
   - It **holds strawberries through price crashes**: 32-34 stock-turns per unit sold, scripts 20-26.
   - It **never plants tomatoes**, it dumps wool, and it sells 92-98 units in the last 5 turns.
   - It beat MMPQ on land, labour and strawberry timing. It lost to DECEM on crop mix.
7. **The most exploitable weaknesses** (measured on Sep 26-27 games):
   - **Boey in tomato towns.** With 2 or more crop shops among the first 5 it wins 36% (-$3.3k); against DSM 29%, against
     MMPQ 25%. DECEM beats it there by $10.8k on average, with $17k of tomatoes.
   - **DECEM, Vadim and MMPQ skip tomatoes** even in crop-shop towns (0.5-2.7 plants). DSM plants 11.9 and earns $8.8k.
     DSM's tomato line against Vadim on Sep 27 was +$6.3k per game.
   - **MMPQ's Sep 28 third field** buys SE in towns without crop shops.
8. **Our NPS's biggest measurable gaps now:**
   - **No tomato table**: 2.8 tomato plants and $2.6k in crop-shop towns, against DSM's 11.9 and $8.8k. The old 20MW8
     planted 25.3 there; the new-plan fine-tune lost the conditional.
   - **Accidental escapes**: 1.38 per game while the product still sold at ≥ 0.6× base, before day 27 (leaders
     0.07-0.25), over 42 NPS games. Geese 36, sheep 12, cows 10. 14 of the 58 escaped the day after they were placed and
     were never fed (11 geese, 3 cows), and 7 of 42 games lose an animal on day 7.
   - **Output per animal-day** 1.62 / 0.99 / 0.98 (eggs / milk / wool), about 8% below Boey.
9. **Ranked recommendations** (section 7):
   1. A DSM-calibrated tomato table in crop-shop towns (W6).
   2. SE_SHOPS=2, also in the SE-off slot as "SE only in crop-shop towns" (W2).
   3. FEED_GUARD, with a check that it covers animals placed today or yesterday (W3).
   4. Optionally a care guard.
   - Not recommended: a sale price floor, Boey's wash trading, Boey's opening, a hire cap.

---

## 1. Data and exactness

| Check | Result |
|---|---|
| Replays | All 7 games (the 3 here + 4 of study A) reproduce the recorded rewards exactly. Money of both seats and market inventory equal the recorded observation at every step (0 mismatches over 3 × 720 steps) |
| Lockstep market port (`kagjax/market_ref.process_market`) | Equals the engine's `after_market` on every counterfactual turn used (0 mismatches) |
| Single-product market model for products that cannot be bought (`pmarket.py`) | Reproduces both seats' revenue exactly for all 38 product-seat series in the 3 games and for 165 live games of ours |
| Buyable-product model for wheat and fertilizer (`wmarket.py`) | Reproduces both seats' cash exactly in 4 games (16 series) |
| Engine detail that matters | In a same-slot race both seats are quoted the **same** price per round, computed from the inventory before either commits. Only a strictly earlier slot sells first. Most races in these games were same-slot |
| Time use (recorded `remainingOverageTime`) | MMPQ 21.5 s, only at step 0: it no longer overspends at hour 0 every day as on Sep 25. DSM 15.8 s and Boey 15.4-15.7 s, both at step 0 only; DECEM 0 s |

---

## 2. Game by game

### 2.1 G1 114531598: MMPQ 93,876 vs Boey 95,472 (Boey +1,596)

Shops, from the start of each day: SMOOTHIE d3, BAKERY d6, YARN d9, BRUNCH d12, PET_CAFE d15, SMOOTHIE d18, SMOOTHIE d21,
BRUNCH d24. There are no crop shops (PIZZA / FARMERS_MARKET) among the first 3.

**Net cash by line** (sales + product buys + seeds for crops + animal purchases for animal products):

| Line | MMPQ | Boey | Boey − MMPQ |
|---|---|---|---|
| WHEAT | +12,748 | +3,063 | **−9,685** |
| CARROT | +2,397 | +5,652 | **+3,255** |
| TOMATO | +4,033 | 0 | **−4,033** |
| STRAWBERRY | +27,296 | +32,281 | **+4,985** |
| MELON | +14,062 | +13,030 | −1,032 |
| EGG | +7,585 | +8,103 | +518 |
| MILK | +14,192 | +13,787 | −405 |
| WOOL | +11,464 | +11,750 | +286 |
| FERTILIZER | +11,717 | +13,357 | +1,640 |
| LAND | −7,000 (3 fields) | −3,000 (2) | **+4,000** |
| HIRES | −7,618 (300) | −5,551 (283) | **+2,067** |
| **Bank** | 93,876 | 95,472 | **+1,596** |

Revenue split for products that cannot be bought (Boey − MMPQ):

| Product | Units | Average price | Volume effect | Price effect |
|---|---|---|---|---|
| STRAWBERRY | 227 vs 231 | $155.0 vs $130.7 | −571 | **+5,556** |
| WOOL | 202 vs 142 | $82.9 vs $115.9 | +5,966 | −5,680 |
| CARROT | 226 vs 81 | $30.9 vs $35.5 | +4,818 | −703 |
| MELON | 72 vs 72 | $194.3 vs $208.6 | 0 | −1,032 |
| MILK | 192 vs 185 | $86.4 vs $91.8 | +624 | −1,029 |
| TOMATO | 0 vs 95 | − | −4,733 | − |

**Build order:**

| | MMPQ | Boey |
|---|---|---|
| Land | d6h4 (cash before $2,883), d8h6, **d10h12** (cash before $4,940) | d6h1, d8h8 |
| Day 0 | 2 cows, 3 sheep; melon 6, wheat 13; 4 hires | 3 cows, 2 sheep; melon 7, wheat 13; 5 hires at step 0 |
| Cows | 2 on d0, 1 on d2, 2 on d3, 2 on d6: 7 | 3 on d0, then one a day d3-d6: 7 |
| Sheep | 3 on d0, 3 on d9, 4 on d10 (YARN_STORE from d9): 10 | 2 on d0, 1 on d7, 5 on d9, 2 on d10: 10 |
| Geese | 3 on d6, 3 on d8: 6 | **2 on d2**, then 1 each on d6, d7, d8: 5 |
| Melons | 6 d0, 1 d1, 3 d2, 2 d6: 12 | 7 d0, 3 d2, 1 d6, 1 d8: 12 |
| Strawberry / tomato / carrot / wheat plantings | 29 / **14** / 24 / 218 | 29 / **0** / **67** (d17-26, after PET_CAFE d15) / 141 |
| Hires on days 10-27 | 11.9 a day (12-13); cost beyond 11 a day **$2,715** | 11.2 a day; beyond 11: $1,008 |
| First melon harvest / sale | d10h3 / **d10h6** at $272 | d10h4 / d10h7 at $266 |

**Market micro:**
- **Strawberries.** Both farms flooded the market on days 16-20 (price 200 → 41). MMPQ kept drip-selling 2-6 units every
  4 hours at $41-116 through days 19-24. Boey stopped selling at d20h10, **held up to 80 strawberries** until the town's 5
  strawberry buyers (3 SMOOTHIE, 2 BRUNCH) had pulled the price back to $135-179, then sold, including a 50-unit dump on
  day 29 (h13-h22, $179 → $139).
  - Exact single-product counterfactual: Boey's recorded schedule earned **$13.4k more** than selling on arrival. Without
    its hold, MMPQ would also have earned $6.6k less.
  - With Boey's schedule fixed, no hold rule would have helped MMPQ: reservation prices of 0.4-0.9× base give −$65 to
    −$3,641 against its recorded schedule.
- **Wool.** Boey's better care made 202 wool against MMPQ's 142 from the same 10 sheep, but with one YARN_STORE the
  extra 60 units floored the price. Boey sold 121 wool units below 0.3× base.
- **Wash.** Boey sold 3,233 and bought 3,040 wheat. 2,355 of these were same-turn round trips, with a within-turn effect of
  Boey +$197 and MMPQ −$8. Section 5 covers the rest.
- **Slots and lists.**
  - MMPQ: 1.51 orders a turn, every SELL in slot 0-0.6, 0 NOOPs.
  - Boey: 8.51 orders a turn, full 10-slot lists in 60% of turns, 986 NOOPs, 2,334 phantom SELLs (empty shed), real SELLs
    in slots 0.6-3.8.
  - Races: slot priority would have been worth +$102 to MMPQ and +$84 to Boey.
- **Last day.**
  - Stock at the start of day 29: MMPQ $2,414 (40 wheat, 22 eggs); Boey **$8,782** (48 strawberries).
  - Both ended with 0 stock.

**Animals and labour:**

| | MMPQ | Boey |
|---|---|---|
| Animal-days G / C / S | 131 / 183 / 174 | 119 / 181 / 199 |
| Fed share G / C / S | 0.86 / 0.69 / 0.78 | **0.97** / 0.77 / 0.81 |
| Cared share G / C / S | 0.82 / 0.69 / 0.74 | **0.95** / 0.75 / 0.81 |
| Units sold per animal-day G / C / S | 1.60 / 1.01 / 0.82 | 1.76 / 1.06 / 1.02 |
| Escapes | 10, all floored culls (sheep d19-28) | 8, all culls |
| Crop units per planted tile | Wheat 4.56, strawberry 7.97, melon 6.00; 8 plants dried to weeds | 4.40, 7.83, 6.00; 0 dried |

MMPQ's 12th hire ($144) works 17-19 turns a day and its 13th ($233) about 3.7 turns a day, about $63 a turn. The late
hires are as productive per turn as the others (57-62% productive actions).

**Where the 3rd field went** (production valued at the seat's own average sale price, seeds deducted):
- MMPQ's SE produced $17.2k net on 9% of its unit-actions (wheat 293, carrots 41, strawberries 24, wool 31).
- But MMPQ's first three fields produced **$13.7k less** than Boey's three ($127.8k vs $141.5k).
- The 4-field farm out-produced Boey's 3-field farm by only $4.9k gross, while paying $4,000 more for land, $2,067 more for
  hires and $1,320 of SE seeds.
- Inference: the labour spread over 4 fields cost the core farm about as much as SE earned.

**What would have flipped it** (MMPQ needed $1,597):
- (a) Skipping SE, if its core fields then produce like Boey's. The land, hires and seeds come to $7.4k against SE's $4.9k
  gross edge. This is an inference; the script cannot be re-run.
- (b) Nothing in MMPQ's own sale timing: the best single-product counterfactual (reservation, sell on arrival, shifts) gains only +$36 of margin (exact product models).
- (c) Boey's strawberry price effect (+$5.6k) alone exceeds the margin, and so do its land and hire savings (+$6.1k).

### 2.2 G2 114532100: DECEM 93,873 vs Boey 93,055 (DECEM +818)

Shops: YARN d3, PET_CAFE d6, ICE_CREAM d9, YARN d12, BAKERY d15, SMOOTHIE d18, PET_CAFE d21, BAKERY d24. No crop shops.

| Line | DECEM | Boey | DECEM − Boey |
|---|---|---|---|
| WHEAT | +9,603 | +2,600 | **+7,003** |
| CARROT | +5,996 | +10,239 | **−4,243** |
| STRAWBERRY | +10,693 | +10,096 | +597 |
| MELON | +13,539 | +14,350 | −811 |
| EGG | +5,690 | +5,900 | −210 |
| MILK | +7,128 | +8,311 | −1,183 |
| WOOL | +34,778 | +33,653 | **+1,125** |
| FERTILIZER | +12,204 | +12,929 | −725 |
| LAND | −3,000 | −3,000 | 0 |
| HIRES | −5,758 (291) | −5,023 (279) | −735 |
| **Bank** | 93,873 | 93,055 | **+818** |

**Build order:**

| | DECEM | Boey |
|---|---|---|
| Land | d6h3, d8h6 (2 fields: DECEM now plays the new plan) | d6h1, **d9h1** |
| Day 0 | 2 cows, 3 sheep; melon 6, wheat 12 | 3 cows, 2 sheep; melon 7, wheat 13 |
| First shop YARN on d3 | Sheep one a day d3-d5, +5 on d6, +3 on d12 (2nd YARN): 14 | Sheep one a day d3-d5, +3 d6, +1 d7, +2 d8, +4 d12: 15 |
| Cows / geese | 2 + 1 (d2) + 4 (d9): 7 / 5 (d7-d9) | 3 + 3 (d9): 6 / **2 on d2** + 2 on d9: 4 |
| Melons | 6 d0, 2 d1, **5 d6**: 13 | 7 d0, 3 d2, 1 d4, 2 d6: 13 |
| Plantings: wheat / carrot / strawberry | **190** / 54 / 19 | 139 / **101** / 23 |
| Hires on days 10-27 | 11.2 a day | 11.1 a day |
| First melon sale | **d10h6** $272 | d10h7 $266 |

**Market micro:**
- **Strawberries.** Only ICE_CREAM d9 and SMOOTHIE d18 buy strawberries, and the price crashed on day 18 and never
  recovered (it sat at $37-68 to the end).
  - Boey again held (hold 57.8 stock-turns per unit against DECEM's 13.3). It carried 39 strawberries into day 29 and
    **dumped 41 on the final turn (step 718), $76 → $26 → $1**.
  - Exact counterfactual, measured in margin: no alternative schedule helps Boey.
    - Sell on arrival: −$2,414 own.
    - Reservation at 0.4-0.7× base: −$0.6k to −$1.6k own.
    - One turn earlier: +$2 own.
    - Reservation at 0.8-0.9× base raises Boey's own revenue by $0.2-0.7k, but DECEM's by more (margin −$0.4k to −$0.6k).
  - So the dump was the least bad exit, not a fixable leak.
- **Wool** (2 YARN stores, one of them on d3): DECEM sold 330 at $126.6, Boey 314 at $131.1. DECEM's line is +$1,125,
  because it had one sheep fewer ($500) and sold 16 more units.
- **Wash.** Boey sold 2,525 and bought 2,346 wheat; 1,832 were same-turn round trips, worth Boey +$5 and DECEM −$5.
- **Slots.** Races: DECEM +$218 and Boey −$85 from slot priority. Boey: 9.02 orders a turn, 79% full lists, 876 NOOPs,
  2,848 phantom SELLs.
- **Final turn.** DECEM sold wool and fertilizer at $1.

**Animals.** Fed share G / C / S: DECEM 0.90 / 0.63 / 0.89, Boey **0.96** / 0.73 / 0.82. Units sold per animal-day: DECEM
1.66 / 0.93 / 1.11, Boey 1.76 / 1.05 / 1.03. Escapes: 16 and 12, all culls (9 sheep each on d28). Boey lost 3 strawberry
and 5 carrot plants to drying.

**Why DECEM won: crop mix.**
- Both farms made 276 plantings. Boey planted 47 more carrots and 51 fewer wheat (PET_CAFE opened d6 and again d21).
- Wheat is also feed. Both fed about 450 wheat, so DECEM's surplus to sell was 453 units against Boey's 179: +$7.0k on the
  wheat line.
- Boey's extra carrots, sold into its own flood at $38.6, returned only +$4.2k.

**What would have flipped it** (Boey needed $819):
- About 20-25 of the carrot plantings as wheat. Inference, from the per-plant nets: wheat about $128 and carrot about
  $102 per plant.
- Not the strawberry dump (see above), and not the wheat churn. The churn is worth +$0.47k of margin to Boey against a
  lean alternative (section 5).

### 2.3 G3 114536000: DSM 112,811 vs MMPQ 110,535 (DSM +2,276)

Shops: PET_CAFE d3, SMOOTHIE d6, ICE_CREAM d9, BRUNCH d12, FARMERS_MARKET d15, YARN d18, BRUNCH d21, ICE_CREAM d24. No
crop shop among the first 3; FARMERS_MARKET opens on d15.

| Line | DSM | MMPQ | DSM − MMPQ |
|---|---|---|---|
| WHEAT | +4,628 | +10,626 | **−5,998** |
| CARROT | +8,283 | +8,806 | −523 |
| TOMATO | 0 | +4,178 | **−4,178** |
| STRAWBERRY | +42,683 | +41,426 | +1,257 |
| MELON | +12,915 | +14,835 | −1,920 |
| EGG | +9,197 | +6,820 | **+2,377** |
| MILK | +14,212 | +13,712 | +500 |
| WOOL | +12,540 | +10,299 | **+2,241** |
| FERTILIZER | +12,972 | +11,822 | +1,150 |
| LAND | −3,000 | −7,000 | **+4,000** |
| HIRES | −4,619 (278) | −7,989 (303) | **+3,370** |
| **Bank** | 112,811 | 110,535 | **+2,276** |

**Build order:** the plans are almost the same, apart from the third field and the labour.

| | DSM | MMPQ |
|---|---|---|
| Land | d6h3, d8h6 | d6h4, d8h6, **d10h9** (cash before $5,088) |
| Day 0 | 2 cows, 3 sheep; melon 6 | 2 cows, 3 sheep; melon 6 |
| Cows | one a day d2-d4, +2 d6, +2 d9: 9 | one a day d2-d4, +2 d6, +2 d9: 9 |
| Geese | **4 on d6**, 3 on d8: 7 | 2 on d6, 3 on d8, 2 on d10: 7 |
| Sheep | +1 on d18 (YARN from d18): 4 | +2 on d18: 5 |
| Melons | 6 d0, 2 d1, 4 d6: 12 | 6 d0, 1 d1, 3 d2, 2 d3, 2 d6: 14 |
| Plantings: wheat / carrot / tomato / strawberry | 122 / 66 / 0 / 40 | 159 / 78 / **10** / 40 |
| Hires on days 10-27 | 10.8 a day; beyond 11: $144 | **12.1 a day; beyond 11: $3,003** |
| First melon sale | **d10h6** $272 | d10h8 $266 |

**Market micro.**
- Prices were nearly equal: strawberry $151.6 vs $150.4, milk $86.0 vs $87.0, carrot $43.1 vs $44.3.
- DSM's wool: $137.2 vs $128.0, with 4 sheep against 5 and the same 106 wool harvested.
- Races: slot priority would have been worth DSM +$168 and MMPQ +$85. Both sell in slot 0-0.5 with about 1.3 orders a turn.

**Animals.**
- Geese fed / cared: DSM **0.92 / 0.91**, MMPQ 0.77 / 0.73. Eggs per goose-day 1.72 vs 1.45, so 267 vs 212 eggs from 7
  geese each.
- Escapes: DSM 7 and MMPQ 9. The 4-6 "valuable" sheep escapes on day 28 are harmless end-game culls.

**Where the 3rd field went.**
- MMPQ's SE: $25.4k net of seeds (carrots 111, wheat 148, eggs 59, strawberries 56) on 10% of unit-actions.
- MMPQ's 4 fields together: $167.4k against DSM's 3 fields at $157.9k. That is +$9.5k gross, against $4,000 of land, $3,370
  of hires and $1,270 of seeds.
- Its core fields and animals lost ground to DSM's: eggs −55 units, wool price −$9.

**What would have flipped it** (MMPQ needed $2,277):
- (a) DSM-level goose feeding: +55 eggs, about +$2.3k. Borderline on its own.
- (b) Skipping SE and the 12th-13th hires: +$7.4k of costs against its SE output. Inference.
- (c) Buying 1 sheep instead of 2 on d18: +$0.5k.

---

## 3. Each team's plan today (Sep 26 plan kept? what is new?)

From exact replays of all their Sep 26-27 games in `tapes_top12`, plus the 7 Sep 28 study games. "Crop shops" means
PIZZA_SHOP or FARMERS_MARKET (the tomato buyers).

| | DSM | DECEM | Vadim | MMPQ | Boey | **Us (NPS live)** |
|---|---|---|---|---|---|---|
| Games (Sep 26 / Sep 27) | 505 / 170 | 311 / 266 | 356 / 232 | 252 / 290 | 81 / 79 | 34 profiled (42 for escapes, 39 for melons) |
| Win % (Sep 26 / 27) | 91% / 87% | 75% / 78% | 77% / 70% | 69% / 79% | 58% / **46%** | 36-3 live (2,703) |
| 3rd field (Sep 26 / 27 / 28 sample) | 17% / 8% / 0 of 4 | 30% / **0%** / 0 of 1 | 21% / **0%** / 0 of 2 | 0% / 2% / **4 of 4** | 0% / 0% / 0 of 2 | 1 of 34 |
| Land 1 / land 2 | d6h3 / d8h6 | d6h3 / d8h6 | d6h3 / d8h6 | d6h3 / d8h6 (Sep 28: d6h4 / d8h6 / d10h9-12) | **d6h1 / d8h7-8** | d6h3 / d8h6 |
| Day-0 herd | 2 cows, 3 sheep | same | same | same | **3 cows, 2 sheep** | 2 cows, 3 sheep |
| Early animals, days 1-5 | cows 3.5 | cows 3.6 | cows 3.6 | cows 3.5 | cows 1.9, sheep 0.8, **geese 2.4** | cows 3.1 |
| Melons d0 / d1 / d2-5 / d6-9 | 6 / 2.0 / 0 / 3.8 | 6 / 2.0 / 0 / 3.8 | 6 / 2.0 / 0 / 3.9 | 6 / 2.1 / 1.4 / 2.3 | **7 / 0.2 / 3.4 / 2.5** | 6 / 2.1 / 0.1 / 3.8 |
| Tomato plants (0 / 1 / 2+ crop shops by d15) | 2.9 / 4.7 / **11.9** ($8.8k) | 0 / 0.4 / 2.7 | 0 / 0.1 / 0.5 | 0 / 0.5 / 0.8 (Sep 28: 8-14) | **0 always** | 2.2 / 1.4 / **2.8** ($2.6k) |
| Hires a day, days 12-27 | 11.1 | 11.0 | 11.0 | 11.0 (Sep 28: 11.9-12.1) | 10.8 | 11.1 |
| First melon sale | d10h6 (49/60), h8 | d10h6 (60/60) | d10h6 (60/60) | d10h6 (55/60) | **d10h7** (60/60) | d10h6 (38/39) |
| Geese fed | 0.90 | 0.91 | 0.91 | 0.89 | **0.96** | 0.88 |
| Eggs / milk / wool per animal-day | 1.67 / 1.04 / 1.01 | 1.69 / 1.03 / 1.01 | 1.68 / 1.01 / 0.97 | 1.61 / 1.01 / 0.99 | **1.77 / 1.08 / 1.06** | 1.62 / 0.99 / 0.98 |
| Escapes before d27 while the product still sold ≥ 0.6× base | 0.13 | 0.12 | 0.25 | 0.10 | 0.08 | **1.38** |
| Strawberry hold (stock-turns per unit sold) | 20.6 | 20.3 | 21.1 | 24.0 | **31.7** | 16.2 |
| Share of wool sold below 0.5× base | 33% | 31% | 34% | 32% | **40%** | 18% |
| Units sold in the last 5 turns (steps ≥ 714) | 74 | 61 | 75 | 67 | **98** | 77 |
| Overage (Sep 28 games) | 15 s at step 0 | 0 | 13-15 s at step 0 | 21.5 s at step 0 (no daily hour-0 overage any more) | 15 s at step 0 | − |

Reading:
- **The Sep 26 DSM plan is now the consensus.**
  - DSM keeps it: 2 fields, early cows, the day-6 melon wave, tomatoes only in crop-shop towns, the melon sold at h6.
  - DECEM and Vadim copied it on Sep 27: 3rd field 30% and 21% → 0%, early cows 2.1 and 2.5 → 3.2, melon wave 2.3 and
    3.1 → 3.8.
  - DECEM also moved its melon harvest to h3 and first sale to h6. That is new since the Sep 23-25 forensics (h5 / h9).
- **MMPQ has two versions.** Its Sep 26-27 version is the same 2-field plan. The 3-field version with SE on day 10,
  tomatoes and 12-13 hires a day appears from late Sep 27 and covers all 4 sampled Sep 28 games (section 4).
- **Boey has not changed plan** between Sep 26 and Sep 28, but its results fell: 106 of 113 wins on Sep 24, 62 of 88 on
  Sep 25, 47 of 81 on Sep 26, 36 of 79 on Sep 27 (`tapes_top12`).
- **Our NPS copies the consensus**, including the melon rush that the forensics' R3 wanted.
  - The exception is tomatoes. The new-plan fine-tune learned DSM's "no tomatoes" average and lost the conditional: 2.8
    plants in crop-shop towns, where our old 20MW8 planted 25.3.

---

## 4. Exploitable weaknesses of each top team

| Team | Weakness (measured) | How to exploit it | Evidence |
|---|---|---|---|
| **Boey** | **Never plants tomatoes.** With ≥ 2 crop shops among the first 5 it wins 36% (−$3.3k, n=53); against DSM 29% (n=21), against MMPQ 25% (n=12); against DECEM it wins 5 of 9 but loses −$10.8k on average (DECEM's tomatoes $17.0k) | A tomato table in crop-shop towns (R1) | `agg.py`, section 3 |
| Boey | Sells its first melon one turn after everyone else: d10h7, price $266 against $272 at h6 | We already sell at h6 (NPS 38/39) | `melon_first.py` |
| Boey | Floods wool: 40-43% of its wool sells below 0.5× base, average $122-125 against the scripts' $136-151 | Nothing to do; it depresses its own wool line | section 3 |
| Boey | Holds strawberries into crashes: it wins big when the town has many strawberry buyers (G1) and dumps down to $1 when it does not (G2) | Do not race it into a crash; sell before its dump | sections 2.1-2.2 |
| **MMPQ (Sep 28 version)** | Buys SE on day 10 whatever the shops, plus 12-13 hires a day ($2.7-3.0k beyond 11 a day). Over its 9 three-field games: 1-5 in towns with no crop shop among the first 3, 2-1 otherwise | Skip SE in those towns (SE_SHOPS=2); where it pays, take SE + tomatoes too | section 0 item 4 |
| MMPQ | Weaker goose care: fed 0.77-0.89, eggs per goose-day 1.45-1.61; DSM's egg line against it is +$1.4-1.6k a game (n=55) | Our goose care is 0.88 (at MMPQ's level): R3/R4 | `agg.py` head-to-heads |
| **DECEM, Vadim** | Almost no tomatoes even with 2+ crop shops (DECEM 2.7, $4.2k; Vadim 0.5). On Sep 27 DSM's tomato line was +$2.1k a game against DECEM and **+$6.3k** against Vadim; DSM beat Vadim 20-0 | R1 | section 3 |
| Vadim | The most accidental escapes of the scripts: 0.25-0.33 a game | − | section 3 |
| **DSM** | No clear weakness. Against Boey it loses carrots (−$1.0-1.2k), melons (−$0.6-0.7k) and hires (−$0.5k), and still wins 56-68% | Match its conditional tomatoes (R1) and its care | section 3 |

Head-to-heads, Sep 26-27 (`tapes_top12`, exact replays; A minus B, mean per game; lands shown as (A, B)):

| Pair | n | A wins | Margin | Largest lines |
|---|---|---|---|---|
| Boey − MMPQ | 29 | 41% (Sep 27: 2 of 12) | −593 | tomato −2,555, carrot +2,100, wheat −1,848, egg +1,339, fertilizer +602, hires +397 |
| DECEM − Boey | 21 | 24% | +3,251 | **tomato +7,488**, melon −1,264, wool −1,263, wheat +1,210, carrot −958 |
| DSM − MMPQ | 55 | 80% | +2,335 | egg +1,566, wheat −1,399, strawberry +1,107, carrot +927 |
| DSM − Boey | 66 | 56% (Sep 27: 68%) | +887 | wheat +1,705, strawberry +1,457, carrot −1,183, tomato +734, melon −701, hires −609 |
| DSM − DECEM | 32 | 91% | +5,263 | milk +3,131, wheat −2,914, tomato −1,670, strawberry +1,632, land +1,500, egg +1,371 |
| DSM − Vadim | 55 | 89% (Sep 27: 20-0) | +3,787 | tomato +2,535 (Sep 27: +6,290), wheat −1,096, strawberry +917 |
| Boey − Vadim | 22 | 68% | +1,548 | melon +763, carrot +752, wool +635, tomato −913, wheat −878 |

---

## 5. Boey: the RL edge, and how to copy or counter it

What it does differently from the scripts (measured):

| Area | Boey | Scripts (DSM family, MMPQ) |
|---|---|---|
| Market lists | Fixed 10-slot lists (60-79% of turns full), 14-16% NOOPs, 2,334-2,848 **phantom SELLs** a game (nothing in the shed: harmless), real SELLs in slots 0.4-4.1 | 1.3-1.5 orders a turn, SELLs in slots 0-0.8, no NOOPs |
| Wheat / fertilizer | Sells 2,500-3,200 wheat and buys back 2,300-3,000 a game; **77-78% are same-turn round trips, exactly neutral** for both seats | Buy only for feed (112-176 wheat), never buy fertilizer |
| Shed use | 25-29 wheat in the shed at day end, **17-30 strawberries** | 31-41 wheat, 6-13 strawberries |
| Selling | **Holds strawberries** through crashes (31.7-33.5 stock-turns per unit sold); dumps wool; 92-98 units in the last 5 turns | Drip-sells 2-4 units at t%4==1; 49-76 units in the last 5 turns |
| Animals | Geese fed 0.96-0.97 and cared 0.93-0.95. Output per animal-day about 4-10% above the scripts. 0.07-0.08 accidental escapes a game | Geese fed 0.89-0.91 |
| Opening | 5 hires at step 0; 3 cows + 2 sheep; 7 melons on d0 + 3 on d2; **2 geese on d2**; a cow a day from the first milk shop or a sheep a day from the first YARN | 2 cows + 3 sheep; 6 + 2 melons; cows d2-4; geese from d6 |
| Crops | **Never tomatoes**; many carrots when PET_CAFE opens (60-67 plantings; G2: 101) | Tomatoes conditional (DSM only) |
| Land | 2 fields, d6h1 and d8h7-8 (it pre-sells a lot to fund land at h1) | d6h3 and d8h6 |

**The wash / churn, in detail:**
- **Same-turn round trips** (Boey sells X and buys X in the same turn): with the lockstep port and the round trip replaced
  by one net order, the effect is Boey +$197 / +$5 and opponent −$8 / −$5 per game. The fertilizer churn is ±$1.
  - It does not block the opponent's sales. Different seats' slots do not compete, and both seats get the same quote in a
    shared slot.
- **Cross-turn trading** (exact wheat-market model, opponent fixed). A lean policy that buys only the next day's feed
  shortfall and sells only the surplus would have:
  - **Lowered Boey's margin** by $378-843 (G1) and $428-637 (G2).
  - Changed the scripts' margins by −$120 to +$450.
  - So Boey's churn is not a leak; if anything it gains about $0.5k. Our guess at the mechanism: it sells surplus wheat
    early in the day (67-70% at h0-11), before the DSM family's evening wheat sales, and keeps its shed light. This is an
    inference.
- **Fertilizer**: the lean baseline is worse for every seat, so it is not a fair benchmark there.
- **Consistent with the closed scripted-era finding** (memory `market-channel-facts`: E452/E453 wash −$170/−296 a game on
  our agents). Do not copy.

**Strawberry holding, in detail** (exact single-product counterfactuals, opponent schedule fixed):

| Game (strawberry buyers) | Boey's recorded schedule vs selling on arrival | What the hold did to the rival |
|---|---|---|
| G1 (3 SMOOTHIE + 2 BRUNCH: the price recovered from $41 to $179) | **+$13.4k own** | The rival earned $6.6k more than if Boey had dumped |
| G2 (ICE_CREAM + SMOOTHIE: no recovery) | +$2.4k own (but a 41-unit dump down to $1) | The rival earned $2.8k more |

**How Boey beat MMPQ (G1) and lost to DECEM (G2):**
- G1: MMPQ spent $6.1k more on land and labour for a 3rd field whose net output was smaller than that. Boey's strawberry
  hold paid in a 5-buyer town, and its carrots took the PET_CAFE market.
- G2: DECEM is on the same 2-field plan, so there was no land or labour gap to take. Boey's carrot-for-wheat swap cost
  about $2.8k on the two lines, and DECEM's wool edge decided it.
- In general (Sep 26-27):
  - Boey beats DECEM and Vadim in towns without crop shops: 11 of 12 against DECEM with ≤ 1 crop shop among the first 5.
  - It loses badly in tomato towns.
  - MMPQ beat it 10-2 on Sep 27.

**What we can copy or counter:**
- **Copy: care diligence.** Its output per animal-day is about 9% above ours. This is unit routing, the network's job; a
  rule can at most guard it (R3, R4).
- **Copy, maybe later: strawberry holding.** It only pays in towns with many strawberry buyers; see R6 for why we do not
  ship a floor now.
- **Counter:** tomatoes in crop-shop towns (R1). Selling melons at h6 we already do.
- **Do not copy:** the wash, the phantom SELLs, the 3-cow opening. Boey's own results are falling (46% on Sep 27).

---

## 6. Our agent against these leaders (NPS live 34-42 games; 20MW8 live 126-127 games)

| Measure | NPS | 20MW8 | Leaders | Gap source |
|---|---|---|---|---|
| Plan (fields, early cows, melon wave, melon at h6) | matches DSM | old plan (3 fields, melon at h9) | − | fixed by NPS |
| Tomato plants with 2+ crop shops by d15 | **2.8** ($2.6k) | 25.3 ($12.2k) | DSM 11.9 ($8.8k), of which 8.3 on d15-18 in place of wheat | new-plan fine-tune |
| Accidental escapes a game (product ≥ 0.6× base, before d27) | **1.38** over 42 games: geese 36, sheep 12, cows 10. 14 escaped the day after placement, never fed (11 geese, 3 cows); 7 games lose an animal on day 7 (5 cows, 4 geese) | 1.48 (geese 106 of 188) | 0.07-0.25 | unit routing (animals placed and not fed) |
| Eggs / milk / wool per animal-day | 1.62 / 0.99 / 0.98 | 1.53 / 0.98 / 0.93 | DSM 1.67 / 1.04 / 1.01; Boey 1.77 / 1.08 / 1.06 | care diligence |
| Strawberry price (vs 2400+) | $138.4 ($125.2, n=9) | $131.9 ($127.9) | $140-151 | 21-28% of strawberries sold below 0.5× base (leaders 10-15%) |
| Hires a day, days 12-27 | 11.1 ($747 beyond 11 a day) | 11.7 ($1,951) | 11.0-11.1 ($413-650) | fine now |

---

## 7. Ranked rules and tunes for our agent

Our rule layer is `il/features.py`: `_plan_fix`, `_feed_guard`, `midnight_fix`, and `decode_market` with
`SELL_ORDER="held"`. The live pick is NPS = new-plan soup + held + SE guard off. W-numbers refer to memory
`live-weaknesses`.

| # | Rule | Evidence here | Expected $/game | How to test / what to check |
|---|---|---|---|---|
| **R1** | **Tomato table (W6, coded as `TOMATO_SHOPS=2`)**, calibrated on DSM. When ≥ 2 PIZZA_SHOP / FARMERS_MARKET are known (DSM's switch point is ~d15), turn the policy's WHEAT PLANT orders on d15-18 into TOMATO, about 8 tiles (DSM: 8.3 of 11.9 plants in that window; 10-11 with 3-4 crop shops). Buy seeds ($50) the hour before, as in MELON_WAVE. The coded rule (8 per tomato shop, cap 32, fires from d6) is much more aggressive than DSM: bench both | DSM $8.8k of tomatoes against our $2.6k in 2+ towns. Boey never competes there (29-36% wins). DECEM and Vadim barely compete. Our old 20MW8 made $12.2k there. The NPS loss to Planned Economy (114547731) was a tomato town | **+$3-5k in 2+ crop-shop towns** (about 1/3 of games) → **+$1.0-1.5k a game** on average. Inference from DSM's realised $/tile minus displaced wheat and seeds | 464 split by crop-shop count (the 2+ subset carries the effect; 0-1 towns must be flat), both leader tests, the live-loss set incl. 114547731. Check tomato units per plant (DSM 7.4 fertilized), wheat feed stock, escapes |
| **R2** | **SE_SHOPS=2 (W2)** for the SE-on slot, and also for NPS as "SE only in crop-shop towns" (`SE_GUARD=1,SE_SHOPS=2` on the soup, the planned evening test) | MMPQ's Sep 27-28 3-field version: 1-5 with no crop shop in the first 3, 2-1 otherwise. Cross-team, observational: 3 fields won 52% / 61% against 2 fields' 80% / 78% with 0 / 1 crop shops, and 84% against 75% with 2+ (n=38, confounded by version). The 750-world study: +$7.4k / +21.3k with 2 / 3 crop shops | Already estimated (memory): the rule beats always-on and always-off on every ruler. Pairs with R1: SE + tomatoes is what won MMPQ its crop-town games (+$3.0-3.7k against DSM) | As planned (`srv_ruletest.sh ... SE_GUARD=1 SE_SHOPS=2`). Report the 2+ crop-town subset separately |
| **R3** | **FEED_GUARD=1 (W3)**, and make sure it covers animals placed today or yesterday: 14 of our 58 accidental escapes are animals never fed after placement (11 geese, 3 cows) | 1.38 accidental escapes a game against 0.07-0.25 for the leaders (42 NPS games); geese 36 of 58 | +$0.3-0.8k (about 1.2 animals a game, about $0.3-0.6k each) | Built as `NEURAL_20MW8NPS_SOH_FG` (first rulers +4, 464 flat). Count accidental escapes in the test games; target ≤ 0.4 |
| R4 | **Care guard** (new, optional): from about hour 16, an animal not fed or not cared today gets FEED + CARE from the nearest unit that is only moving | Boey's output per animal-day is about 9% above ours; DSM's about 3-4% | An upper bound of +$2-4k gross if we reached Boey's level (prices would fall); a realistic guard: small | Only if R1-R3 are in. 464 + leader tests; check waterings and harvests do not drop |
| R5 | Keep SELL_ORDER=held, the h6 melon sale (already there) and no wash | Slot races ≤ $0.22k a seat a game here; the wash is neutral; everyone sells the melon at h6 except Boey | − | − |
| R6 | **Not now:** a price floor (hold STRAWBERRY / MILK / WOOL below 0.3× base until step 714) | Open loop (`an8_ours_hold.py`, `an8b_shed.py`): +$1.19k margin a game against 2400+ on 20MW8 tapes (se 0.15k), but only **+$0.22k (se 0.21k) on NPS tapes**. The shed overflows in **87-94%** of games (85-115 units a game), and the sale-timing family has realised −$992..+$38 before. Strawberry only at 0.3×: NPS +$199 (se 122) | ≈ 0 to +0.2k, high risk | Only as a strawberry-only, shed-aware variant after R1-R3 |
| R7 | **Not recommended:** Boey's opening (3 cows, 7 melons, geese on d2), wash trading, a hire cap | Boey's results are falling; the wash was closed; NPS already hires 11.1 a day | − | − |

A note on R1 and R2 against specific teams: in crop-shop towns they take the tomato market from Boey (none), DECEM and
Vadim (little), and match DSM. In towns without crop shops, SE_SHOPS keeps us off the losing branch of MMPQ's new version.

---

## 8. Caveats

- Three games decide nothing by themselves. Every general claim above is cross-checked on 2,128 Sep 26-27 leader games or
  our 165 live games; where a claim rests on the 3 games only, it says so.
- MMPQ's 3-field version: n=9. 7 of the 9 games were against DSM. The shop split (1-5 against 2-1) is suggestive, not
  proof.
- The per-quadrant "production" values use each seat's own average sale price. The SE "what would have flipped it" items
  are inferences: a script cannot be re-run with different land decisions.
- The product-market counterfactuals are exact but open loop: the opponent keeps its recorded schedule, and shed capacity
  is not modelled (reported separately). The wheat "lean" baseline is one reasonable policy, not the optimum.
- Tape rulers cannot react. R1 and R2 add supply in markets where Boey, DECEM and Vadim do not compete, which is the
  favourable case for open-loop estimates. They still need closed-loop tests (`judge-top-band`: 2400+ games and the leader
  tests first).

---

## Appendix: scripts (session scratchpad `.../scratchpad/topvstop_B/`)

Run with `nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 <script>` from that folder.

| Script | What | Time |
|---|---|---|
| `make_tapes.py` | Raw replays (read-only) → compact tapes in `./tapes/` (seed, teams, rewards, shops, actions, overage) | 20 s |
| `verify.py` | Exactness: rewards, money and inventory at all steps | 1 s |
| `lib.py` | Replay helpers, ledgers, the market-port wrapper (`run_market`) | − |
| `an1_decomp.py` | Net cash ledger and product lines, unit flows, end stock | 1 s |
| `an2_calendar.py` | Build order by day: land, hires, animals, plants, harvests, sales, herd, shop reveals | 1 s |
| `an3_market.py` | Sales by product (price, hold, post-drain share, slot, hours), races, list stats, last day | 1 s |
| `an4_wash.py` | Same-turn wash, exact net-order counterfactual (lockstep port) | 1 s |
| `an5_farm.py` | Production by quadrant, SE payback, animal care, escapes, hires | 1 s |
| `an6_hold.py`, `pmarket.py` | Exact single-product market model; sell-on-arrival, reservation and shift counterfactuals | 1 s |
| `an7_labour.py` | What each hire index does, days 10-27 | 1 s |
| `an8_ours_hold.py`, `an8b_shed.py` | Price-floor counterfactual on our live tapes (per product; joint with shed check) | 30-60 s |
| `an9_melon.py`, `melon_first.py` | Melon race: first harvest and sale hours (games; leaders Sep 27; ours) | 1 s / 40 s |
| `an10_wheat.py`, `wmarket.py` | Exact buyable-product model; Boey's churn against a lean policy | 1 s |
| `an11_volprice.py` | Volume and price split of revenue differences | 1 s |
| `an12_races.py` | Per-seat value of slot priority in races (exact port) | 1 s |
| `xprof.py`, `agg.py` | Profiles of 2,128 leader games (Sep 26-27), 166 live games of ours and the 7 study games → team plans, head-to-heads, our gaps | 6 min + 5 s |
| `esc_ours.py`, `shedcomp.py` | Our accidental escapes by animal and day; shed occupancy and composition at day end | 20 s |

---

## 9. Addendum, Sep 28 ~20:00 PKT (after the evening box results)

The live pair changed after this report was written: NPS was retired; the pair is NPSE (new-plan net + SE) and SOH
(v13_p3 + held + SE). The box winner NEURAL_20MW8_SOH_ST2 is v13_p3 + held + SE + STRADDLE=2. Consequences for sections 6
and 7:

- **The tomato gap belongs to the new-plan net only.** v13_p3 (20MW8 column) already plants about 25 tomatoes in crop-shop
  towns. That matches the box result: `tom` was −5 / −8 wins on v13_p3.
  - R1 therefore applies to NPSE-type nets, not to the SOH_ST2 line.
  - R2 (se2) and R3 (fg) were neutral or worse on v13_p3 on the box.
- **New for the v13_p3 line: the melon rush (forensics R3, never built).** Exact melon-market counterfactual on the live
  SOH tapes (`an14_melonrush.py`; open loop, rival on its recorded schedule):
  - Live SOH sells its first melon at **d10h9 in 93 of 96 games**. Every leader sells at h6 (Boey at h7), and so does NPSE
    (87 of 88).
  - Moving SOH's day 10-11 melon harvest and sales earlier, with the sales in slot 0:

    | Shift | Margin per game, all 96 games | Games improved | Margin per game, 76 games vs 2400+ |
    |---|---|---|---|
    | 1 turn | +$294 (se 20) | 94% | +$271 |
    | 2 turns | +$482 (se 31) | 94% | +$439 |
    | 3 turns (h9 → h6, the leaders' timing) | **+$653 (se 38)** | 95% | **+$604 (se 36)** |

  - Needs a unit override on days 10-11: the farmer plus about 3 hands on the ripe melons nearest the shed from hour 0,
    drop by hour 5-6. NPSE's net does this natively, so the layout allows it.
  - It is different from `m10` (a second melon wave), which the box rejected.
- **Boey's wheat churn and the drain straddle** (`an13_straddle.py`): Boey's net wheat flows lean only weakly on the
  drain phase.
  - Net sells: 33-42% at t%4==1, right after the drain (uniform would be 25%).
  - Net buys: 29-30% at t%4==0.
  - This replaces the "sells before the evening" guess in section 5. Boey half-exploits the drain; STRADDLE=2 does it
    cleanly. The scripts show no such pattern.
