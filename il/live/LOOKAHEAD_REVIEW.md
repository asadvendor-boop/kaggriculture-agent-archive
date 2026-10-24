# Look-ahead search: second review and redesign

Written Sep 27 2026, ~21:00 PKT. Read-only review of the Sep 26 look-ahead ceiling study (agent L,
`il/live/lookahead_ceiling.py`), the Kaggle time budget, and our value functions. The only new computation was four
value-function probes on the Mac (1.1-4.6 s each, 1 core, about 1,000 forward rows in total, section C.2). Probe
scripts: `/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/vprobe{,2,3,4}.py`.
They are session-local; the method is in the appendix.

## 0. Bottom line

- **The Sep 26 verdict "not deployable" was right. The number behind the interest was not a ceiling.** The
  +$6.7k/game "upper bound" mixes three things:
  1. Selection on chaotic noise: the max over deterministic full-game rollouts, with no placebo arm.
  2. Future leakage: the rival's recorded future sales, and the recorded future shop unlocks.
  3. A strawman option: "sell all" includes feed WHEAT and FERTILIZER.
- **The first study did not underestimate search. It overestimated it.** Doing it right does not reveal a hidden
  large gain, for three reasons:
  - **Compute budget.** On Kaggle-like x86 CPUs the 20M numpy runtime takes 66-189 ms per forward
    (`il/logs/server2/timing_*.log`). The approved plan assumed about 0.01 s. That leaves room for only about
    5-15 extra forwards per decision, so rollouts longer than a few game-hours do not fit.
  - **Evaluators.** Both value functions we own misjudge value (probe C.2). The IL value head and the JAX RL critic
    count $1 of extra cash as $0.4-0.6. With cash counted exactly, the IL head is erratic: |Q| is 1.5x the stake at
    the median and 5.4x at p90. The critic undervalues stock by about half, so it prefers "sell all" in 85% of
    states; that blanket rule lost in the rollout study. The two agree on sell-or-hold only 35% of the time.
  - **Nothing left to add.** The network already sees the market model: the features6 price forecasts, the dump
    price, and both farms' pending supply. The rule layer already computes the exact short-horizon mechanics:
    midnight overflow, final turn, homing and the SE guard.
- **Realistic gain of any search we could deploy by Sep 30: $0-300/game (at most about 10 rating points), with real
  downside risk.** One wrong switch costs $300-925 on average (the one-step deviation values). The chance of building,
  validating and uploading a positive version by Sep 30 23:59 UTC (Oct 1 04:59 PKT) is about 5-10%.
- **Recommendation: do not build a deployable search for this competition.** To close the question for good, run
  Stage 1a (placebo arm, about 5-10 core-hours, section D) on the next CPU box. Otherwise spend the remaining days on
  the final pair and on exact rules from loss autopsies. LESSONS.md lists these as the things that moved the ladder.

## A. The first study

### A.1 What it did

| Item | Sep 26 design | Source |
|---|---|---|
| Decision | End-of-day market order only, at hour 23 (step 24d+23), days 1-28 | `lookahead_ceiling.py` l.288-290 |
| Options | A0 = policy action after the rule layer<br>A1 hold = A0 without SELLs, then `midnight_fix`<br>A2 sell all = SELL every product in the shed, including WHEAT (feed) and FERTILIZER<br>A3 sell half was coded but never run | l.238-267; CANON l.45 |
| Branching | The C++ engine has no state clone, so each branch is a fresh game replayed from step 0 | l.12-13, 182-191 |
| Rollout | To the end (step 718). Our seat uses an exact greedy replica of the benchmark policy (20M g20, team 10, tau 0.5, rule layer). Forwards are lockstep and padded to 6 rows for bitwise parity | l.54-144, 202-218 |
| Opponent | The recorded tape: it does not react and its future is known. The rollouts also see the recorded future shop unlocks (forced shops), the rival's exact hidden shed and seeds, and the weed RNG | l.161-175 |
| Selection | Max final margin; ties keep A0; the chosen branch becomes the main line (the sequential rollout algorithm) | l.318-322 |
| Worlds | 32 of the 219 held-out 2400+ worlds (first job of 8 shards). The control already won 30/32 with a mean margin of +$15,564 | `report` output |
| Compute (days 8-28) | 1,339 s per game per worker; 6,247 policy rows per game; the forward pass is 98% of the time; about 12 core-hours for 32 worlds | `report` output |

### A.2 What it found

Re-derived today with `lookahead_ceiling.py report` and `combine`:

| Run | n | Gain vs control | Own bank | Rival bank | Notes |
|---|---|---|---|---|---|
| Days 8-28, sequential (server) | 32 | **+4,379** (se 609, median +4,030) | +2,078 | -2,301 | 32/32 improved; wins 30 -> 32 |
| Days 1-7 | 32 | **+2,281** (se 701, median 0) | +1,056 | -1,225 | 10/32 improved; day 4 alone gives +1,451 |
| Sum ("ceiling") | 32 | **+6,660** (se 1,066) | | | An approximation (`combine`) |
| Full days 1-28 on the Mac | 4 | +1,516 | +760 | -756 | **Different worlds** (eps 112273533, 112286813, 112097389, 112107323) |
| srvtest, days 27-28 | 8 | +506 (se 261) | | | |
| gputest, days 24-28 (CUDA) | 8 | +1,003 (se 340) | | | GPU numerics differ from the CPU benchmark |

- **One-step deviation values (days 8-28):**
  - Hold: -303 per decision (better 20%, worse 41%, equal 40%).
  - Sell all: -925 per decision (better 25%, worse 68%).
  - Sell all, days 1-7: -4,044.
- **Unfinished runs.** The full days 1-28 server run on the same worlds (`g20_s012_sh*_srv`, started 18:39) and the
  464-world GPU run (`il/logs/server2/lookahead_g20_sell012.log`) never finished.

### A.3 What it got right

- **Exactness.**
  - The control reproduced the stored benchmark margins, and the CHECK_A0 re-roll found 0 mismatches.
  - It found and neutralised the x86 batch-size numerics trap: logits from 1-3 rows differ bitwise from logits from
    5+ rows, hence PADMIN=6.
- **The right skeleton.** It is a sequential rollout improvement, and ties keep the policy's action.
- **Blanket baselines were measured.** Hold and sell-all lose, which matches the scripted-era lessons.
- **It flagged its own weak spots.** It named the tape oracle and the share of the gain that is the rival's loss.
  For full-game rollouts it concluded "not deployable" correctly: they need 10^2-10^3 forwards per decision against
  an affordable ~10 (C.1).

### A.4 Flaws and gaps (with evidence)

1. **Chaos-selection bias, and no placebo arm.** The max over deterministic full-game outcomes harvests
   butterfly-effect noise, so its expected value is positive even when the options are equal in expectation.
   - One end-of-day market choice moves the final margin by far more than its direct stake. Mean |Δ| of the
     alternatives vs the policy, by day: d4 $8,987, d8 $3,485, d10 $2,486, d12 $1,669, d16 $1,550, d20 $851,
     d24 $664, d28 $132.
   - The direct stake is usually a few hundred dollars. In the recorded hour-23 sheds that I replayed, the stock is
     mostly WHEAT and FERTILIZER, and sellable produce is worth $0-131. Caveat: those are the recorded actions of
     the older submissions that played the ladder games, not the 20M's.
   - The gain is concentrated where noise is largest:
     - 62% of the days 8-28 gain comes from days 8-11.
     - 68% of it comes from the switches worth more than $2k (out of 161 switches in 32 games).
     - Opposite deviations (hold vs sell all) have positively correlated outcomes (+0.10 to +0.50). In 5-14% of
       decisions both beat the policy.
   - Per-world outcomes are known to be near coin flips between near-equal models. In the Yarneo world six near-equal
     models ranged from -9,771 to +9,973 (memory `next-iteration-plan-sep26`).
2. **Future leakage beyond the rival's sales.**
   - The rollouts know the recorded future shop unlocks (`set_forced_shops`), the rival's exact hidden shed and
     seeds, and the rival's future sales.
   - Half of the gain is the rival's bank loss (-$2.3k and -$1.2k). The oracle sells just before sales it knows
     about.
   - In the scripted era, front-running predictable rivals with a real predictor realised **+$38/game against a +$224
     oracle** (`...Recovery/memory/market-channel-facts.md` l.231-233). In the same file, the sell-timing oracle
     bound was +$11-20k (l.207) while realised retiming lost $7.6-12.8k (l.259) and a drain-matched drip lost $992
     (l.280).
3. **Scope: one decision at one hour, with blunt options.**
   - The live NEURAL_20MW5_SE sells on 368 of 719 turns (51%), spread across all hours; hour 23 has only **3%** of
     the sell turns (my count on 97 live games, `data/tapes_n20mw5`).
   - A2 sells feed wheat and fertilizer. The scripted-era facts say "selling them starves the farm", so part of the
     "-925 for blanket sell-all" is feed damage.
4. **The evaluation design can never be deployed, and no deployable variant was tested.**
   - Full-game rollouts cost about 365 rows per decision on average and 1,000-1,300 early in the game.
   - A short horizon plus a leaf evaluator was never tried.
   - Every checkpoint already has a value head `v` and aux heads (`train6.py` l.114 and l.122: 0.1·MSE of the final
     margin / 2e4, clipped ±3; aux = margin and own-money change at +24, +72 and +168 steps, l.33 and l.62-70).
     They were never validated. `np_runtime.NumpyNet.forward` does not compute `v` (l.34-53), and `export.py` drops
     `aux.*` (l.15).
5. **The time budget was never measured.** `next-iteration-plan-sep26` l.29 says "our model uses ~0.01 s". The
   measured 20M numpy turn on x86 (server2, 1 thread) is:
   - 66-97 ms mean for the B64 package, 107-189 ms on a loaded box.
   - p99 up to 236 ms and a maximum of 130-260 ms.
   - The 10M takes 84-88 ms.
   - The Kaggle agent machine (1.6 vCPU) has never been measured.
6. **Sample and ruler.**
   - The 32 worlds were comfortable wins (30/32 won, +$15.6k), so at most 2 flips were possible. The leader tests,
     where the flips are (71% and 61% win rates), were not used.
   - "Not additive" was inferred from 4 *different* worlds with no standard error, so it is unsupported either way.
7. **The CPU and GPU runs are not comparable.** The GPU control is not bitwise equal to the CPU benchmark (docstring
   l.26-28), and gputest and srvtest cover different day ranges.

### A.5 Was the conclusion justified?

- "Not deployable": **justified.**
- "Realistic ~+$2k/g for RL to capture": **not supported.** Chaos, leakage and history all point to at most about
  $0.5k/game realisable for this decision family. LESSONS.md l.21 already says "Market planners (knowable gain <=
  $0.5k/game)".

## B. Which decisions are candidates for search?

Per-game frequencies are measured on the 97 taped live games of NEURAL_20MW5_SE (68-29).

| Decision | Per game (live) | Practical branching | When consequences show | Short rollout / value head? | Verdict |
|---|---|---|---|---|---|
| Sell timing, any hour | 862 SELL orders on 368 turns | 3-5 (policy, hold, all, premium only) | Hours to 2 days (shop drain every 4 turns, town every 24, rival dumps); later cascades through cash | A 1-day exact market model: yes, cheap. Value heads: no (C.2) | This is the "market planner" family: at most about $0.5k/g realisable |
| Midnight overflow | 28 | Exact | Same night | Exact rule (`midnight_fix`, +$2.2k/g) | Done; no search needed |
| Final-day liquidation | Last 1-2 days | Small | At most 48 turns | Exact except the rival | Done by rules (final turn +$195, homing +$475). Oracle for days 27-28: +$506 on 8 worlds, including leakage |
| BUY_LAND | About 4 orders, on days 6 and 9-10 (SE guard on day 11) | 2 (now or wait) | 4-16 days (crop and animal payback) | No: needs 100-400 turns; no calibrated critic | SE guard already covers it (+$288-696/g) |
| BUY_ANIMAL | 22.5 orders on 19 turns, mostly days 5-9 | 3 species × 4 quantities | 4-8 days to first yield, then ongoing | No | Sheep caps: 12 was wrong, 15 was slightly negative |
| HIRE (hands are hired daily) | 320 orders (about 11.4 hands/day) | 0-10 | Same day | Needs 1-day rollouts with the policy (24+ forwards per option) | Scripted era: extra hands had zero marginal product (-$2,306/g) |
| Crop choice and BUY_SEED | 274 seed orders plus per-unit PLANT | 5 crops × tiles | 2-16 days | No | Strategic; belongs to IL/RL |
| BUY_PRODUCT wheat / fertilizer | 34 | 18 / 4 quantity classes | 1-2 days (feed) | An exact feed projection is a rule, not search | Mid-game escapes cost about $1k/g: a **rule** candidate |
| Strategy (wool / milk specialist) | 1-3 | 48 styles | Whole game | No | Shop-router test; drop if under $1k |

Reading of the table:
- Every decision with a short, computable horizon is already an exact rule.
- Every decision with a long horizon needs a value function we do not have (C.2).
- Our losses are strategic (specialists: wool in 22 of 75 losses, milk 12, wheat 9), and those are not search
  problems.

## C. A deployable design, and why it is not worth building now

### C.1 Time budget per decision

| Quantity | Value | Source |
|---|---|---|
| Per-turn limit | 1 s (`actTimeout`), 720 turns | `kaggle_environments/envs/kaggriculture/kaggriculture.json` |
| Overage pool | 60 s per game. Each turn drains max(0, t - 1 s). The pool is readable in `obs.remainingOverageTime`. The local `schemas.json` default is 12 s, but Kaggle games show 60 | `core.py` l.631-632; `agent.py` l.220; recon tapes; memory `kaggle-act-timeout-1s` |
| 20M forward (x86, 1 thread, numpy) | 0.07-0.19 s. Plan for 0.1-0.2 s on Kaggle | `timing_NEURAL_20M_SE_B64.log`, `timing_NEURAL_20M_SE.log` |
| Package load at step 0 | 1.0-2.9 s | `timing_NEURAL_20M_SE_B64.log` |
| Safe in-turn headroom at a decision turn | About 0.5 s, i.e. 2-5 extra forwards | 1 s minus the normal turn minus a 0.3 s margin |
| Safe overage for search | About 25 s per game (keep at least 30-35 s reserve), i.e. 125-250 extra forwards | |
| **Per decision (28 per game)** | **About 6-15 forwards of the 20M** | |
| Sep 26 design | About 365 rows per decision on average, 1,000-1,300 early (25-250x over budget) | A.1 |

Ways out of the budget, none of them feasible by Sep 30:
- A distilled rollout policy of 1-2M parameters (5-10 ms per forward) would allow 3 options × 24-turn rollouts
  (about 0.4-0.7 s per decision). It needs GPU distillation plus validation.
- A full-state in-agent engine is possible in principle: the official `kaggriculture.py` is pure Python, and
  `kagjax/market_ref.py` is a pure-Python market port verified on 26k scenarios. But it needs the rival's hidden shed,
  inventories and seeds, plus the future shop draws. `kagjax/state.obs_to_game` needs *both* seats' observations.

### C.2 Leaf evaluators

Measured today on the Mac. The replays use the recorded actions of both seats in held-out 2400+ ladder worlds; the
models are `il/ckpt/ilsp_v13_p3.pt` (NEURAL_20MW8_SE) and the critic `vstate` in `il/ckpt/jax1_final.pt`.

| Probe | IL value head `v` | RL critic (jax1_final, return-to-go) | A coherent leaf would give |
|---|---|---|---|
| Δ estimate for +$1,000 cash (48 states) | median **+636** (IQR 388-715) | median **+415** (IQR 316-605) | about +1,000 |
| Sell all (excluding wheat/fertilizer), cash counted exactly, money inputs held equal (82 states, median stake $398) | Median \|Q\|/revenue **1.46**, p90 **5.43**. Erratic: $172 of goods moved v by -$1,056, and 3 FERTILIZER ($102) moved it by +$1,694 | Median \|Q\|/revenue 0.57 and **prefers sell-all in 85% of states**, because it undervalues stock | Small \|Q\| vs revenue, no systematic side |
| Agreement between the two on sell vs hold | **35%** (worse than chance) | | |

Neither network can serve as a search leaf as-is:
- The IL head would make near-random switches, each costing about $300-925 on average.
- The critic would turn search into the losing "sell all" blanket rule.

### C.3 The only affordable design

This is a market-model lookahead, i.e. the market-planner family. Build it only if Stage 1 passes (section D).

- **Trigger.** Hours 22-23 when the shed holds at least $500 of sellable goods and the time guard is OK. Optionally,
  on turns where the market head fires a SELL of a premium product.
- **Options.**
  - A0: the policy action.
  - Hold: A0 without SELLs, then `midnight_fix`.
  - Sell all, keeping 2 days of feed wheat and the fertilizer needed for planted crops.
  - Sell premium products only (strawberry, melon, milk, wool).
- **Transition model.** Pure Python, under 20 ms for 4 options × 24 turns. It uses only public or own data and no
  tape:
  - The exact lockstep price curve (`features6.price`, `MKT`) and the exact shop and town drain schedule
    (`features6.drain`).
  - Our own production from our tiles (the features6 pending supply).
  - The rival's sales by hour, inferred from public market-inventory deltas: Δinventory minus our sales plus the known
    drain.
- **Opponent model.** An exponential average of the rival's inferred sales by hour of day over the last 2-3 days,
  plus their public pending supply. The rival's hidden shed is ignored, or estimated as inferred harvests minus
  inferred sales. No clone forward: at 0.1-0.2 s it is unaffordable.
- **Leaf.** Cash plus the exact revenue of the remaining stock sold over the next day at the drain ticks, interleaved
  with the predicted rival sales, minus overflow losses. No network in the leaf (C.2).
- **Fallbacks.**
  - Compute the policy action first and use search only to override it.
  - Skip search if `remainingOverageTime` < 40 s, or if the call has already used more than 0.3 s.
  - Cap search at 20 s per game.
  - Wrap it in try/except that returns the policy action.
  - No randomness and no printing.
  - Ties keep A0, and require a margin of at least $100 before switching.

Why I do not expect it to pay:
- The network already sees these quantities (the features6 docstring: price forecasts at +4/12/24/48 turns, dump
  price, both farms' pending supply).
- Every past member of this family was measured at +$38 to -$992/game (A.4 #2).

## D. Staged plan (only if the operator wants closure)

This plan avoids the LESSONS.md traps: no tape future inside the search, no near-clone head-to-heads, 200+ worlds,
wins before coins, one pre-registered variant, and a placebo arm.

| Stage | What | Where / cost | Go / no-go |
|---|---|---|---|
| 0 (done) | Value-leaf probes (C.2) | Mac, under 10 s | **Failed.** Network leaves are ruled out as-is |
| **1a (cheapest decisive test)** | Placebo arm in a copy of `lookahead_ceiling.py`. Add two economically trivial alternatives: A0 with its largest SELL -1 unit and +1 unit (or 1 unit of the cheapest shed product). Run them non-sequentially at days 8, 12, 16 and 20 on the same 32 worlds, together with the real A1/A2 at the same decisions. Compare E[max(0, Δ)] and \|Δ\| | CPU box: about 92k policy rows per arm pair, about 5-10 core-hours (about 15 min on 32-64 cores); or about 0.5 GPU-hour | **Stop the sell-decision search for good** if placebo E[max(0, Δ)] ≥ 50% of the real options'. My prior: about 75% that it stops here |
| 1b (optional) | No-leak ceiling: same as Sep 26, but inside the rollouts the rival's market orders come from the inferred-habit model and shops are unknown (resample future unlocks). The main line stays vs the true tape. 32 worlds + 64 leader-test worlds | About 30-40 core-hours | Stop if the main-line gain is below +$1k/game or flips are below +2/96 |
| 2 | Deployable prototype (C.3) as a rule-layer module called from `features.midnight_fix`, so the harness, `bench_gov` and the export all run it. Paired vs the same model without it on the 464 + top:10 + top2:10 | About 1 day of engineering + about 20 core-hours (the 464 is about 9 core-h, the leader tests about 7) | **Go only if** 2400+ wins ≥ +3/219, the leader tests ≥ +5/260 combined with neither below -1, and the top-band paired margin ≥ +$500/g with z ≥ 2 |
| 3 | Package: `RULES` flag in `export.py`, `pkg_parity` 12/12 exact, `turn_timing.py` on an x86 box pinned to 1 core (max turn under 0.8 s, overage used under 10 s/game), `official_check.py` both seats | About 2 core-hours | Then the upload decision with the operator. It retires the older active submission; LESSONS says fix the pair early |

Timeline:

| Stage | UTC | PKT |
|---|---|---|
| 1a | Sep 28 00:00-06:00 | 05:00-11:00 |
| 2 | Sep 28 06:00-18:00 | 11:00-23:00 |
| 3 | Sep 29 00:00-12:00 | 05:00-17:00 |
| Upload decision, at the latest | Sep 29 18:00 | Sep 29 23:00 |

What would justify building the deployable version before the deadline: 1a shows the real options ≥ 2x the placebo,
**and** Stage 2 passes the thresholds above. Anything less is not worth an upload that retires a rated submission.

## E. Honest bottom line

- **Expected gain from decision-time search by Sep 30:**
  - Probability-weighted: about +$50/game (0-2 rating points).
  - If it succeeds: +$300-800/game, about +5-15 rating points.
  - It can go negative if the evaluator is biased: one wrong switch costs about $300-925.
- **Probabilities:**
  - About 60-70% that the Stage 2 prototype can be built and tested by Sep 29.
  - About 10-15% that it is positive on the operator's rulers.
  - **About 5-10% overall** that a verified positive version is uploaded by Sep 30.
- **Search is not the lever for the top 10.** Our losses are strategic (specialist wool/milk/wheat, about $10k less
  production). Search at decision time only works with an evaluator that is better than the policy's own judgement,
  and we do not have one.
- **What could make search real later (multi-day projects):**
  - An afterstate value trained on our own policy with cash counted exactly.
  - A 1-2M distilled rollout policy.
  - A rival-sales predictor validated on the 22.7k daily tapes.
  - A numpy full-state engine with determinized rival hidden state.
- **Side intel, weak evidence.**
  - Some top-10 teams spent 14-21 s of their overage in single sampled games: Majkel1337 had 39.4 s left, seek
    inspiration 41.8 s, Vadim 45.6 s (`data/recon/tapes`, older games, n=1 each). That hints at planning-heavy
    agents.
  - It can be checked cheaply by reading `remainingOverageTime` in a few recent replays of DSM and Boey on a
    server.

## Appendix: reproduce

```
cd /Users/asad.ali/kaggriculture/il/live
python3 lookahead_ceiling.py report 'g20_s012d8_sh*_srv'       # +4,379, hold -303, sell-all -925, 1,339 s/game
python3 lookahead_ceiling.py report 'g20_s012d1to7_sh*_srv'    # +2,281, median 0
python3 lookahead_ceiling.py combine 'g20_s012d1to7_sh*_srv' 'g20_s012d8_sh*_srv'   # +6,660
```

Value probes: `.venv/bin/python` with OMP=1 and nice 15.
- Replay a ladder world with `kagsim_trace.Game(seed)`, `set_forced_shops`, and the weed prefix, stepping the recorded
  actions of both seats.
- Take `observe(seat)` and build variants: +$1k cash; the sold shed at the exact `features6.price` curve, with the
  inventory raised and money held equal.
- Run `Net(...).forward(..., team=10, ret_hm=True)[3] * 2e4`:
  - IL: `live_rollout._net(ilsp_v13_p3.pt)`.
  - Critic: `Net(320, 10, 48, g_in=179, n_aux=6)` loaded with `jax1_final.pt['vstate']`.
- For the critic: estimate = margin_now + V (return-to-go).
