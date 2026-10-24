# Look-ahead and scheduling: why the leaders can, why we cannot, and a design that can

Written Sep 28 2026 (early morning PKT) for the operator's question of Sep 28: "why can we not look ahead /
schedule while our opponents have done it successfully?". Read-only research. Computation was light (1 core, nice 10,
about 6 core-minutes of exact C++ replays and zip streaming). Scripts are in the session scratchpad
(`.../scratchpad/plan/*.py`, listed in the appendix). Web sources are cited inline. Earlier review of decision-time
search: `il/live/LOOKAHEAD_REVIEW.md` (Sep 27); this document does not repeat it.

## 0. Bottom line

- **The leaders do look ahead inside their agents, and it shows in the data.** In the 577 Kaggle games of Sep 25,
  five top-15 teams (M&M&P&Q, THIRD FARM CLUB, mtmr_s1, 吃白饭的大肥鱼, Kaggledew) go over the 1 s limit at hour 0
  of almost every day after day 9-12. They spend 1.4-5 s per morning, 14-50 s of the 60 s overage pool per game,
  apparently re-planning the day's hires and seeds after a fixed opening. M&M&P&Q also spends about 33 s on its first
  turn. DSM, Boey, Vadim, UMG and Majkel spend 14-21 s once at the start and stay under 1 s after (2.1).
- **Publicly, the top 10 explain nothing** (one top-8 player says "RL"). Ryo Hasegawa's "1st place (previously)" post
  is about ratings and submission hygiene, not decisions. The public meta is recorded routes plus rules, with routers
  that pick routes by shop world; in-agent search is claimed only below the top 20. So the leaders' daily computation
  is most likely route/plan selection with a cheap model, not deep search (1).
- **Their plans are build orders, re-tuned between uploads.** On Sep 26 DSM (like M&M&P&Q) dropped the third land,
  planted a melon wave on the new NE land on day 6, bought a cow a day on days 2-4 and bought SW a day earlier.
  DSM's wins went 86% -> 89% and its margin +14.2k -> +21.3k (2.2).
- **Our live agent plays DSM's Sep 25 build order almost move for move**: the imitation works. But it never plants a
  second melon wave (0.3 late tiles per game against 2.5-5.2 for the leaders; rivals sell 76 melons per game to our
  62, -$2.6k), never buys early cows, and the SE guard forces the third land in 100% of games (2.2-2.3).
- **Why we cannot look ahead** (4): imitation rewards copying, not looking ahead; thresholded greedy decoding erases
  the conditional and minority macro moves, and the network cannot tell which version of a leader is current; the
  policy has no plan state; search over the 20M costs 25-250x the budget and our value heads misprice cash; and our
  explicit planners (E300-E311, E380, E487, HYB12) failed because they also had to run the units, not because of
  their look-ahead. A planner that only sets macro targets for a leader-level executor has never been built.
- **Comparable competitions (3):** economy and build-order games (Halite I-IV, Kore 2022, Lux S2) were won by
  hand-written planners with a stripped exact forward model and heuristic scoring, mostly in Python within 1-3 s per
  turn. Pure RL won where tactics and compute dominate (Lux S1/S3, Orbit Wars, Football), and search on top of a net
  gave modest gains (Orbit Wars 6th +30-40 points in 2-player only). The design below follows the economy-game
  pattern.
- **Design (5): macro planner + neural executor.** Plan 3-4 macro decisions per game (melon second wave, land plan,
  early cows, optionally geese) at hour 1 of days 2-13 with the pure-Python farm-and-market model we already own and
  verified (`planner/model.py`: exact prices, drains, crop and animal steppers, the rival's supply from its public
  tiles; melon price error $7 at +7 days). Inject the decisions through the rule layer like the SE guard; the network
  does everything else. Cost: 22-45 ms per trigger turn on the Mac, zero extra network forwards, zero overage.
- **Stage 0 preview, run today (5.5):** the model values the 1st-4th extra melon tile on day 6 at $838-605 net, where
  our policy plants wheat (about $0.2k per 5-day cycle). On DSM's Sep 26 games it is about 17% optimistic against the
  realised $685 per late tile. Predicted gain of 4 melons instead of wheat: median +$600 per game over 210 live games
  and +$1,163 in our 22 losses to 2,700+ opponents. These are model numbers, not measured gains.
- **Plan (6):** Route A = the leaders' Sep 26 plan as fixed rules (R1 melons, R2 no SE, R3 SW on day 8, R4 early
  cows, R5 all four), benched on the A100 box CPUs by about Sep 28 14:00 PKT. Route B = the planner chooses per game,
  by Sep 29 08:00 PKT, kept only if it beats the best fixed rule. Package by Sep 30 12:00 PKT, latest upload Sep 30
  17:00 PKT. Zero-cost check: bench the fresh3 IL models with the SE guard off, because the guard may fight the newer
  leaders' plan.
- **Odds (my judgement):** a fixed rule passes the top-band gates about 30-40% (then about +$0.5-1.5k per game,
  roughly +10-30 rating points); the planner beats the fixed rules about 15-25%; nothing passes about 55-65%.

## 1. What top players say publicly

Research by a sub-agent (Sep 28): 214 of the 217 forum threads with comments, the notebook lists (titles, votes and
tables of contents; notebook bodies do not render through the reader), GitHub and web search. Public pages only.
Rank badges on forum pages show today's rank, not the rank at posting time. D/ = https://www.kaggle.com/competitions/kaggriculture/discussion/ ,
K/ = https://www.kaggle.com/code/ , GH/ = https://github.com/ . [I] marks an inference.

**The top 10 say almost nothing.** Leaderboard on Sep 28: 1 M&M&P&Q 3,032.0, 2 DECEM 3,027.3, 3 Boey 2,999.8, 4 Vadim
Vasilenko 2,963.4, 5 UMG, 6 DSM, 7 Majkel1337, 8 Just A game on your lips, 9 Anton Tikhonov, 10 My second life.
- Boey, Vadim, DSM, Majkel1337, Anton Tikhonov, UMG, Just A game on your lips and My second life: no posts about method,
  no outside write-ups. The #1 team's only visible post (masspeaks, rank badge 1st, so probably M&M&P&Q [I], ~Sep 19)
  warns about a fake staff account (D/742035). DECEM only
  cheered under an RL post (D/743716).
- One top-8 player (Sayaka Miki, team not shown) confirmed the agent is RL, gave no details, and said she chose RL after
  rules plus search went badly for her in Orbit Wars (D/741792). Earlier she reported behaviour cloning on average
  replays as very poor (D/736567).
- **Ryo Hasegawa's "1st Place (previously) - Submission Strategy for beginners" is a discussion post** (D/736219) about
  Elo, the final Bradley-Terry fit and submission hygiene. It says nothing about how his agent decides, and he did not
  answer a question about RL versus rules there. (In our older recon tapes his one sampled game used about 44 s of
  overage; n = 1.)

**The public meta is recorded routes plus rules.**
- Top public notebooks (votes on Sep 28): boatlee V16-RC5 (320, score 2,913), ahmedberatozer V38 (240), kaitofukami
  "25/27 Strict-Future" (214, score 3,090: one collapsed opening plus a sparse controller), raykkretzschmar "Findings
  from Zero to Top Meta" (202), thomastschinkel "The 2945 Farm" (157: a recorded route with reflexes), tetsutani
  "Adaptive Farming" (145: demand chooses the route), yhay81 "Shop Router 0909" (135), "Six-Day Fieldbook" (131: 4
  decision points, 8 routes, 30 paths), "Three-Day Shop Router" (124). RL and MCTS notebooks have 5-20 votes. [I] None
  shows in-agent look-ahead.
- Players who measured the top: the Sep 1 #1 was fully adaptive (diverged from turn 0, 40-0, median +$13k) and was then
  withdrawn; the Sep 2 #1 was a near-fixed route per shop world with small repairs (0.6-3.9% of turns vary), forking at
  turns 72-144 on the unlocked shops (destbreso, D/739179). Top players were byte-identical over turns 1-51 in early
  August (onlysmrtsumx, D/733055); top agents were "essentially fixed 720-move policies" in late August (Michael Timbs,
  D/737955). 25 of 26 opponents around 2,400 matched public tapes on over 60% of farm turns (D/740437).
- Routers in the meta re-plan by switching routes: "Shop Router 0908" branches on the wool shop at day 6 and picks a
  sell method from the egg stock at day 27 (GH/lhzwb2008/kaggriculture); another router switches each morning to the
  closest of about 280 recorded games.

**In-agent look-ahead is claimed only below the top 20.** Michael Timbs (about 130th) added inference-time search to his
agent (D/742341) and says 1-2 s per turn and the 100 MB limit leave no room for deep MCTS (D/737801). Others: a rolling
3-day NPV simulation (D/734637), 5-turn plot intents plus an executor (D/738619), a depth-2 route look-ahead
(D/741935), MCTS at 40 iterations per turn, never submitted (GH/akmalkhaniub/kaggriculture-mcts-agent). Dropped: a
HiGHS solver (too slow, D/742005) and beam search (does not fit in 1 s, D/738325).

**RL and imitation below the top 10.**
- atsushi11o7 (16th), public repo GH/atsushi11o7/kaggriculture: behaviour cloning in JAX, then PPO self-play on a JAX GPU
  simulator; a transformer fills 25 unit and 10 market slots with rule-based legality masks; **a rule-based opening to
  day 11** copied from the public hybrid2965 agent with forced land buys on days 9 and 11; rule-based liquidation on
  days 28-29.
- Snorlax (41st-49th): cloning warm-up, then PPO over macro actions, about 300k games (D/741743).
- dzjiann: PPO with about 1M games on 196 cores peaked around the top 100, and was weaker against strong players than
  his own earlier DP and math-based agents (D/743716).
- Aaweg Bhaladhare (26th): flat end-to-end PPO has not matched a strong engineered policy; he favours a daily or phase
  plan plus an executor, and "search-generated value targets" (simulate a few full alternatives from real states and
  train a selector on the outcomes) (D/740792, D/743716).
- Gengsr (29th): hand-edited strategy, gave up RL after a week (D/740022).

**Market and opponent modelling.** hwe owe (8th around Sep 4): an ML model to time sales (D/738619). Yujin Cha
(D/743231): the public hybrid2965 agent changes its opening depending on the opponent's turn-0 orders; games between
forks of the same tape are decided by sale timing on days 17-25 (wool fell from 226 to 31 to 1 when both dumped);
splitting the day-27+ bulk sells into slices lost about $1k. Mid-ranked schedule notes: no melon sowing after day 8
(+$8.7k for that builder over 96 games, GH/yasumorishima/kaggle-competitions/pull/16); about 40 strawberry tiles by
day 14; land on days 6 and 9; 8 cows + 6 sheep.

**How this fits our measurements.** Nobody at the top says they plan inside the agent, yet five top-15 teams spend
1.4-5 s every morning (2.1). The public record points to the likely content of those computations: route selection and
repair keyed on shops and market state, and daily plans for crops and labour, not deep search [I]. It also supports
the pattern of our design: a fixed opening (atsushi11o7 to day 11, the leaders to about day 9), then macro decisions per
day, with rules at the end.

Side note, for information only: a reply in Ryo's thread (D/736219, posted about Sep 27-28, apparently answering a
comment of ours) says one of our agents belongs to the old public family "abo_v41" (identical for 251 steps, about
day 10). Our neural agent opens with the common meta opening, so a match over the first days is plausible [I].

## 2. What the leaders actually do (our measurements)

### 2.1 They plan inside the agent: time use in 577 Kaggle games

Source: the Kaggle daily episodes dataset of Sep 25 (`data/kaggle_daily/kaggriculture-episodes-2026-09-25.zip`, 577
episodes, streamed, nothing extracted). Every observation carries `remainingOverageTime`; it only falls when a turn
takes longer than the 1 s `actTimeout`. So this measures time **above** 1 s per turn. A team that plans inside 1 s shows
zero. Script: `overage.py`, `overage2.py`.

| Team (games) | First turn | In-game overage per game | When | Per planning call |
|---|---|---|---|---|
| M&M&P&Q (117) | **about 33 s** (60 -> 26.7 s left) | 13.7 s (median 17.4, max 55) | hour 0 (765 of 771 events), days 12-29 | about 2.2 s (1 s + 1.2 s) |
| THIRD FARM CLUB (23) | 0 | **39.9 s** (median 37) | hour 0 only, days 10-29 | about 2.9 s |
| mtmr_s1 (28) | 0 | **49.8 s**, ends with about 10 s left | hour 0 only, days 11-29 | 3.1-5.1 s, growing late |
| 吃白饭的大肥鱼 (55) | 2 s | 26.5 s | hour 0 (66%), days 12-28 | about 8 s on day 12, then 1-5 s |
| Kaggledew Valley (31) | 1.6 s | 16.6 s | hour 0 only, days 9-28 | 1.4-2.3 s |
| Fourth Quadrant (87) | 11-12 s | 0.9 s | scattered | under 1.7 s |
| DSM (129), Boey (87), Vadim (158), Unknown Mother-Goose (144), Majkel1337 (71) | **13.7-20.6 s** | 0.0 s | - | every turn under 1 s |
| DECEM (152) | 0 | 0 | - | every turn under 1 s |

Reading:
- Five teams from the top 15 run an **expensive computation at hour 0 of almost every day after day 9-12**, and they
  pay for it on purpose from the 60 s overage pool (1-4 s per day). The hour-0 and hour-1 orders are the day's hires
  (8-9 HIREs) and that day's seed list (e.g. mtmr_s1 day 13: WHEAT 8, TOMATO 7, STRAWBERRY 6). This looks like a
  **daily re-plan of labour and crops after a fixed opening**. That is an inference from timing and orders; their
  code is not public.
- DSM, Boey, Vadim, UMG and Majkel spend 14-21 s once, on the first turn, and never exceed 1 s afterwards. That is
  either a big model load or a one-off precomputation. It cannot tell us whether they plan per turn inside 1 s.
- Our package: 1.0-2.9 s load, 66-189 ms per turn (x86, `il/logs/server2/timing_*.log`). We leave about 0.8 s per
  turn and 57 s of overage unused.

### 2.2 Their schedules are build orders, and they change them between uploads

Exact replays (C++ `kagsim_trace`, receipts) of `data/tapes_top12` (3,859 top-12 games, Sep 23-26) and of our 97 live
NEURAL_20MW5_SE games. Mean executed actions per game by day. Scripts `macro_sched.py`, `melon_replay.py`,
`dsm26.py`, `ours_days.py`.

| Per game | Ours (20MW5_SE, live) | DSM Sep 25 | DECEM Sep 25 | **DSM Sep 26** | M&M&P&Q Sep 26 |
|---|---|---|---|---|---|
| Melon tiles day 0 / day 1 / day 6 | 6.0 / 4.0 / 0 | 6.0 / 3.8 / 0 | 6.0 / 4.0 / 0 | 6.0 / 1.9 / **3.9** | 6.0 / 2.9 / 2.0 |
| Strawberries days 2-5 / day 6 | 9.7 / 7.5 | 10.1 / 7.9 | 9.8 / 7.0 | 8.0 / **11.8** | 6.8 / 14.9 |
| Wheat day 6 / day 8 / day 9 | 4.4 / 3.6 / 13.6 | 5.2 / 4.2 / 12.7 | 5.6 / 4.3 / 12.8 | 2.6 / **14.8** / 4.4 | 1.9 / 13.1 / 4.5 |
| Cows bought days 2-5 | 0 | 0 | 0 | **3.3** | 3.2 |
| Lands (median days) | 3 (6.2 / 9.1 / 10.6) | 3 (6.2 / 9.1 / 10.5) | 3 (6.2 / 9.1 / 10.6) | **2 (6.1 / 8.2)** | 2 (6.1 / 8.2) |
| Tomatoes planted | 20.6 | 21.3 | 20.3 | **0.5** | 1.9 |
| All games of that team that day (all opponents) | 68-29 live | 86% wins, +14.2k (n=137) | 83%, +15.1k (n=261) | **89%, +21.3k** (n=188) | 81%, +16.9k (n=151) |

Build-order columns: n=97 (ours) and the first 60 games per team and day. Result row: every game of that team on that
day in `tapes_top12`.

- **Our live agent plays DSM's Sep 25 build order almost move for move** (also DECEM's and Vadim's): the imitation
  worked. On Sep 26 DSM switched to M&M&P&Q's Sep 24 plan: no third land (orders for a land on day 10+ in 85% of games
  on Sep 25, 0% on Sep 26), a melon wave on the new NE land on day 6, one cow a day on days 2-4, the SW land a day
  earlier (day 8.2), no tomatoes. M&M&P&Q made the same move (94% on Sep 24, 24% on Sep 25, 0% on Sep 26).
- The switch cost DSM nothing: 86% -> 89% wins, margin +14.2k -> +21.3k. Of the teams that kept the old plan, DECEM
  went 83% -> 71% and Vadim 63% -> 41%, while UMG held 79% -> 77%. Head to head, DSM beat DECEM 11-0 (+3.9k) on
  Sep 25 and 9-1 (+6.3k) on Sep 26, and Vadim 6-0 and 5-0 (`h2h.py`). The field changes daily, so this is evidence,
  not proof, that the new plan is better. The Sep 28 leaderboard still has DECEM 2nd (3,027) and Vadim 4th (2,963),
  and DECEM's own Sep 26 games already show a partial switch (land orders on day 10+ in 58% of games instead of 100%,
  late melons 1.4 per game instead of 0.1). Our tapes end on Sep 26; the Sep 27 games on the A100 box will show whether
  more teams followed.
- Our SE guard forces the third land in 100% of our games. It was validated on rulers built before this switch.

### 2.3 Melons: the clearest schedule gap, and it is exactly executed when planted

| Per game (exact replays) | Ours | DSM | M&M&P&Q | Boey | Majkel1337 | UMG | Vadim |
|---|---|---|---|---|---|---|---|
| Melon tiles planted | 10.3 | 11.3 | 13.1 | 12.4 | 13.2 | 12.5 | 10.1 |
| Units per tile | 6.02 | 6.00 | 6.02 | 5.97 | 5.90 | 5.89 | 5.99 |
| Units sold days 10-13 (avg price) | 59.5 ($209) | 51.8 ($226) | 51.3 ($223) | 59.8 ($212) | 62.0 ($213) | 60.4 ($214) | 60.0 ($217) |
| Units sold from day 14 on | 2.7 | 15.8 | 27.9 | 14.5 | 16.2 | 13.1 | 0.6 |

- The melon price is a shared, nearly non-draining resource: only the town centre buys melons (1 a day) and above the
  base inventory the price is 250 - 0.01 k^2 (k = units above 10,000). Both farms dump about 60 each on days 10-13,
  the book sits at +95..+125 for the rest of the game, and a second-wave melon sells at about $107-145. A tile of
  late melons still brings about $640-870 of revenue ($560-790 after the $80 seed) for about 10 unit-actions.
- The neural policy waters and harvests melons perfectly (6.02 units per tile). What it never does is plant the second
  wave: in live play it planted 0.3 late melon tiles per game, against 2.5-5.2 for the leaders.
- In our live games against opponents rated 2,700+ we went 11-22 (`loss_opp.py`, 33 games of 20MW5_SE and 20MW_SE).
  The biohub memory's cash decomposition of 8 recent losses: MELON -3.1k (volume), third field -2.5k (earned back),
  MILK -2.0k, WOOL -1.7k.

### 2.4 Where DSM beats a rival that plays the same build order

53 direct games of DSM against DECEM / Vadim / UMG (Sep 24-26, exact replays, `dsm_vs.py`): DSM 52-1, +$5,750 per
game. Sources: EGG +$2.2k (238 vs 189 units: more geese), MILK +$1.8k (same units, $111 vs $105 per unit), land
-$1.7k and seeds -$0.8k and hires -$0.6k cheaper, FERTILIZER +$0.8k, STRAWBERRY +$0.7k; TOMATO -$2.4k. Melons, wool,
wheat and carrots are even. So the edge is a sum of small macro choices (skip SE and tomatoes, more geese) plus better
milk sale prices, not one big trick.

## 3. What winners of comparable competitions did

Research by a sub-agent (Sep 28, public pages only, Kaggle pages read through r.jina.ai; ranks checked on Kaggle
leaderboards and CLIST). The numbers are as the writeups state them; "(inf.)" marks an inference.

| Competition (turn limit) | Top solution | Look-ahead | Forward model / evaluation | Language, per-turn compute | Reported gain from planning | Source |
|---|---|---|---|---|---|---|
| **Kore 2022** (3 s + 60 s) | 1st Harm Buisman: rule bot in 6 phases (defence, attacks, expansion, mining, spawning) | Scores almost every possible fleet route for each objective | Exact: fleets fly fixed plans, so a per-cell, per-future-turn damage table for both sides; heuristic values | Python (inf.); needed faster route search and caching to fit the limit | The planner is the whole agent | https://www.kaggle.com/competitions/kore-2022/discussion/340035 |
| Kore 2022 | 4th qihuazhong: rule-based task sequence | About 80k mining routes precomputed offline, about 30k scored per step | Per-turn damage-potential map; heuristic | Python, about 30k evaluations per 3 s step | - | https://www.kaggle.com/c/kore-2022/discussion/340157 |
| **Halite IV** (Kaggle 2020) | 1st Tom Van de Wiele: over 11k lines of rules, scores every target for every ship, builds a conflict-free plan; gave up deep RL after a month | 1-, 2- and N-step safety checks against worst-case replies | Local threat model; heuristic | Python | 16 of his 22 submissions would have placed 1st | https://www.kaggle.com/competitions/halite/discussion/183543 |
| Halite III / II / I | Greedy target assignment with economic projections (spawn or build a dropoff when the projected gain beats the cost) | Economic projections, no multi-turn simulation | Heuristic | Java / C++ / C# | Best Halite II ML bot about 62nd | https://github.com/teccles-halite/halite3-bot |
| **Lux AI S2** (2023) | 1st ry_andy_: stateful rule bot with roles and goals; units claim cells and targets | Projects all own units and factories 5-50+ steps ahead, in a loop that stops when time runs out | Simplified: opponent's lichen held fixed, its moves guessed only in local fights; heuristic | Python, about 2.9 s per turn | - | https://www.kaggle.com/competitions/lux-ai-season-2/writeups/ry-andy-1st-place-solution |
| Lux AI S1 (2021), S3 (2025) | Deep RL (IMPALA / PPO), no search | None | Neural net | PyTorch, 2-2.5 s per step (S1) | S1: their own rule bot lost to RL within a month | https://www.kaggle.com/competitions/lux-ai-2021/writeups/toad-brigade-toad-brigade-s-approach-deep-reinforc |
| **Hungry Geese** (2021; 1 s + 60 s) | 1st HandyRL: RL net + MCTS (decoupled UCT); 2nd Goosebumps: cloned policy/value net + PUCT | MCTS | Exact; a linear model for simulations and evaluation (1st) or the net's value (2nd) | 2nd: Python/PyTorch, 20-50 iterations per second, 0.93 s base, extends while the choice is unstable | Search fixed the net's blind spots; 1st and 2nd almost tied | https://www.kaggle.com/competitions/hungry-geese/writeups/goosebumps-goosebumps-solution-2nd-place |
| Hungry Geese | 11th: CNN choice x MCTS score | MCTS with playouts of at most 10 turns driven by a rule bot | Exact | 600 playouts per turn in Python, 4,000 in C++ | - | https://www.kaggle.com/competitions/hungry-geese/discussion/263655 |
| ConnectX | Negamax alpha-beta, bitboards, transposition table, opening book | Full solve | Exact | Python, about 0.3 s per move on Kaggle (0.08 s locally) | Plays perfectly | https://www.kaggle.com/c/connectx/discussion/188305 |
| **Orbit Wars** (2026; 1 s + 60 s) | 1st-3rd, 5th, 7th: pure self-play RL; 1st falls back to a 5M model when its time bank runs low | None | Neural net | CPU | - | https://tufalabs.ai/research/orbit-wars/ |
| Orbit Wars | 6th LeFiz: 5 sampled own actions x the opponent's argmax, 2 greedy steps | Rollouts | Exact env; value head | Not stated | **+30-40 points in 2-player**; worse on the 4-player board | https://www.kaggle.com/competitions/orbit-wars/writeups/6th-rl-league-search-with-a-custom-edge-atte |
| Orbit Wars | 13th: two checkpoints propose; if they differ, play each 2-3 turns | Rollouts, depth by remaining time bank | numpy engine copy; averaged value heads | About 20 ms per turn | Mixed: 1,620 with search vs 1,640 without on the board | https://www.kaggle.com/competitions/orbit-wars/writeups/13th-place-solo-gold-solution |
| Google Research Football (2020) | PPO + LSTM, league, GAIL; no search at any top team | None | - | - | - | https://www.kaggle.com/c/google-football/discussion/202232 |
| CodinGame "Witches' Brew" (2020, an ingredient economy) | 1st pb4git: beam width 400, about 30k simulations per turn; 2nd Agade: 10 ms beam to predict the opponent's plan, then its own beam against it | Beam search | Exact; heuristic (ingredient tiers, time-discounted potion value) | Compiled (Agade: C++), tens of ms | pb4git about +200 Elo in the last 5 days, nothing above 30k simulations; opponent prediction about +30% self-play win rate | https://github.com/pb4git/Fall-Challenge-2020 |

Classic references that fit a farm economy:
- **StarCraft build-order search** (Churchill & Buro 2011, the BOSS library): depth-first branch and bound, anytime,
  macro actions, and a fast-forward to the moment the next action becomes affordable (no "wait" actions). It plans in
  60-120 s increments at about 1.5% of the plan's own duration in CPU time, and its plans match professionals' lengths
  about 90% of the time. https://ojs.aaai.org/index.php/AIIDE/article/view/12435 . The 2019 continuous version
  maximises the area under a value curve instead of reaching a fixed goal: https://ieeexplore.ieee.org/document/8848109/
- **Search over scripts, not raw actions**: Puppet Search and Hierarchical Portfolio Search (Prismata, top 25% of the
  human ladder): https://ojs.aaai.org/index.php/AIIDE/article/view/12787
- **Rolling-horizon evolution** (40 ms per tick in GVGAI): https://arxiv.org/abs/2003.12331 ; **receding-horizon
  planning in a competitive supply-chain game**: https://arxiv.org/abs/2401.09853
- **AlphaStar** conditions its policy on a build-order statistic z sampled from human replays, with no search:
  https://www.nature.com/articles/s41586-019-1724-z . This is our team id, used as a style switch; a "plan id" is the
  imitation-friendly way to switch build orders.

What transfers to Kaggriculture:
1. **Economy and build-order games were won by hand-written planners** with a stripped but exact forward model and
   heuristic scoring (Halite I-IV, Kore, Lux S2), mostly in Python within 1-3 s. Pure RL won where tactics and compute
   dominate (Lux S1/S3, Orbit Wars, Football). Search on top of a learned net gave modest gains (Orbit Wars 6th
   +30-40 points in 2-player only; 13th mixed; Hungry Geese used it to patch the net's blind spots).
2. **Plan the macro choices only**, fast-forward between decision points (BOSS), and score a few options or scripts.
3. **Make the own side exact and the opponent simple**: fixed opponent assumptions (Lux S2 1st), 2-3 rival scenarios,
   or a cheap predictor (Agade's +30%).
4. **Heuristic values are the norm**: cash plus inventory at projected prices plus discounted future output.
5. **Anytime with a clock and a fallback** (13th, Goosebumps, Tufa's small-model fallback).
6. **Model quality beats simulation count** (pb4git: nothing above 30k simulations; Goosebumps: a 10x faster search
   gave little). Python is fast enough at the macro level.
7. **2-player favours search** (two Orbit Wars teams gained in 2-player and lost in 4-player). Kaggriculture is
   2-player.

Our design (section 5) is the economy-game pattern: an exact own-farm model, the rival's supply from its public
tiles, heuristic cash values, a handful of macro options, and the imitation policy as the fallback and the executor.

## 4. Why our agent cannot look ahead

1. **Nothing in our training rewards looking ahead.** Imitation predicts the teacher's next action. The 20M learned
   "what DSM, DECEM and Vadim did in this state", averaged over a data mix dominated by the Sep 25-era build order. It
   does receive the market model as input (features6: prices after +4/12/24/48 turns, dump price, both farms' pending
   supply), but only uses it to predict actions. Our RL runs on top (thousands to tens of thousands of games each)
   moved the 464 by -8..+2 wins (RL4, RL5, jax1_final against the 448 of their start). Orbit Wars winners needed
   10^9-10^10 steps to learn plans from reward.
2. **Thresholded greedy decoding keeps only the majority macro move.** A market head fires only when P(order) > tau
   (0.5). A move that the teachers make in some games, or only in their newest version, never fires. Examples: the
   second melon wave (ours 0.3 late tiles per game against 2.5-5.2 for the leaders), early cows (0 against 3.2-3.3).
   The network also cannot know which version of a leader is current: the data mixes several days and versions. The
   only way to change its plan is to retrain (hours, and +-10 wins of seed noise on a leader test).
3. **No plan state.** The policy is Markov: one observation in, one action out, plus the previous op. It cannot commit
   to "melons on NE on day 6" and carry it through except by what the farm happens to show. The leaders' daily
   planners keep a plan.
4. **Search over the network is unaffordable, and our evaluators are wrong** (`LOOKAHEAD_REVIEW.md`): 66-189 ms per
   20M forward on x86 leaves 6-15 forwards per decision against the 10^2-10^3 that rollouts need; the IL value head
   and the RL critic value +$1,000 of cash at +$415-636; the +$6.7k "ceiling" was mostly chaos-selection and tape
   leakage.
5. **Our explicit planners failed on execution, and we filed that as "planners do not work".** E300, E302, E304,
   E311, E380, E487 and HYB12 were whole-farm controllers: they also had to route and schedule every unit. They never
   reached farm scale (6.8 crops/day against r6's 39.8; 466-614 tasks against about 1,400; HYB12 46-154 against V94,
   then -$5k/game after its leaks were fixed). HYB12's own measurements say where the loss was: with the tape's units
   and the planner's sales the margin was -$1.3k / -$0.6k, with the planner's units about -$8k "whatever the sales
   policy" (`planner/planner.py` docstring). The market-and-farm model inside it (`planner/model.py`) passed every
   exactness check. **A planner that only sets macro targets for an executor that plays at leader level has never
   been built.** Before Sep 23 we had no such executor; now the 20M is one (it reproduces DSM's build order and gets
   6.02 units per melon tile).
6. **The leaders' look-ahead is cheap because it runs in the right units.** Crop growth and animal production are
   deterministic, the price function is public, the rival's crops and animals are visible, and shop unlocks are the
   only randomness. A day-level model of the own farm plus the book costs milliseconds per option. That is what a
   1-5 s hour-0 computation (section 2.1) buys: a few hundred to a few thousand option evaluations, with no neural
   value function and no rollouts of the opponent.

## 5. A design that can: macro planner + neural executor

### 5.1 Principle

Plan where the game is deterministic and cheap to simulate: days, tiles, herds, lands and the market book. Execute
where the network is strong: units, routes and per-turn choices. **Never search over the network.** This is the
leaders' pattern from section 2.1 (fixed opening, daily re-plan with a cheap hand model), except that our executor is
the 20M instead of their hand-written unit scheduler.

### 5.2 Components

| Part | What | Reuse / new | Cost per call (Mac; Kaggle x86 about 2-3x) |
|---|---|---|---|
| A. Macro forward model | Day-level projection of both farms and the book to day 29: exact price function, exact town drain for the known shops and the expected drain of future unlocks (`drain_path`, `shop_scenarios`), our tiles and animals through exact steppers (`crop_outcome`, `animal_outcome`, `project_farm_output`), the rival's supply from its public tiles (`rival_sales_schedule`) and its past sales read exactly from public book changes (`FlowTracker`), marginal revenue of extra units (`Forecast`, `value_of_crop_on_tile`) | **Exists**: `planner/model.py`, standard library only, about 1,600 lines. `tools/planner/test_model.py`: ALL PASS on 60 archive games; melon yield 0.996 of model; price error at +168 steps with the rival schedule: melon $7, milk $14, strawberry $17, wool $17 (melon $36 if the rival is ignored) | forecast 0.4 ms; rival schedule 0.3-1 ms; `rank_crops` over 5 crops 9-16 ms (measured today on 3 live states); J1: 3 plant days 57-110 ms |
| B. Policy profile | What the 20M does anyway: which tiles it plants per day and with what, units per tile (melon 6.02), hires, feed and wheat buys. The planner changes *what* is planted and bought, never *how much work* is done | New, from our replays (half a day) | offline |
| C. Planner | At hour 1 of days 2-13, evaluate a few pre-registered macro options (5.3) with A. Score = our projected final cash + 0.5 x the rival's projected revenue loss on shared markets (the game is won on margin). Evaluate under two rival scenarios (as visible; rival copies our change). Keep the policy's own choice unless an option gains >= $300-500 in both | New, about 300 lines | 22-45 ms per trigger turn on the Mac (5.5), so about 0.05-0.15 s on x86; 12 trigger turns per game |
| D. Injector | The rule layer, exactly like the SE guard (`features._plan_fix` already hosts SE guard, sheep cap, seed budget): (i) market orders: BUY_SEED for the planned crops one turn before planting (the market runs after the units, so seeds bought at step t can be planted from t+1); trim the policy's own seed orders it will no longer use; BUY_LAND now / later / never; BUY_ANIMAL; (ii) unit orders: swap the crop of PLANT orders the policy already issues (same labour), respecting the atomic PLANT rule (the SEED_BUDGET logic) | Extend `_plan_fix` behind a flag (MACRO=1) | under 1 ms per turn |
| E. Guards | Policy action first; planner only on trigger turns; stop planning after 0.4 s of wall time in the turn; skip if `remainingOverageTime` < 45 s; try/except returns the policy action; deterministic; no printing; flags live in `features.py`, so `live_rollout`, `bench_gov`, `export.py` and `pkg_parity` all run the same code | Existing pattern | - |

### 5.3 Pre-registered decisions

| # | Decision | Options | Evidence it matters |
|---|---|---|---|
| D1 | Crop upgrade on free tiles, days 2-10: melon second wave (and strawberry vs wheat/carrot) | k extra melon tiles in {0, 2, 4, 6}, planted where the policy plants anyway | Leaders plant 2.5-5.2 late melon tiles, we plant 0.3 (2.3); late melons sell at $107-145 (about $560-790 per tile after the seed); the model values the 1st / 2nd / 4th / 6th extra melon tile on day 6 at $838 / $766 / $605 / $429 net (median of 97 live states) and a wheat tile at about $0.2k per 5-day cycle (5.5); MELON is the largest cash gap in 8 analysed NEURAL_20MW8_SE losses (-$3.1k) and -$2.6k per game over all 97 live 20MW5_SE games (rivals sell 76 melons, we sell 62) |
| D2 | Land plan | SW on day 8 or 9; SE on days 10-13 or never (a conditional SE guard) | DSM and M&M&P&Q dropped SE on Sep 26 and bought SW on day 8.2 (2.2); our SE guard was +$288..+1,289 per game on the older rulers, and switching it off on v13_p3 (Sep 27) cost 2 + 1 leader-test wins and was neutral on the live losses. So no-SE only makes sense together with what DSM spends the $4,000 on (early cows, melons) |
| D3 | Early herd | one cow a day on days 2-4 when cash allows, or not | DSM Sep 26 and M&M&P&Q +3.2-3.3 cows by day 5, ours 0; MILK is -$2.0k in the 8 analysed 20MW8_SE losses |
| D4 (optional) | Geese on days 6-10 | policy, +1, +2 | DSM's EGG edge over same-plan rivals +$2.2k (2.4) |

Out of scope on purpose: sale timing (the market-planner family measured at most +$0.5k/game, and every realised
version was -$992..+$38, `LOOKAHEAD_REVIEW.md` A.4), hires (extra hands had zero marginal product in the scripted
era), unit routing (the network's job).

### 5.4 Per-turn budget on Kaggle (1.6 vCPU)

| Item | Time |
|---|---|
| 20M forward + decode (every turn) | 66-189 ms mean, max 260 ms (x86, 1 thread) |
| Rule layer (every turn) | under 1 ms |
| Planner (hour 1 of days 2-13 only) | measured on the Mac: 22-45 ms for forecast + `rank_crops` over 5 crops + 6 k-steps of extra melon tiles (5.5). Budget 0.1-0.3 s on x86 with a second rival scenario, D2 (one `rank_crops` over the SE tiles plus a labour check) and D3/D4 (`animal_outcome` against the milk/egg forecast) |
| Worst trigger turn | about 0.5 s, under the 1 s limit, **zero overage by design**. Headroom if a later version needs it: we leave about 57 s of the pool unused, so 1-2 s per morning for 20 days (what M&M&P&Q and THIRD FARM CLUB spend) would still keep a 20-35 s reserve |
| Package | `planner/model.py` inlined (standard library, about 70 KB more `main.py`; the HYB12 build already inlined it once) |

### 5.5 Stage 0 preview (run today)

`audit1.py`: the model at hour 1 of days 2/6/8/10 in our 97 live games, with the forecast built like HYB12's
(`Planner._forecast`: observed book, exact drain, the rival's schedule from its public tiles, **plus our own projected
harvests**). Marginal net value of one more tile, median:

| Day | Melon | Strawberry | Tomato | Wheat (one 5-day cycle) | Carrot (one cycle) | What our policy planted that day |
|---|---|---|---|---|---|---|
| 2 | $907 | $2,045 | $473 | $174 | $160 | 3.6 strawberry, 0.3 melon |
| 6 | $838 | $1,913 | $852 | $217 | $134 | 7.5 strawberry, **4.4 wheat** |
| 8 | $778 | $1,665 | $840 | $236 | $134 | **3.6 wheat**, 0.3 strawberry |
| 10 | $778 | $1,611 | $577 | $255 | $157 | 8.6 wheat, 2.3 strawberry, 2.0 carrot, 1.0 tomato |

- The k-th extra melon tile on day 6 is worth $838 / $766 / $687 / $605 / $520 / $429 (k = 1..6; p10-p90 of k=4:
  $429-838). The realised value of the leaders' late melon tiles is about $560-790 after the seed (2.3), inside that
  range: a first calibration check passed.
- A melon holds its tile for about 12 days; wheat fits 2-3 cycles in that time (about $450-650, plus its feed value)
  and needs about twice the unit-actions. So 4 melon tiles in place of 4 wheat tiles are worth roughly +$0.4-1.2k per
  game **in the model's eyes**, before the damage to the rival's late melon sales. That is the size of gain to test,
  not a measured one.
- On 60 DSM Sep 26 states the same model gives $907 / $838 / $766 / $687 for k = 1..4 (mean $800). DSM planted 3.9
  late tiles per game in those games, and they earned $765 of revenue, **$685 net per tile** (6.0 units at $127,
  `calib_late.py`). So the model is about 17% optimistic there: inside the 20% gate, and a reason for a switching
  margin. A threshold of about $600-700 per tile would reproduce DSM's choice (speculation).
- Predicted gain of D1 per game (`audit2.py`; own cash only; 4 melon tiles instead of wheat on day 6, wheat valued at
  2.4 cycles per displaced tile): median **+$600** over 210 live games (20MW5_SE + 20MW_SE); **+$1,163** in the 22
  losses to 2,700+ opponents (73% of them >= $500); +$990 in the 11 wins against 2,700+; +$803 against 2,400-2,700.
  Discounted by the 17% optimism above, that is roughly +$0.5-1.0k per game in the top band, if the network executes
  the extra melons as well as the first wave.
- Strawberries look even better in the model ($1.6-2.0k), but the strawberry price forecast is the least accurate one
  (MAE $17 at +168 steps), depends on random shop unlocks, and the policy already plants 34 per game (leaders about 30).
  Keep strawberries out of D1 until Stage 0 calibrates them.
- Cost: 22-45 ms per trigger turn on the Mac.
- My first run of this audit left our own harvests out of the forecast and valued the first extra melon at $1,356.
  Any production version must include them.

### 5.6 Why this is not a repeat of what failed

| Past failure | Cause | This design |
|---|---|---|
| E300-E311, E380, E487, HYB12 | The planner also ran the units and never reached farm scale | The 20M runs the units; the planner touches only purchases and the crop of PLANT orders already issued |
| Scripted opening + handover (DSMR0: 18-26/219 at 2,400+) | Six days of scripted play moved the farm off the network's training distribution | Each change is one the leaders themselves make in the training data (melons on NE on day 6, early cows, no SE), one at a time |
| Full RACE5 stack on R3 (358 -> 268) | Many layers at once | One decision per test, each paired, like the SE guard |
| Decision-time search (Sep 26) | 10^2-10^3 network forwards per decision; biased value heads | Zero extra forwards; the leaf is exact cash from a verified model |
| Bolt-on supply changes ("supply is interference") | They withheld, delayed or paced supply, and the rival sold into the gap | D1 adds supply to the one market where we leave about 15 late melons per game to the rival |

### 5.7 What it will not fix

- Specialist losses (wool, milk and wheat specialists: 43 of 75 live losses) are strategy match-ups. D3/D4 move the
  product mix only a little.
- The executor's own gaps (milk sale prices, fertilizer channel, labour efficiency) stay as they are.
- Rival re-planning: M&M&P&Q, THIRD FARM CLUB and mtmr_s1 re-plan every morning. None of our rulers let the rival
  react (section 7).

## 6. Plan to Sep 30

Deadline: Sep 30 23:59 UTC = Oct 1 04:59 PKT (about 75 h from 02:00 PKT Sep 28). The current pair is 56595847
NEURAL_20MW5_SE (82-45, 2,719 at 20:57 UTC Sep 27) and 56614508 NEURAL_20MW8_SE (53-12, 2,737). Any upload retires
56595847.

Two routes, run in parallel. Route A copies the leaders' newest plan (the output of their look-ahead). Route B builds
our own look-ahead. Both go through the same rule layer and the same rulers.

| Stage | When (PKT) | Where / cost | What | Go / no-go |
|---|---|---|---|---|
| 0 | Sep 28 03:00-08:00 (partly done, 5.5) | Mac, 1 core, about 1 core-hour | (a) **Offline audit**: run `planner/model.py` on our recorded live states at hour 1 of days 2-13 (97 live games + the 2,700+ losses): what D1-D3 would choose and the predicted gain; compare with what DSM Sep 26 / M&M&P&Q did in the same situations. (b) **Calibration**: the model's predicted revenue of late melon tiles at planting time vs the realised revenue in `tapes_top12` (leaders' day 2-10 melon tiles) | Continue Route B only if the prediction error of late-melon revenue is <= 20% (DSM Sep 26: +17%, passed), the predicted D1 gain is >= $500 per game in at least half of the 2,700+ losses (73%, passed), and the model agrees with the leaders' revealed choices in >= 60% of cases |
| 1 (Route A) | Sep 28 06:00-14:00 | Rule code 3-4 h; A100 box CPUs about 3 h (each variant about 30-40 min at 40 workers: 464 + old leaders + top-10 + live losses) | **Fixed rules**, each paired against v13_p3 + SE (the live 20MW8_SE): R1 melons: buy 4 melon seeds when NE is bought and plant them instead of the first 4 WHEAT or CARROT PLANTs on days 6-8 (our policy plants 4.4 wheat on day 6 and 3.6 on day 8; strawberries stay, the model values them highest); R2 no SE: SE guard off and the policy's BUY_LAND blocked from day 10 (guard-off alone was already tested on Sep 27: 98 -> 96 and 74 -> 73 leader-test wins, so R2 is expected to matter only inside R5); R3 SW on day 8 when cash allows; R4 one cow a day on days 2-4 when cash >= $400 + reserve; R5 = R1-R4 together (the DSM Sep 26 bundle). Zero-cost extra: bench the fresh3 IL models (n1/n3/specft on the A100 box) with SE_GUARD 0 and 1: newer data may already carry the new plan, and the guard may be fighting it | Operator rule (`judge-top-band`): 2,400+ wins >= +3/219; leader tests >= +5/260 combined, neither below -1; paired top-band margin >= +$500/game with z >= 2; live-loss replay flips positive; the 464 not below -2 wins. Check melon units per tile (about 6), escapes and dead plants in the test games |
| 2 (Route B) | Sep 28 14:00 - Sep 29 08:00 | About 8-10 h of engineering: `il/live/macro_plan.py` (C of 5.2) + injector flags; A100 box CPUs about 3 h | **The planner chooses D1-D3 per game** (thresholds fixed before the test). Unit tests: 200 recorded states with no exceptions; time per trigger turn <= 0.3 s on x86 1 core; identical actions in `live_rollout` and the exported `main.py` | Planner >= the best Stage 1 rule by +$300/game (z >= 2) or +3 top-band wins; otherwise ship the fixed rule: the look-ahead added nothing measurable |
| 3 | Sep 29 08:00 - Sep 30 12:00 | About 2 core-hours | Package the winner: `export.py` RULES/MACRO defaults flipped, `pkg_parity` 12/12 exact, `turn_timing.py` on x86 pinned to 1 core (max turn < 0.8 s, overage < 5 s per game), `official_check.py` both seats, 200 games with zero exceptions | Upload decision with the operator. If Stage 1 passes early (Sep 28 evening), upload then: more rated games before October |
| Latest upload | Sep 30 17:00 PKT (12:00 UTC) | | | |

Implementation notes:
- Keep the Stage 1 rules **stateless**: they read their state from the farm (e.g. "melon tiles planted since day 2"
  from the tiles' `planted_day`), exactly like the SE guard. Then they run unchanged in `live_rollout`'s lockstep
  batches, in `bench_gov` and in the exported `main.py` (`export.py` embeds `features.py` and flips flags through the
  `RULES` environment variable).
- The Stage 2 planner needs per-game memory (its daily plan, the `FlowTracker`). Key it by seat and reset it at step 0,
  and pass it explicitly in `live_rollout`, which plays many games in one process. `export.py` must also embed
  `planner/model.py`, the same way it embeds `features_ar.py`.
- Every injected purchase goes after the policy's own sells and hires and keeps a cash reserve, like the SE guard. PLANT
  swaps must respect the atomic PLANT rule: never more swapped PLANTs of a crop than seeds held at the start of the turn.

Optional Route A' (GPU, if the A100 has a free slot): fine-tune v13_p3 for one epoch on the Sep 26-27 games of DSM
and M&M&P&Q under a **new style id** (their new plan), SE guard off, and bench it like the Stage 1 rules. This lets
the network learn the new plan's unit work as well, instead of having it injected. It needs the fresh3 data already on
the box, about 1-2 GPU hours and the usual benches; seed noise (+-10 leader-test wins) makes a single run weak evidence.

Plan watch (cheap, recommended whatever else happens): run `macro_sched.py` / `dsm26.py` on each new batch of
top-10 games (the fresh3 pull on the A100 box, the Kaggle daily dataset) and diff the build orders by team and day
(20-60 s on one core). DSM's Sep 26 switch was visible in the data two days before this report; the leaders' offline
look-ahead is readable from their tapes.

What each stage costs if it fails: Stage 0 one core-hour; Stage 1 about 7 h of work and 3 A100-box CPU hours; Stage
2 about 10 h and 3 more. Stop Route B at the first failed gate.

Estimates (my judgement, not measured):
- Stage 1 (a fixed rule or the bundle) passes the top-band gates: about 30-40%. If it passes: about +$0.5-1.5k per game
  in the top band, roughly +10-30 rating points.
- Stage 2 beats the best fixed rule: about 15-25%, for about +$0.2-0.5k per game more.
- Nothing passes: about 55-65%. Then the pair stays as it is, and we keep a verified macro model and a measured answer
  to "does look-ahead pay here".

## 7. Risks

| Risk | How it would show | Containment |
|---|---|---|
| **Rulers cannot react.** The 464 worlds and both leader tests replay recorded opponents. A change that adds supply (more melons) is scored against rivals whose sales are fixed, while M&M&P&Q, THIRD FARM CLUB and mtmr_s1 re-plan every morning in live play | Gains on tape that shrink live | Prefer changes the leaders themselves adopted (proven live); require the top band and the leader tests, not the 464 average; count live-loss flips both ways; watch the first 40-60 live games of an upload |
| **Rulers are older than the meta.** The 464 worlds and the old-leader test (138 worlds, Sep 23-25) predate DSM's Sep 26 switch. The top-10 test (fresh2, 122 worlds) has 52 Sep 26 worlds, of which only 17 face the new plan (10 DSM, 7 M&M&P&Q) | A rule tuned to the old field | Weigh the fresh2 leader test and the live-loss set most; build a fresh3 leader test from the Sep 27 games already on the A100 box if time allows |
| **Closed-loop drift.** The network meets farms it rarely produced (melons on NE on day 6, cows on day 2) | Fewer than 6 units per melon tile, escapes, dead plants, idle units | These states exist in its training data (DSM Sep 26, M&M&P&Q). Measure yields, escapes and dead plants in every test |
| **Model bias.** The macro model ignores the rival's future plantings and our own labour limits | The planner over-plants melons | Two rival scenarios, a $300-500 switching threshold, the policy's choice on ties, and the Stage 0 calibration gate |
| **Cash.** Early cows and extra seeds compete with the day-6 land and animals | Missed NE purchase or animals | Every injected purchase keeps the policy's own purchases funded first (reserve), as the SE guard does |
| **Time-outs.** A slow trigger turn on the 1.6 vCPU machine | Overage use; a TIMEOUT loses the game | 0.4 s wall-time stop, overage floor of 45 s, try/except returning the policy action, x86 timing test |
| **Upload churn** (LESSONS) | A new submission restarts at 600 and retires 20MW5_SE | Upload only after all gates pass; if Stage 1 passes, upload early rather than late |
| **The history** says that planners, bolt-on supply changes and neural + script hybrids lost | The same thing happens again | Section 5.6: this design keeps the network as the executor, changes one decision at a time and adds supply instead of withholding it. It is still the least-proven part, hence Stage 0 and the fixed-rule baselines |

## Appendix: reproduce

All scripts are in `/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/plan/`
(session-local). Run them with `/Users/asad.ali/kaggriculture/.venv/bin/python` (arm64; the system python3 is x86 and
cannot load `kagsim_trace`), `OMP_NUM_THREADS=1 nice -n 10`.

| Script | What | Time |
|---|---|---|
| `macro_sched.py` | Orders per day by team from the action tapes (no replay): melon waves, plants, land, animals, hires | 17 s |
| `melon_replay.py 80 rep80.json` + `ana1.py`, `ana2.py` | Exact C++ replays of 80 games per leader + our 97 live games: units, prices, revenue by product, melon sales by day, book, land days | 11 s + 1 s |
| `se_bydate.py`, `dsm26.py`, `h2h.py` | Plan changes by date (third land, late melons), DSM / M&M&P&Q / DECEM build orders Sep 24-26, head-to-heads | 30 s |
| `dsm_vs.py` | DSM against same-plan rivals: product decomposition of the margin | 3 s |
| `loss_opp.py`, `ours_days.py` | Our live games by opponent rating band; our executed build order | 10 s |
| `overage.py`, `overage2.py` | Overage use per team from the Sep 25 Kaggle daily dataset (zip streamed, nothing extracted) | 75 s each |
| `bench_model.py` | Timing and crop values of `planner/model.py` on three live states | 2 s |
| `audit1.py` | Stage 0 preview: marginal crop values with our own output in the forecast (97 live games, 60 DSM Sep 26 games); k-th extra melon tile | 60 s |
| `calib_late.py` | Realised revenue of the leaders' late melon tiles (days 2-9) | 20 s |
| `audit2.py` | Predicted D1 gain per live game by opponent band (imports `audit1.py`, which re-runs its own report first) | 2 min |
| `audit0.py` | The first, wrong audit (forecast without our own harvests): kept only to show the error | 30 s |

Web research notes: `.../scratchpad/forum_research/notes.md` plus per-thread excerpts (forum sub-agent). The
competition sub-agent wrote no notes; its WebFetch calls saved two papers automatically into the session's
tool-results folder (Churchill & Buro 2011; Chan et al. 2007).
