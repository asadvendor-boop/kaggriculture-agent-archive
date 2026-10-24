# Winner-imitation fine-tune of v13_p3 — RUNBOOK (training box, Sep 29)

Goal: fix live weaknesses (wool in yarn towns, melon timing, eggs, milk per cow, cows/strawberries in demand towns) by
imitating the WINNER of each of our own live games: games we won → our own moves; games we lost → the opponent's moves.
Low LR, ≤ 2 passes, rehearsal of v13_p3's own training data, checkpoints every half pass, then the usual fair tests on the
test box (fresh3 = today's top-7 games, the 464 + its 2400+ band, reactive head-to-heads). Nothing here uploads anything.

Scripts (this folder; on the box it lives at `$K/il/ilwin/`, `K=/Users/asad.ali/kaggriculture`):

| file | what |
|---|---|
| `build_winner_ds.py` | tapes → training seats (exact `build_dataset3.py` + `srv_build_x6.py` format, label hygiene) |
| `train_winner.py` | `il/train6.py` + rehearsal groups, half-pass checkpoints, ignore labels, team override |
| `agree_check.py` | drift gauge: per-head decision agreement new vs v13_p3, with weakness slices |
| `finetune_winner.sh` | box driver: build (once) → train → agree checks; log `/root/winft.log` |
| `push_winner_box.sh` | Mac → box copy of scripts, tapes, rehearsal subset |
| `make_reh_subset.py`, `reh_lists/` | the rehearsal subset lists (already made: 1.39 GB) |
| `pkg_src/` | the rule-layer source embedded in each live package (ST2, SOH, 20MW8_SE, NPSE, NPS, 20MW5_SE) |

## 1. How v13_p3 was made (from `il/logs/h100/ilsp_v13_p3.log`, `il/logs/h100_final2/root/il_sprint.log`)
- Trainer `il/train6.py` on the 8x H100 box, Sep 27 14:12–14:55 UTC. Common env:
  `BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 CHUNK=100 VAL_MAX=180 AUX_W=0.1 LOAD_THREADS=4 PREFETCH=1`, width 320, BS 512,
  AdamW(wd 1e-4) + OneCycleLR(pct 0.05), no AMP.
- Stage 1 (“p3”): `INIT=ckpt/g20_w5_s0.pt` + **3 passes over `il/data/fresh2`** (8,440 train seats) at **MAX_LR 1.5e-4**, seed 0,
  `RATING_W=il/data/rating_w_fresh2.json` → `ilsp_v13_p3_p.pt` (35,556 it, 38 min, val 0.756).
- Stage 2 (“FT2”): **2 epochs over `il/data/dsm_ft2`** (1,502 DSM seats, team 10) at **MAX_LR 2e-4**, seed 0, no rating weights
  → `ilsp_v13_p3.pt` (4,218 it, 4.5 min, val 0.4275 = last). Checkpoint: width 320, blocks 10, n_teams 48, g_in 179, n_aux 6, feat 6.
- Played live with TEAM=10, TAU=0.5, greedy + rules. ST2 = `SE_GUARD=1,STRADDLE=2` + `SELL_ORDER=held` (+ MIDNIGHT/FINAL_FIX/HOMING).
- Speed reference: 64 ms/iteration (BS 512) on an H100 NVL.

## 2. Data (winner seats), counts on the Mac at Sep 28 23:05 PKT (ST2 tapes still growing)
All = every game; rec = recommended filters (opponent rating ≥ 2,400, |margin| ≥ $500). W = we won (our moves), L = we lost (their moves).

| tape dir (what) | all W / L | rec W / L | used in the recommended set |
|---|---|---|---|
| tapes_st2 (live ST2 = v13_p3 + held + SE + STRADDLE=2) | 57 / 22 | 33 / 22 | W + L |
| tapes_soh (v13_p3 + held + SE) | 68 / 39 | 45 / 37 | W + L |
| tapes_n20mw8 (v13_p3 + SE, canonical) | 85 / 44 | 57 / 42 | W + L |
| tapes_npse (new-plan net + held + SE) | 66 / 22 | 40 / 20 | L only |
| tapes_nps (new-plan soup, SE off) | 44 / 8 | 17 / 7 | L only |
| tapes_n20mw5 (older 20MW5 net + SE) | 98 / 60 | 70 / 57 | L only |
| tapes_neural (Sep 24–25 nets, weaker field; 30 games are 464 worlds) | 130 / 56 | 56 / 27 | no |
| tapes_stack1 / race / ladder (scripted, Sep 20–24) | — | — | **never**: they are the 464's source games |
| tapes_today / tapes_top12 (leader games) | — | — | no (fresh3 source; the newplan lesson) |

Recommended set = **135 won games (our moves) + 185 lost games (their moves) = 320 games ≈ 230k samples** (719 steps each).
Why: all dirs used are Sep 27–28 (current meta; the whole set is small anyway, so volume from older dirs would mostly add
old-meta/weak-field games). Wins only from the v13_p3 family: a win of another net would pull v13_p3 toward that net
(the new-plan net lost the tomato conditional and was 111 vs 131 on fresh3). Losses from every recent net: those opponents
beat us in states our nets share. Wins vs < 2,400 are the rating-ramp games (nothing to learn). All 464 / fresh / fresh2 /
fresh3 / fresh_heldout episodes are excluded by the builder automatically. **The llall ruler (our live losses) IS in the
training data → it is contaminated for these candidates: do not judge on it.**

## 3. Label hygiene (what the builder does; all verified in the dry run)
Format: identical to `il/build_dataset3.py` + `srv_build_x6.py` — checked byte-for-byte on a fresh2 seat (all 9 arrays + x6
+ meta[0:8] exact, `test_format_parity.py`).
- **STRADDLE legs removed exactly (ST2 wins).** In STRADDLE=2 mode the legs go LAST in the order list (FIRST only once the
  rival is seen straddling); e.g. ep 114804416 step 300: `[SELL WHEAT 4, BUY_SEED WHEAT 1, BUY_PRODUCT WHEAT 79]`, step 301
  `[..., SELL WHEAT 79]`. The builder inverts the package's own `straddle_fix` (code extracted from the live ST2 package)
  step by step: candidates “no leg / last order / first order” must reproduce the tape's list exactly AND leave a list the
  policy could have emitted (decode grammar: one order per head, wheat BUY in {1..16, 35}, SELL in {1..16, 1000} except at
  hour 23 / step 718) — without the grammar, “the policy bought 79 wheat itself” is also self-consistent (first dry run:
  0 legs found, verify 750‰); ties keep ≤ 8 branches, more legs first. Labels = the pre-straddle list; the input features use `straddle_obs` (what the live policy saw: shed without
  the straddle wheat, cash + sell-back value) — otherwise the net would learn “don't sell the extra wheat”.
- **SE-guard BUY_LAND dropped** (default `--se-label drop`): at each step where the guard could fire (3 quadrants, day 10
  h≥12..day 13, cash ≥ $4,300) and the list holds a BUY_LAND, v13_p3 is run on that exact state; if the policy did not
  order the land, the label is `midnight_fix` with SE_GUARD=0 (exact list without the rule's land). Reason: SE stays the
  rule layer's decision (SE_GUARD, maybe SE_SHOPS=2 tomorrow); a net that learned “buy SE on day 10” would bypass SE_SHOPS.
- **Kept** (consistent with the rules we keep): SELL_ORDER=held (slot order is invisible to the 21 quantity heads),
  MIDNIGHT overflow sells (the rule only adds what is still over after the net's own sells → no double selling),
  HOMING/FINAL_FIX (idempotent end-of-game rules; the last turn's market is overwritten at play time anyway).
- **Opponent straddlers** (5–8 games per dir): BUY_PRODUCT WHEAT ≥ 10 at a drain step + SELL WHEAT ≥ half of it next step
  → those two market heads are IGNORED (label -100): our STRADDLE rule already does it; learning it too would double it.
- **Failed orders** (engine receipts, both seats): unit ops the engine did not execute → op/qty -1 (ignored; ~1.5% ours, ~3%
  theirs: duplicate WATER/CARE, empty HARVEST…); market heads whose orders were ALL rejected (INSUFFICIENT_CASH, SHED_FULL,
  LAND_EXHAUSTED, UNIT_CAP, MALFORMED, DEAD_SLOT) → -100 (ignored). Phantom SELLs (SHED_EMPTY, ~38% of our orders) keep
  their requested quantity — standing sell orders are the leaders' style v13_p3 was trained on. Only the first 10 orders
  count (the engine ignores the rest). Steps without a recorded action → ignored.
- Full-pipeline check (`--verify 1`): v13_p3 is replayed on every step of our won games and the package's rules applied;
  the log reports the share of steps where policy + rules reproduce the tape exactly (dry run: see §11).

## 4. Team labelling — recommendation: (a) team 10 for every winner seat
- (a) `WIN_TEAM_MODE=ours`: the winners' moves train the conditional we PLAY with (team 10). Direct effect on our play;
  risk = style averaging, bounded by the rehearsal (DSM seats are team 10 too) and our own wins (self-distillation).
- (b) `WIN_TEAM_MODE=true N_TEAMS=49`: opponents keep their vocab id (67 of 185 losses) or a new generic id 48 (mean
  of the table + noise); our own moves stay team 10. The team-10 play changes only through the shared trunk: weaker
  intended effect, yet the Sep 28 specft run (true ids, specialists' wins) still lost leader strength (98→86, 74→67).
  Use (b) only as a control if the box has time.

## 5. Copy to the training box (Mac upload is slow; sizes)
1. `bash il/live/srv_bootstrap5.sh <port> <host>` (≈0.5 GB: engine source + build + parity, il code, v13_p3, tapes, rulers).
   On a pure training box you may skip its kaggle-environments part; it does not hurt.
2. `bash <this folder>/push_winner_box.sh <port> <host>`: scripts (0.4 MB), tapes_st2 + fresh copies of the 5 other live
   tape dirs (≈13 MB), rows_*.json, rating_w_fresh2.json, and the **rehearsal subset 1.39 GB**: `il/data/reh_fresh2`
   (3,000 fresh2 seats incl. all 224 of its validation seats, 0.57 GB), `il/data/reh_dsm2` (all 1,537 dsm_ft2 seats,
   0.29 GB), their v6 features (4,537 x6 files, 0.53 GB). At ~2 MB/s ≈ 12 min. (Full fresh2 + dsm_ft2 would be ~3.0 GB.)
   Needed checkpoint: only `il/ckpt/ilsp_v13_p3.pt` (77 MB).
3. Setup: the Vast PyTorch image already has torch + CUDA; bootstrap5 installs `pybind11 numpy scipy orjson` and builds
   `engine/v2/kagsim_trace*.so`. Check: `cd $K/il && /venv/main/bin/python -c "import sys; sys.path.insert(0,'../tools'); import harness, torch; print(torch.cuda.is_available())"`.

## 6. Commands on the training box
```bash
K=/Users/asad.ali/kaggriculture; cd $K/il/ilwin
# A (recommended): team 10, LR 5e-5, 2 passes, rehearsal 1.0 x fresh2 + 0.5 x dsm_ft2 per winner seat, saves every 1/2 pass
setsid -f bash finetune_winner.sh a > /root/winft_a.out 2>&1 < /dev/null
tail -f /root/winft.log
# B (stronger, newplan-strength): 1 pass at 1e-4 (reuses the built data)
setsid -f bash finetune_winner.sh b MAX_LR=1e-4 PASSES=1 > /root/winft_b.out 2>&1 < /dev/null
# C (control, true team ids): setsid -f bash finetune_winner.sh c WIN_TEAM_MODE=true N_TEAMS=49 > /root/winft_c.out 2>&1 < /dev/null
```
Run A first; B/C only after A finished (one GPU job at a time; the data build happens once, in A).
The build alone: `/venv/main/bin/python build_winner_ds.py --out $K/il/data/win29 --tapes <dirs as in finetune_winner.sh> --min-opp-rating 2400 --min-margin 500 --list-only`
prints the selection. Other knobs: `--since 2026-09-27T12:00` (recency), `MERGE_FLAGS="--margin-weight loss"` (weight
losses by margin/5k, clipped 0.5–2), `--w-loss 1.5`, `--se-label keep`, `--exclude-llall 1` (keeps llall clean, loses 128 losses).
Outputs: `il/ckpt/ilsp_v13_win_a_h1.pt … _h4.pt` (= 0.5, 1, 1.5, 2 passes; `ilsp_v13_win_a.pt` = `_h4`), logs
`il/logs/winft_a.log`, agreement reports `il/ckpt/ilsp_v13_win_a_h*_agree.{log,json}`.

## 7. Expected times on an RTX 4090 (single GPU, 24 GB; A100 ≈ 1.3–1.5x faster)
- Build (320 games, 8 workers): measured on the Mac 0.35 s/game without the full verify, ~10 s/game with it on 2 CPU threads;
  on the box with the verify on the GPU ≈ 1–3 min in total.
- Training A: per pass 320 winner + 480 rehearsal seats ≈ 1,120 iterations (BS 512); 2 passes ≈ 2,250 it. At ~0.13 s/it
  (≈2x the H100's 64 ms) ≈ 5 min + chunk loading/validation ≈ 4 min → **≈ 10 min**. GPU memory ≈ 6 GB.
- Agreement checks: 4 checkpoints + 1 calibration x ~30 s ≈ 3 min. **Whole run A ≈ 15 min; A + B + C ≈ 40 min.**

## 8. Reading the output (before spending test-box time)
- `winft_a.log`: `[save k]` lines give winner-val and rehearsal-val loss/accuracy. Rehearsal val (v13_p3's own leader
  validation seats) must stay close to its start value (`[start]` line): a rise > ~0.02 = forgetting → prefer an earlier `_h`.
- `*_agree.log`: the first run also writes `== CALIBRATION: newplan vs v13_p3` = the drift of the Sep 28 plan-changing
  fine-tune (1 epoch @1e-4; lost the tomato conditional, fresh3 111 vs 131) — a level that was TOO MUCH. A good candidate
  keeps `[reh]` (leader states) agreement clearly closer to 1 than newplan's (rough guide: op ≥ ~0.97, all-market-heads
  ≥ ~0.90), while `[win]` shows its changes concentrated in the slices (yarn → BUY_ANIMAL:SHEEP fire rate, milk → COW,
  eggs → GOOSE, melon → melon seeds, straw → strawberry seeds), moving toward the label fire rate printed next to it.
- Send to the test box: `_h2` (1 pass) and `_h4` (2 passes) of A (+ `_h2` of B if its drift is still small).

## 9. Hand-off to the test box
Copy: `rsync -a il/ckpt/ilsp_v13_win_a_h2.pt il/ckpt/ilsp_v13_win_a_h4.pt root@<test>:$K/il/ckpt/` (box-to-box or via the Mac).
On the test box (srv_bootstrap5 state; the Sep 28 v13_p3 ST2 baselines `logs/*ilsp_v13_p3_t10_soh_st2*.json` are there):
```bash
cd $K/il
# all rulers with the ST2 rules (SE_GUARD=1 SELL_ORDER=held + STRADDLE=2), GPU rollouts
bash live/srv_gpu_chain.sh ilsp_v13_win_a_h2 soh "st2|STRADDLE=2"
bash live/srv_gpu_chain.sh ilsp_v13_win_a_h4 soh "st2|STRADDLE=2"
# paired vs the live ST2 (v13_p3 + same rules); judge on fresh3 + 464 (2400+ band); ignore llall (contaminated)
for m in ilsp_v13_win_a_h2 ilsp_v13_win_a_h4; do for d in fresh3 fresh fresh2; do
  /venv/main/bin/python live/pair_cmp.py logs/top_${d}_ilsp_v13_p3_t10_soh_st2.json logs/top_${d}_${m}_t10_soh_st2.json; done
  /venv/main/bin/python live/pair_cmp.py logs/ladder_ilsp_v13_p3_t10_r3_tau0.5_soh_st2_test.json logs/ladder_${m}_t10_r3_tau0.5_soh_st2_test.json; done
# reactive head-to-heads (same seeds 900000.. as the Sep 28 st2 baselines h2h_st_st2_vs20mw8 / _vsV94)
env SE_GUARD=1 SELL_ORDER=held STRADDLE=2 NW=20 LOCKSTEP=4 /venv/main/bin/python live/h2h_rules.py ckpt/ilsp_v13_win_a_h2.pt $K/agents/ref/V94_2945_main.py 40 win_a_h2_vsV94
env SE_GUARD=1 SELL_ORDER=held STRADDLE=2 NW=40 LOCKSTEP=2 /venv/main/bin/python live/h2h_rules.py ckpt/ilsp_v13_win_a_h2.pt $K/package/NEURAL_20MW8_SE/main.py 40 win_a_h2_vs20mw8
/venv/main/bin/python live/pair_cmp.py logs/h2h_st_st2_vs20mw8.json logs/h2h_win_a_h2_vs20mw8.json
# most direct: vs the LIVE ST2 package itself
env SE_GUARD=1 SELL_ORDER=held STRADDLE=2 NW=40 LOCKSTEP=2 /venv/main/bin/python live/h2h_rules.py ckpt/ilsp_v13_win_a_h2.pt $K/package/NEURAL_20MW8_SOH_ST2/main.py 40 win_a_h2_vsST2
# package a winner (never uploads; the operator uploads; FIFO retires the OLDER active submission SOH 56631711)
bash live/srv_mkpkg.sh ilsp_v13_win_a_h2 NEURAL_20MW8W_ST2 "SE_GUARD=1,STRADDLE=2" "SELL_ORDER=held" logs/ladder_ilsp_v13_win_a_h2_t10_r3_tau0.5_soh_st2_test.json
```
(`srv_gpu_chain.sh`'s own “paired vs soh” lines compare against the new checkpoint's missing base run and stay empty —
use the pair_cmp lines above. Set `ROLLOUT_DEVICE=cuda` for h2h_rules.py on a GPU box if its CPU is weak.)
Accept only if fresh3 wins do not drop, the 464 2400+ band holds (≥ ST2's 210) with a non-negative margin, and the reactive
h2h vs the ST2 package is ≥ 50%. Remember Sep 27–28: offline gains did not transfer live twice.

## 10. Caveats
- The new signal is small: ~185 lost games (~130k samples) of opponents' moves; the ~135 won games are mostly
  self-distillation (v13_p3's own greedy choices) + kept rule edits. Expect modest changes; `agree_check` shows where.
- Precedent: specft (v13_p3 + 1 epoch @1e-4 on specialists' winning seats, no broad rehearsal) cost leader strength
  (98→86 / 74→67) and did not fix live losses. This run differs (LR 5e-5, 60% rehearsal, our own wins as anchor,
  half-pass checkpoints) but the same failure mode is the main risk → fresh3 + 464 + h2h decide, not the loss replays.
- Val loss does not rank playing strength (Sep 27); checkpoints are chosen by the rulers.
- Opponent straddle detection is heuristic (labels ignored, not rewritten). Opponents' same-step wash trades stay as labels.
- Near the game end (steps ≥ 690) prev-op features follow the tape (post-HOMING) while live play feeds the pre-rule ops —
  same convention as the original leader data; affects < 30 steps per game.
- Rehearsal subset = 3,000 of 7,948 non-DSM fresh2 seats (random, seed 29) + all dsm_ft2: enough for ≤ 2 passes without
  repeats (A draws ≈ 960 rehearsal seats per 2 passes).
- The scripts live in the Mac scratchpad (`/private/tmp/...`, survives until reboot): copy the folder into the repo
  (e.g. `il/ilwin/`) if it must persist; `pkg_src/` must travel with `build_winner_ds.py`.

## 11. Mac dry run (Sep 28 23:10–23:17 PKT, one process at a time, nice 15, 4-process cap respected)
Folder `dryrun/` (logs `build.log`, `rest.log`, `fix.log`, `se.log`; post-fix seat sets `ds_fix/`, `ds_se/` kept; the pre-fix set and big files deleted).
- Format parity vs an existing fresh2 seat (tapes_top12 ep 113672447 seat 0): all 9 arrays + x6 + meta[0:8] byte-exact.
- Build 3 games (ST2 win, SOH loss, N20MW8 win) with `--verify 1`: 22 s. **N20MW8 win: v13_p3 + the package's rules reproduce
  719/719 steps exactly.** ST2 win: first version found 0 straddle legs (verify 750‰) → grammar fix → re-run on 2 ST2 wins:
  **1,438/1,438 steps exact, 178 + 180 legs removed** (≈ 89 round trips per game; census ≈ 85). 90 / 55 failed unit ops masked,
  55 / 31 market heads ignored per game. SOH loss (opponent 2,633): 48 unit ops masked, 1 head ignored, true team = generic 48.
- SE path (12 SOH/N20MW8 wins, 4 s): 8 land buys in the guard window → 5 by the policy itself (kept), 3 by the guard
  (removed; exact reconstruction with SE_GUARD=0 in all 3).
- Training (CPU, BS 16, 6 steps, rehearsal read in place from il/data/fresh2 + dsm_ft2): 17 s; half-pass saves `_h1` (it 3),
  `_h2` (it 6) + final; rehearsal val 0.940 → 0.943 → 0.946 (256 states).
- Play: the new checkpoint through `il/live_rollout.run_job` (ST2 rules from env) vs V94 on seed 900001: +17,449 in 12 s
  (v13_p3 on the same world +14,333) → loads with the same Net() arguments (width 320, blocks 10, teams 48, g_in 179).
- agree_check (400 states per set, 23 s): op agreement 0.994 (win) / 0.995 (leader val), all 21 market heads 0.948 / 0.940
  after only 6 tiny steps → the thresholded market heads are sensitive; judge drift against the newplan calibration.
- Export (`il/export.py`, B64=1, RULES `SE_GUARD=1,STRADDLE=2`, RULES_STR `SELL_ORDER=held`): 93.9 MB main.py, loads the Kaggle
  way (`harness.mk`) and plays 30 turns → the checkpoint is package-ready for `srv_mkpkg.sh`.

