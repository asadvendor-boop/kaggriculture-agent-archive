# kagjax/train: self-play PPO in JAX (part D)

Goal: large-scale self-play on GPU(s), warm-started from our best imitation model, exported to the Kaggle numpy
runtime with the existing `il/export.py`. Everything runs on the accelerator: B games in lockstep under `jit`,
trajectories stay on device, the PPO update runs on device.

## Files

| file | what |
|---|---|
| `ppo.py` | the trainer: config (env vars), rollout (`lax.scan` over steps), PPO update (`lax.scan` over epochs x minibatches), opponent pool + PFSP, evaluation, checkpoints, torch export, multi-GPU (`shard_map`) |
| `sample.py` | decisions (T-sampling of unit ops / quantities, sharpened market fires, greedy mode) and `logp_terms` (= `il/ppo.logp_terms`) |
| `actions.py` | engine-native action arrays (`Act`), policy classes -> unit commands (qty clamp), canonical market decode, and the **rule layer** (`apply_rules` = `features.midnight_fix`: plan fix SE guard / sheep cap / seed budget, final fix, homing, midnight overflow sale) |
| `backend.py` | the batched Backend interface the trainer uses + loader (`real` / `mock`) |
| `real_backend.py` | Agent A's engine + Agent C's encoder / legal mask / network, glued by exact array adapters (`raw_from_state`, `view_from_raw`, `to_engine`) |
| `mock_env.py` | a toy environment with the same interface and the real input shapes (learnable objective) |
| `ckpt_io.py` | training-state npz (resume), snapshot npz, torch `.pt` in the `ppo_live` format (-> `il/export.py`), round-trip check |
| `run_gpu.sh` | detached launcher for a GPU server (1 GPU, data parallel over N GPUs, or one seed per GPU) |
| `bench_gpu.py` | GPU throughput / memory benchmark (real engine + real checkpoint) |
| `test_rules.py`, `test_loss.py`, `test_real_backend.py`, `smoke_test.py` | verification (below) |
| `run1.sh` | one-core wrapper for the shared Mac |

## Interfaces used (A and C)

- Engine (A, `kagjax/engine.py`, `state.py`, `spec.md`): `EnvState(t, g)`, **one step counter `t` shared by the batch**
  (lockstep; day / decay / town logic are real branches). `engine._step_jit(tables, state, a0, a1, None)`
  (training randomness from each game's key), `state.Actions` (40 units, 10 market slots), money = reward.
  Steps 0..718 are played; after step 718 the state is done and further steps are no-ops.
- Policy (C, `kagjax/policy/`): `net.forward` / `load_ckpt` / `to_state_dict` / `convert_state_dict`,
  `encode.encode(RawObs, prev_ops, tables)` (features6, bit-exact), `act.legal_mask(RawObs)`.
- My adapters (`real_backend.py`): A's `GameState` -> C's `RawObs` per seat (tile-kind remap, animal ids, field
  masking exactly like `raw_from_obs`), `RawObs` -> rule-layer `View`, `Act` -> A's `Actions`. All three are
  bit-identical to the dict-based references on real C++ game states (`test_real_backend.py`).
- Tables (A's price tables, C's ~10 MB encoder tables) are passed INTO the jitted programs as an argument
  (`Backend.consts()`), never closed over and never created inside a trace.

## One rollout step (all B games at once)

1. `observe` both seats (A state -> C `RawObs`), pick the learner's / opponent's view per game (learner seat
   alternates game by game, so each opponent block has both seats).
2. `encode` (features6 with each seat's previous op classes), `legal` masks, `view` for the rules.
3. Forward passes: learner (current params, team `TEAM`=10) on all B games; opponents per block (each block: one
   bank slot + one team id) with `vmap` over the K blocks (`OPP_MAP=map`: sequential `lax.map`, less memory).
4. Decisions (`sample.decide`): learner sampled like `live_rollout` (unit ops ~ softmax(masked/T), PICKUP/PLACE
   quantity ~ softmax(qty/T), market head fires ~ Bernoulli(sigmoid(KSHARP (logit P(order) - logit tau))), class =
   argmax of the non-zero classes); opponents greedy (as `BCAgent`), `OPP_SAMPLE=1` samples them too.
5. Actions (`sample.build_act`): class -> engine command with the quantity clamp, `decode_market` (canonical order,
   <= 10 orders), then the **rule layer** (`MIDNIGHT/FINALFIX/HOMING`, optional `SE_GUARD`, `SHEEP_CAP`,
   `SEED_BUDGET`) for both sides.
6. Engine step; reward of the learner = change of (own money - rival money) / 2e4 (the terminal step's change is the
   final margin's last piece, so the undiscounted return is the final margin), + `WINB` x sign(final margin).
7. Previous op classes (pre-rule, as the reference agents keep them) are carried per seat for the encoder.
Recorded per learner decision: planes uint8, glob f32, pos int8, uin f32 (`UIN_DTYPE`), legal mask, op / qty classes,
market fires, reward, done, valid, and the behavior log-probs of the sampled decisions (unit op, quantity, market
fires) computed by `sample.logp_terms` - the same function the update uses (~11 KB per sample).

## Episodes, segments, memory

An episode is 720 slots (steps 0..718 + one dummy slot where the done engine does nothing; the dummy sample is
masked out). Because the engine keeps all games at the same `t`, every game is at the same phase:
- `SEG=720` (default): one update per episode, on all 719 decisions of every game; no bootstrap needed; every
  update sees all game phases. Memory: 720 x B samples (B=1024: 737k samples, ~6.3 GB trajectories + ~2 GB of
  reference unit log-probs + nets/optimizer ~1.5 GB + activations of one minibatch).
- `SEG` = 360 / 240 / 180 / 90 (must divide 720): truncated segments with value bootstrap V(s_SEG); less memory,
  more updates per episode, but each update sees only one phase of the game (all games are synchronous). Use it only
  if memory forces it.
Games (and opponent assignments) are refreshed when the episode ends.

## PPO update (identical to il/ppo_live.py, verified against the torch code)

- Old log-probs = the behavior log-probs stored at rollout (`OLD_LOGP=behavior`, default: one function, the same
  inputs, so at LR = 0 every ratio is 1 up to float32 rounding of differently batched kernels), or recomputed from the
  stored inputs before the epochs (`OLD_LOGP=recompute`, the ppo_live way); `DIAG_OLD=1` reports max |stored -
  recomputed|. Before the epochs (chunks of MB, `lax.map`): reference log-probs (frozen `REF`, default INIT;
  `REF_ONLINE=1` computes them per minibatch instead of storing [N,16,44]) and old values (value net).
- Market fires: P(order) enters as logit P(order) = logsumexp(z[1:]) - z[0] (`sample.MARKET_FORMULA="stable"`,
  exact identity, no cancellation when P(no order) -> 1; the torch code's 1 - softmax[0] and log(1 - pf) lose up to
  1.3e-2 in the fire log-prob, measured against float64 in test_loss.py); sampling and log-probs use the same
  clamped formula.
- GAE with gamma = 1, `LAM`=0.95 (time-major reverse scan), advantages normalized over the batch (unbiased std,
  as torch).
- Loss = clipped surrogate on unit ops (valid units) + 0.5 x clipped surrogate on quantities + clipped surrogate on
  market fires, + `BETA` x (KL(pi || ref) on unit ops + Bernoulli KL of the fires); separate value network (MSE,
  `VLR`), both with Adam (torch semantics) and grad-norm clip 1.0; `coords` frozen; `VWARM` iterations update the
  value net only. `EPOCHS` x (N / `MB`) minibatch steps per iteration.
- `test_loss.py`: with the torch arithmetic (`MARKET_FORMULA="torch"`) logp terms rel. err ~1e-7, loss equal to 1e-7,
  every gradient within 3e-6 relative, GAE exact, Adam+clip within 6e-8 of torch; greedy decisions identical to
  `live_rollout`; sampled frequencies match; the stable formula is within 2e-7 of float64.
- Safety: every update checks that all params and statistics are finite (else the run stops before saving);
  `stats.jsonl` logs per iteration: KL (unit / market), clip fraction, max |ratio - 1| (first minibatch and all),
  entropy, value loss, loss per epoch, timings.

## Worlds and seeds

Kaggriculture's only world randomness is the end-of-day generator (weed spawns p = 0.005 per empty tile, one shop
unlock draw every 3 days); every game starts from the same initial state (money 3000, NW quadrant, farmer at the
shed). A's engine consumes that generator as 624 raw MT19937 words per day:
- training: `words=None`, each game draws i.i.d. uniform words from its own PRNG key at each day end (same
  distribution as Kaggle's `random.Random((seed*1000003) ^ day)`). The trainer's key stream (`SEED`) splits a fresh
  key per game for every new episode (`Backend.reset(key, n)`), so every episode is a new world;
- evaluation: a fixed key (`PRNGKey(10000 + SEED)`) -> the same worlds at every evaluation (paired comparison);
- tests: exact words of real Kaggle seeds (`rng.batch_words(seeds)` in `cs["words"]`) reproduce the C++ / Kaggle world
  of that seed bit for bit (`test_e2e_game.py`).
The learner's seat alternates game by game (half of every opponent block in each seat).

## Opponent pool, PFSP, evaluation

No Python scripted bots can run inside the JAX engine; every opponent is a network from the bank, conditioned on a
style id:

- Bank (stacked params on device): slot 0 = LATEST (current learner, refreshed every iteration = mirror
  self-play), slot 1 = IL (the frozen imitation model, default the multi-style `g20_u_s0` before the DSM fine-tune,
  same architecture as the learner; opponents play it as the style ids of `IL_TEAMS`, default DSM 10 x3, Vadim 5 x2,
  DECEM 3 x2, UMG 4 x2, Boey 0 x2), slots 2.. = SELF snapshots (past selves) every `SNAP_EVERY` iterations, taken at
  an episode boundary (ring of `SELF_MAX`, snapshot 0 = the start policy is always kept).
- TOP (`tapes.py`, OFF by default until re-audited): recorded leader games (data/tapes_top12, leader = a current top
  team): the leader's seat plays its recorded actions, the learner the other seat (ppo_live's TOP worlds), in the
  recorded world PINNED like the C++ reference (topruler.run_tape: recorded shop sequence; leader in seat 1 rolls its
  weeds at the recorded stream positions). A's engine shares one end-of-day stream between both farms and the shop,
  so natural words alone shift the leader's world as soon as the learner's farm differs (audit, Sep 26);
  `RealBackend.step_tapes` pins by stepping twice at day ends (A's `with_info` gives the empty-tile counts, then the
  day's words are rebuilt). test_tape_pin.py: learner = PASS or a foreign recorded player, JAX == C++ pinned in 12/12
  worlds each (unpinned: 2/12 and 0/12). Every episode of the benchmark files (leader tests, ladder test, held-out lists; 1,249 on the Mac) is
  excluded; ep % 10 == 0 is held out for the `TOP` evaluation (314 leader seats) and the rest (2,379) is the
  training bank (`TAPE_MAX` of them on the device, ~0.25 MB each). test_tapes.py: replaying both seats through the
  trainer's rollout reproduces the recorded Kaggle rewards exactly.
- Each of the K blocks of B/K games gets one entry (kind by `OPPS` weights, e.g. `LATEST=2,SELF=2,IL=4`; SELF picks a
  random snapshot, IL a style id by weight) for a whole episode; new entries are drawn when the episode ends.
  Opponents act greedily (as `BCAgent`: argmax, tau threshold) with the same rule layer; `OPP_SAMPLE=1` samples.
- `OPPS` weights per kind; `PFSP=1` multiplies by (0.25 + recent loss rate of that kind), as ppo_live.
- Evaluation every `EVAL_EVERY` iterations: greedy learner vs each `EVAL_OPPS` entry (`IL@<team>`, `SNAP0`) on
  fixed worlds (fixed keys; both seats), paired against the first evaluation; `best.pt` keeps the best score.
- The real benchmarks (464 ladder worlds, leader tests, live-loss replays) run on the CPU pipeline from the exported
  `.pt` files (`pt/it<N>.pt`, `best.pt`), unchanged.

## Checkpoints and export

- `state_<it>.npz`: policy, value, both Adam states, pool, iteration (resume with `RESUME=`).
- `pt/it<it>.pt`, `best.pt`: torch checkpoints in the ppo_live format (`state`, `vstate`, width, blocks, n_teams,
  g_in, n_aux, feat, team, it): `python il/export.py <pt> <main.py> 0.5 10` builds the Kaggle numpy runtime
  (add `RULES="SE_GUARD=1"` to bake the SE guard in); `ckpt_io.py to_pt <state.npz> <out.pt>` converts any
  training state. Round trip torch -> JAX -> torch is bit-exact (`ckpt_io.py check`).

## Throughput and memory (measured on the shared A100-40GB: 100% busy with other jobs, JAX capped at 8 GB)

Per lockstep step at B = 1024 games, 20M model (profile, `prof_gpu.py`): learner forward 117 ms (TF32 default;
378 ms full fp32; 46 ms bf16), opponent forwards 151 ms with `lax.map` over 4 blocks (275 ms with `vmap`, so
`OPP_MAP=map` is the default), engine step 4.9 ms (A's `CFG_GPU` loop structure; the sequential default is ~10x
slower on GPU), encode + legal + view 1.7 ms per seat, decisions 1.1 ms, actions + rule layer 2.5 ms: the two
network forwards are ~95% of a step. f32 matmul probe: 53 TFLOP/s default (TF32), 6.6 full fp32, 57 bf16.
Training (update of N samples): ~15 forward-equivalents per sample with EPOCHS=2 and a 20M value network
(reference + value + optional recompute pre-pass, then 2 x policy fwd/bwd + value fwd/bwd); measured 410 samples/s
at MB=256 on the shared GPU. The knobs were measured on Sep 26-27 night: see "Update speed" below (value head,
1 epoch x MB 4096, 25% KL anchor, bf16 gradients: 4.5-6x; a fresh small value net does not fit in time; full bf16
fails the LR-0 check).
Trajectory memory: ~11.3 KB per sample + 2.9 KB reference terms (`REF_ONLINE=1` drops the latter): B=1024 x 720 =
9.8 GB, B=2048 x 720 = 19.5 GB, B=4096 x 720 = 39 GB (80 GB GPUs; else `SEG=360`).

## Update speed (Sep 26-27 night; shared A100-40GB, 8 GB cap, other jobs + the overnight run on the same GPU)

New options (all off by default; the default path is bit-identical to the gate trainer):
- `EPOCHS=1`, `MB=4096` with `MICRO=512`: gradient accumulation over MB / MICRO microbatches with minibatch-global
  normalizers (the loss of one 4,096-sample minibatch, exactly; float order differs by ~1e-7 rel.). LR by the
  square-root rule from the gate's MB=256: `LR` = 5e-6 x sqrt(MB / 256) = 2e-5, `VLR` = 4e-4. With it the first
  update from INIT moves the policy as far as the gate's 2 x 360 steps (KL to INIT 0.00050 vs 0.00057, clip 0.0050
  vs 0.0059, same data). MICRO bounds memory: 20M at MICRO=1024 needs a 4.2 GiB temp buffer (ran out of memory under
  the 8 GB cap twice); 80 GB GPUs: `MICRO=1024`.
- `REF_FRAC=0.25`: the KL anchor's reference terms on a uniform subsample of M = N/4 samples drawn without replacement
  once per update, each KL term weighted N/M: every sample's inclusion probability is exactly M/N, so the weighted
  sum is an unbiased estimate of the full KL and (the draw does not depend on the parameters) of its gradient.
  Measured on 4,096 real samples (g20_w_dsm_s0 vs its IL parent, KL 0.095 units / 0.161 market): mean of 2,000
  draws / full = 1.0014 / 1.0047 (+0.8 / +1.8 se) at f = 0.25, 0.9992 / 0.9987 at f = 0.5; per-draw relative sd
  8% / 12% (f = 0.25); gradient of BETA x KL: per-draw cosine to the full gradient 0.75-0.85, the mean of 8 draws
  0.967 (relative error 0.27 = per-draw 0.78 / sqrt 8: noise only, no bias). Saves 3/4 of the reference forward.
- `VALUE_INIT=head VALUE_TARGET=final`: the value is the policy's own IL-trained `v` head on its trunk features
  (`hm`, stop-gradient; recorded at rollout, so V_old needs no forward), predicting the final margin (V = V' -
  margin); only the head (257 numbers) is trained, at `VLR`. No value network forward / backward at all.
- `UPDATE_PRECISION=bf16grad`: forward values from the TF32 forward (ratios, clip, KL, losses exactly as TF32),
  gradients from a bf16 forward (value = sg(a - b) + b); `UPDATE_PRECISION=bf16`: everything in bf16.
- `OPP_GROUPS=G`: G opponent param groups per episode (a group draws one entry; IL groups give each block its own
  style id), one opponent forward per group with per-row team ids; `OPP_PRECISION=bf16`: opponents' rollout
  forwards in bf16 (the learner stays TF32; evaluation is fp32 for both seats).
- `EVAL_BASE=<eval_0.npz>`: reuse another run's iteration-0 evaluation (same INIT / worlds).

Measured with `speed_chain*.sh` (`python kagjax/train/speed_report.py` prints everything from
kagjax/logs_server/D/speed; summary in speed_report.txt / speed_summary.json). Update: one recorded 128-game rollout
(92,160 samples) updated by every variant; "LR-0" = ratio statistics of the first minibatch, before any parameter
change. The shared GPU's load changed by the minute (TF32 matmul probe 23-65 TFLOP/s), so the reference numbers are
the back-to-back repeat at a steady probe of ~61 TFLOP/s (00:52-01:02 UTC) and the round-robin rollout timing.

| 20M, back-to-back (probe 60-62) | update samples/s | x gate | end-to-end game-steps/s (rollout 4,711) | dedicated A100 (x125/61) |
|---|---|---|---|---|
| gate: 2 epochs x MB 256, 20M value net (to-go) | 531 | 1.0 | 477 | update ~1,100, end-to-end ~980 |
| 1 epoch x MB 4096 (MICRO 512) + anchor 25%, 20M value net | 1,114 | 2.1 | 901 | ~2,300 / ~1,850 |
| + value head (final margin) instead of the value net | 2,379 | 4.5 | 1,581 | ~4,900 / ~3,240 |
| + `UPDATE_PRECISION=bf16grad` | 3,306 | 6.2 | 1,943 (2,206 with bf16 opponents) | ~6,800 / ~3,980 (~4,500) |

Other update variants (20M, measured 22:17-23:16 under heavier / changing load; same-block rows comparable):
1 epoch x MB 256 738 (gate 475 in that block); 1 epoch x MB 4096 1,576 (temp 3.5 GiB); value head with the anchor on
all samples 1,128 vs 25% 1,568 (MICRO 512, temp 2.2 GiB; MICRO 1,024: 1,753, temp 4.2 GiB, out of memory 2 of 3 times
under the 8 GB cap); 2 epochs 1,056 (2,112 sample-passes/s); small 128x4 value net 1,393; `bf16grad` 2,239 (temp 1.3
GiB); **`bf16` 3,541 but LR-0 fails: 2.1% of unit ratios off by > 5% (max 1.09), market 0.43% (max 1.87), clip
fraction 0.38% before any update** (TF32 and `bf16grad`: 0-2e-5 above 5%, max 0.04-0.07, clip 0); `bf16` with
recomputed old log-probs 2,196 (ratios exactly 1 at LR 0, but only because both sides are the bf16 policy: the 2.1%
gap to the behavior policy is then silently ignored). 10M (rf_w_dsm_s0): gate 740, 1 epoch x MB 4096 + anchor 25%
1,452, + value head 2,707 / 3,046, + `bf16grad` 3,343, `bf16` 8,031 (LR-0 fails the same way: 2.2%, max 1.17).

Value options on the same 4 saved rollouts vs IL clones (policy frozen, value-only updates; explained variance of
the Monte-Carlo return BEFORE each update, i.e. on data the value has not seen yet; initial -> rollouts 1 / 2 / 3;
advantage sd = GAE advantages before normalization, rollouts 1-3):

| value | EV initial | EV r1 / r2 / r3 | held-out mean | adv sd |
|---|---|---|---|---|
| 20M value net, to-go, gate optimizer (2 x MB 256, VLR 1e-4) | 0.318 | 0.342 / 0.398 / 0.412 | 0.384 | 0.125 |
| 20M value net, to-go, 1 x MB 4096, VLR 4e-4 | 0.318 | 0.351 / 0.403 / 0.415 | 0.390 | 0.141 |
| 20M value net, final margin, 1 x MB 4096 | 0.357 | 0.418 / 0.424 / 0.432 | **0.425** | 0.096 |
| value head, final margin (VLR 4e-4 or 1e-3: same) | 0.357 | 0.388 / 0.378 / 0.346 | 0.370 | 0.133 |
| value head, to-go | 0.318 | 0.272 / 0.300 / 0.278 | 0.283 | 0.181 |
| small 128x4 net (fresh), final | -0.374 | -0.318 / -0.277 / -0.283 | -0.293 | 0.119 |
| small 128x4 net (fresh), to-go | 0.001 | 0.005 / 0.026 / 0.046 | 0.026 | 0.073 |

The final-margin target beats to-go for both; the value head (linear on the IL trunk, no forward of its own) starts
where the full net starts but cannot improve much, the full net (0.425, advantage sd 0.096 vs 0.133) is better at
2.1x the update cost; small fresh nets are useless for many iterations. Next step if the head's value quality
matters: unfreeze (value-loss gradient into the trunk at a small weight), as in the research plan.

Rollout (20M, B = 1024 unless noted, no recording, 3 round-robin rounds at probe ~61; the 10M at probe ~27):
| mode | 20M game-steps/s | 10M | 20M dedicated (x125/61) |
|---|---|---|---|
| one opponent forward per block (`lax.map`, default) | 4,711 | 4,504 | ~9,600 |
| `OPP_GROUPS=2` | 4,768 (+1%) | 4,689 (+4%) | |
| `OPP_GROUPS=1` (one forward over all opponent rows, per-row style ids) | 4,937 (+5%) | 5,088 (+13%) | |
| `OPP_PRECISION=bf16` (learner TF32) | **6,635 (+41%)** | **6,552 (+45%)** | ~13,500 |
| B = 2048 | 4,915 (+4%) | 5,169 (+15%) | |

Why the per-block opponent forward costs more than the learner forward over the same rows (profile: 151 vs 117 ms
at B = 1024): K sequential forwards of B/K rows (`lax.map`), each with its own fixed kernel sequence, smaller
convolutions / GEMMs at lower occupancy, no overlap between blocks (the per-block parameter gather is a negligible
77 MB copy; `vmap` is worse: grouped convolutions). Merging them into one forward (`OPP_GROUPS=1`) needs one
parameter set per group and gains only 5-13%; the opponents' share of a step is ~45%, which is why bf16 opponents
(2.5x faster convolutions) gain 41-45%. `OPP_PRECISION=bf16` changes only the opponents' play (their greedy moves
differ from TF32 in a few % of decisions); the learner's behavior log-probs stay TF32.

KL anchor subsample (`exp_value_kl.py EXP=kl`): see the `REF_FRAC` bullet above (unbiased; gradient noise only).

3-iteration sanity runs from g20_w_dsm_s0 (128 games, same seed; greedy fp32 evaluation on 128 fixed worlds, paired
vs the gate run's iteration 0; KL / clip of iterations 2 / 3; time = 3 iterations + evaluations, busy GPU):

| run | KL units it 2 / 3 | clip | vs SNAP0 | vs IL@10 | time | peak GiB |
|---|---|---|---|---|---|---|
| gate-5 (Sep 26, the gate) | 0.0006 / 0.0013 | 0.006 | +597 (se 459) | +1,857 (se 662) | 18 min | 3.9 |
| S0 gate settings (tonight) | 0.0006 / 0.0011 | 0.006 | -924 (se 514) | -1,853 (se 744) | 1,385 s | 3.9 |
| S1 1 epoch, MB 4096, anchor 25%, 20M value net | 0.0005 / 0.0009 | 0.005 / 0.004 | -195 (se 513) | -687 (se 725) | 702 s | 5.5 |
| S2 + value head (final) | 0.0005 / 0.0010 | 0.005 / 0.003 | -776 (se 410) | -1,043 (se 816) | 535 s | 4.3 |
| S4 + `bf16grad` | 0.0005 / 0.0011 | 0.006 / 0.004 | -43 (se 525) | -444 (se 790) | 498 s | 4.3 |
| S3 = S2 + `OPP_GROUPS=2` | 0.0005 / 0.0010 | 0.005 / 0.004 | -295 (se 436) | -1,196 (se 779) | 524 s | 4.3 |

All finite; per iteration the fast settings move the policy as far as the gate settings (KL to INIT, clip). The
evaluations after 3 iterations scatter by +-2 se between identical configurations (gate-5 vs S0: XLA autotuning picks
different kernels per process, runs are not bit-reproducible), so 3 iterations show "not broken" and equal KL, not
a learning-speed ranking; learning per hour is for the long runs (the default 8-GPU plan keeps a gate control).

### Recommended launch (dedicated A100 80 GB, one run per GPU)

`run_gpu.sh` `PRESET=fast` (now the default): `EPOCHS=1 MB=4096 MICRO=1024 LR=2e-5 VLR=4e-4 BETA=0.05 REF_FRAC=0.25
VALUE_INIT=head VALUE_TARGET=final VWARM=1 UPDATE_PRECISION=bf16grad`, rollouts TF32, evaluation fp32,
`OLD_LOGP=behavior`, `OPP_MAP=map`, `OPP_GROUPS=0`, SEG=720. 20M: B=2048, K=8 (256 rows per opponent block);
10M: B=3072, K=8. Memory (20M, B=2048): trajectories ~18 GB + anchor terms ~1 GB + update buffers ~5 GB.
Options: `OPP_PRECISION=bf16` (+41-45% rollout, opponents' play slightly changed; off by default as rollouts stay
TF32 until decided); `VALUE_INIT=policy` with `VALUE_TARGET=final` (better value, EV 0.425 vs 0.370, ~40% slower end
to end). Not adopted: `UPDATE_PRECISION=bf16` (fails LR-0), `OPP_GROUPS` (+1-5% for the 20M, fewer opponents per
episode), `EPOCHS=2` (2x the update time for 1.3x the KL per iteration).

Check of the projection method: from 01:10 UTC the overnight run (gate settings, B=256) had the A100 nearly to
itself: rollout 24 s + update 151 s per 184,320 samples = 1,055 game-steps/s end to end (update 1,220 samples/s,
recorded rollout 7,700 game-steps/s even at B=256), vs ~1,000 / ~1,100 projected for the gate settings.

Projected per dedicated A100 (probe-normalized, +-25%): 20M rollout ~9,600 game-steps/s, update ~6,800 samples/s,
end-to-end ~4,000 game-steps/s (gate settings ~1,000; with bf16 opponents ~4,500). 10M: rollout ~20,000, update
~10,000 (C10M: 3,343 at probe 39), end-to-end ~6,700 (~1.7x the 20M, not 2.5x: the engine and rule layer do not
shrink with the net, and the 10M update gains less from bf16 gradients). The research estimate (update 410 ->
2,000-2,800, end-to-end ~370 -> 1,400-2,200 per A100) holds in shared-GPU units (531 -> 2,379-3,306 and 477 ->
1,581-2,206) and is ~2x conservative for a dedicated GPU.

## Runs A and B (operator plan Sep 27: 10M only, dedicated 8x A100 80 GB or 4x H100)

New options (all off by default; the default path is still bit-identical to the gate trainer, re-checked Sep 27):
- `INIT=random[:<template ckpt>]`: fresh torch-default init (weights and biases uniform +-1/sqrt(fan_in), team table
  N(0,1)) with the template's architecture (default template: IL, i.e. rf_w_s0 = 256x8, 48 teams, 6 aux heads). The
  exported .pt is a normal ppo_live checkpoint: gate_cpu.py roundtrip on a random 10M (il/export.py B64=1 SE_GUARD=1
  -> numpy package in 2 C++ games): 2,876 / 2,876 decisions identical to the JAX greedy pipeline; loads in
  il/live_rollout.py (Net width 256, 48 teams) and il/agent.BCAgent.
- `ENT_COEF`: loss -= ENT_COEF x (H(unit ops) + 0.5 H(quantities) + H(market fires)) of the sampling distributions
  (temperature T over legal ops; Bernoulli of the sharpened fire probability), each averaged like its surrogate term;
  logged as `ent u/q/m`.
- `BETA_END` + `BETA_DECAY_SAMPLES` (or `BETA_DECAY_ITERS`) + `BETA_SCHEDULE=exp|linear`: the KL weight per iteration
  (geometric BETA -> BETA_END, then held); passed into the jitted update as an argument (no recompilation). BETA = 0 at
  an iteration switches the reference forward and the KL terms off (static switch, one recompilation); `KL_MAX` only
  applies while the anchor is on.
- `WINB_RAMP_SAMPLES` / `WINB_RAMP_ITERS`: linear ramp 0 -> WINB of the terminal win bonus (a rollout argument).
- `VALUE_GRAD=c` (value head): share c of the value-loss gradient that reaches the policy trunk (forward unchanged).
- Opponent kind `PASS` (training and evaluation): the opponent seat never acts (no unit command, no order = the C++
  PASS); a passive farmer ends every game with its $3,000 start money.
- Tapes: `TAPE_DIRS` (leader tapes; seats of the `TOP_TEAMS` names, default the 15 leader teams + the 8 named
  specialists elmo, Aaweg, Yannik Schiffner, SpaTaro, lingxiaojun, yarneo, Snorlax, Russell Kirk; "*" = any team) and
  `SPEC_DIRS` (our own live games vs specialists: every non-"us:" seat) form ONE device bank; kind `TOP` draws leader
  seats (all seats when SPEC is not in OPPS), kind `SPEC` specialist seats (own PFSP loss rate and W/L in the log);
  evaluations `TOP` / `SPEC` play every held-out seat (ep % 10 == 0) once. Benchmark episodes are excluded, an episode
  present in two dirs is used once, recordings that do not replay exactly in C++ are dropped (none so far).
  Sep 27: data/tapes_top12 = 2,423 training + 317 held-out leader seats (19 teams); data/tapes_spec = 322 training +
  34 held-out specialist seats (272 + 34 players; 90 of its 446 episodes are benchmark episodes); all replay exactly.
  `python kagjax/train/tapes.py check --spec <dir>` validates a new dir (data/tapes_spec_more is picked up once it has
  an index.json).
- `PT_EVERY_SAMPLES=1e8`: a .pt every 100M learner samples (the sample counter is saved in the state, RESUME keeps it).
- Data parallel: `[dp]` log lines = max |difference| of the replicated params between the devices (iterations 1-3 and
  every 20th; must stay 0).
- Tests: test_ab_options.py (unit: entropy terms vs numpy, the bonus raises entropy, schedules; mock runs of A- and
  B-like settings incl. resume and 2-device data parallel: replicas identical), all PASS on the Mac.

TOP / SPEC gate (the auditor's re-check, `recheck_tape_pin.py`, Mac CPU, Sep 27): 60 leader + 40 specialist seats,
learner = PASS and = V94 (a reactive scripted bot; its moves from the C++ pinned run replayed in JAX): the trainer's
step_tapes gives exactly the C++ pinned final banks in 100 / 100 worlds for both learners; the unpinned (pre-fix)
worlds differ in 98 / 100 and 95 / 100 (kagjax/logs_server/D/tape_recheck.json). -> TOP and SPEC are in both recipes.

Recipes (`run_gpu.sh RECIPE=A|B`, layered on `PRESET=fast`; any variable can still be overridden):

| | A: warm start | B: from scratch |
|---|---|---|
| INIT | rf_w_dsm_s0 (team 10) | random 10M (architecture of rf_w_s0) |
| sampling | T 0.2, KSHARP 10, TAU 0.5 (as imitation) | T 1, KSHARP 1 (plain softmax; fire prob = P(order)), TAU 0.5 |
| KL anchor | BETA 0.05 -> 0.005, geometric over the first 1B learner samples, then held | none (no reference forward) |
| win bonus | WINB 0.5 from the start | WINB 0 -> 0.5 linearly over the first 300M samples |
| entropy bonus | - | ENT_COEF 0.01 |
| value | IL head, final-margin target, stop-gradient | head, return-to-go target, VALUE_GRAD 0.5 |
| LR / VLR | 2e-5 / 4e-4 | 1e-4 / 4e-4 |
| league (OPPS, x PFSP) | LATEST 2, SELF 2, IL 3, TOP 2, SPEC 2 | LATEST 2, SELF 2, IL 2, TOP 1, SPEC 1, PASS 1 |
| evaluation | SNAP0, IL@10, IL@0, TOP, SPEC | PASS, IL@0, IL@10, TOP, SPEC |

Common: B 3072 games per GPU, K 8, SEG 720, EPOCHS 1, MB 4096, MICRO 1024, REF_FRAC 0.25, UPDATE_PRECISION bf16grad,
rollouts TF32, evaluation fp32 (128 games per GPU per opponent every 5 iterations), SNAP_EVERY 5, SELF_MAX 8,
CKPT_EVERY 5, all tape seats on the device (TAPE_MAX / SPEC_MAX 4096, ~0.3 MB each), a .pt every 100M samples.
With 4 GPUs in data parallel one iteration = 12,288 games = 8.8M samples (~4-5 min if dp scales: 10M ~9k samples/s
per dedicated A100 measured Sep 27).

Win bonus: the reward is margin / 2e4 per step plus WINB x sign(final margin) at the end.  Per-game margin sd vs a
peer (the DSM 20M vs IL@10, 128 fixed worlds, 5 evaluations): $8.6-9.8k (0.46 in reward units); vs its own mirror
$4.6-5.8k.  One more dollar of final margin is worth 1/2e4 = 5.0e-5 through the margin term and 2 WINB f(0) through the
win term (f(0) = density of the final margin at 0: ~4.3e-5 per $ for sd 9.2k): equal weight at WINB ~0.6.  The torch
runs' 0.3 gave wins only ~half the weight of margins (consistent with "RL raised margins, not wins"); 0.5 (= $10k per
side, a $20k swing between a narrow loss and a narrow win) gives wins ~0.9x the margin's weight vs peers and ~1.6x in
near-mirror games, nothing in lopsided games (where the margin still teaches).  1.0 would make wins dominate (1.7x vs
peers) at the cost of noisier advantages (return sd 0.46 -> ~1.1).  B ramps it in linearly over its first 300M
samples: a random policy loses every game against everything but PASS, so the win term carries no information early.

### From-scratch sanity test (B, first hour)

Reference points (greedy fp32, 16 fixed worlds each, Mac CPU, Sep 27): a PASS player ends every game with its $3,000
start money.  Random-init 10M: bank ~$235 (it fires ~90% of the market heads and burns its money): vs PASS -$2,765
(W 0%), vs IL@0 -$85,522 (W 0%), vs IL@10 -$144,193 (W 0%).  IL@0 (the Boey-style clone, the weakest style) vs PASS
+$90,620 (W 100%), vs itself -$1,037 (W 44%).  The warm starts beat IL@0 by ~+$23-27k (20M evaluations).  So: no
economy ~$0-3k, IL@0 economy ~$90k, leaders $100-150k.

`launch_multi.sh <p> bsweep`: four single-GPU B runs (B=2048: 1.47M samples per iteration, evaluation vs PASS / IL@0 /
IL@10 on 256 fixed worlds every 3 iterations, paired vs iteration 0): LR 1e-4 (the recipe), LR 3e-4, ENT_COEF 0.03,
separate 10M value network.  After ~45-60 min (~20-25M samples each; `run_report.py kagjax/runs/<p>_Bsw_*`):
- learns at all (the gate): by ~5M samples the margin vs PASS rises above the random -$2.8k; by 20M: vs PASS W >= 90%
  with margin >= +$10k (it farms and sells), vs IL@0 paired gain >= +$10k and rising at every evaluation;
- healthy: unit entropy (T=1) falling from ~1.9 but > 0.3, clip fraction 0.02-0.3, approx. KL per update < 0.05, value
  EV (MC) > 0.2, no stop; `[dp]` 0 in data parallel;
- by 50M (B in data parallel: ~30 min): vs PASS >= +$30k, vs IL@0 >= -$50k (gap closed >= 40%), still rising.
  Beating IL@0 (W > 50%) is not expected before ~0.2-0.5B samples - the A-vs-B comparison point of the plan.
These numbers are expectations for a working setup, not measurements (nothing can learn at scale on the Mac): the
hard signal is a monotone paired gain vs PASS within ~10-20M samples.  Flat or falling vs PASS at 20M = broken
settings (or a bug): that variant stops; if every variant fails, B is dropped and its GPUs join A (RESUME A on 8 GPUs).
The best variant's settings go into B's data-parallel run (a plan line with its LR / ENT_COEF / VALUE_INIT).

## Sprint (operator plan Sep 28: 8x H100 NVL 94 GB, ~2.5-8 h; hour 1 = A on all GPUs, hour 2 = B on all GPUs, compare,
## the winner resumes on all GPUs; the last ~10% win-only)

`sprint.sh` wraps everything (all runs = ONE data-parallel process over the 8 GPUs, `kagjax/runs/sprint_A|B`):

    bash kagjax/train/sprint.sh preflight              # ~10 min: tools, data, option tests, 20-world pin re-check, recipe smokes
    bash kagjax/train/sprint.sh memprobe               # ~3-5 min on GPU 0: memory estimate for 2048/3072/4096/5120 games per GPU
    bash kagjax/train/sprint.sh A 60 4096              # hour 1: warm start, stops itself by 60 min (final eval, state, pt, FINAL.json)
    bash kagjax/train/sprint.sh B 60 4096              # hour 2: from scratch (start it after sprint_A/FINAL.json appears)
    bash kagjax/train/sprint.sh status                 # any time: iterations, W by opponent per iteration, evaluations, exports
    bash kagjax/train/sprint.sh resume A 270 3e9       # the winner (A here): +270 min, schedules over a 3B-sample horizon
    bash kagjax/train/sprint.sh resume A 30 3e9 win    # the last ~10%: win-only (MARGIN_W=0, WINB=1, 1 value-only iteration)

Settings (RECIPE=A|B in run_gpu.sh, on PRESET=fast; each overridable on the command line, e.g. `... A 60 4096 LR=1e-5`):
- Games per GPU B=4096 (K=8 blocks of 512): mem_probe.py (compile-only XLA buffer sizes, calibrated on the A100 measurement
  B=2048 -> 20.4 GiB peak in use; the Mac CPU compile gives 19.3 GiB + 1 GiB slack for 2048): ~8.4 MB of trajectory per
  game + ~2 GiB update temporaries + ~0.8 GiB tapes -> 4096 ~ 38 GiB (40% of 94 GB), 5120 ~ 47 GiB (50%); both keep far
  more than 15% headroom (the A100 OOM at 3072 happened at ~80% of its pool).  One iteration = 8 x 4096 x 720 = 23.6M
  learner samples (~3 min at the expected 120-160k samples/s on 8 H100 NVL).
- Global minibatch = 8 GPUs x MB 4096 = 32,768 samples; `LR_SCALE=sqrt` multiplies LR / VLR by sqrt(global / 4096) =
  2.83: A 2e-5 -> 5.7e-5 (VLR 1.1e-3), B 1e-4 -> 2.8e-4 (VLR 1.1e-3); a resume on another GPU count rescales by
  itself.  Watch A's first policy iterations: clip fraction > 0.1 or approx. KL per update > 0.02 -> resume with LR=1e-5.
- Throughput: TF32 rollouts, bf16grad updates, OPP_PRECISION=bf16 (opponent forwards; the learner stays TF32; +41-45%
  rollout on the shared A100), the persistent XLA cache kagjax/train/.jax_cache (shared by all runs; resumes reuse the
  compiled programs).  Optional, untested, off: `XLA_EXTRA="--xla_gpu_enable_latency_hiding_scheduler=true"` (overlaps
  the gradient all-reduce with compute) - try it only as a 10-minute A/B (`sprint.sh A 10` with / without, compare the
  samples/s of iterations 2-3) if there is time.
- League (OPPS x PFSP (0.25 + recent loss rate)): A `LATEST=3,SELF=3,IL=1.5,TOP=1,SPEC=1.5` -> ~60% self / past selves,
  ~40% clones + leader tapes + specialist tapes at A's current loss rates; B `LATEST=6,SELF=4,IL=1,TOP=0.5,SPEC=0.5,
  PASS=0.5` with IL_TEAMS=0:1 (the weakest clone only) -> ~70% self / past, ~12% IL@0, ~6% each leader / specialist
  tapes and PASS (PFSP raises the kinds it loses to).  SELF = a ring of 16 snapshots, one every 2 iterations.
- Schedules over HORIZON_SAMPLES (default 1e9 for the 1-hour runs; the resume passes the new total): A's KL weight 0.05 ->
  0.005 geometric until 0.9 x horizon (~0.016 after one hour), B's entropy bonus 0.01 -> 0.001 linear until 0.9 x horizon,
  B's win bonus 0 -> 0.5 over the first 300M samples; a RESUME continues each curve from its saved value to the end value
  over the remaining horizon (never tightens again).  Win-only phase = a resume with MARGIN_W=0 WINB=1 (the value head
  first refits for one iteration: VWARM_RESUME=1).
- Evaluation every 12 min + a final one (fp32 'highest', 64 games per GPU = 512 per opponent, paired vs iteration 0; a
  resume reuses eval_0.npz): A vs SNAP0 (its start) and IL@10; B vs PASS, IL@0, IL@10.  Every iteration also logs the
  training W / margin per opponent kind (log.txt and games.jsonl; `sprint.sh status` prints the last 12 rows).
- Exports: pt/it<N>.pt every 100M samples (ring of the last 6 + one per 500M + the final), state every 2 iterations,
  FINAL.json at the end.  SE guard OFF, midnight / final-fix / homing rules ON (as in training so far).
- Hour-2 B sanity (from the "From-scratch sanity test" above, at 8-GPU speed): the training W vs PASS must rise within
  the first 2-3 iterations (~50-70M samples) and pass 90% by ~150M; IL@0 margin gap closing at every evaluation.

## Multi-GPU

`DEVICES=N` (default: all local devices): `shard_map` data parallelism in one process. Games are sharded
(B per device), params / optimizer / bank are replicated, gradients and advantage statistics are averaged with
`pmean` / `psum`, so all replicas stay identical. Alternative `MODE=seeds` in `run_gpu.sh`: one independent run per
GPU with different seeds (no communication; choose the best by evaluation).

## Differences from the CPU pipeline (by design or not yet ported)

- No scripted opponents (STACK1_RACE4, V94, ...) and no pinned tape worlds (LADDER / LOSSES / TOP): the JAX
  engine plays networks only. Pinned worlds are possible later (A's engine accepts the exact end-of-day words;
  the opponent's recorded actions would be a per-step action tape).
- World randomness in training is the engine's key-based mode (same distribution as the Kaggle generator).
- Rollout / update math runs in TF32 on Ampere by default (`ROLLOUT_PRECISION` / `UPDATE_PRECISION=highest` for
  full fp32; `ROLLOUT_PRECISION=bf16` for speed); the exported policy runs in fp32 numpy as before.
- Opponent blocks are fixed for an episode; the learner seat alternates by game.

## Verification status (Sep 26-27)

| check | result |
|---|---|
| rule layer vs `features.midnight_fix` / `_plan_fix` on recorded top games (test_rules.py, 12 games, recorded + random + shed-stress actions) | 12,900 cases x 3 rule configs, 0 mismatches (midnight 207+, homing 283, final 67, SE-guard insertions 4,563); decode_market 0/4,000; qty clamp 0/20,147 |
| rule layer on real decision states of g20_u_dsm_s0 and rl2_it24 games (gate_cpu.py rules) | 2,876 states + 583 ten-order variants x 3 configs (default, SE_GUARD, SE+cap15+seed budget): 0 mismatches |
| PPO math vs il/ppo.py + il/ppo_live.py (test_loss.py) | logp 1e-7, loss 1e-7, gradients 3e-6 rel., GAE exact, Adam+clip 6e-8 |
| adapters vs dict references on real states (test_real_backend.py) | raw_from_state, view_from_raw, to_engine: bit-identical; greedy JAX pipeline = numpy Kaggle runtime 12/12 |
| whole games: trainer rollout (JAX engine, exact world) vs C++ engine + numpy runtime (test_e2e_game.py) | 2/2 games identical at every step (learner seat 0 and 1) |
| mock smoke (Mac CPU; server CPU with 2 devices) | PPO and value loss fall within every update; greedy margin vs the start policy -2 -> +1,138 (paired se 72); resume + torch export OK |
| LR = 0 on the mock | stored = recomputed log-probs (0 / 0 / 5e-11), max |ratio-1| 2.4e-7, clip 0, KL 1e-10 |
| gate 1, LR = 0 on the real engine + 20M on the A100 (TF32) | params bit-unchanged; clip fraction 0 in every minibatch; KL 1.6e-7; mean ratio 1.000001; max |ratio-1| 0.047-0.051 (unit ops), 0.013-0.017 (qty), 0.054-0.11 (market fires) with stored or recomputed old log-probs alike: TF32 numerics of differently compiled kernels, amplified by 1/T=5 and KSHARP=10 on rare low-probability actions |
| gate 2, rules on SE-guard states (model games with the 3rd land withheld) | 576 states, 326 guard-eligible (guard fired on all 326, 6 of them needing the 10-order drop): 0 mismatches |
| gate 3, rewards on a real 64-game episode (WINB 0.3) | sum of rewards vs final margin / 2e4 + WINB sign: max error 1.3e-7 (= $0.0025); GAE(lambda=1) returns = reward-to-go within 4.6e-6; dummy slot masked; done only at step 718 |
| gate 5, learning sanity (g20_w_dsm_s0, B=128, 3 iterations, 18 min) | KL 0.0013 / 0.0007, clip 0.006, value loss falls within every update, finite; greedy on 128 fixed worlds, paired vs before: vs the frozen start +$597 (se 459, W 0.35 -> 0.54), vs the IL clone as DSM +$1,857 (se 662, W 0.56 -> 0.66) |
| recorded leader tapes (test_tapes.py, Mac and server) | no overlap with the 1,317 benchmark episodes; whole recorded games replayed through the trainer's rollout reproduce the Kaggle rewards exactly (3/3 Mac, 4/4 server) |
| gate 1 control, same LR = 0 run in full fp32 (`*_PRECISION=highest`) | max |ratio-1| 1.3e-4 / 4e-5 / 1.3e-4, stored-vs-recomputed 1.3e-4, KL 6e-11: the TF32 deviations are kernel numerics, not logic |
| gate 4, export round trip (a GPU JAX-trained 20M snapshot, 2 updates) | `.pt` -> il/export.py (B64=1, RULES="SE_GUARD=1") -> numpy package in 2 C++ games: 2,876 / 2,876 decisions identical to the JAX greedy pipeline; the `.pt` loads in il/live_rollout.py and il/agent.BCAgent |
| TOP pinning (test_tape_pin.py, server, 60 leader seats, 32 in seat 1) | learner = PASS or a foreign recorded player: JAX == C++ pinned in 60/60 and 60/60 (unpinned 1/60 and 3/60, leader bank -$21k / -$15k) |

## Decisions after the gate (Sep 26-27)

- Old log-probs: `OLD_LOGP=behavior` (the exact probabilities the actions were sampled with, stored at rollout by the
  same `logp_terms`); with TF32 both choices show the same ~0.05 worst-case log-prob noise (recompute runs a
  differently compiled forward too), so the true behavior probabilities and one forward pass less win. The gap to a
  recompute is measured every `DIAG_EVERY=10` iterations (`old_diff_*` in stats.jsonl).
- Precision: TF32 (default) for rollouts and updates, fp32 softmax / log-probs / losses; greedy evaluation and
  checkpoint selection at `EVAL_PRECISION=highest` (TF32 flips ~0.3% of greedy action dicts, C's measurement).
  bf16 rollouts: not advised (audit: 3.2% of greedy ops flip, 0.69% of unit ratios start outside the clip band).
- planes / 255 through the exact table (`Q255`), not a division XLA may turn into a reciprocal multiply.
- Opponents: `OPP_MAP=map` (one forward per block, B/K >= 256). TOP (recorded leader worlds) is implemented and
  pinned, OFF by default until re-audited; enable with `OPPS=...,TOP=2` and `EVAL_OPPS=...,TOP`.
- Resume advances the random streams (fold in the iteration); SNAP0 is a fixed bank slot holding INIT.

## Launch (new machine, operator plan Sep 27: runs A and B, 10M only)

Setup: kagjax/, il/ (ckpt/rf_w_dsm_s0.pt, rf_w_s0.pt, export.py, np_runtime.py, features*.py), tools/, engine/v2 built for
the Python that runs JAX (`PY`, default /opt/jaxenv/bin/python: JAX with CUDA; the trainer replays the tapes in C++ at
start-up), data/tapes_top12, data/tapes_spec (+ tapes_spec_more when filled).  All commands from the repo root.

First hour:
1. `bash kagjax/train/preflight.sh`  (~10 min: GPUs, checkpoints, tape dirs, C++ engine, option tests, a 20-world
   pinning re-check, 2 iterations of each recipe on GPU 0) -> "PREFLIGHT OK".
2. `bash kagjax/train/dp_probe.sh 0,1,2,3`  (~15 min: recipe A, B=1024 per GPU, 3 iterations on 1 GPU and as one
   data-parallel run on 4) -> "DP OK" (finite, replicas identical, >= 3x the 1-GPU samples/s) or "DP FAIL".
   In parallel on the other half: `bash kagjax/train/launch_multi.sh sep28 bsweep`  (4 from-scratch variants, one per
   GPU 4-7, B=2048, eval vs PASS / IL@0 / IL@10 every 3 iterations; judge after ~45-60 min, see "From-scratch sanity").
3. DP OK: `bash kagjax/train/launch_multi.sh sep28 ab10m` once the sweep is judged (A on 0-3, B on 4-7, each one
   data-parallel run; B takes the sweep winner's settings via a plan file, e.g. `echo "4,5,6,7 B10m RECIPE=B SEED=2
   LR=3e-4" > plan_b; bash kagjax/train/launch_multi.sh sep28 plan_b`; A can start right after the probe with a plan
   line `0,1,2,3 A10m RECIPE=A SEED=1`).  DP FAIL: `bash kagjax/train/launch_multi.sh sep28 ab10m_seeds` (one run per
   GPU: A seeds on 0-3, B seeds on 4-7).  4x H100: the same commands (A on 0-1, B on 2-3; the sweep uses 2 variants).
Then:
4. Watch `kagjax/runs/sep28_*/log.txt` (per iteration: W/L by opponent kind incl. TOP / SPEC / PASS, KL, beta, entropy,
   clip, value fit, timings, [dp] replica check) and `eval.jsonl` (greedy fp32 on fixed worlds, paired vs iteration
   0; TOP / SPEC = every held-out tape world once).  Runs stop by themselves on NaN (and on KL > KL_MAX while an anchor
   is on).  `python kagjax/train/run_report.py kagjax/runs/<run>` prints the per-iteration table and the checkpoints.
5. Every 100M learner samples `pt/it<N>.pt` appears (+ `best.pt`): `B64=1 RULES="SE_GUARD=1" python il/export.py
   <pt> <main.py> 0.5 10`, then the CPU rulers (2400+ top band, both leader tests; judge on WINS).
6. Moving GPUs from the slower run to the faster one: stop both, `RESUME=kagjax/runs/<run>/state_<it>.npz` with the
   new GPU list (DEVICES may change: params / Adam / pool / sample counter are device-independent).
