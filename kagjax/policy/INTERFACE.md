# kagjax/policy: JAX port of the neural policy (Agent C). Interface for the trainer / engine

Status (Sep 26): network, v6 encoder, legal masks, greedy + sampled decisions, market decode and PPO log-probs are ported
and parity-tested against the production code (numbers in the results_*.json files next to this note):

| test | reference | result |
|---|---|---|
| test_net.py | torch Net (NNPACK off) + np_runtime.NumpyNet, fr_t_dsm_s0 (10M), g20_u_dsm_s0 (20M), rl2_it24 (policy + vstate) | real inputs: max abs logit diff <= 9.2e-5 (batched B=16 and per-sample); params round trip exact |
| test_encode.py | features6.encode_obs, every step, both seats: 24 games on the Mac (34,512 obs) + 200 games on the Linux server (287,600 obs) | planes, glob (bitwise), pos, unit features: 0 mismatches in 322,112 observations |
| test_act.py | np_runtime.NumpyAgent (greedy, tau 0.5, team 10, no rule layer), 3 checkpoints, Mac 25,488 + Linux server 37,000 action dicts | legal masks (644k unit rows), decode on identical logits and end-to-end decisions: 100% identical |
| test_act.py (C) | ppo.logp_terms (torch) | lpu / lpq <= 9.2e-5, pf <= 1e-5; lpm 6e-5 vs float64 (torch's float32 log(1-pf) is 3e-2 off near saturation) |
| test_adapter.py | engine state built by kagjax.state.obs_to_game from 3,456 real seat views | RawObs, encoder outputs, prices, Act -> engine Actions: exact |
| test_encode_fuzz.py | features6.encode_obs on 6,144 perturbed observations (up to 30 units, extreme money / market / tiles) | exact |
| test_sampling.py | live_rollout sampling (softmax(masked/T), rollout.sharpen fires) | 48 states x 4000 draws: op / qty TV <= 0.02 (noise), fire rates within 2.4 s.e.; rollout log-probs == logp_terms |
| test_selfplay.py | C++ engine + NumpyAgent (both seats), exact day words | full games on device (engine + adapter + encoder + net + decide in ONE jitted step) identical to the Kaggle stack: every decision, observations, final banks |
| test_selfplay.py RULES=1 | C++ engine + NumpyAgent WITH midnight_fix (= the Kaggle submission) | policy + train/actions.apply_rules on device: identical games, banks (the rule layer active: +$1-4k per game) |

## End-to-end identity at scale (Sep 26 evening, operator request; results_e2e_summary.json, e2e_out/)
JAX engine + this policy (greedy) + train/actions.apply_rules with the DEPLOYED rule set (MIDNIGHT, FINAL_FIX,
HOMING, SE_GUARD=1) vs the C++ engine + the REAL exported packages (main.py exec'd per game and seat exactly as
Kaggle / the harness load them), same seeds, exact day randomness; every action dict of both seats compared at every
step (92,032 per 64-game run) plus the final banks (e2e_jax.py / e2e_ref.py / e2e_compare.py / e2e_verify.py):

| run (64 games each) | A100 fp32 ("highest") | Mac CPU fp32 |
|---|---|---|
| rl2_it24 self-play (packages/RL24_SE: il/export.py, RULES=SE_GUARD=1, tau 0.5, team 10) | 64/64 identical | 63/64 |
| g20_u_dsm_s0 self-play (package/NEURAL_20M_SE_B64, the deployed 20M) | 63/64 | 63/64 |
| g20 (team 10) vs fr_t_s0 (team 0, packages/FRTS0_T0_SE), learner in both seats, two param sets in one program | 63/64 | 64/64 |

All 4 divergences (3 distinct states) were replayed and verified: inputs bit-identical up to the step, the flipped
decision is a float tie in the numpy runtime itself: NORTH vs HARVEST gap 9.5e-7, EAST vs COLLECT_FERTILIZER gap
1.9e-6, one market head at |P(order) - tau| = 3.9e-7; the two backends' logits agree to <= 4e-5 at those states.
Calibration: the deployed numpy runtime differs between the Mac (Accelerate) and Linux x86 (OpenBLAS) by up to 1.2e-4
on identical inputs (only 46% / 15% of op / market logits bit-identical), so two copies of the Kaggle runtime flip at
such ties too; bit-identity at a 1e-6 tie is not something any backend can promise.

TF32 ("default" precision on the A100) vs fp32 on the same seeds (fp32 shadow decisions on the same states):
rl2 60/64 and g20 59/64 games diverge (mean first divergence step 391 / 337; only 4/64 games stay identical to the
deployed runtime); 0.31% / 0.32% of action dicts flip (median 4-5 of 1,438 per game); per-game bank change vs fp32
-2,128 (se 1,333) / -27 (se 1,093), margin change +159 (se 334) / +853 (se 524): no significant bias, but rollouts are
NOT the deployed behaviour at ~0.3% of decisions.

## Modules## Modules

- `net.py`: `load_ckpt(path, which="state"|"vstate") -> (params, meta)`; `forward(params, planes, glob, pos, uin, team,
  precision=HIGHEST)` -> dict(op [N,U,44], qty [N,U,12], mk [N,249], v [N], aux [N,6]); `convert_state_dict` /
  `to_state_dict` (torch layout, exact round trip, for exporting trained params to torch / the numpy Kaggle runtime);
  `save_npz` / `load_npz` (torch-free checkpoints). Blocks are stacked ([L,3,3,W,W]) and run with `lax.scan`.
- `encode.py`: `RawObs` (documented field by field in the module docstring), `raw_from_obs(obs_dict)` (host),
  `encode(raw, prev_ops, tabs)` for ONE game (vmap it; `encode_batch` is the jitted vmap) -> dict(planes uint8
  [47,10,10], glob f32 [179], pos int32 [16,2], uf f32 [16,76], valid bool [16]); `net_inputs(enc, tabs)` -> the
  network inputs (planes / 255 via an exact table); `tables_jnp()` = the lookup tables (pass them as an argument into
  big jitted functions).
- `adapter.py`: engine GameState -> RawObs: `raw_from_game(g, t, seat, prices)` (one game), `raw_from_state(g, t,
  seat, prices)` (batch), `raw_both_seats(g, t, prices)` ([2B]: seat 0 rows then seat 1 rows); `actions_from_policy(act)`
  -> engine `state.Actions` (40 unit slots, units >= 16 PASS). Prices = the engine's table lookup (see
  test_selfplay.prices_tb for the in-jit version). (train/real_backend.py has its own raw_from_state; both are exact.)
- `act.py`: `legal_mask(raw)` [16,44]; `decide(raw, out, key, cfg)` (ONE game; greedy or sampled) -> Act + op / qact /
  qcmd / fire / mclass / log-probs; `policy_step(params, raw[B], prev_ops[B,16], key, team[B], cfg, tabs)` = encode +
  forward + decide for a batch (+ value, the encoder outputs for the buffer, `next_prev`); `logp_terms(...)` = ppo.py's
  logp_terms (differentiable); `decode_market(classes)`; `act_to_dict(act)`; `Cfg(greedy, T, tau, ks)` (static; tau
  may be a tuple of 21 per-head thresholds).

## Action format
`act.Act` has exactly the fields of `kagjax/train/actions.Act` (engine-native: u_op/u_arg/u_n/n_cmd, o_op/o_item/
o_n/n_ord; C++ op codes; items in engine order; BUY_SEED item = crop id). Convert with `train.actions.Act(*act)`.
Market orders are compacted in the leaders' canonical order (sells FERTILIZER WHEAT STRAWBERRY MILK WOOL EGG CARROT
TOMATO MELON, hires, land, animals, seeds, BUY_PRODUCT WHEAT / FERTILIZER), > 10 orders drop trailing SELLs first.

## Semantics to keep (they matter for parity with the Kaggle runtime)
- `prev_ops`: the op class each unit chose LAST step as the NETWORK chose it (before the rule layer), -1 for units
  that did not exist, all -1 at step 0. `policy_step` returns it as `next_prev`.
- The rule layer (features.midnight_fix: final fix, homing, midnight sale, plan rules) is not in act.py; apply
  `train/actions.apply_rules` to the Act. Greedy decisions of `decide` == NumpyAgent before midnight_fix.
- Quantity: `qact` is the sampled / argmax class BEFORE the clamp (what PPO's qact records); the command uses
  max(0, min(qc, have-1)) with have = shed (PICKUP) or the unit's inventory (PLACE) from the observation.
- Market sampling: fire ~ Bernoulli(sigmoid(ks*(logit P(order) - logit tau))), class = argmax of the non-zero classes;
  logit P(order) is computed as logsumexp(z[1:]) - z[0] (no float32 cancellation); pf clamped to [1e-6, 1-1e-6] as
  ppo.py. Use the same function (decide or logp_terms) for old and new log-probs.
- Cross-check of the trainer's own sampler (test_crosscheck_train.py, train/sample.py read-only): greedy decisions ==
  NumpyAgent on 3,700 decisions, unit log-probs identical to act.py; but its market log-prob uses float32
  `1 - softmax[0]` and `log(p/(1-p))`, which is up to 5e-2 off (float64 reference) for heads near saturation
  (act.py: 2e-6). Suggest `act.market_fire_logits` + `jax.nn.log_sigmoid(+-a)` (same clamps) so PPO ratios of
  saturated heads are not noise-dominated.
- Exactness trap: XLA rewrites `x / c` into `x * (1/c)` (not correctly rounded; e.g. step/720 is 1 ulp off in
  222 of 800 values). The encoder therefore uses exact lookup tables for every quotient; for planes / 255 use
  `net_inputs` (or `tabs["q255"][planes]`), not `planes / 255.0`. bfloat16 rollouts will not reproduce greedy decisions.
- The torch PPO buffer stores uin as float16 (x/9 etc. rounded); the JAX path keeps float32 (what the runtime sees).
- Lookup-table ranges (exact inside, documented fallbacks outside, never reached in real games): market inventory for
  the price forecast [-4096, 36864), money [0, 2^20) for money/1e5 and log1p(money)/12, observed prices < 4096.
  The tables are cached per platform (`_tables_v3_<OS>_<arch>.npz`, built in ~2 s on first use) because they embed
  that machine's float64 libm results, exactly what its numpy reference computes (Linux x86_64 = Kaggle's platform).
- menu=8 / ar (order-by-order market head) models are not ported (none of the shipped v6 models use them).

## Speed
Mac, 1 CPU core, jitted, B=64: forward 10M 340 ms (5.3 ms/sample), 20M 625 ms; encoder 0.45 ms (7 us/obs); decide
~1 ms.  At B=256 on CPU: encode 2.3 ms, decide 3.7 ms, forward 1364 ms (the forward is everything).

A100 40GB, SHARED with the live RL job at ~100% utilization (contended numbers, they fluctuate ~30%):

| | fp32 "highest" | TF32 "default" | bf16 |
|---|---|---|---|
| 10M forward B=64 | 22.7 ms (2.8k/s) | 8.0 ms (8.0k/s) | 4.0 ms (15.9k/s) |
| 10M forward B=256 | 52.5 ms (4.9k/s) | 14.1 ms (18.2k/s) | 7.8 ms (32.7k/s) |
| 10M forward B=1024 | 244 ms (4.2k/s) | 76 ms (13.4k/s) | - |
| 20M forward B=256 | 130 ms (2.0k/s) | 36 ms (7.0k/s) | 15.8 ms (16.2k/s) |
| 20M forward B=1024 | 410 ms (2.5k/s) | 140 ms (7.3k/s) | 62 ms (16.5k/s) |
| policy_step (encode+forward+sample) 10M B=256 | 48 ms | 25 ms | |
| policy_step 20M B=256 / B=1024 | 97 / 611 ms | 29 / 142 ms | |
| full game step on device (engine + both seats' policy + decide), 256 games, 10M | 108 ms = 2.35k game-steps/s (719-step batch: 78 s) | | |
| same, 20M | 274 ms = 936 game-steps/s (197 s) | | |

GPU vs torch CPU fp32 logits (128 real states): highest 1.5e-4 (10M) / 1.45e-4 (20M), greedy actions 128/128 equal
to CPU fp32; TF32 0.09-0.17, greedy actions 127/128 and 126/128; bf16 ~1.3 (not usable where decisions must match).
