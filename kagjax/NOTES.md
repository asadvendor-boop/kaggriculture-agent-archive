# kagjax NOTES (running log; newest status first)

## Status (Sep 26 evening; verification items below)
- Engine complete: all rules of sim_trace.hpp ported (unit phase, lockstep market, town, decay, end of day, weeds,
  shops, midnight capped drop with insertion order, atomic PLANT, dead slots, 10-order cap, cash/shed checks).
- Parity (full observe() dict of BOTH seats compared after EVERY step, incl. inventory key order; final rewards vs
  the tapes' Kaggle rewards): recorded top-12 games 0-49: 50/50 exact through all 720 steps (results/parity_top12_0_50.json).
- Market closed form vs literal lockstep port: 26k random scenarios, 0 mismatches (incl. 17k SELL/SELL, 3.2k
  BUY/BUY, 13k SELL-vs-BUY same slot same item, 1.2k at the FERTILIZER $1 floor); mutation check: 3/3 non-equivalent
  mutants caught (the 4th was an equivalent mutant: redundant stop flag).
- Fuzz (bots + random/malformed perturbations + forced same-slot interactions): 20/20 games exact (incl. 8 rich
  games with $50k-$135k banks).
- obs->state->obs round trip 20/20; restarting JAX mid-game (steps 100/300/500/650) from reference observations and
  continuing to the end: 20/20 exact.
- CPU throughput (1 core, realistic ~21 units/farm mid-day, 2-10 orders/turn): ~35k game-steps/s at B=256,
  ~50k at B=1024.

## Operator verification items (Sep 26 ~19:20 UTC request) - status (summary: results/verification_summary.json)
1. Full recorded-game parity, ALL 3,857 recorded games (data/tapes_top12 on the server; the 2 other files there are
   index.json / episodes.json):
   (a) digest route (gpu_verify.py build on the Mac from a copy of the server tapes: per-step C++ digest of the full
   state; replay as ONE batch): 3,857/3,857 identical to C++ at EVERY step (2.78M game-steps), final money = C++
   and = Kaggle's recorded rewards for 3,857/3,857; same with CFG_SEQ (42 s), CFG_GPU (119 s) and CFG_PAR (52 s),
   CPU. Divergences: none. -> results/replay_cpu_tapes_all3857_{seq,gpu,par}.json
   (b) independent dict route (parity.py on the server CPU: full observe() dicts of both seats incl. inventory key
   order, compared after every step): CFG_SEQ tapes 0-1929: 1,930/1,930 exact; CFG_PAR tapes 1930-3856:
   1,927/1,927 exact; final rewards = Kaggle 3,857/3,857. Divergences: none.
   -> results/server/parity_srv_seq_0_1930.json, results/server/parity_srv_par_1930_3860.json
   Earlier Mac dict runs: 250 + 25 + 10 + 13 tapes exact.
2. GPU = C++ (A100, cuda:0): datasets with per-step C++ digests of the full state (gpu_verify.py build, Mac):
   2,048 recorded games + 512 bot/perturbation fuzz games (61k SELL/SELL, 18.6k BUY/BUY, 13.6k SELL-vs-BUY same-slot
   same-item pairs, 294k hires, 11.5k land buys, ~300k shed ops, malformed actions). Replayed on the GPU as ONE batch
   each: tapes 2,048/2,048 identical to C++ at EVERY step with CFG_GPU (14 s incl. compile), CFG_SEQ (62 s) and
   CFG_PAR (77 s), final money = Kaggle 2,048/2,048; fuzz 512/512 identical with all three configs. Same datasets on
   CPU: identical too (so GPU = CPU = C++). -> results/server/replay_gpu_*.json, results/replay_cpu_*.json
3. Training randomness (rng_stats.py, deterministic JAX wheat bot in both seats, 4,096 exact-seed games vs 4,096
   jax.random games, CPU): no difference. final money KS p 0.66-0.87; weeds/game 7.55 vs 7.63 (KS p 0.50);
   weed rate 0.004975 vs 0.005023 over 6.2M rolls each (two-sample p 0.23, both consistent with 0.005);
   per-day weeds Fisher p 0.26; shop draws homogeneity p >= 0.31 at every unlock, (1st,2nd) pair p 0.27.
   Training mode: init(B, seed) = identical official initial states (the game has no per-seed initial state);
   per-game keys = split(PRNGKey(seed), B); step(words=None) uses the key ONLY at day ends (split, then
   jax.random.bits(624) uniform). Nothing else in init/step is random. -> results/rng_stats_cpu_4096.json
4. Batch independence (gpu_verify.py batchindep: target games alone B=1, together B=K, and among 4,096-K different
   re-seeded companion games at random batch positions; per-step digests): identical across batch sizes and
   identical to C++ at every step - CPU: CFG_SEQ / CFG_GPU / CFG_PAR with K=7 (B=1/7/4096); GPU (cuda:0): CFG_GPU
   K=7 (B=1/7/4096), CFG_SEQ and CFG_PAR K=3 (B=1/3/4096). -> results/batchindep_cpu_*.json,
   results/server/batchindep_gpu_*.json
Code audit for GPU hazards: the only scatters with possibly repeated indices (parallel unit rounds, shed-op order)
are unique by construction with out-of-range sentinels dropped; every lax.cond predicate and every loop bound is
unbatched (t or a batch maximum) or static; all gathers are clipped; integer-only arithmetic.

## Re-running the checks on a fresh machine (e.g. the multi-GPU host)
Needs: the kagjax/ directory; Python with jax (CUDA build on GPU), numpy, scipy; the C++ reference module
kagsim_trace (engine/v2/*.so for that Python/OS, or build engine/src_v2/kagsim_trace.cpp with pybind11) and
tools/harness.py (parity/fuzz/build import it; it hard-codes ROOT=/Users/asad.ali/kaggriculture, so mirror that path,
as the Vast server did); data/tapes_top12 (recorded games); for fuzz, the bot files listed in harness.AGENTS (missing
ones are skipped). The engine itself (consts/state/engine/rng/digest) needs only jax + numpy.
```
cd /Users/asad.ali/kaggriculture/kagjax          # on the Mac prefix python with ./env1.sh (single core)
python test_market.py --n 20000 && python test_market.py --n 20000 --count          # market oracle (CPU)
python test_units.py --n 20000 && python test_loopcfg.py --B 64 --steps 72           # unit phase, loop configs
python test_obs_roundtrip.py --n 6                                                   # obs<->state, mid-game restart
python parity.py --n 1930 --start 0    --batch 50 --cfg seq --out results/parity_seq_a.json   # dict parity (CPU,
python parity.py --n 1930 --start 1930 --batch 50 --cfg par --out results/parity_par_b.json   #  ~0.7 s/game/core)
python gpu_verify.py build --src tapes --start 0 --n 3857 --out results/ds_tapes_all.npz     # C++ digests (~17 min)
python gpu_verify.py build --src fuzz --n 512 --seed 101 --out results/ds_fuzz512.npz        # bots+fuzz (~13 min)
for c in gpu seq par; do python gpu_verify.py replay --data results/ds_tapes_all.npz --cfg $c \
    --out results/replay_tapes_$c.json; python gpu_verify.py replay --data results/ds_fuzz512.npz --cfg $c \
    --out results/replay_fuzz_$c.json; done                                          # device replays vs C++
python gpu_verify.py batchindep --data results/ds_tapes_all.npz --k 7 --sizes 1,7,4096 --cfg gpu --out results/bi_gpu.json
python gpu_verify.py batchindep --data results/ds_tapes_all.npz --k 3 --sizes 1,3,4096 --cfg seq --out results/bi_seq.json
python rng_stats.py --n 4096 --cfg gpu --out results/rng_stats.json                 # training randomness
python bench.py --B 16384 --steps 96 --reps 2 --full --cfg gpu                       # throughput
python summarize_verification.py                                                     # -> results/verification_summary.json
```
Multi-GPU: the engine is single-device; run one process per GPU (CUDA_VISIBLE_DEVICES) on disjoint games, or shard
the batch axis (t is shared and must stay replicated). On a GPU the default loop structure is CFG_GPU automatically.

## GPU (A100 40GB on the Vast server, SHARED: another job keeps it at ~100% util; all runs via srv_jax.sh --gpu)
| config | B | ms/step | game-steps/s |
|---|---|---|---|
| CFG_SEQ (default; dynamic loops + lax.cond) | 1,024 / 4,096 / 16,384 | 70 / 50-110 / 66 | 15k / 37-80k / 247k (full ep.) |
| CFG_PAR (round-based units, dynamic) | 4,096 / 16,384 | 71-111 / 94 | 37-57k / 174k |
| **CFG_GPU** (static trip counts, no conds) | **4,096 / 16,384** | **10.6 / 22.4** | **388k / 733k** |
| CFG_UNROLLED (fully unrolled) | 4,096 | 26 (with conds) | compile takes minutes |
Findings: on this shared GPU every device->host sync (a `while` with a data-dependent trip count, or a `lax.cond`)
costs ~1 ms (time-slicing with the other process), so the step must be sync-free: static trip counts (XLA knows
the trip count and does not sync) + `no_cond` (end of day computed every step and selected). CFG_GPU compiles in
~30-45 s. Per-phase (CFG_SEQ, B=4096): units 25 ms, market 21 ms, end of day 3 ms; unrolled: 2.6 / 5.5 / 1.3 ms.
Use CFG_GPU on GPU, CFG_SEQ on CPU. All configs are bit-identical (test_loopcfg.py: 12 variants).

## Design decisions
- State = `EnvState(t, g)`, `t` a scalar shared by the batch (synchronous episodes). All t-dependent branches
  (decay on even steps, town intervals, end of day, game over) are `lax.cond` on an UNBATCHED predicate, so under
  vmap they stay real branches (the end of day costs nothing on the other 23 steps).
- Loop bounds are batch maxima computed outside vmap: units loop to `max(n_units)`, market slots to the last
  non-empty slot in the batch -> no per-element select on loop carries.
- Tiles: int32[.., 100, 13] (one row per tile) so the sequential unit loop does one row gather + one row scatter.
  Hidden fields are canonical (C++ `Tile{}` on every kind change), so the observation fully determines the state.
- Unit inventory dict order (load-bearing when the 100-cap binds on DROP / midnight) = per-item insertion stamp
  (= step index; a unit inserts at most one new key per step). Capped pours use a prefix-sum closed form.
- Market: prices from an exact lookup table P[item, inv] (built with the C++ formula in float64 + round-half-even,
  inventory range [-10000, 50000], monotone asserted). Every slot is closed form: seeds/animals (no market
  interaction), solo SELL (prefix sums, $1 floor adds no supply), solo BUY_PRODUCT (binary search over prefix sums,
  cap <= 100), SELL/SELL same item (stride-2 sums, floor), BUY/BUY (stride-2 search then solo), SELL vs BUY
  (constant price while sale > $1; FERTILIZER at the $1 floor: a descent phase, then constant, else two solos).
  Prefix sums wrap mod 2^32 in int32; differences are exact (all real sums < 2^31).
- Randomness = raw MT19937 words of the day generator (uint32[624]); exact from the seed via CPython's random
  (rng.py), or i.i.d. from jax.random for training. Weed test done in integers (a*2^26+b <= 45035996273704).
- Money int32 (C++ double, but always integral).

## Faithfulness notes (C++ is the spec; where it differs from the official Python)
- C++ parses `["PLANT"]` / `["PICKUP"]` / `["PLACE"]` without an item as item 0 (WHEAT); official Python no-ops.
  We follow the C++ (never seen in recorded games; policy arrays avoid it anyway).
- Hinge price: C++ computes `8*e*e`, Python `8*(e**2)`; checked: identical after rounding over the whole table.
- C++ caps hands at 39 action entries and units at 40; Fibonacci hire costs make >25 hires/day unaffordable.

## Files
consts.py (ids, tables, exact price tables), state.py (layout, dict->arrays, obs<->state), engine.py (JAX engine),
rng.py (exact day words), parity.py (recorded-game parity), fuzz.py (adversarial fuzz), market_ref.py +
test_market.py (market oracle), test_obs_roundtrip.py, bench.py, env1.sh (single-core runner), spec.md.

## Next
- Optional: MT19937 seeding on device (exact Kaggle worlds without host-side rng.py words; training does not need it).
- Multi-GPU sharding of the batch axis for the rented host (not implemented; one process per GPU works today).
- Full-episode GPU throughput on an unshared GPU (all numbers above were taken on a GPU shared with a live RL run).
