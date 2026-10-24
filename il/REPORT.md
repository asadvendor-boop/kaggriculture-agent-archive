# Neural Kaggriculture agent: imitation learning, then RL (night of Sep 23-24, 2026)

## What was built
- **Data**: 4,758 public games of the current top-10 teams (Boey, M&M&P&Q, nah id win, DECEM, Unknown Mother-Goose,
  Vadim Vasilenko, 吃白饭的大肥鱼, THIRD FARM CLUB, Kaggledew Valley, mtmr_s1), pulled from Kaggle's public episode API
  and streamed into 20 KB tapes (`data/tapes_pull`, ~100 MB; raw 35 MB replays never written to disk), plus the Sep 19-22
  top-replay sets. Every leader seat was replayed in the C++ engine (all exact) and encoded:
  **6,298 leader seats, ~4.5M decision steps** (`il/data/leaders`, 1.1 GB).
- **Encoder** (`il/features.py`): 47 board planes (own + rival farm: tiles, crops, animals, watering/feeding state,
  yields, units), 98 global features (clock, money, seeds, shed, prices, market inventory, shops, hires, land), and
  per-unit features (index, position, inventory, previous command).
- **Network** (`il/model.py`, 1.3M parameters): residual CNN on the 10x10 farm with the global state broadcast into
  every cell, a per-unit head (reads the cell under each unit, 44 commands + quantity), a market head (21 order types),
  a value head, and an 11-way **team-style embedding** (which leader to play like).
- **Training** on the Apple M5 GPU (MPS), chunked streaming from disk (`il/train2.py`).
- **Runtime**: pure numpy (`il/np_runtime.py`), exported to a self-contained Kaggle `main.py` with float32 weights
  (`il/export.py`, `il/package.sh`); identical play to the torch model, ~1.3 s per full game on the M5.

## Imitation learning (IL)
| model | data | held-out command accuracy | closed loop |
|---|---|---|---|
| v1 DSM-only | 430 DSM games | 90.5% | 77% of DSM's bank in DSM's own worlds |
| v2 DSM-only, market orders weighted x3 | same | 90.7% | 84% of DSM's bank (tau 0.72) |
| v3a multi-leader | 1,785 older seats | 84.7% | DSM style 20/48 vs leader tapes, -$14.6k |
| v3b | 4,096 seats | 83.8% | DSM style 21/48, +$5.7k |
| v3c (2 epochs, all 6,121 seats) | all | 84.5% | Vadim style 26/48, +$26.3k |
| **v3d (4 epochs)** | all | 84.8% | see below |

Key lessons: rare-but-decisive orders (hiring, seeds) need a probability threshold instead of argmax; masking to legal
actions helps (+16 points) but forcing "another legal action" when the preferred one is illegal hurts; more data and
more epochs were the biggest levers; Vadim Vasilenko's style is the strongest one to imitate.

## RL (after imitation matured)
All RL ran in the C++ engine against top leaders' recorded games (our agent in the opponent's seat, world pinned),
starting from the imitation weights, with a KL anchor to the imitation policy.
- **PPO, exploratory sampling (T 0.4-0.5)**: flat or slightly worse; PPO improves the *sampled* policy while we deploy
  the *greedy* one.
- **Accept/reject updates on a 48-world test**: too noisy (per-game swings $40-60k); it chased noise.
- **Direct search on decision thresholds**: +$4.7k on its own 96 worlds, did not transfer to held-out worlds.
- **Self-imitation** (clone the best of several sampled games): best-of-2 beats greedy by ~$19k (lots of headroom), but
  cloning the winners destabilised the policy.
- **PPO with near-greedy sampling (T 0.2, sharp market decisions), from v3d** - the one that worked:
  iteration 30 vs v3d, paired: 300 training worlds +$1.6k (SE 0.9k); fresh games vs V94 **+$3.8k (SE 1.2k)**, 56/100 wins
  (v3d 41/100); fresh games vs STACK1_RACE +$1.5k (SE 1.45k). Pooled ~ **+$2.2k/game +- 0.65k**. Iteration 60 drifted back
  (-$0.4k), so iteration 30 is kept. A second PPO round from iteration 30 (lower learning rate) gave +$0.2k (SE 0.6k)
  after 20 more iterations and -$0.1k after 30: RL has converged at this level with these settings.

## Results of the delivered model
**Delivered: `package/NEURAL_RL30.tar.gz`** = imitation v3d (6,298 top-10 leader seats, 4 epochs) + near-greedy PPO
(30 iterations), Vadim Vasilenko style, market threshold 0.6. Numpy-only `main.py` (6.2 MB, float32 weights), plays
identically to the torch model, ~1.1 s per game on the M5; official kaggle_environments runner: 2/2 games exact.

| test (fresh closed-loop games unless noted) | v3d (IL) | **RL30 (IL + RL)** |
|---|---|---|
| vs STACK1_RACE (our live bot), 100 games | 46/100, -$4.3k | 43/100, **-$2.8k** |
| vs V94 (public 2945 Farm), 100 games | 41/100, -$2.8k | **56/100, +$0.9k** |
| vs V54 (public clone), 100 games | - | **59/100, +$1.1k** |
| vs top-10 leader recordings, 48 untouched test worlds | 25/48, +$23.3k | 25/48, +$23.8k |
| vs leader recordings, 300 paired training worlds | 150/300, +$21.0k | 155/300, +$22.5k |

Reference on the same leader-recording worlds: our scripted STACK1_RACE 30/48, +$37.5k (first 48 eval worlds).
Other models tried: v4 (2.3M params, 3 epochs from scratch) -$1.4k vs v3d; v3c+v3d ensemble -$0.3k (no gain).

**Honest verdict**: a pure neural agent trained overnight now plays at the level of the public top-family bots
(beats V54 and V94 slightly) and is close to, but still below, our scripted STACK1_RACE (-$2.8k/game, 43%).
It is not a replacement for STACK1_RACE on the leaderboard; it is a working base for further RL.

## Files
- Checkpoints: `il/ckpt/bc_lead_v3d.pt` (IL), `il/ckpt/rl_v3d_it30.pt` (IL+RL), `il/ckpt/bc_lead_v4.pt`.
- Packages: `package/NEURAL_RL30.tar.gz` (best), `package/NEURAL_V3D.tar.gz` (IL only).
- Logs: `il/logs/` (training, RL, evaluations); rows of all paired tests in `il/logs/big_eval_*.json`, `h2h_*.json`.
