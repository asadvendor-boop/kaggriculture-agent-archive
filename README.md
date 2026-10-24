# Kaggriculture agents: archive (Kaggle game-agent benchmark, 2026)

This archives our agents for **Kaggriculture**, Kaggle's farming-market game-agent benchmark (Aug–Oct 2026). We finished **#84 of 10,148 (silver)**.

## Final agents

These are the two submissions that counted for the final ranking. They are attached to the **v1.0 release**:

| Agent | What it is | Live rating | Final score |
|---|---|---|---|
| `NEURAL_D63_P2_DSMVADIM.tar.gz` | Behaviour cloning on top leader games (DSM and Vadim styles), 10-block width-320 network, 63 team embeddings | 2525.2 | 4829.8 |
| `NEURAL_D63_RL_H3R3.tar.gz` | The same network fine-tuned with JAX PPO (learning-rate rounds 1e-5 → 5e-5 → 1e-4; coordinates frozen) | 2499.7 | 4799.2 |

sha256:
- `NEURAL_D63_P2_DSMVADIM.tar.gz`: `77c60c0b5e1e53c50783494b7abd88433faa267a8201e6a9eb0489c2f8a1c253`
- `NEURAL_D63_RL_H3R3.tar.gz`: `c6bf2ee3628512ea52850aa3121eafd6c3ea905d050e2f9a61b9fa4b6d8c8ce3`

Each archive holds one self-contained `main.py`. It contains the feature encoders, a numpy-only policy runtime, and the weights embedded as a compressed blob, so no torch is needed at inference.

## Code

| Folder | What it holds |
|---|---|
| `il/` | Imitation learning: dataset builders from game replays, training (`train2.py`), PPO fine-tuning (`ppo.py`, `ppo_live.py`), export to numpy (`export.py`), the ladder benchmark (`live/ladder.py`), and ops scripts (server IPs redacted). |
| `kagjax/` | A JAX port of the game engine plus a PPO trainer, for fast self-play. |
| `engine/`, `planner/` | Engine helpers and a look-ahead planning prototype. |

**Not included:**
- game replays and other competition data, which the rules do not allow us to redistribute;
- intermediate checkpoints, logs and other competitors' agents.

## Licence

- **Code:** MIT (`LICENSE`).
- **The two agent archives** (code plus embedded weights): CC BY 4.0, matching the benchmark's winner-licence type.

Author: Syed Asad Ali.
