"""GPU throughput / memory benchmark of the self-play trainer on the real engine and a real checkpoint.

  1. rollout-only (greedy learner vs greedy IL blocks, no recording): lockstep steps/s and game-steps/s for each B
  2. training iterations through ppo.main (rollout with recording + PPO update) with the given env config
Env: INIT (checkpoint), BENCH_B (comma list of batch sizes for part 1, default 256,1024), BENCH_L (steps, default 48),
     BENCH_TRAIN (0/1, default 1) + the usual ppo.py env vars for part 2 (B, K, SEG, MB, EPOCHS, ITERS, ...).
Writes a JSON summary to $RUN_DIR/bench.json.  Server: bash kagjax/srv_jax.sh --gpu python kagjax/train/bench_gpu.py
"""
import os, sys, json, time, dataclasses
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import ppo as P


def peak_gib():
    try:
        return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2 ** 30
    except Exception:
        return 0.0


def main():
    cfg = P.Cfg()
    os.makedirs(cfg.run_dir, exist_ok=True)
    out = {"devices": str(jax.devices()), "init": cfg.init, "rollout": [], "train": None}
    Ls = int(os.environ.get("BENCH_L", "48"))
    for B in [int(x) for x in os.environ.get("BENCH_B", "256,1024").split(",") if x]:
        c = dataclasses.replace(cfg, games=B, blocks=min(cfg.blocks, B), mb=min(cfg.mb, B * Ls))
        tr = P.Trainer(c)
        be = tr.be
        params, _ = be.load_params(cfg.init, "state")
        bank = jax.tree_util.tree_map(lambda p: jnp.stack([p, p]), params)
        cs = be.consts()
        K = c.blocks
        oi = jnp.asarray([1] * K, jnp.int32); ot = jnp.asarray(([10, 5, 0, 3] * K)[:K], jnp.int32)
        carry = tr.fresh_games(jax.random.PRNGKey(0), B)
        t0 = time.time()
        carry, _, _, _ = tr._rollout(params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(1), L=Ls, learner_greedy=False, record=False)
        jax.block_until_ready(carry[0].t); t_c = time.time() - t0
        t0 = time.time()
        carry, _, _, _ = tr._rollout(params, bank, cs, None, oi, ot, carry, jax.random.PRNGKey(2), L=Ls, learner_greedy=False, record=False)
        jax.block_until_ready(carry[0].t); dt = time.time() - t0
        r = dict(B=B, L=Ls, compile_s=round(t_c - dt, 1), s_per_step=dt / Ls, steps_per_s=Ls / dt, game_steps_per_s=B * Ls / dt,
                 t_end=int(carry[0].t), peak_gib=round(peak_gib(), 2))
        print("[rollout]", json.dumps(r), flush=True)
        out["rollout"].append(r)
        del carry, bank, params, tr
    if int(os.environ.get("BENCH_TRAIN", "1")):
        t0 = time.time()
        P.main(cfg)
        lines = [l for l in open(os.path.join(cfg.run_dir, "log.txt")) if l.startswith("it ")]
        out["train"] = dict(cfg=dataclasses.asdict(cfg), wall_s=time.time() - t0, iters=[l.strip()[:220] for l in lines], peak_gib=round(peak_gib(), 2))
    json.dump(out, open(os.path.join(cfg.run_dir, "bench.json"), "w"), indent=1)
    print("[bench] done", json.dumps({k: v for k, v in out.items() if k != "train"}), flush=True)


if __name__ == "__main__":
    main()
