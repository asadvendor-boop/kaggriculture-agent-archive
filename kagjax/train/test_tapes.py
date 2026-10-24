"""Recorded leader worlds (tapes.py + RealBackend.step_tapes + the trainer's rollout):
  1. no tape of the training / held-out banks is a benchmark episode (leader tests, ladder test, held-out lists);
  2. replay: both seats replay a recorded game through the trainer's own rollout (tape ids, learner seat = the
     non-leader seat, the seed's exact words) -> the final money of both seats equals the recorded Kaggle rewards;
  3. a learner (tiny network) playing against tape opponents runs (rollout with recording + one PPO update).
run: kagjax/train/run1.sh kagjax/train/test_tapes.py [n_tapes]
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import ppo as P
import tapes as TP


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ok = True
    bench = TP.benchmark_episodes()
    tr_sel, _ = TP.select([f"{TP.R}/data/tapes_top12"], TP.LEADERS_DEFAULT, "train")
    he_sel, _ = TP.select([f"{TP.R}/data/tapes_top12"], TP.LEADERS_DEFAULT, "heldout")
    eps = lambda sel: {int(os.path.basename(p).split(".")[0]) for p, _ in sel}
    clash = (eps(tr_sel) | eps(he_sel)) & bench
    print(f"[bench] {len(bench)} benchmark episodes; leader seats: train {len(tr_sel)}, held-out {len(he_sel)}; overlap with benchmarks: {len(clash)}")
    ok &= not clash and not (eps(tr_sel) & eps(he_sel))
    bank = TP.build_bank(split="train", max_n=n, both=True)
    cfg2 = P.Cfg(backend="real", init="tiny:8x1", games=n, blocks=1, seg=720, mb=720, epochs=1, eval_every=0)
    tr = P.Trainer(cfg2)
    be = tr.be
    params = be.init_params(jax.random.PRNGKey(0), width=8, blocks=1)
    pbank = jax.tree_util.tree_map(lambda p: jnp.stack([p, p]), params)
    cs = be.consts()
    dev = {k: jnp.asarray(v) for k, v in TP.device_arrays(bank).items()}
    tid = np.arange(n); ls = bank["lseat"]
    carry = tr.fresh_games(jax.random.PRNGKey(1), n, tid, ls)
    t0 = time.time()
    carry, traj, (fin, mar), _ = tr._rollout(params, pbank, cs, dev, jnp.asarray([1]), jnp.asarray([10]), carry, jax.random.PRNGKey(2),
                                             L=720, learner_greedy=True, record=False)
    money = np.asarray(be.money(carry[0]))
    for i, (ep, team, s, rec_leader, rec_learner, *_) in enumerate(bank["meta"]):
        same = money[i, s] == rec_leader and money[i, 1 - s] == rec_learner
        print(f"[replay] ep {ep} leader {team} (seat {s}): engine {money[i].tolist()} vs recorded leader {rec_leader} learner {rec_learner}: {'IDENTICAL' if same else 'DIFFERENT'}")
        ok &= bool(same)
    print(f"  ({time.time() - t0:.0f}s)")
    # 3. tiny learner vs the tapes (leader actions only), with recording and one update
    dev2 = {k: v for k, v in dev.items() if not k.startswith("l") or k == "lseat"}
    carry = tr.fresh_games(jax.random.PRNGKey(3), n, tid, ls)
    carry, traj, (fin, mar), boot = tr._rollout(params, pbank, cs, dev2, jnp.asarray([1]), jnp.asarray([10]), carry, jax.random.PRNGKey(4),
                                                L=720, learner_greedy=False, record=True)
    st = tr._update(jax.tree_util.tree_map(jnp.array, params), jax.tree_util.tree_map(jnp.array, params), P.adam_init(params), P.adam_init(params),
                    params, traj, boot, jax.random.PRNGKey(5), train_policy=True)[4]
    print(f"[learner vs tapes] final learner margins {np.asarray(mar).round(0).tolist()}; update loss {float(st['loss']):+.4f} finite {float(st['finite'])}")
    ok &= float(st["finite"]) == 1.0
    print("TAPES OK" if ok else "TAPES FAILED", flush=True)


if __name__ == "__main__":
    main()
