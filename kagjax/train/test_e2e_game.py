"""End-to-end game parity: the trainer's own rollout code (ppo.Trainer.rollout_local, greedy learner vs a greedy opponent
block, rule layer on) playing whole games in the JAX engine with the exact Kaggle world randomness of each seed
(real_backend: cs["words"]), versus the C++ engine played by the numpy Kaggle runtime (il/np_runtime.NumpyAgent with
features6 + midnight_fix) with the same models, teams, seats and seeds.  Compares the money margin after every step.
A difference can only come from a near-tied decision (fp32 accumulation order differs between numpy and XLA);
the step and size of the first difference are reported.
run (server CPU): bash kagjax/train/srv_launch.sh <log> INIT=<ckpt> python kagjax/train/test_e2e_game.py <n_games>
"""
import os, sys, time, copy
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
sys.path.insert(0, os.path.dirname(HERE))                     # kagjax/ (Agent A's modules: rng, engine, state, consts)
sys.path.insert(0, HERE)                                      # first: this trainer's ppo.py, not il/ppo.py
import numpy as np
import jax
import jax.numpy as jnp
import ppo as P
import rng as RNG                       # Agent A's exact end-of-day words
import actions as A


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ckpt = os.environ.get("INIT", "/Users/asad.ali/kaggriculture/il/ckpt/rf_w_dsm_s0.pt")
    team_l, team_o, tau = 10, int(os.environ.get("OPP_TEAM", "5")), 0.5
    seeds = [int(s) for s in os.environ.get("SEEDS", "424242,77001,5150,90210,31337,271828")[:1000].split(",")][:n]
    cfg = P.Cfg(backend="real", init=ckpt, games=n, blocks=1, seg=720, mb=720, epochs=1, eval_every=0, tau=str(tau), opp_tau=tau,
                rollout_precision="highest")
    tr = P.Trainer(cfg)
    params, _ = tr.be.load_params(ckpt, "state")
    bank = jax.tree_util.tree_map(lambda p: jnp.stack([p, p]), params)
    cs = dict(tr.be.consts(), words=jnp.asarray(RNG.batch_words(seeds)))
    carry = tr.fresh_games(jax.random.PRNGKey(0), n)                 # learner seat: game i -> i % 2
    t0 = time.time()
    carry, traj, (fin, mar), _ = tr._rollout(params, bank, cs, None, jnp.asarray([1], jnp.int32), jnp.asarray([team_o], jnp.int32), carry,
                                             jax.random.PRNGKey(1), L=720, learner_greedy=True, record=True)
    marg_j = np.cumsum(np.asarray(traj["rew"]) * cfg.rscale, 0)       # learner margin after each step [720, n]
    fin_j = np.asarray(mar)
    print(f"[jax] {n} games in {time.time() - t0:.0f}s, final learner margins {fin_j.round(0).tolist()}", flush=True)

    # C++ engine + numpy Kaggle runtime
    import torch
    import features6 as F6, np_runtime as NR
    from harness import kagsim_trace, PASS
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    W = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
    net = NR.NumpyNet(W, width=c["width"], blocks=c["blocks"], n_teams=c["n_teams"])
    ok = True
    for i, seed in enumerate(seeds):
        ls = i % 2
        ag = [None, None]
        ag[ls] = NR.NumpyAgent(net, F6, tau=tau, team=team_l); ag[1 - ls] = NR.NumpyAgent(net, F6, tau=tau, team=team_o)
        g = kagsim_trace.Game(int(seed), 720, False)
        t1 = time.time(); first = None; margins = []
        for t in range(720):
            o = [g.observe(0), g.observe(1)]
            a = [ag[p](copy.deepcopy(o[p])) for p in (0, 1)]
            g.step(a[0], a[1])
            m = g.reward(ls) - g.reward(1 - ls) if g.done else o_money(g, ls)
            margins.append(m)
            if first is None and abs(m - marg_j[t, i]) > 0.5:
                first = (t, m, float(marg_j[t, i]))
            if g.done:
                break
        fc = g.reward(ls) - g.reward(1 - ls)
        same = first is None and abs(fc - fin_j[i]) < 0.5
        ok &= same
        print(f"[game {i}] seed {seed} learner seat {ls}: C++/numpy final margin {fc:+,.0f} | JAX {fin_j[i]:+,.0f} | "
              + ("identical at every step" if same else f"first difference at step {first[0]}: C++ {first[1]:+,.0f} vs JAX {first[2]:+,.0f}")
              + f" ({time.time() - t1:.0f}s)", flush=True)
    print("E2E IDENTICAL" if ok else "E2E DIFFERENCES (see above)", flush=True)


def o_money(g, seat):
    o = g.observe(seat)
    return float(o["farms"][seat]["money"] - o["farms"][1 - seat]["money"])


if __name__ == "__main__":
    main()
