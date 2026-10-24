"""Smoke test of the self-play trainer on the mock environment (CPU, tiny net, ~2-4 minutes on one core).

Checks: jit / shapes of the whole loop (rollout -> update -> pool -> eval -> checkpoints), that PPO learns the toy
objective (fraction of units taking the right op for their tile rises from chance; greedy margin vs the frozen start
policy rises; value loss falls), the resume path (state_*.npz) and the torch export (best.pt loads into il/model.Net,
forward equal to the JAX forward).
run: kagjax/train/run1.sh kagjax/train/smoke_test.py [iters]          (DEVICES=2 + XLA_FLAGS_EXTRA=
     --xla_force_host_platform_device_count=2 exercises the shard_map data-parallel path)
"""
import os, sys, json, time, dataclasses, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import jax
import jax.numpy as jnp
import ppo as P
import actions as A
import mock_env as M
import sample as S

SP = os.environ.get("SMOKE_DIR", "/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/smoke")


def toy_accuracy(be, params, n=64, seed=123):
    """Greedy policy: fraction of live units whose chosen op is the right one for the tile under them."""
    st = be.reset(jax.random.PRNGKey(seed), n)
    raw = be.observe(st, 0)
    ob = be.encode(raw, -jnp.ones((n, 16), jnp.int32))
    out = be.net_apply(params, ob.planes.astype(jnp.float32) / 255.0, ob.glob, ob.pos, ob.uin, jnp.full((n,), 10, jnp.int32))
    d = S.decide(out, be.legal(raw), ob.nu, jax.random.PRNGKey(0), 0.2, 10.0, jnp.full(21, 0.5), True)
    cls_good = jnp.asarray([5, 6, 8, 13])                                # policy classes of WATER HARVEST DIG CARE
    g = raw.g
    cell = g.pos[:, 0, :, 1] * 10 + g.pos[:, 0, :, 0]
    code = jnp.take_along_axis(g.tile[:, 0], cell, 1)
    live = jnp.arange(16)[None] < ob.nu[:, None]
    return float(((d.op == cls_good[code]) & live).sum() / live.sum())


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    shutil.rmtree(SP, ignore_errors=True)
    cfg = P.Cfg(backend="mock", episode_slots=48, init="tiny:16x1", games=32, blocks=4, seg=24, mb=128, epochs=4, iters=iters,
                T=1.0, lr=1e-3, vlr=1e-3, beta=0.0, rscale=100.0, eval_every=max(1, iters // 3), eval_n=32, eval_opps="SNAP0",
                snap_every=4, self_max=3, ckpt_every=max(1, iters // 2), pt_every=0, run_dir=SP,
                devices=int(os.environ.get("DEVICES", "1")))
    be = M.MockBackend(episode_slots=48)
    p0 = be.init_params(jax.random.PRNGKey(cfg.seed + 1), width=16, blocks=1)
    acc0 = toy_accuracy(be, p0)
    t0 = time.time()
    out = P.main(cfg)
    dt = time.time() - t0
    acc1 = toy_accuracy(be, jax.device_get(out["params"]))
    log = open(os.path.join(SP, "log.txt")).read().splitlines()
    its = [l for l in log if l.startswith("it ")]
    ep = [l.split("epochs loss ")[1].split(" | ")[0] for l in its]           # "+a->+b v c->d"
    dl = np.array([float(e.split()[0].split("->")[1]) - float(e.split()[0].split("->")[0]) for e in ep])
    dv = np.array([float(e.split()[2].split("->")[1]) - float(e.split()[2].split("->")[0]) for e in ep])
    fit = [float(l.split("(res ")[1].split()[0]) / max(1e-9, float(l.split("/ var ")[1].split(")")[0])) for l in its]
    ev = [json.loads(l) for l in open(os.path.join(SP, "eval.jsonl"))]
    evl = [l for l in log if l.startswith("[eval")]
    k = len(fit) // 3 or 1
    print(f"\n[smoke] {iters} iterations in {dt:.0f}s on {jax.devices()}")
    print(f"[smoke] PPO loss on each iteration's batch, first -> last epoch: mean change {dl.mean():+.4f} (decreased in {(dl < 0).mean():.0%} of iterations)")
    print(f"[smoke] value loss on each batch, first -> last epoch: mean change {dv.mean():+.4f} (decreased in {(dv < 0).mean():.0%})")
    print(f"[smoke] value fit (residual / return variance): first third {np.mean(fit[:k]):.2f} -> last third {np.mean(fit[-k:]):.2f}")
    print(f"[smoke] toy objective (greedy units on the right op): {acc0:.2f} -> {acc1:.2f}")
    print(f"[smoke] greedy eval vs the frozen start policy (margin, $): " + " -> ".join(f"{e['SNAP0'][1]:+.0f}" for e in ev))
    print("[smoke] " + evl[-1])
    se = float(evl[-1].split(" se ")[1].split(")")[0].replace(",", "")) if " se " in evl[-1] else 1e9
    ok = dl.mean() < 0 and dv.mean() < 0 and acc1 > acc0 and (ev[-1]["SNAP0"][1] - ev[0]["SNAP0"][1]) > 3 * se

    # resume from the last training state for 2 more iterations
    st = max((f for f in os.listdir(SP) if f.startswith("state_")), key=lambda f: int(f[6:-4]))     # newest (numeric)
    cfg2 = dataclasses.replace(cfg, resume=os.path.join(SP, st), iters=int(st[6:-4]) + 2, eval_every=0)
    P.main(cfg2)
    print(f"[smoke] resume from {st}: OK")

    # torch export of best.pt: loads into il/model.Net and matches the JAX forward
    try:
        import torch
        sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
        from model import Net
        c = torch.load(os.path.join(SP, "best.pt"), map_location="cpu", weights_only=False)
        n = Net(width=c["width"], blocks=c["blocks"], n_teams=c["n_teams"], g_in=c["g_in"], n_aux=c["n_aux"]); n.load_state_dict(c["state"]); n.eval()
        from kagjax.policy import net as N
        pj = jax.tree_util.tree_map(jnp.asarray, N.convert_state_dict(c["state"]))
        rng = np.random.RandomState(0)
        pl = (rng.rand(4, 47, 10, 10) < 0.2).astype(np.float32); gl = rng.rand(4, 179).astype(np.float32)
        pos = rng.randint(0, 10, (4, 16, 2)); uin = (rng.rand(4, 16, 76) < 0.2).astype(np.float32); tm = np.full(4, 10)
        with torch.no_grad():
            to = n(torch.from_numpy(pl), torch.from_numpy(gl), torch.from_numpy(pos), torch.from_numpy(uin), torch.from_numpy(tm))
        jo = N.forward(pj, jnp.asarray(pl), jnp.asarray(gl), jnp.asarray(pos), jnp.asarray(uin), jnp.asarray(tm), precision=jax.lax.Precision.HIGHEST)
        err = max(float(np.abs(to[i].numpy() - np.asarray(jo[k_])).max()) for i, k_ in enumerate(("op", "qty", "mk", "v")))
        print(f"[smoke] best.pt (it {c['it']}, team {c['team']}) loads into il/model.Net; torch vs JAX forward max abs diff {err:.2e}")
        ok &= err < 1e-4
    except Exception as e:
        print("[smoke] torch export check failed:", repr(e)); ok = False
    print("SMOKE OK" if ok else "SMOKE FAILED", flush=True)


if __name__ == "__main__":
    main()
