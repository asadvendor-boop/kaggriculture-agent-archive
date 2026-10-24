"""Device-scale verification against the C++ reference, via per-step canonical state digests (digest.py).

  build     (CPU)  encode games into a compact dataset: seeds, every step's actions of both seats (exactly as
                   state.encode_action parses them) and the C++ reference digest of the full state after every step.
                   --src tapes: recorded games (data/tapes_top12, sorted, [start, start+n));
                   --src fuzz : bots (tools/harness, read-only) + random/malformed perturbations + forced same-slot
                                market interactions (fuzz.py), played in the C++ engine.
  replay    (any device) run the whole dataset as ONE batch through the JAX engine (exact day words from the seeds),
                   digest after every step, compare with the C++ digests: reports every game's first mismatching step.
  batchindep (any device) the first K games of a dataset alone (B=1 each), together (B=K) and embedded in large batches
                   of different companion games (other dataset games re-seeded): per-step digests must be identical
                   to each other and to the C++ digests.

usage: python gpu_verify.py build --src tapes --start 0 --n 2048 --out logs_server/ds_tapes.npz
       python gpu_verify.py replay --data logs_server/ds_tapes.npz --cfg par
       python gpu_verify.py batchindep --data logs_server/ds_tapes.npz --k 7 --sizes 1,7,4096 --cfg seq
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consts as C  # noqa: E402
import state as S  # noqa: E402
import rng as R  # noqa: E402

ACT_DT = [("uop", np.int8), ("uarg", np.uint8), ("un", np.int16), ("mtype", np.int8), ("mitem", np.uint8),
          ("mn", np.int32)]


def _enc(d):
    return S.encode_action(d)


def _pack(acts):
    """acts: list over steps of (a0, a1) encoded tuples -> dict of compact arrays [T,2,...]."""
    out = {}
    for k, (name, dt) in enumerate(ACT_DT):
        out[name] = np.stack([np.stack([a0[k], a1[k]]) for a0, a1 in acts]).astype(dt)
    return out


# ------------------------------------------------------------------------------------------------ build
def build(args):
    sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
    from harness import kagsim_trace, PASS
    import digest as D
    n = args.n
    seeds, names, digs, finals, tape_rewards = [], [], [], [], []
    packs = {name: [] for name, _ in ACT_DT}
    t0 = time.time()
    if args.src == "tapes":
        files = sorted(glob.glob(os.path.join(args.dir, "*.json.gz")))[args.start:args.start + n]
        n = len(files)
    else:
        import harness as H
        import fuzz as F
        r = random.Random(args.seed)
        bots = [b for b in F.DEFAULT_BOTS if os.path.exists(H.AGENTS.get(b, b))]
    for i in range(n):
        if args.src == "tapes":
            t = json.load(gzip.open(files[i]))
            seed = int(t["seed"])
            A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][k]] for k in (0, 1)]
            g = kagsim_trace.Game(seed, 720, False)
            acts, dg = [], []
            for s in range(720):
                acts.append((_enc(A[0][s]), _enc(A[1][s])))
                g.step(A[0][s], A[1][s])
                dg.append(D.ref_digest([g])[0])
            names.append(os.path.basename(files[i]))
            tape_rewards.append([float(x) for x in t["rewards"]])
        else:
            seed = r.randrange(2 ** 31)
            g = kagsim_trace.Game(seed, 720, False)
            pair = []
            for seat in range(2):
                if r.random() < args.p_random:
                    pair.append(("random", None))
                else:
                    nm = r.choice(bots)
                    pair.append((nm, H.mk(nm)()))
            obs = [g.observe(0), g.observe(1)]
            acts, dg = [], []
            for s in range(720):
                pa = []
                for seat in range(2):
                    ag = pair[seat][1]
                    if ag is None:
                        a = F.random_agent(r, obs[seat])
                    else:
                        try:
                            a = ag(obs[seat])
                        except Exception:
                            a = dict(PASS)
                    pa.append(F.perturb(r, a, obs[seat], args.p))
                F.entangle(r, pa[0], pa[1], args.p)
                if r.random() < 0.5:
                    F.entangle(r, pa[1], pa[0], args.p)
                acts.append((_enc(pa[0]), _enc(pa[1])))
                g.step(pa[0], pa[1])
                obs = [g.observe(0), g.observe(1)]
                dg.append(D.ref_digest([g])[0])
            names.append(f"fuzz_{seed}_{pair[0][0]}_{pair[1][0]}")
            tape_rewards.append([float("nan"), float("nan")])
        seeds.append(seed)
        digs.append(np.stack(dg))
        finals.append([g.reward(0), g.reward(1)])
        for k, v in _pack(acts).items():
            packs[k].append(v)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"built {i + 1}/{n} games, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(args.out, seeds=np.array(seeds, np.int64), names=np.array(names),
                        ref_digest=np.stack(digs, axis=1).astype(np.uint32),      # [720, N, 2]
                        ref_final=np.array(finals, np.float64), tape_reward=np.array(tape_rewards, np.float64),
                        **{k: np.stack(v, axis=1) for k, v in packs.items()})      # [720, N, 2, ...]
    print("saved", args.out)


# ------------------------------------------------------------------------------------------------ replay
def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def pick_cfg(name):
    import engine as E
    return {"seq": E.CFG_SEQ, "par": E.CFG_PAR, "gpu": E.CFG_GPU, "unrolled": E.CFG_UNROLLED}[name]


_RUNNERS = {}


def make_runner(cfg):
    """Returns run(state, acts_chunk[T,...,B,2,...], words[B,30,624]) -> (state, digests[T,B,2]) (cached per cfg,
    so a new batch shape only retraces)."""
    if cfg in _RUNNERS:
        return _RUNNERS[cfg]
    import jax
    import jax.numpy as jnp
    from jax import lax
    import engine as E
    import digest as D
    stepf = E.make_step(cfg)
    tb = E.tables()

    @jax.jit
    def run(st, ch, words):
        def body(st, a):
            a0 = S.Actions(*[x[:, 0].astype(jnp.int32) for x in a])
            a1 = S.Actions(*[x[:, 1].astype(jnp.int32) for x in a])
            st = stepf(tb, st, a0, a1, words)
            return st, D.digest(st.g)
        return lax.scan(body, st, ch)
    _RUNNERS[cfg] = run
    return run


def run_dataset(ds, idx, seeds, cfg, chunk=24):
    """Replay dataset games idx (with the given world seeds) as one batch; returns (digests[720,B,2], money[B,2])."""
    import jax
    import jax.numpy as jnp
    import engine as E
    B = len(idx)
    run = make_runner(cfg)
    words = jnp.asarray(np.moveaxis(R.batch_words(seeds), 0, 1))           # [B,30,624]
    st = E.init(B)
    out = []
    for s0 in range(0, 720, chunk):
        ch = tuple(jnp.asarray(ds[name][s0:s0 + chunk][:, idx]) for name, _ in ACT_DT)
        st, dg = run(st, ch, words)
        out.append(np.asarray(dg))
    return np.concatenate(out), np.asarray(st.g.money)


def compare(mine, ref, names):
    bad = []
    neq = np.any(mine != ref, axis=2)                                        # [720,B]
    for b in np.nonzero(np.any(neq, axis=0))[0]:
        bad.append({"game": str(names[b]), "first_step": int(np.argmax(neq[:, b]))})
    return bad


def replay(args):
    import jax
    ds = load(args.data)
    N = ds["seeds"].shape[0]
    n = N if args.n <= 0 else min(args.n, N)
    idx = np.arange(n)
    t0 = time.time()
    dig, money = run_dataset(ds, idx, [int(s) for s in ds["seeds"][:n]], pick_cfg(args.cfg))
    dt = time.time() - t0
    bad = compare(dig, ds["ref_digest"][:, :n], ds["names"][:n])
    fin_ok = int(np.sum(np.all(money.astype(np.float64) == ds["ref_final"][:n], axis=1)))
    tape = ds["tape_reward"][:n]
    has_tape = ~np.isnan(tape[:, 0])
    tape_ok = int(np.sum(np.all(money.astype(np.float64)[has_tape] == tape[has_tape], axis=1)))
    res = {"data": args.data, "device": str(jax.devices()[0]), "cfg": args.cfg, "games": n, "steps": 720,
           "games_all_steps_identical_to_cpp": n - len(bad), "final_money_equal_cpp": fin_ok,
           "final_money_equal_kaggle_tape": f"{tape_ok}/{int(has_tape.sum())}", "wall_s": round(dt, 1),
           "mismatches": bad[:50]}
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}))
    for m in bad[:10]:
        print("MISMATCH", m)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


def batchindep(args):
    import jax
    ds = load(args.data)
    N = ds["seeds"].shape[0]
    K = args.k
    seeds = [int(s) for s in ds["seeds"]]
    cfg = pick_cfg(args.cfg)
    ref = ds["ref_digest"][:, :K]
    runs = {}
    for size in [int(x) for x in args.sizes.split(",")]:
        if size == 1:
            dg = np.concatenate([run_dataset(ds, np.array([b]), [seeds[b]], cfg)[0] for b in range(K)], axis=1)
        else:
            assert size >= K
            comp = np.arange(K, K + size - K) % N                            # companions: other games...
            comp_seeds = [(seeds[c] + 7919 * (1 + j // N)) % (2 ** 31) for j, c in enumerate(comp)]   # ...re-seeded
            order = np.random.RandomState(size).permutation(size)            # targets at random batch positions
            idx = np.concatenate([np.arange(K), comp])[order]
            sd = [(seeds[:K] + comp_seeds)[o] for o in order]
            dg_all = run_dataset(ds, idx, sd, cfg)[0]
            pos = [int(np.nonzero(order == b)[0][0]) for b in range(K)]
            dg = dg_all[:, pos]
        runs[size] = dg
        same_ref = int(np.sum(np.all(dg == ref, axis=(0, 2))))
        print(f"B={size}: {same_ref}/{K} target games identical to C++ at every step", flush=True)
    sizes = list(runs)
    pair_ok = all(np.array_equal(runs[sizes[0]], runs[s]) for s in sizes[1:])
    res = {"data": args.data, "device": str(jax.devices()[0]), "cfg": args.cfg, "targets": K, "sizes": sizes,
           "identical_across_batch_sizes": pair_ok,
           "identical_to_cpp": {str(s): int(np.sum(np.all(runs[s] == ref, axis=(0, 2)))) for s in sizes}}
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "replay", "batchindep"])
    ap.add_argument("--src", default="tapes", choices=["tapes", "fuzz"])
    ap.add_argument("--dir", default=os.path.join(HERE, "..", "data", "tapes_top12"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.12)
    ap.add_argument("--p_random", type=float, default=0.15)
    ap.add_argument("--data", default="")
    ap.add_argument("--cfg", default="seq")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--sizes", default="1,7,4096")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    {"build": build, "replay": replay, "batchindep": batchindep}[args.mode](args)


if __name__ == "__main__":
    main()
