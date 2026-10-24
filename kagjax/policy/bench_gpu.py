"""GPU numbers for the JAX policy (run on the A100 via srv_jax.sh --gpu; keep each run to a few minutes).

  speed   : jitted forward throughput of the 10M / 20M nets for several batch sizes and precisions (HIGHEST = full
            fp32, DEFAULT = XLA default = TF32 matmuls/convs on Ampere, BF16 = params and inputs cast to bfloat16);
            policy_step (encoder + forward + decide) and one full on-device game step (engine + both seats' policy).
  parity  : GPU forward (HIGHEST / DEFAULT / BF16) vs torch CPU (fp32, NNPACK off) on real encoded observations, and
            greedy decisions on GPU vs on CPU (JAX CPU fp32 == Kaggle numpy runtime per test_act.py).

usage: python -m kagjax.policy.bench_gpu speed|parity [ckpt ...]
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import jax
import jax.numpy as jnp
from kagjax.policy import net as N, encode as E, act as A, refdata

CK10 = "/Users/asad.ali/kaggriculture/il/ckpt/fr_t_dsm_s0.pt"
CK20 = "/Users/asad.ali/kaggriculture/il/ckpt/g20_u_dsm_s0.pt"
OUT = os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))), "results_gpu.json")
PRECS = {"highest": N.HIGHEST, "default": None}
BS = [int(x) for x in os.environ.get("BENCH_BS", "64,256,1024,2048").split(",")]
REPS = int(os.environ.get("BENCH_REPS", "20"))
GAMES = [int(x) for x in os.environ.get("BENCH_G", "128,512").split(",")]
NPAR = int(os.environ.get("N_PARITY", "256"))
SKIP_FWD = int(os.environ.get("BENCH_SKIP_FWD", "0"))


def real_raws(n, every=13, games=6):
    raws, prevs = [], []
    for f in refdata.tape_files(games, start=3):
        for t, s, obs, prev in refdata.iter_tape_obs(f, every=every):
            raws.append(E.raw_from_obs(obs)); prevs.append(E.prev_array(prev))
    idx = np.linspace(0, len(raws) - 1, n).astype(int) if len(raws) >= n else np.arange(n) % len(raws)
    return E.stack([raws[i] for i in idx]), E.stack([prevs[i] for i in idx])


def fwd_fn(prec):
    if prec == "bf16":
        def f(p, pl, gl, pos, uin, team):
            c = lambda a: a.astype(jnp.bfloat16) if a.dtype == jnp.float32 else a
            out = N.forward(jax.tree_util.tree_map(c, p), c(pl), c(gl), pos, c(uin), team, precision=None)
            return {k: v.astype(jnp.float32) for k, v in out.items()}
        return jax.jit(f)
    return jax.jit(lambda p, pl, gl, pos, uin, team: N.forward(p, pl, gl, pos, uin, team, precision=PRECS[prec]))


def timeit(f, args, reps=None):
    reps = reps or REPS
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def speed(cks):
    dev = jax.devices()[0]
    res = {"device": str(dev), "forward": {}, "policy_step": {}, "game_step": {}}
    tabs = E.tables_jnp()
    raw_all, prev_all = real_raws(256)
    for ck in cks:
        params, meta = N.load_ckpt(ck)
        name = os.path.basename(ck)
        for B in ([] if SKIP_FWD else BS):
            idx = np.arange(B) % 256
            raw = jax.tree_util.tree_map(lambda a: jnp.asarray(a[idx]), raw_all); prev = jnp.asarray(prev_all[idx])
            enc = jax.jit(jax.vmap(E.encode, in_axes=(0, 0, None)))(raw, prev, tabs)
            pl, gl, pos, uin = E.net_inputs(enc, tabs)
            team = jnp.full(B, 10, jnp.int32)
            for prec in ("highest", "default", "bf16"):
                try:
                    t = timeit(fwd_fn(prec), (params, pl, gl, pos, uin, team))
                except Exception as e:                                      # OOM under the 20% cap
                    print(f"  {name} B={B} {prec}: failed {type(e).__name__}", flush=True); continue
                res["forward"][f"{name}|B={B}|{prec}"] = dict(ms=t * 1e3, samples_per_s=B / t)
                print(f"  forward {name} B={B:5d} {prec:8s}: {t*1e3:8.2f} ms  {B/t:10.0f} samples/s", flush=True)
            if B in (BS[1:3] if len(BS) > 2 else BS):
                for prec in ("highest", "default"):
                    cfg = A.Cfg(greedy=False, T=0.5, tau=0.5, ks=4.0)
                    f = jax.jit(lambda p, r, pr, k, tm: A.policy_step(p, r, pr, k, tm, cfg, tabs, PRECS[prec]))
                    t = timeit(f, (params, raw, prev, jax.random.PRNGKey(0), team))
                    res["policy_step"][f"{name}|B={B}|{prec}"] = dict(ms=t * 1e3, decisions_per_s=B / t)
                    print(f"  policy_step (encode+forward+sample) {name} B={B:5d} {prec:8s}: {t*1e3:8.2f} ms  {B/t:10.0f} decisions/s", flush=True)
        # full on-device game step: engine + adapter + policy for both seats (greedy, exact day words not needed)
        try:
            from kagjax.policy.test_selfplay import make_step
            from kagjax import engine as EN
            loops = [("seq", None)] + ([("par", EN.CFG_PAR)] if hasattr(EN, "CFG_PAR") else [])
            for G, (lname, loop) in [(G, l) for G in GAMES for l in loops]:
                st = EN.init(G)
                step = make_step(G, loop=loop)
                prev = jnp.full((2 * G, 16), -1, jnp.int32)
                words = jnp.zeros((G, 30, 624), jnp.uint32)
                key = jax.random.PRNGKey(0)
                st2, prev2, _, _ = step(params, st, prev, words, key)       # compile
                jax.block_until_ready(st2.g.money)
                for _ in range(30):                                          # advance into a mid-game-like state
                    st, prev, _, _ = step(params, st, prev, words, key)
                jax.block_until_ready(st.g.money)
                ts = []
                for _ in range(max(3, REPS // 2)):
                    t0 = time.perf_counter()
                    st, prev, _, _ = step(params, st, prev, words, key)
                    jax.block_until_ready(st.g.money); ts.append(time.perf_counter() - t0)
                t = float(np.median(ts))
                res["game_step"][f"{name}|games={G}|engine_{lname}"] = dict(ms=t * 1e3, game_steps_per_s=G / t, decisions_per_s=2 * G / t)
                print(f"  game step (engine[{lname}] + both seats' policy, fp32) {name} games={G}: {t*1e3:.1f} ms  "
                      f"{G/t:.0f} game-steps/s  ({2*G/t:.0f} decisions/s; a 719-step game batch takes {719*t:.0f}s)", flush=True)
        except Exception as e:
            print("  game step benchmark failed:", type(e).__name__, str(e)[:200], flush=True)
    return res


def parity(cks, n=None):
    n = n or NPAR
    import torch
    torch.set_num_threads(1)
    torch.backends.nnpack.set_flags(False)
    from model import Net
    tabs = E.tables_jnp()
    raw, prev = real_raws(n)
    cpu = jax.devices("cpu")[0] if any(d.platform == "cpu" for d in jax.devices("cpu")) else None
    res = {}
    for ck in cks:
        name = os.path.basename(ck)
        c = torch.load(ck, map_location="cpu", weights_only=False)
        tn = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"),
                 n_aux=c.get("n_aux", 0)); tn.load_state_dict(c["state"]); tn.eval()
        params, meta = N.load_ckpt(ck)
        enc = jax.device_get(jax.jit(jax.vmap(E.encode, in_axes=(0, 0, None)))(jax.tree_util.tree_map(jnp.asarray, raw), jnp.asarray(prev), tabs))
        pl = np.asarray(tabs["q255"])[enc["planes"].astype(np.int32)]
        team = np.full(n, 10, np.int64)
        with torch.no_grad():
            ref = {}
            for b0 in range(0, n, 64):  # torch CPU reference in chunks
                o = tn(torch.from_numpy(pl[b0:b0 + 64]), torch.from_numpy(enc["glob"][b0:b0 + 64]),
                       torch.from_numpy(enc["pos"][b0:b0 + 64].astype(np.int64)), torch.from_numpy(enc["uf"][b0:b0 + 64]),
                       torch.from_numpy(team[b0:b0 + 64]))
                for k, v in zip(("op", "qty", "mk", "v"), o):
                    ref.setdefault(k, []).append(v.numpy())
            ref = {k: np.concatenate(v) for k, v in ref.items()}
        valid = enc["valid"]
        # greedy decisions on CPU fp32 (== numpy runtime) as the reference for decision agreement
        cfg = A.Cfg(greedy=True, tau=0.5)
        rawj = jax.tree_util.tree_map(jnp.asarray, raw)
        dec_ref = None
        if cpu is not None:
            with jax.default_device(cpu):
                pc = jax.device_put(params, cpu)
                dec_ref = jax.device_get(jax.jit(lambda p, r, pr: A.policy_step(p, r, pr, jax.random.PRNGKey(0), jnp.full(n, 10, jnp.int32), cfg, E.tables_jnp(), N.HIGHEST))(pc, jax.device_put(rawj, cpu), jax.device_put(jnp.asarray(prev), cpu)))
        for prec in ("highest", "default", "bf16"):
            f = fwd_fn(prec)
            got = jax.device_get(f(params, jnp.asarray(pl), jnp.asarray(enc["glob"]), jnp.asarray(enc["pos"]), jnp.asarray(enc["uf"]),
                                   jnp.full(n, 10, jnp.int32)))
            d = {k: float(np.abs(np.where(valid[..., None], ref[k] - got[k], 0)).max()) if k in ("op", "qty")
                 else float(np.abs(ref[k] - got[k]).max()) for k in ("op", "qty", "mk", "v")}
            agree = None
            if dec_ref is not None:
                if prec == "bf16":
                    dg = None
                else:
                    dg = jax.device_get(jax.jit(lambda p, r, pr: A.policy_step(p, r, pr, jax.random.PRNGKey(0), jnp.full(n, 10, jnp.int32), cfg, tabs, PRECS[prec]))(params, rawj, jnp.asarray(prev)))
                if dg is not None:
                    same = [all(np.array_equal(np.asarray(a)[i], np.asarray(b)[i]) for a, b in zip(dg["act"], dec_ref["act"])) for i in range(n)]
                    agree = f"{sum(same)}/{n}"
            res[f"{name}|{prec}"] = dict(max_abs_diff_vs_torch_cpu=d, greedy_actions_equal_to_cpu=agree)
            print(f"  parity {name} GPU {prec:8s} vs torch CPU fp32: " + ", ".join(f"{k} {v:.2e}" for k, v in d.items())
                  + (f" | greedy actions equal to JAX-CPU fp32: {agree}" if agree else ""), flush=True)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    cks = sys.argv[2:] or [CK10, CK20]
    print("devices:", jax.devices(), flush=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    key = mode + ":" + "+".join(os.path.basename(c).replace(".pt", "") for c in cks) + (":game_step_only" if SKIP_FWD and mode == "speed" else "")
    res[key] = speed(cks) if mode == "speed" else parity(cks)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)
