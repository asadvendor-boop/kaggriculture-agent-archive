"""Parity: JAX forward vs the PyTorch Net on the same inputs (random + real encoded observations), batched and
per-sample, fp32 on CPU; plus the params round trip (JAX -> torch state dict) and the speed of a jitted B=64 forward.

run: kagjax/policy/run1.sh -m kagjax.policy.test_net [ckpt ...]
"""
import sys, time, json, os
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import torch
torch.set_num_threads(1)
# torch's batched CPU conv goes through NNPACK (Winograd): its B>1 logits differ from its own B=1 logits by ~1e-3 on real
# inputs (measured 1.29e-3 on fr_t_dsm_s0), while the B=1 path agrees with the numpy Kaggle runtime to 4e-5.  The
# reference below therefore runs torch with NNPACK off, and the numpy runtime (np_runtime.NumpyNet) is checked too.
NNPACK_DEFAULT = torch._C._get_nnpack_enabled() if hasattr(torch._C, "_get_nnpack_enabled") else None
torch.backends.nnpack.set_flags(False)
import np_runtime
import jax
import jax.numpy as jnp
from kagjax.policy import net as N

CK = ["/Users/asad.ali/kaggriculture/il/ckpt/fr_t_dsm_s0.pt", "/Users/asad.ali/kaggriculture/il/ckpt/g20_u_dsm_s0.pt",
      "/Users/asad.ali/kaggriculture/il/ckpt/rl2_it24.pt"]
REAL = "/private/tmp/claude-501/-Users-asad-ali-kaggriculture/8f03e978-4361-433a-9817-8d9c5ae8de86/scratchpad/real_inputs.npz"


def torch_net(path, which="state"):
    from model import Net
    c = torch.load(path, map_location="cpu", weights_only=False)
    n = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"),
            n_aux=c.get("n_aux", 0), ar=c.get("ar", 0), menu=c.get("menu", 0))
    n.load_state_dict(c[which]); n.eval()
    return n, c


def random_inputs(rng, b, g_in=179, nu=16):
    planes = rng.randint(0, 256, (b, 47, 10, 10)).astype(np.float32) / 255.0
    planes *= (rng.rand(b, 47, 10, 10) < 0.3)                              # sparse like real planes
    glob = (rng.rand(b, g_in) * 2).astype(np.float32)
    pos = rng.randint(0, 10, (b, nu, 2)).astype(np.int64)
    uin = (rng.rand(b, nu, 76) < 0.15).astype(np.float32) * rng.rand(b, nu, 76).astype(np.float32) * 2
    team = rng.randint(0, 48, b).astype(np.int64)
    return planes, glob, pos, uin, team


def run_torch(n, planes, glob, pos, uin, team):
    with torch.no_grad():
        o = n(torch.from_numpy(planes), torch.from_numpy(glob), torch.from_numpy(pos), torch.from_numpy(uin),
              torch.from_numpy(team) if n.n_teams else None, aux=n.aux is not None)
    keys = ["op", "qty", "mk", "v", "aux"]
    return {k: v.numpy() for k, v in zip(keys, o)}


def run_jax(params, planes, glob, pos, uin, team):
    o = N.forward_jit(params, jnp.asarray(planes), jnp.asarray(glob), jnp.asarray(pos, jnp.int32), jnp.asarray(uin),
                      jnp.asarray(team, jnp.int32) if "team" in params else None)
    return {k: np.asarray(v) for k, v in o.items()}


def maxdiff(a, b):
    return max(float(np.abs(a[k] - b[k]).max()) for k in a if k in b)


def check(path, which="state", report=None):
    tn, c = torch_net(path, which)
    params, meta = N.load_ckpt(path, which)
    tag = os.path.basename(path) + ("" if which == "state" else ":" + which)
    print(f"== {tag}: width {meta['width']} blocks {meta['blocks']} teams {meta['n_teams']} g_in {meta['g_in']} "
          f"aux {meta['n_aux']}  params {N.n_params(params):,}", flush=True)
    rng = np.random.RandomState(0)
    res = {}
    sets = [("random", random_inputs(rng, 8))]
    if os.path.exists(REAL):
        z = np.load(REAL)
        sel = np.linspace(0, len(z["glob"]) - 1, 16).astype(int)
        sets.append(("real", (z["planes"][sel].astype(np.float32) / 255.0, z["glob"][sel], z["pos"][sel].astype(np.int64),
                              z["uin"][sel].astype(np.float32), np.full(16, 10, np.int64))))
    for name, inp in sets:
        ref = run_torch(tn, *inp)
        got = run_jax(params, *inp)
        d_b = {k: float(np.abs(ref[k] - got[k]).max()) for k in ref}
        d_s = 0.0; d_bs = 0.0
        for i in range(inp[0].shape[0]):
            one = tuple(a[i:i + 1] for a in inp)
            r1 = run_torch(tn, *one); g1 = run_jax(params, *one)
            d_s = max(d_s, maxdiff(r1, g1))
            d_bs = max(d_bs, max(float(np.abs(got[k][i] - g1[k][0]).max()) for k in g1))
        scale = max(float(np.abs(ref[k]).max()) for k in ref)
        P = {k: v.numpy() for k, v in c[which].items()}
        nn = np_runtime.NumpyNet(P, c["width"], c.get("blocks", 5), c.get("n_teams", 0))
        d_np = 0.0
        for i in range(inp[0].shape[0]):
            lop, lq, lm = nn.forward(inp[0][i], inp[1][i], inp[2][i], inp[3][i], int(inp[4][i]))
            d_np = max(d_np, float(np.abs(lop - got["op"][i]).max()), float(np.abs(lq - got["qty"][i]).max()),
                       float(np.abs(lm - got["mk"][i]).max()))
        with torch.backends.nnpack.flags(enabled=True):
            rn = run_torch(tn, *inp)
        d_nn = max(float(np.abs(rn[k] - got[k]).max()) for k in rn)
        print(f"   {name:6s} batched B={inp[0].shape[0]} max|jax-torch| " + " ".join(f"{k} {v:.2e}" for k, v in d_b.items())
              + f" | per-sample B=1 {d_s:.2e} | jax batch-vs-single {d_bs:.2e} | vs numpy runtime {d_np:.2e}"
              + f" | (vs torch+NNPACK batched {d_nn:.2e}) | max|logit| {scale:.1f}", flush=True)
        res[name] = dict(batched=d_b, per_sample=d_s, batch_vs_single=d_bs, numpy_runtime=d_np, torch_nnpack_batched=d_nn, scale=scale)
        # real inputs: absolute 1e-4; random (out-of-distribution, logits up to ~1e3): relative 1e-5 of max |logit|
        tol = 1e-4 if name == "real" else 1e-5 * max(1.0, scale)
        assert max(d_b.values()) < tol and d_s < tol and d_np < tol, (tag, name, d_b, d_s, d_np, tol)
    # round trip: JAX params -> torch state dict -> identical tensors
    sd = N.to_state_dict(params)
    ts = c[which]
    rt = max(float(np.abs(sd[k] - ts[k].numpy()).max()) for k in ts)
    assert set(sd) == set(ts.keys()), set(ts.keys()) ^ set(sd)
    print(f"   round trip to_state_dict: max diff {rt:.1e} over {len(sd)} tensors", flush=True)
    res["roundtrip"] = rt
    if report is not None:
        report[tag] = res
    return params, meta


def speed(params, b=64, reps=20):
    rng = np.random.RandomState(1)
    planes, glob, pos, uin, team = random_inputs(rng, b)
    args = (jnp.asarray(planes), jnp.asarray(glob), jnp.asarray(pos, jnp.int32), jnp.asarray(uin), jnp.asarray(team, jnp.int32))
    out = {}
    for prec_name, prec in (("highest", N.HIGHEST), ("default", None)):
        f = jax.jit(lambda p, a, b_, c, d, e: N.forward(p, a, b_, c, d, e, precision=prec))
        t0 = time.time(); jax.block_until_ready(f(params, *args)); tc = time.time() - t0
        ts = []
        for _ in range(reps):
            t0 = time.time(); jax.block_until_ready(f(params, *args)); ts.append(time.time() - t0)
        out[prec_name] = (tc, float(np.median(ts)))
        print(f"   speed B={b} precision={prec_name}: compile {tc:.1f}s, median {np.median(ts)*1e3:.1f} ms/forward "
              f"({np.median(ts)/b*1e3:.2f} ms/sample), 1 core", flush=True)
    return out


if __name__ == "__main__":
    cks = sys.argv[1:] or CK
    report = {}
    for p in cks:
        if not os.path.exists(p):
            print("missing", p); continue
        params, meta = check(p, "state", report)
        c = torch.load(p, map_location="cpu", weights_only=False)
        if "vstate" in c:
            check(p, "vstate", report)
        del c
        report[os.path.basename(p) + ":speed"] = speed(params)
    json.dump(report, open(os.path.join(os.environ.get("KJP_RESULTS", os.path.dirname(os.path.abspath(__file__))), "results_net.json"), "w"), indent=1)
    print("ALL NET PARITY CHECKS PASSED")
