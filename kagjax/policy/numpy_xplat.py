"""Calibration: the deployed numpy runtime (il/np_runtime.NumpyNet, the code inside every exported package) on the
same real inputs on two machines (Mac: Accelerate BLAS; Linux x86 server: OpenBLAS, like Kaggle), plus the JAX fp32
forward.  Shows how far the reference itself moves across platforms, i.e. how often argmax near-ties can flip even
between two copies of the Kaggle runtime.

usage: python -m kagjax.policy.numpy_xplat run <ckpt> <out.npz>      (on each machine)
       python -m kagjax.policy.numpy_xplat cmp <mac.npz> <linux.npz>
"""
import sys, json
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")


def run(ckpt, out, n=256):
    import torch
    torch.set_num_threads(1)
    import np_runtime, features6 as F6, platform
    from kagjax.policy import refdata
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    P = {k: v.numpy() for k, v in c["state"].items()}
    net = np_runtime.NumpyNet(P, c["width"], c.get("blocks", 5), c.get("n_teams", 0))
    ops, mks, keys = [], [], []
    files = [f"{refdata.TAPES}/{x}" for x in ("112251604.json.gz", "112917757.json.gz", "113018253.json.gz", "113163447.json.gz")]
    for f in files:                                          # present on both machines
        for t, s, obs, prev in refdata.iter_tape_obs(f, every=11):
            pl, g, pos, uf = F6.encode_obs(obs, prev)
            lop, lq, lm = net.forward(pl.astype(np.float32) / 255.0, g, pos.astype(np.int64), uf, 10)
            o = np.zeros((16, 44), np.float32); o[:len(lop)] = lop
            ops.append(o); mks.append(lm.astype(np.float32)); keys.append(f"{f.split('/')[-1]}:{t}:{s}")
            if len(ops) >= n:
                break
        if len(ops) >= n:
            break
    np.savez(out, op=np.stack(ops), mk=np.stack(mks), keys=np.array(keys), platform=np.array(platform.platform()))
    print("saved", out, len(ops), platform.platform())


def cmp(a, b):
    A, Bz = np.load(a), np.load(b)
    assert list(A["keys"]) == list(Bz["keys"])
    d_op = np.abs(A["op"] - Bz["op"]); d_mk = np.abs(A["mk"] - Bz["mk"])
    print(f"numpy runtime {str(A['platform'])} vs {str(Bz['platform'])} on {len(A['keys'])} real states: "
          f"max |logit diff| ops {d_op.max():.2e}, market {d_mk.max():.2e}; identical logits in "
          f"{np.mean(d_op == 0) * 100:.1f}% (ops) / {np.mean(d_mk == 0) * 100:.1f}% (market) of entries")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        cmp(sys.argv[2], sys.argv[3])
