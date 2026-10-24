"""Export path of JAX-trained params: perturbed INIT params (a stand-in for trained weights, every tensor changed)
-> train/ckpt_io.save_pt (the ppo_live torch format) -> the exact loading il/export.py does (state minus coords /
aux, feat / width / blocks / n_teams from the file) -> il/np_runtime.NumpyNet (the Kaggle runtime) forward, vs the
JAX forward (fp32 HIGHEST) of the same perturbed params on real encoded observations; plus the metadata export.py
reads, and export.py itself run on the file (the embedded weights decoded back and compared bit for bit).
usage (server, CPU): python export_check.py <init.pt> <prec_inputs.npz>
"""
import io
import os
import re
import subprocess
import sys
import zlib
import base64

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, R)
sys.path.insert(0, f"{R}/il")
sys.path.insert(0, f"{R}/kagjax/train")
os.environ.setdefault("JAX_CACHE_DIR", "off")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
from kagjax.policy import net as N  # noqa: E402
import ckpt_io  # noqa: E402
import np_runtime as NR  # noqa: E402


def main():
    init, inp = sys.argv[1], sys.argv[2]
    params, meta = N.load_ckpt(init, "state")
    key = jax.random.PRNGKey(3)
    leaves, tdef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    pert = [x * (1 + 0.05 * jax.random.normal(k, x.shape)) + 0.01 * jax.random.normal(jax.random.fold_in(k, 1), x.shape)
            for x, k in zip(leaves, keys)]
    p2 = dict(jax.tree_util.tree_unflatten(tdef, pert))
    p2["coords"] = params["coords"]                     # a buffer, never trained
    out_pt = os.path.join(HERE, "out", "export_check.pt")
    ckpt_io.save_pt(out_pt, p2, None, team=10, it=1)
    c = torch.load(out_pt, map_location="cpu", weights_only=False)
    print("meta:", {k: c.get(k) for k in ("width", "blocks", "n_teams", "g_in", "n_aux", "feat", "menu", "ar", "team")})
    st = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
    blocks = len({k.split(".")[1] for k in st if k.startswith("blocks.")})
    net = NR.NumpyNet(st, width=c["width"], blocks=blocks, n_teams=c.get("n_teams", 0))
    z = np.load(inp)
    idx = np.arange(0, z["planes"].shape[0], z["planes"].shape[0] // 24)[:24]
    planes = z["planes"][idx].astype(np.float32) / 255.0
    o = N.forward_jit(p2, jnp.asarray(planes), jnp.asarray(z["glob"][idx]), jnp.asarray(z["pos"][idx]),
                      jnp.asarray(z["uin"][idx]), jnp.full((len(idx),), 10, jnp.int32))
    worst = {"op": 0.0, "qty": 0.0, "mk": 0.0}
    for j, i in enumerate(idx):
        n = int(z["nu"][i])
        a, b, m = net.forward(planes[j], z["glob"][i], z["pos"][i, :n].astype(np.int64), z["uin"][i, :n], 10)
        worst["op"] = max(worst["op"], float(np.abs(a - np.asarray(o["op"][j, :n])).max()))
        worst["qty"] = max(worst["qty"], float(np.abs(b - np.asarray(o["qty"][j, :n])).max()))
        worst["mk"] = max(worst["mk"], float(np.abs(m - np.asarray(o["mk"][j])).max()))
    print("numpy runtime (exported weights) vs JAX fp32 forward, max abs logit diff:", worst)
    # il/export.py on the file, then decode the embedded blob and compare with the saved state
    main_py = os.path.join(HERE, "out", "export_check_main.py")
    r = subprocess.run([sys.executable, f"{R}/il/export.py", out_pt, main_py, "0.5", "10"], capture_output=True, text=True,
                       env=dict(os.environ, B64="1"))
    print("export.py:", r.stdout.strip()[-200:], r.stderr.strip()[-300:])
    src = open(main_py).read()
    blob = re.search(r"_BLOB = '([^']*)'", src).group(1)
    W = dict(np.load(io.BytesIO(zlib.decompress(base64.b64decode(blob)))))
    same = all(np.array_equal(W[k], st[k]) for k in st) and set(W) == set(st)
    print("embedded weights == saved state:", same, "| v6 encoder embedded:", "features6.py" in src,
          "| team/tau line:", re.search(r"_AGENT = .*", src).group(0))
    os.remove(main_py); os.remove(out_pt)


if __name__ == "__main__":
    main()
