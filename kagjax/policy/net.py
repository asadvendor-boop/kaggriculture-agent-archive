"""il/model.py Net in plain JAX (functional; params = dict pytree of float32 arrays).  menu 0 / ar 0 models only
(every v6 model we ship: fr_t_*, g20_*, rl2_*).

Architecture (identical to the torch module):
  glob' = [glob, team_emb[team]]                    (team embedding only when n_teams > 0)
  g     = relu(g2(relu(g1(glob'))))                  [N, gdim=160]
  x     = relu(stem([planes, coords]) + film(g))     3x3 conv, padding 1, NHWC here
  x     = R x (relu(x + c2(relu(c1(x)))))            residual blocks (lax.scan over stacked weights)
  unit  u = [cell under the unit (y*10+x, clamped 0..99), g, uin]  -> relu(u2(relu(u1(u)))) -> uop [44], uqty [12]
  market hm = relu(m2(relu(m1([mean_cells(x), max_cells(x), g]))))  -> mk [249], v [1], aux [n_aux]

API
  params, meta = load_ckpt(path, which="state")   torch checkpoint -> JAX params ("vstate" = the PPO value network)
  out = forward(params, planes, glob, pos, uin, team)   planes float [N,47,10,10] in 0..1 (= uint8 / 255, reference
        layout), glob [N,179], pos int [N,U,2] (x, y), uin [N,U,76], team int [N] (or None when n_teams == 0)
        -> dict(op [N,U,44], qty [N,U,12], mk [N,249], v [N], aux [N,n_aux] if present, hm [N,256] if with_hm)
  to_state_dict(params) -> {torch name: numpy array} (inverse of convert_state_dict; export trained params back to
        the torch checkpoint / numpy Kaggle runtime)
Precision: every dot / conv takes `precision` (default HIGHEST = full fp32; on GPU the XLA default would use TF32 on
Ampere, ~1e-3 relative error).  Pass precision=None to get the backend default for speed.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

HIGHEST = lax.Precision.HIGHEST
LINEARS = ("g1", "g2", "film", "u1", "u2", "uop", "uqty", "m1", "m2", "mk", "v", "aux")


def convert_state_dict(sd):
    """torch state_dict (tensors or numpy arrays) -> params pytree of numpy float32 arrays."""
    sd = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float32) for k, v in sd.items()}
    for k in sd:
        if k.startswith("ar_") or k.startswith("ar."):
            raise ValueError("autoregressive market head (ar) models are not supported by the JAX port")
    p = {}
    if "team.weight" in sd:
        p["team"] = sd["team.weight"]                                        # [n_teams, 16]
    p["coords"] = np.transpose(sd["coords"], (1, 2, 0)).copy()               # [2,10,10] -> [10,10,2] (x/9, y/9)
    p["stem"] = {"w": np.transpose(sd["stem.weight"], (2, 3, 1, 0)).copy(), "b": sd["stem.bias"]}   # OIHW -> HWIO
    nb = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    blk = {"c1w": [], "c1b": [], "c2w": [], "c2b": []}
    for i in range(nb):
        for c in ("c1", "c2"):
            blk[c + "w"].append(np.transpose(sd[f"blocks.{i}.{c}.weight"], (2, 3, 1, 0)))
            blk[c + "b"].append(sd[f"blocks.{i}.{c}.bias"])
    p["blocks"] = {k: np.stack(v) for k, v in blk.items()}                   # [L,3,3,W,W], [L,W]
    for n in LINEARS:
        if n + ".weight" in sd:
            p[n] = {"w": sd[n + ".weight"].T.copy(), "b": sd[n + ".bias"]}   # [in, out]
    return p


def to_state_dict(params):
    """params pytree -> {torch parameter name: numpy float32 array} (loadable with Net.load_state_dict)."""
    P = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float32), params)
    sd = {}
    sd["coords"] = np.transpose(P["coords"], (2, 0, 1)).copy()
    if "team" in P:
        sd["team.weight"] = P["team"]
    sd["stem.weight"] = np.transpose(P["stem"]["w"], (3, 2, 0, 1)).copy()
    sd["stem.bias"] = P["stem"]["b"]
    for i in range(P["blocks"]["c1w"].shape[0]):
        for c in ("c1", "c2"):
            sd[f"blocks.{i}.{c}.weight"] = np.transpose(P["blocks"][c + "w"][i], (3, 2, 0, 1)).copy()
            sd[f"blocks.{i}.{c}.bias"] = P["blocks"][c + "b"][i]
    for n in LINEARS:
        if n in P:
            sd[n + ".weight"] = P[n]["w"].T.copy()
            sd[n + ".bias"] = P[n]["b"]
    return sd


def load_ckpt(path, which="state"):
    """Torch checkpoint -> (params as jnp arrays, meta).  which: "state" (policy) or "vstate" (PPO value network)."""
    import torch
    c = torch.load(path, map_location="cpu", weights_only=False)
    if int(c.get("ar", 0) or 0) or int(c.get("menu", 0) or 0):
        raise ValueError(f"{path}: menu/ar variants are not supported (menu={c.get('menu')}, ar={c.get('ar')})")
    meta = {k: c.get(k) for k in ("width", "blocks", "n_teams", "g_in", "n_aux", "feat", "menu", "ar", "team")}
    params = convert_state_dict(c[which])
    return jax.tree_util.tree_map(jnp.asarray, params), meta


def save_npz(path, params, meta=None):
    """Portable checkpoint without torch: flat npz of the torch-layout state dict (+ meta as a json string)."""
    import json
    sd = to_state_dict(params)
    np.savez(path, __meta__=np.array(json.dumps(meta or {})), **sd)


def load_npz(path):
    import json
    z = np.load(path)
    meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
    sd = {k: z[k] for k in z.files if k != "__meta__"}
    return jax.tree_util.tree_map(jnp.asarray, convert_state_dict(sd)), meta


def _lin(p, x, precision):
    return jnp.dot(x, p["w"], precision=precision) + p["b"]


def _conv(x, w, b, precision):
    y = lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=precision)
    return y + b


def trunk(params, planes, glob, team=None, precision=HIGHEST, unroll=1):
    """-> grid features [N,10,10,W] (NHWC), global embedding g [N,gdim]."""
    relu = jax.nn.relu
    n = planes.shape[0]
    if "team" in params:
        if team is None:
            raise ValueError("this model has a team embedding: pass team ids")
        team = jnp.broadcast_to(jnp.asarray(team, jnp.int32), (n,))
        glob = jnp.concatenate([glob, params["team"][team]], 1)
    g = relu(_lin(params["g2"], relu(_lin(params["g1"], glob, precision)), precision))
    x = jnp.transpose(planes, (0, 2, 3, 1))
    x = jnp.concatenate([x, jnp.broadcast_to(params["coords"], (n,) + params["coords"].shape)], -1)
    x = relu(_conv(x, params["stem"]["w"], params["stem"]["b"], precision) + _lin(params["film"], g, precision)[:, None, None, :])

    def block(x, bp):
        h = relu(_conv(x, bp["c1w"], bp["c1b"], precision))
        return relu(x + _conv(h, bp["c2w"], bp["c2b"], precision)), None

    x, _ = lax.scan(block, x, params["blocks"], unroll=unroll)
    return x, g


def forward(params, planes, glob, pos, uin, team=None, precision=HIGHEST, with_hm=False, unroll=1):
    relu = jax.nn.relu
    x, g = trunk(params, planes, glob, team, precision, unroll)
    n, w = x.shape[0], x.shape[-1]
    flat = x.reshape(n, 100, w)                                            # cell index y*10 + x
    pos = jnp.asarray(pos, jnp.int32)
    idx = jnp.clip(pos[..., 1] * 10 + pos[..., 0], 0, 99)                   # [N,U]
    cell = flat[jnp.arange(n)[:, None], idx]                               # [N,U,W]
    nu = idx.shape[1]
    u = jnp.concatenate([cell, jnp.broadcast_to(g[:, None, :], (n, nu, g.shape[-1])), uin], -1)
    h = relu(_lin(params["u2"], relu(_lin(params["u1"], u, precision)), precision))
    pooled = jnp.concatenate([flat.mean(1), flat.max(1), g], -1)
    hm = relu(_lin(params["m2"], relu(_lin(params["m1"], pooled, precision)), precision))
    out = {"op": _lin(params["uop"], h, precision), "qty": _lin(params["uqty"], h, precision),
           "mk": _lin(params["mk"], hm, precision), "v": _lin(params["v"], hm, precision)[:, 0]}
    if "aux" in params:
        out["aux"] = _lin(params["aux"], hm, precision)
    if with_hm:
        out["hm"] = hm
    return out


forward_jit = jax.jit(forward, static_argnames=("precision", "with_hm", "unroll"))


def n_params(params):
    return int(sum(np.prod(a.shape) for a in jax.tree_util.tree_leaves(params)))
