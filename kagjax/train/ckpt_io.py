"""Checkpoints of the JAX self-play trainer.

1. Training state (resume): one .npz with the policy / value params and both Adam states, stored in the torch
   state-dict layout (kagjax.policy.net.to_state_dict, so every array is named like the torch module), plus a JSON
   "__meta__" (iteration, config, opponent pool, Adam step counts, RNG key).
2. Torch checkpoint (.pt) in the il/ppo_live.py format {width, blocks, n_teams, team, g_in, n_aux, feat, menu, ar,
   state, vstate, it, score} -> il/export.py <pt> <main.py> <tau> <team> builds the Kaggle numpy runtime unchanged,
   and every CPU benchmark (bench_gov, pair_cmp, live_rollout NN:<pt>:<team>:<tau> opponents) can load it; ppo_live
   can even resume from it (vstate included when the value net has the policy's architecture).
3. Snapshot params (.npz, policy only) for the opponent pool.

CLI:  ckpt_io.py to_pt <train_state.npz> <out.pt>          (export a training checkpoint for il/export.py)
      ckpt_io.py check <init.pt>                            (round trip torch -> JAX -> torch, bit-exact check)
"""
import os, sys, json
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _N():
    from kagjax.policy import net as N
    return N


def to_sd(params):
    return _N().to_state_dict(params)


def from_sd(sd):
    return jax.tree_util.tree_map(jnp.asarray, _N().convert_state_dict(sd))


def arch_of(params):
    """(width, blocks, n_teams, g_in, n_aux) of a params pytree."""
    w = int(params["stem"]["w"].shape[-1]); nb = int(params["blocks"]["c1w"].shape[0])
    nt = int(params["team"].shape[0]) if "team" in params else 0
    td = int(params["team"].shape[1]) if "team" in params else 0
    g_in = int(params["g1"]["w"].shape[0]) - td
    n_aux = int(params["aux"]["w"].shape[1]) if "aux" in params else 0
    return dict(width=w, blocks=nb, n_teams=nt, g_in=g_in, n_aux=n_aux)


def _is_net(tree):
    return isinstance(tree, dict) and "stem" in tree


def _flat(tree, prefix=""):
    """Generic nested dict of arrays -> {"a.b": array} (small non-network trees, e.g. the VALUE_INIT=head value head)."""
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = np.asarray(v)
    return out


def _unflat(flat):
    out = {}
    for k, v in flat.items():
        d = out
        parts = k.split(".")
        for q in parts[:-1]:
            d = d.setdefault(q, {})
        d[parts[-1]] = jnp.asarray(v)
    return out


def save_train_state(path, params, vparams, opt_p, opt_v, meta):
    """opt_* = dict(m=pytree, v=pytree, t=int).  Network trees are stored in the torch state-dict layout (prefixes p/ v/
    pm/ pv/ vm/ vv/); a non-network value tree (the value head) under g-prefixes (vg/ vmg/ vvg/)."""
    arrs = {}
    for pre, tree in (("p/", params), ("v/", vparams), ("pm/", opt_p["m"]), ("pv/", opt_p["v"]), ("vm/", opt_v["m"]), ("vv/", opt_v["v"])):
        items = to_sd(tree).items() if _is_net(tree) else _flat(tree).items()
        if not _is_net(tree):
            pre = pre[:-1] + "g/"
        for k, a in items:
            arrs[pre + k] = a
    meta = dict(meta, opt_p_t=int(opt_p["t"]), opt_v_t=int(opt_v["t"]))
    tmp = path + ".partial"                        # (not matched by state_*.npz globs if a kill interrupts the write)
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=np.array(json.dumps(meta)), **arrs)
    os.replace(tmp, path)


def load_train_state(path):
    z = np.load(path)
    meta = json.loads(str(z["__meta__"]))
    grp = {}
    for k in z.files:
        if k == "__meta__":
            continue
        pre, name = k.split("/", 1)
        grp.setdefault(pre, {})[name] = z[k]
    params = from_sd(grp["p"])
    net_v = "v" in grp
    vl = (lambda k: from_sd(grp[k])) if net_v else (lambda k: _unflat(grp[k + "g"]))
    vparams = vl("v")
    opt_p = dict(m=from_sd(grp["pm"]), v=from_sd(grp["pv"]), t=jnp.int32(meta["opt_p_t"]))
    opt_v = dict(m=vl("vm"), v=vl("vv"), t=jnp.int32(meta["opt_v_t"]))
    return params, vparams, opt_p, opt_v, meta


def save_params_npz(path, params, meta=None):
    tmp = path + ".tmp.npz"
    np.savez(tmp, __meta__=np.array(json.dumps(meta or {})), **{"p/" + k: v for k, v in to_sd(params).items()})
    os.replace(tmp, path)


def load_params_npz(path, which="state"):
    """Snapshot / training-state npz -> (params, meta).  which: "state" (policy, prefix p/) or "vstate" (prefix v/)."""
    z = np.load(path)
    meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
    pre = "v/" if which == "vstate" else "p/"
    sd = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
    return from_sd(sd), meta


def save_pt(path, params, vparams=None, team=10, it=0, score=None, extra=None):
    """Torch checkpoint in the ppo_live format (loadable by il/model.Net, il/export.py, il/ppo.load)."""
    import torch
    a = arch_of(params)
    sd = {k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in to_sd(params).items()}
    d = dict(width=a["width"], blocks=a["blocks"], n_teams=a["n_teams"], team=int(team), g_in=a["g_in"], n_aux=a["n_aux"],
             feat=6 if a["g_in"] == 179 else 5, menu=0, ar=0, state=sd, it=int(it), score=score)
    if vparams is not None:
        vsd = {k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in to_sd(vparams).items()}
        if arch_of(vparams) == a:
            d["vstate"] = vsd                       # ppo_live can resume from this file
        else:
            d["vstate_other"] = vsd; d["vmeta"] = arch_of(vparams)
    d.update(extra or {})
    tmp = path + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)


def check_roundtrip(pt_path):
    """torch ckpt -> JAX params -> save_pt -> torch: every tensor bit-identical; the saved file loads into il/model.Net
    (strict) with the metadata export.py reads."""
    import torch, tempfile
    sys.path.insert(0, os.path.join(ROOT, "il"))
    from model import Net
    c = torch.load(pt_path, map_location="cpu", weights_only=False)
    params, meta = _N().load_ckpt(pt_path, "state")
    out = os.path.join(tempfile.mkdtemp(), "rt.pt")
    save_pt(out, params, params, team=meta.get("team") or 10)
    c2 = torch.load(out, map_location="cpu", weights_only=False)
    bad = [k for k in c["state"] if not torch.equal(c["state"][k].float(), c2["state"][k])]
    miss = sorted(set(c["state"]) ^ set(c2["state"]))
    n = Net(width=c2["width"], blocks=c2["blocks"], n_teams=c2["n_teams"], g_in=c2["g_in"], n_aux=c2["n_aux"], ar=c2["ar"], menu=c2["menu"])
    n.load_state_dict(c2["state"])                  # strict
    for k in ("width", "blocks", "n_teams", "g_in", "n_aux", "feat"):
        assert c2[k] == c.get(k, c2[k]), (k, c2[k], c.get(k))
    os.remove(out)
    return dict(tensors=len(c["state"]), mismatched=bad, key_diff=miss)


if __name__ == "__main__":
    if sys.argv[1] == "to_pt":
        p, v, _, _, meta = load_train_state(sys.argv[2])
        if not _is_net(v):                                  # value head: exported inside the policy's "v"
            p, v = dict(p, v=v), None
        save_pt(sys.argv[3], p, v, team=meta.get("team", 10), it=meta.get("it", 0), score=meta.get("score"))
        print("wrote", sys.argv[3], arch_of(p), "it", meta.get("it"))
    elif sys.argv[1] == "check":
        print(check_roundtrip(sys.argv[2]))
