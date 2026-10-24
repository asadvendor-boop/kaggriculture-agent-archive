"""Weight-averaged model ("uniform soup", Sep 27): checkpoints fine-tuned from the SAME parent (here g20_w5_s0) sit in one
basin, so the mean of their weights is usually at least as good as the average member and smooths out seed luck.
The network has no normalisation layers, so every float tensor is simply averaged.

  python live/soup.py <out.pt> <a.pt> <b.pt> ...          write the soup
  VAL=<data dir> python live/soup.py <out.pt> <a.pt> ...   also print the validation loss (train6's exact loss and
                                                          validation split) of the soup and of every member on that data
Run from il/ with the training env (BLOCKS=10 N_TEAMS=48 MK_NZ_W=3 LOSS_W=0.7 VAL_MAX=180 AUX_W=0.1) when VAL is set.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # il/: model, features
out, cks = sys.argv[1], sys.argv[2:]
assert len(cks) >= 2, "need at least two checkpoints"
cs = [torch.load(c, map_location="cpu", weights_only=False) for c in cks]
base = cs[0]
for k in ("width", "blocks", "n_teams", "g_in", "n_aux", "feat"):
    assert all(c.get(k) == base.get(k) for c in cs), f"members differ in {k}"
assert all(set(c["state"]) == set(base["state"]) for c in cs), "members differ in parameter names"
st = {}
for k, v in base["state"].items():
    st[k] = (sum(c["state"][k].double() for c in cs) / len(cs)).to(v.dtype) if v.is_floating_point() else v
soup = {kk: vv for kk, vv in base.items() if kk not in ("state", "val_loss", "it", "val_files")}
soup["state"] = st
soup["soup_of"] = cks
torch.save(soup, out)
num = sum(float(((c["state"][k].double() - st[k].double()) ** 2).sum()) for c in cs for k in st if st[k].is_floating_point())
den = sum(float((st[k].double() ** 2).sum()) for k in st if st[k].is_floating_point())
print(f"soup of {len(cs)} -> {out}; mean member distance from the soup {(num / len(cs) / den) ** 0.5:.4f} of its norm", flush=True)

if os.environ.get("VAL"):
    # train6's own data loading and loss (everything above its network construction), so the numbers match its logs
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train6.py")).read()
    head = src[:src.index('\nBLOCKS = int(os.environ.get("BLOCKS", "5"))')]
    sys.argv = ["train6.py", "/dev/null", "0", str(base["width"]), os.environ["VAL"]]
    g = {"__name__": "train6_head", "__file__": "train6.py"}
    exec(compile(head, "train6.py", "exec"), g)
    VA = g["load"](g["val_files"])
    NV = VA["op"].shape[0]

    def val_loss(state):
        net = g["Net"](width=base["width"], blocks=base["blocks"], n_teams=base["n_teams"], g_in=base["g_in"], n_aux=base["n_aux"])
        net.load_state_dict(state)
        net = net.to(g["dev"]).eval()
        stats = {k: 0 for k in ("op_ok", "op_n", "m_nz_ok", "m_nz_n", "m_fp", "m_z_n")}
        tot, nb = 0.0, 0
        with torch.no_grad():
            for b in range(0, NV, 2048):
                with g["amp_ctx"]():
                    l, _ = g["losses"](net, VA, torch.arange(b, min(NV, b + 2048), device=g["dev"]), stats)
                tot += l.item(); nb += 1
        return tot / nb, stats["op_ok"] / stats["op_n"]

    for name, state in [("SOUP", st)] + [(os.path.basename(c), cc["state"]) for c, cc in zip(cks, cs)]:
        vl, acc = val_loss(state)
        print(f"  val {name:<28} loss {vl:.4f}  op acc {acc:.4f}", flush=True)
