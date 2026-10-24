"""Function-preserving growth of a trained policy network (Net2Net-style): wider trunk and extra residual blocks that
compute EXACTLY the same outputs at step 0, so continued training starts from the trained model instead of scratch.

Widening W -> W2: old weights are copied; new channels get fresh (random) incoming weights but ZERO outgoing weights
(every consumer of the new channels - the next convolutions, the unit head u1 and the market head m1 - reads them
with zero weight), so the old channels and every head see exactly what they saw before.  Deepening: each new
residual block has its last convolution zeroed, so it is relu(x + 0) = x (x >= 0 after the previous block's relu).
The zero outgoing weights receive gradient from the first step, so the new capacity starts learning immediately.

Usage: grow.py <in.pt> <out.pt> <new_width> <new_blocks>
"""
import sys
import torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from model import Net


def grow(c, W2, B2, seed=0):
    torch.manual_seed(seed)
    W, B = int(c["width"]), int(c.get("blocks", 5))
    assert W2 >= W and B2 >= B
    kw = dict(n_teams=c.get("n_teams", 0), g_in=c.get("g_in"), n_aux=c.get("n_aux", 0), ar=c.get("ar", 0), menu=c.get("menu", 0))
    old = Net(width=W, blocks=B, **kw); old.load_state_dict(c["state"])
    new = Net(width=W2, blocks=B2, **kw)                    # fresh init supplies the random incoming weights
    so, sn = old.state_dict(), new.state_dict()
    out = {}
    for k, v in sn.items():
        if k not in so:
            out[k] = v.clone(); continue
        o = so[k]
        if o.shape == v.shape and not (k.startswith(("stem.", "film.", "blocks.", "u1.", "m1."))):
            out[k] = o.clone(); continue
        t = v.clone()
        if k in ("stem.weight", "film.weight") or k in ("stem.bias", "film.bias"):
            t[:W] = o                                        # old output channels; new ones keep fresh init
        elif k.startswith("blocks.") and k.endswith(".weight"):   # conv [out, in, 3, 3] of an OLD block
            t[:W, :W] = o; t[:W, W:] = 0.0                   # old outputs ignore the new input channels
        elif k.startswith("blocks.") and k.endswith(".bias"):
            t[:W] = o
        elif k == "u1.weight":                               # input = [cell (W), g (gdim), uin]
            t.zero_(); t[:, :W] = o[:, :W]; t[:, W2:] = o[:, W:]
        elif k == "u1.bias":
            t = o.clone()
        elif k == "m1.weight":                               # input = [mean (W), max (W), g (gdim)]
            t.zero_(); t[:, :W] = o[:, :W]; t[:, W2:W2 + W] = o[:, W:2 * W]; t[:, 2 * W2:] = o[:, 2 * W:]
        elif k == "m1.bias":
            t = o.clone()
        else:
            raise ValueError(f"unhandled parameter {k} {tuple(o.shape)} -> {tuple(v.shape)}")
        out[k] = t
    for b in range(B, B2):                                   # new blocks: identity at step 0
        out[f"blocks.{b}.c2.weight"].zero_(); out[f"blocks.{b}.c2.bias"].zero_()
    new.load_state_dict(out)
    d = {k: v for k, v in c.items() if k not in ("state", "vstate", "opt", "vopt", "val_loss", "it")}
    d.update(state=new.state_dict(), width=W2, blocks=B2, grown_from=(W, B))
    return d, old, new


if __name__ == "__main__":
    src, dst, W2, B2 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    c = torch.load(src, map_location="cpu", weights_only=False)
    d, old, new = grow(c, W2, B2)
    torch.save(d, dst)
    print(f"{src}: width {c['width']} blocks {c.get('blocks', 5)} ({sum(p.numel() for p in old.parameters()):,} params) -> "
          f"width {W2} blocks {B2} ({sum(p.numel() for p in new.parameters()):,} params) -> {dst}")
