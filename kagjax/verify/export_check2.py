"""Second half of export_check.py on the files it left behind: decode the weights embedded by il/export.py (base85 or
base64, whichever the file uses) and compare bit for bit with the .pt state; then delete both large files."""
import io, os, re, sys, zlib, base64
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
pt, main_py = os.path.join(HERE, "out", "export_check.pt"), os.path.join(HERE, "out", "export_check_main.py")
c = torch.load(pt, map_location="cpu", weights_only=False)
st = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
src = open(main_py).read()
blob = re.search(r"_BLOB = '([^']*)'", src).group(1)
dec = base64.b64decode if "_b64.b64decode(_BLOB)" in src else base64.b85decode
W = dict(np.load(io.BytesIO(zlib.decompress(dec(blob)))))
print("decoder:", dec.__name__, "| tensors", len(W), "| embedded weights == saved state:",
      set(W) == set(st) and all(np.array_equal(W[k], st[k]) for k in st),
      "| features6 embedded:", "features6.py" in src, "|", re.search(r"_AGENT = .*", src).group(0),
      "|", re.search(r"_NET = .*", src).group(0))
os.remove(main_py); os.remove(pt)
