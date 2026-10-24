"""Format parity: build_winner_ds.build_game on a leader tape (hygiene off) must reproduce the existing fresh2 seat file
(il/build_dataset3.py output) and its v6 features (il/data/x6) exactly."""
import sys, os, json, collections, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_winner_ds as B
K = B.K
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 113672447
path = f"{K}/data/tapes_top12/{ep}.json.gz"
args = types.SimpleNamespace(fail_mask=0, strip_churn=0, team_mode="true", generic_team=48, se_label="drop", no_policy=True,
                             device="cpu", verify=0, src_index={})
voc = {t: i for i, t in enumerate(json.load(open(f"{K}/il/data/team_vocab.json")))}
stats = collections.Counter()
r = B.build_game(path, "tapes_top12", args, voc, stats)
b, x6, meta, info = r
W = info["winner"]
ref = f"{K}/il/data/fresh_all/{ep}_{W}.npz"
z = np.load(ref); x6r = np.load(f"{K}/il/data/x6/{ep}_{W}.npy")
print("winner seat", W, "steps", info["steps"], "ref T", z["op"].shape[0], "meta ours", list(meta[:8]), "ref", list(z["meta"]))
ok = True
for k in ("planes", "glob", "pos", "ufeat", "prev_op", "op", "qty", "market", "money"):
    same = b[k].shape == z[k].shape and np.array_equal(b[k], z[k])
    ok &= same
    print(f"  {k:8s} {str(b[k].dtype):8s} {b[k].shape} exact={same}" + ("" if same else f" ndiff={int((b[k] != z[k]).sum()) if b[k].shape == z[k].shape else 'shape'}"))
same = x6.shape == x6r.shape and np.array_equal(x6, x6r)
print(f"  x6       {x6.dtype} {x6.shape} exact={same}"); ok &= same
print("FORMAT PARITY", "OK" if ok else "MISMATCH", dict(stats))
