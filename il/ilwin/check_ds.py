"""Integrity + label census of the winner datasets (Sep 29): every seat npz/x6 loads, shapes agree, no NaN/inf, manifest == files.
Usage: check_ds.py <dir> ...   (dir = il/data/win29 etc.)"""
import sys, os, glob, json, numpy as np
K = "/Users/asad.ali/kaggriculture"
for d in sys.argv[1:]:
    seats = sorted(glob.glob(f"{d}/seats/*.npz")); bad = []; n_op = n_mk = n_steps = 0; opq = 0
    for f in seats:
        key = os.path.basename(f)[:-4]
        try:
            z = np.load(f); T = z["planes"].shape[0]
            for k in ("glob", "pos", "ufeat", "prev_op", "op", "qty", "market", "money"):
                assert z[k].shape[0] == T, f"{k} T"
            assert np.isfinite(z["glob"].astype(np.float32)).all() and np.isfinite(z["ufeat"].astype(np.float32)).all(), "nan"
            x = np.load(f"{d}/x6/{key}.npy"); assert x.shape[0] == T and np.isfinite(x.astype(np.float32)).all(), "x6"
            n_steps += T; n_op += int((z["op"] >= 0).sum()); n_mk += int((z["market"] != -100).sum()); opq += int((z["op"] >= 0).sum() > 0)
            assert int(z["meta"][6]) == 10, "team"
        except Exception as ex:
            bad.append((key, str(ex)[:60]))
    man = {f"{g['ep']}_{g['winner']}" for g in json.load(open(f"{d}/manifest.json"))["games"]} if os.path.exists(f"{d}/manifest.json") else set()
    have = {os.path.basename(f)[:-4] for f in seats}
    print(f"{os.path.basename(d)}: seats {len(seats)} steps {n_steps} unit-op labels {n_op} market-head labels {n_mk} | bad {len(bad)} {bad[:3]} | "
          f"manifest-only {len(man - have)} files-only {len(have - man)}")
