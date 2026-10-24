"""Server (gap fix before training, Sep 25): order-by-order labels (il/data/mseq, as build_mseq.py) and extended-menu
labels (il/data/m8, as build_m8.py: padding heads etc.) for every seat file in SEATS dirs, read straight from the tapes
in TAPES dirs (no replay).   Usage: srv_build_labels8.py      env TAPES, SEATS (colon-separated), NP (12)"""
import sys, os, json, gzip, glob, time
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
D = "/Users/asad.ali/kaggriculture/il/data"


def one(item):
    import features_ar as FA, features8 as F8
    ep, seats, tape = item; n = 0
    d = None
    for s, f in seats.items():
        base = os.path.basename(f)[:-4]
        if os.path.exists(f"{D}/m8/{base}.npy"):
            continue
        if d is None:
            d = json.load(gzip.open(tape, "rt"))
        T = np.load(f)["op"].shape[0]
        ms = np.stack([FA.seq_labels((a or {}).get("market") if isinstance(a, dict) else None) for a in d["actions"][s][:T]]).astype(np.int8)
        np.save(f"{D}/mseq/{base}.npy", ms)
        lab = np.zeros((T, F8.N_HEADS), np.int8)
        for t in range(T):
            if ms[t, 0, 0] != 0:
                lab[t] = F8.labels_from_seq(ms[t])
        np.save(f"{D}/m8/{base}.npy", lab); n += 1
    return n


if __name__ == "__main__":
    os.makedirs(f"{D}/mseq", exist_ok=True); os.makedirs(f"{D}/m8", exist_ok=True); t0 = time.time()
    tape = {}
    for d_ in os.environ["TAPES"].split(":"):
        for ep in json.load(open(f"{d_}/index.json")):
            if os.path.exists(f"{d_}/{ep}.json.gz"):
                tape[int(ep)] = f"{d_}/{ep}.json.gz"
    seats = {}
    for d_ in os.environ["SEATS"].split(":"):
        for f in glob.glob(f"{d_}/*.npz"):
            ep, s = os.path.basename(f)[:-4].split("_"); seats.setdefault(int(ep), {})[int(s)] = f
    items = [(ep, v, tape[ep]) for ep, v in seats.items() if ep in tape]
    print(f"{len(items)} games with tapes, {sum(1 for ep in seats if ep not in tape)} without", flush=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "12"))) as pool:
        n = sum(pool.imap_unordered(one, items, chunksize=16))
    print(f"LABELS8 DONE: {n} new seat label files, {time.time() - t0:.0f}s", flush=True)
