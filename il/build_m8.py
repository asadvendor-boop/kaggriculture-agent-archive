"""Menu-8 market labels (features8) for every seat with order-by-order labels: il/data/m8/<ep>_<seat>.npy int8 [T, 23],
derived from il/data/mseq (no replays needed)."""
import sys, os
import numpy as np
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
D = "/Users/asad.ali/kaggriculture/il/data"


def one(f):
    import features8 as F8
    out = f"{D}/m8/{f}"
    if os.path.exists(out):
        return 0
    ms = np.load(f"{D}/mseq/{f}")
    lab = np.zeros((ms.shape[0], F8.N_HEADS), np.int8)
    for t in range(ms.shape[0]):
        if ms[t, 0, 0] != 0:                      # slot 0 STOP = no orders at all
            lab[t] = F8.labels_from_seq(ms[t])
    np.save(out + ".tmp.npy", lab); os.replace(out + ".tmp.npy", out)
    return 1


if __name__ == "__main__":
    os.makedirs(f"{D}/m8", exist_ok=True)
    fs = sorted(f[:-4] + ".npy" for f in os.listdir(f"{D}/union"))
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "3"))) as pool:
        n = sum(pool.imap_unordered(one, fs, chunksize=64))
    print(f"built {n} new label files; total {len(os.listdir(f'{D}/m8'))}", flush=True)
