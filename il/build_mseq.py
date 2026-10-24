"""Order-sequence labels for every training seat: il/data/mseq/<ep>_<seat>.npy = int8 [T, 10, 3] (op, item, qty class of
features_ar), T = the seat file's step count.  Read straight from the tapes (no replay needed).
Usage: build_mseq.py <worker> <n_workers>"""
import sys, os, json, gzip, glob, time
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import features_ar as FA
R = "/Users/asad.ali/kaggriculture"; OUT = f"{R}/il/data/mseq"; os.makedirs(OUT, exist_ok=True)
w, nw = int(sys.argv[1]), int(sys.argv[2])
tape = {}
for d_ in ("data/tapes_archive", "data/tapes_0922", "data/tapes_pull", "data/tapes_daily"):
    for ep in json.load(open(f"{R}/{d_}/index.json")):
        if os.path.exists(f"{R}/{d_}/{ep}.json.gz"): tape[int(ep)] = f"{R}/{d_}/{ep}.json.gz"
seats = {}
for d_ in ("il/data/leaders", "il/data/all"):
    for f in glob.glob(f"{R}/{d_}/*.npz"):
        ep, s = os.path.basename(f)[:-4].split("_"); seats.setdefault(int(ep), {})[int(s)] = f
eps = sorted(seats)[w::nw]; t0 = time.time(); done = miss = 0
for i, ep in enumerate(eps):
    todo = {s: f for s, f in seats[ep].items() if not os.path.exists(f"{OUT}/{ep}_{s}.npy")}
    if not todo: continue
    if ep not in tape: miss += 1; continue
    d = json.load(gzip.open(tape[ep], "rt"))
    for s, f in todo.items():
        T = np.load(f)["op"].shape[0]
        lab = np.stack([FA.seq_labels((a or {}).get("market") if isinstance(a, dict) else None) for a in d["actions"][s][:T]]).astype(np.int8)
        np.save(f"{OUT}/{ep}_{s}.npy", lab); done += 1
    if (i + 1) % 500 == 0: print(f"worker {w}: {i+1}/{len(eps)} games, seats {done}, missing tape {miss}, {time.time()-t0:.0f}s", flush=True)
print(f"worker {w} done: seats {done}, missing tape {miss}, {time.time()-t0:.0f}s", flush=True)
