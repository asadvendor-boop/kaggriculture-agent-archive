"""Server variant of build_extras6.py (identical features, configurable directories): for every seat file in the SEATS
dirs that has no il/data/x6/<ep>_<seat>.npy yet, replay its game exactly (seed, recorded actions, observe before step)
and save features6.future_features.   Usage: srv_build_x6.py <worker> <n_workers>
env TAPES (colon-separated tape dirs with index.json), SEATS (colon-separated seat dirs)"""
import sys, os, json, gzip, glob, time
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from harness import kagsim_trace, PASS
import features6 as F6
R = "/Users/asad.ali/kaggriculture"
OUT = f"{R}/il/data/x6"; os.makedirs(OUT, exist_ok=True)


def main(w, nw):
    tape = {}
    for d_ in os.environ["TAPES"].split(":"):
        for ep in json.load(open(f"{d_}/index.json")):
            if os.path.exists(f"{d_}/{ep}.json.gz"):
                tape[int(ep)] = f"{d_}/{ep}.json.gz"
    seats = {}
    for d_ in os.environ["SEATS"].split(":"):
        for f in glob.glob(f"{d_}/*.npz"):
            ep, s = os.path.basename(f)[:-4].split("_")
            seats.setdefault(int(ep), {})[int(s)] = f
    eps = sorted(seats)[w::nw]; t0 = time.time(); done = miss = bad = 0
    for i, ep in enumerate(eps):
        todo = {s: f for s, f in seats[ep].items() if not os.path.exists(f"{OUT}/{ep}_{s}.npy")}
        if not todo:
            continue
        if ep not in tape:
            miss += 1; continue
        d = json.load(gzip.open(tape[ep], "rt"))
        A = [[x if isinstance(x, dict) else dict(PASS) for x in d["actions"][s]] for s in (0, 1)]
        g = kagsim_trace.Game(d["seed"], 720, False); X = {s: [] for s in todo}
        for t in range(720):
            for s in todo:
                X[s].append(F6.future_features(g.observe(s)))
            g.step(A[0][t], A[1][t])
            if g.done:
                break
        for s, f in todo.items():
            T = np.load(f)["op"].shape[0]
            if T != len(X[s]):
                bad += 1; continue
            np.save(f"{OUT}/{ep}_{s}.npy", np.stack(X[s]).astype(np.float16)); done += 1
        if (i + 1) % 200 == 0:
            print(f"worker {w}: {i+1}/{len(eps)} games, seats {done}, missing tape {miss}, length mismatch {bad}, {time.time()-t0:.0f}s", flush=True)
    print(f"worker {w} done: seats {done}, missing tape {miss}, length mismatch {bad}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
