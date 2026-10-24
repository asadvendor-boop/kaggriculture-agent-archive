"""Strategy profile of our own live bots from their saved live games (same measures as leader_study)."""
import sys, json, gzip, glob, numpy as np, multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from leader_study import study
R = "/Users/asad.ali/kaggriculture"
PROD = ["WHEAT", "STRAWBERRY", "MELON", "MILK", "WOOL", "FERTILIZER", "TOMATO", "EGG", "CARROT"]
if __name__ == "__main__":
    for name, sub, d_ in (("NEURAL_R3 (ours)", 56518491, "data/tapes_neural"), ("STACK1_RACE4 (ours)", 56520632, "data/tapes_race4")):
        eps = {r["ep"] for r in json.load(open(f"{R}/data/rows_{sub}.json"))}
        items = []
        for f in sorted(glob.glob(f"{R}/{d_}/*.json.gz"))[::-1]:
            dd = json.load(gzip.open(f, "rt"))
            if dd["ep"] in eps and len(items) < 40: items.append((f, dd["seat"]))
        with mp.get_context("spawn").Pool(4) as pool: res = pool.map(study, items)
        P = [r[2] for r in res]; tot = np.mean([sum(p[f"rev:{x}"] for x in PROD) for p in P])
        print(f"{name:22s} {len(P):2d} games | bank {np.mean([p['bank'] for p in P])/1e3:4.0f}k | rev share % " + " ".join(f"{x[:3]} {np.mean([p['rev:'+x] for p in P])/max(1,tot)*100:2.0f}" for x in PROD) +
              f" | cows {np.mean([p['animal:COW'] for p in P]):4.1f} sheep {np.mean([p['animal:SHEEP'] for p in P]):4.1f} geese {np.mean([p['animal:GOOSE'] for p in P]):4.1f} land {np.mean([p['land'] for p in P]):3.1f} wheat bought/sold {np.mean([p['wheat_bought'] for p in P]):3.0f}/{np.mean([p['sold:WHEAT'] for p in P]):3.0f}", flush=True)
