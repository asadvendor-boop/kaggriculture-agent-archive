"""Engine parity check for a new machine: replay N recorded games (both seats' recorded actions) and compare the final
banks with the recorded rewards; every game must match exactly.   Usage: engine_parity.py <tape_dir> [N]"""
import sys, json, gzip, glob
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from harness import kagsim_trace, PASS
d_, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30
ok = bad = 0
for f in sorted(glob.glob(f"{d_}/*.json.gz"))[:n]:
    t = json.load(gzip.open(f, "rt")); rw = t.get("rewards") or []
    if len(rw) != 2 or None in rw:
        continue
    A = [[x if isinstance(x, dict) else dict(PASS) for x in t["actions"][k]] for k in (0, 1)]
    g = kagsim_trace.Game(int(t["seed"]), 720, False)
    if t.get("shops") and hasattr(g, "set_forced_shops"):
        pass                                   # shops follow from the seed; forced shops only for pinned worlds
    for s in range(720):
        g.step(A[0][s], A[1][s])
        if g.done:
            break
    got = [g.reward(0), g.reward(1)]
    if abs(got[0] - rw[0]) < 0.5 and abs(got[1] - rw[1]) < 0.5:
        ok += 1
    else:
        bad += 1; print("MISMATCH", f, got, rw, flush=True)
print(f"engine parity: {ok} exact, {bad} different (of {ok + bad} recorded games) -> {'OK' if ok and not bad else 'FAIL'}", flush=True)
