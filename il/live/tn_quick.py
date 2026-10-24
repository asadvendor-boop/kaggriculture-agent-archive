"""Sep 28 evening: quick paired table of tonight's variants vs their base on the rulers finished so far.
Usage: tn_quick.py <ckpt> <base tag> <t1,t2,...> [days, default llall,fresh,fresh2,fresh3] [464]"""
import sys, json, os, math
R = "/Users/asad.ali/kaggriculture/il/logs"
ck, bt, ts = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
days = (sys.argv[4] if len(sys.argv) > 4 else "llall,fresh,fresh2,fresh3").split(",")


def load(f):
    return {(r["ep"], r.get("seat", 0)): r["margin"] for r in json.load(open(f))} if os.path.exists(f) else None


def cell(B, X, band=None):
    if not B or not X:
        return "-"
    k = sorted(B.keys() & X.keys())
    if not k:
        return "-"
    wb = sum(B[i] > 0 for i in k); wx = sum(X[i] > 0 for i in k)
    d = [X[i] - B[i] for i in k]; m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5
    return f"W {wb:3d}->{wx:3d} ({wx - wb:+3d}) {m:+7,.0f} z{m / (sd / math.sqrt(len(d)) + 1e-9):+5.1f} n{len(k)}"


files = {d: (lambda t, d=d: f"{R}/top_{d}_{ck}_t10_{bt}{'_' + t if t else ''}.json") for d in days}
if len(sys.argv) > 5:
    files["464"] = lambda t: f"{R}/ladder_{ck}_t10_r3_tau0.5_{bt}{'_' + t if t else ''}_test.json"
print(f"{'variant':8s} | " + " | ".join(f"{d:38s}" for d in files))
for t in ts:
    print(f"{t:8s} | " + " | ".join(f"{cell(load(f('')), load(f(t))):38s}" for f in files.values()))
