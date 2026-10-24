"""Rebuild data/tapes_pull/index.json from the tape files (two downloaders wrote it concurrently)."""
import json, gzip, glob, os
D = "/Users/asad.ali/kaggriculture/data/tapes_pull"
old = json.load(open(f"{D}/index.json")); idx = {}
for f in glob.glob(f"{D}/*.json.gz"):
    ep = os.path.basename(f)[:-8]
    if ep in old:
        idx[ep] = old[ep]; continue
    try:
        t = json.load(gzip.open(f, "rt"))
    except Exception:
        continue
    idx[ep] = dict(teams=t["teams"], rewards=t["rewards"], day=t.get("day"), statuses=t.get("statuses"))
json.dump(idx, open(f"{D}/index.json", "w"), ensure_ascii=False)
print("tapes_pull index rebuilt:", len(old), "->", len(idx))
