"""Merge the parallel downloaders' index_part*.json files into data/tapes_daily/index.json."""
import json, glob, os
D = "/Users/asad.ali/kaggriculture/data/tapes_daily"
idx = json.load(open(f"{D}/index.json")); n0 = len(idx)
for p in sorted(glob.glob(f"{D}/index_part*.json")):
    idx.update(json.load(open(p)))
idx = {e: m for e, m in idx.items() if os.path.exists(f"{D}/{e}.json.gz")}
json.dump(idx, open(f"{D}/index.json", "w"), ensure_ascii=False)
print("tapes_daily index:", n0, "->", len(idx))
