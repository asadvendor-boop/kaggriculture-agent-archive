"""Sep 30: stream a 'Top episode replays for <day>.zip' (read-only, never extracted) into compact tapes data/<out>/<ep>.json.gz,
skipping episodes already present in any data/tapes_* folder. Usage: zip_to_tapes.py <zip> <out dir name> <day>"""
import sys, os, glob, json, gzip, zipfile
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from srv_fetch_top12 import convert
try:
    import orjson; loads = orjson.loads
except ImportError:
    loads = json.loads
R = "/Users/asad.ali/kaggriculture/data"; zp, outn, day = sys.argv[1], sys.argv[2], sys.argv[3]
out = f"{R}/{outn}"; os.makedirs(out, exist_ok=True)
have = {os.path.basename(f).split(".")[0] for f in glob.glob(f"{R}/tapes_*/*.json.gz") if not f.startswith(out)}
z = zipfile.ZipFile(zp); n = skip = bad = 0
for i in z.infolist():
    if not i.filename.endswith(".json"):
        continue
    ep = os.path.basename(i.filename).split(".")[0]
    if ep in have or os.path.exists(f"{out}/{ep}.json.gz"):
        skip += 1; continue
    try:
        d = loads(z.read(i)); t = convert(int(ep), d, day); del d
        if len(t["teams"] or []) != 2 or None in (t["rewards"] or [None]):
            bad += 1; continue
        with gzip.open(f"{out}/{ep}.json.gz", "wt") as f:
            json.dump(t, f, separators=(",", ":"), ensure_ascii=False)
        n += 1
    except Exception as ex:
        bad += 1; print("bad", ep, str(ex)[:80], flush=True)
    if (n + skip + bad) % 100 == 0:
        print(f"{n} new, {skip} already had, {bad} bad", flush=True)
print(f"DONE {zp}: {n} new tapes, {skip} already had, {bad} bad", flush=True)
