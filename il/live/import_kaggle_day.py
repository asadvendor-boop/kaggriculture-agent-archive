"""Import one day of Kaggle's official top-games dataset (kaggle/kaggriculture-episodes-<day>) into data/tapes_daily:
download the zip with the Kaggle API (no extraction; retried every RETRY seconds until Kaggle has compressed it),
convert only the games we do not have yet straight from the zip (pull_leaders.convert: identical to our own tapes),
write index_part_kaggle_<day>.json, merge the index, then delete the zip.   Usage: import_kaggle_day.py <YYYY-MM-DD>"""
import sys, os, json, gzip, time, zipfile, subprocess, glob
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import pull_leaders as PL
R = "/Users/asad.ali/kaggriculture"; day = sys.argv[1]; RETRY = int(os.environ.get("RETRY", "1200"))
ref = f"kaggle/kaggriculture-episodes-{day}"; out = f"{R}/data/kaggle_daily"; zp = f"{out}/kaggriculture-episodes-{day}.zip"
LOCAL = sys.argv[2] if len(sys.argv) > 2 else None     # a zip the operator already downloaded: read-only, never deleted
if LOCAL:
    zp = LOCAL
KAGGLE = "/usr/local/Caskroom/miniforge/base/bin/kaggle"
while not os.path.exists(zp):
    r = subprocess.run([KAGGLE, "datasets", "download", "-d", ref, "-p", out], capture_output=True, text=True, timeout=7200)
    if os.path.exists(zp) and zipfile.is_zipfile(zp):
        break
    if os.path.exists(zp):
        os.remove(zp)
    print(f"[{time.strftime('%H:%M')}] {ref} not ready ({(r.stdout + r.stderr).strip()[-120:]}); retry in {RETRY}s", flush=True)
    time.sleep(RETRY)
print(f"[{time.strftime('%H:%M')}] downloaded {zp} ({os.path.getsize(zp)/1e9:.2f} GB)", flush=True)
known = set()
for d_ in ("data/tapes_pull", "data/tapes_0922", "data/tapes_archive", "data/tapes_daily"):
    fn = f"{R}/{d_}/index.json"
    if os.path.exists(fn):
        known |= {e for e in json.load(open(fn)) if os.path.exists(f"{R}/{d_}/{e}.json.gz")}
z = zipfile.ZipFile(zp); ip = f"{R}/data/tapes_daily/index_part_kaggle_{day}.json"; idx = {}; n = 0
for name in z.namelist():
    if not name.endswith(".json") or not name[:-5].isdigit() or name[:-5] in known:
        continue
    eid = int(name[:-5])
    try:
        d = json.load(z.open(name)); t = PL.convert(eid, d, day); del d
    except Exception as ex:
        print("fail", eid, str(ex)[:80], flush=True); continue
    with gzip.open(f"{R}/data/tapes_daily/{eid}.json.gz", "wt") as fh:
        json.dump(t, fh, separators=(",", ":"))
    idx[str(eid)] = dict(teams=t["teams"], rewards=t["rewards"], day=day, statuses=t["statuses"]); n += 1
    if n % 50 == 0:
        json.dump(idx, open(ip, "w"), ensure_ascii=False); print(f"  {n} new games taped", flush=True)
json.dump(idx, open(ip, "w"), ensure_ascii=False)
subprocess.run([sys.executable, f"{R}/il/live/merge_daily_index.py"], check=False)
if not LOCAL:
    os.remove(zp)
print(f"[{time.strftime('%H:%M')}] {day}: {n} new games added to data/tapes_daily" + ("" if LOCAL else "; zip deleted"), flush=True)
