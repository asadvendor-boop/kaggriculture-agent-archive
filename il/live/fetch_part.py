"""Parallel downloader: fetch_part.py <episodes.json {ep: day}> <out_dir> <part> <nparts>  (THREADS env).
Downloads episodes with ep % nparts == part (newest first) into compact tapes; writes its own <out_dir>/index_part<k>.json
(no shared-file races); merge the parts into index.json afterwards."""
import sys, os, json, gzip, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import pull_leaders as PL
R = "/Users/asad.ali/kaggriculture"
src, out, part, nparts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
eps = json.load(open(src)); known = set()
for d_ in ("data/tapes_pull", "data/tapes_0922", "data/tapes_archive", "data/tapes_daily"):
    fn = f"{R}/{d_}/index.json"
    if os.path.exists(fn):
        known |= set(json.load(open(fn)).keys())
todo = sorted(((d, int(e)) for e, d in eps.items() if e not in known and int(e) % nparts == part
               and not os.path.exists(f"{out}/{e}.json.gz")), reverse=True)
ip = f"{out}/index_part{part}.json"; idx = json.load(open(ip)) if os.path.exists(ip) else {}
lock = threading.Lock(); t0 = time.time(); done = [0]
print(f"part {part}: {len(todo)} to fetch", flush=True)
def one(item):
    day, eid = item
    try:
        d = PL.fetch_replay(eid); t = PL.convert(eid, d, day); del d
    except Exception as ex:
        print("fail", eid, str(ex)[:80], flush=True); return
    with gzip.open(f"{out}/{eid}.json.gz", "wt") as fh:
        json.dump(t, fh, separators=(",", ":"))
    with lock:
        idx[str(eid)] = dict(teams=t["teams"], rewards=t["rewards"], day=day, statuses=t["statuses"]); done[0] += 1
        if done[0] % 100 == 0:
            json.dump(idx, open(ip, "w"), ensure_ascii=False)
            print(f"part {part}: {done[0]}/{len(todo)} fetched, {time.time()-t0:.0f}s", flush=True)
with ThreadPoolExecutor(int(os.environ.get("THREADS", "4"))) as ex:
    list(ex.map(one, todo))
json.dump(idx, open(ip, "w"), ensure_ascii=False)
print(f"part {part} fetch done {done[0]} {time.time()-t0:.0f}s", flush=True)
