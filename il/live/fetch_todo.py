"""Download a list of episodes {ep: day} (json file) into compact tapes in data/tapes_pull (+ index.json). THREADS env."""
import sys, os, json, gzip, threading, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import pull_leaders as PL
R = "/Users/asad.ali/kaggriculture"; D = f"{R}/data/tapes_pull"
todo = json.load(open(sys.argv[1]))
index = json.load(open(f"{D}/index.json")); lock = threading.Lock(); done = [0]; t0 = time.time()
todo = [(int(ep), day) for ep, day in todo.items() if str(ep) not in index and not os.path.exists(f"{D}/{ep}.json.gz")]
print("to fetch", len(todo), flush=True)
def one(item):
    ep, day = item
    try:
        d = PL.fetch_replay(ep); tape = PL.convert(ep, d, day); del d
    except Exception as ex:
        print("fail", ep, str(ex)[:80], flush=True); return
    with gzip.open(f"{D}/{ep}.json.gz", "wt") as fh:
        json.dump(tape, fh, separators=(",", ":"))
    with lock:
        index[str(ep)] = dict(teams=tape["teams"], rewards=tape["rewards"], day=day, statuses=tape["statuses"]); done[0] += 1
        if done[0] % 100 == 0:
            json.dump(index, open(f"{D}/index.json", "w"), ensure_ascii=False)
            print(f"{done[0]}/{len(todo)} fetched, {time.time()-t0:.0f}s", flush=True)
with ThreadPoolExecutor(int(os.environ.get("THREADS", "8"))) as ex:
    list(ex.map(one, todo))
json.dump(index, open(f"{D}/index.json", "w"), ensure_ascii=False)
print("fetch done", done[0], "index", len(index), f"{time.time()-t0:.0f}s", flush=True)
