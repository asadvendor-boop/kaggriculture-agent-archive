"""Stream Kaggle's daily top-episodes datasets (kaggle/kaggriculture-episodes-YYYY-MM-DD) into compact tapes.
The ~21 GB daily zips are never downloaded: the dataset file list gives the episode ids, and each replay is fetched
from the public episode URL and converted in memory to a ~20 KB tape (same format as data/tapes_pull).

  pull_daily.py list 2026-09-12 2026-09-23   -> data/tapes_daily/episodes.json  {ep: day}
  pull_daily.py fetch                         -> data/tapes_daily/<ep>.json.gz + index.json (5 threads, skips known eps)
"""
import sys, os, json, gzip, time, threading, datetime
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import pull_leaders as PL

R = "/Users/asad.ali/kaggriculture"
D = f"{R}/data/tapes_daily"
os.makedirs(D, exist_ok=True)
lock = threading.Lock()


def do_list(d0, d1):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    fn = f"{D}/episodes.json"
    eps = json.load(open(fn)) if os.path.exists(fn) else {}
    day = datetime.date.fromisoformat(d0)
    while day <= datetime.date.fromisoformat(d1):
        slug = f"kaggle/kaggriculture-episodes-{day.isoformat()}"; tok = None; n = 0
        while True:
            for attempt in range(4):
                try:
                    res = api.dataset_list_files(slug, page_token=tok, page_size=200); break
                except Exception as ex:
                    print("list retry", slug, str(ex)[:80], flush=True); time.sleep(15)
            else:
                break
            for f in res.files or []:
                name = f.name if hasattr(f, "name") else str(f)
                if name.endswith(".json") and name[:-5].isdigit():
                    eps[name[:-5]] = day.isoformat(); n += 1
            tok = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
            if not tok:
                break
        print(f"{day}: {n} episodes (total {len(eps)})", flush=True)
        json.dump(eps, open(fn, "w"))
        day += datetime.timedelta(days=1)


def known_eps():
    have = set()
    for d_ in ("data/tapes_pull", "data/tapes_0922", "data/tapes_archive", "data/tapes_daily"):
        fn = f"{R}/{d_}/index.json"
        if os.path.exists(fn):
            have |= set(json.load(open(fn)).keys())
    return have


def do_fetch():
    eps = json.load(open(f"{D}/episodes.json"))
    idx_fn = f"{D}/index.json"
    index = json.load(open(idx_fn)) if os.path.exists(idx_fn) else {}
    have = known_eps()
    todo = sorted(((day, int(ep)) for ep, day in eps.items() if ep not in have), reverse=True)
    print("to fetch", len(todo), "skipped (already have)", len(eps) - len(todo), flush=True)
    t0 = time.time(); done = [0]

    def one(item):
        day, eid = item
        try:
            d = PL.fetch_replay(eid); tape = PL.convert(eid, d, day); del d
        except Exception as ex:
            print("fetch fail", eid, str(ex)[:80], flush=True); return
        with gzip.open(f"{D}/{eid}.json.gz", "wt") as fh:
            json.dump(tape, fh, separators=(",", ":"))
        with lock:
            index[str(eid)] = dict(teams=tape["teams"], rewards=tape["rewards"], day=day, statuses=tape["statuses"])
            done[0] += 1
            if done[0] % 50 == 0:
                json.dump(index, open(idx_fn, "w"), ensure_ascii=False)
                print(f"{done[0]}/{len(todo)} fetched, {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(int(os.environ.get("THREADS", "5"))) as ex:
        list(ex.map(one, todo))
    json.dump(index, open(idx_fn, "w"), ensure_ascii=False)
    print("fetch done", done[0], "total", len(index), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        do_list(sys.argv[2], sys.argv[3])
    else:
        do_fetch()
