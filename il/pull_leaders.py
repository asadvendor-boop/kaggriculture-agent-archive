"""Pull the top-10 leaders' public games into compact tapes (streamed; raw replays never written to disk).

  pull_leaders.py list        : ListEpisodes for every submission in data/tapes_pull/top10_subs.json -> episodes.json
  pull_leaders.py fetch [n]   : download up to n episodes (newest first), 3 threads, into data/tapes_pull/<ep>.json.gz
Tape format = data/tapes_archive: ep, seed, teams, rewards, statuses, shops, actions[2][720], day.
"""
import sys, os, json, gzip, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import pull
import urllib.request, urllib.error, gzip as _gz
import orjson


def fetch_replay(eid):
    """Download one public replay (gzip transfer) and parse it with orjson (fast, low CPU)."""
    req = urllib.request.Request(f"https://www.kaggleusercontent.com/episodes/{eid}.json",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
    for attempt in range(12):
        try:
            raw = urllib.request.urlopen(req, timeout=300).read(); break
        except urllib.error.HTTPError as ex:           # 429: wait as long as the server asks, then retry
            if attempt == 11:
                raise
            wait = int(ex.headers.get("Retry-After") or 30) if ex.code == 429 else 10 * (attempt + 1)
            time.sleep(wait + (attempt % 4) * 7)
        except Exception:
            if attempt == 11:
                raise
            time.sleep(10 * (attempt + 1))
    return orjson.loads(_gz.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)

D = "/Users/asad.ali/kaggriculture/data/tapes_pull"
SUBS = json.load(open(f"{D}/top10_subs.json"))
lock = threading.Lock()


def do_list():
    eps = {}
    for team, info in SUBS.items():
        for sub in sorted(info["subs"], reverse=True):
            for attempt in range(3):
                try:
                    L = pull.fetch(sub); break
                except Exception as ex:
                    print("list fail", team, sub, str(ex)[:60], flush=True); time.sleep(20)
            else:
                continue
            n = 0
            for e in L.get("episodes", []):
                if e.get("state") != "COMPLETED" or len(e.get("agents", [])) != 2:
                    continue
                r = eps.setdefault(e["id"], {"end": (e.get("endTime") or "")[:16], "teams_top": []})
                if team not in r["teams_top"]:
                    r["teams_top"].append(team)
                n += 1
            print(f"{team:22s} sub {sub}: {n} episodes (unique so far {len(eps)})", flush=True)
            time.sleep(2)
    json.dump(eps, open(f"{D}/episodes.json", "w"), ensure_ascii=False)
    print("unique episodes", len(eps), flush=True)


def convert(eid, d, day):
    steps = d["steps"]
    acts = [[(steps[t + 1][s].get("action") if t + 1 < len(steps) and isinstance(steps[t + 1][s].get("action"), dict) else None)
             for t in range(720)] for s in (0, 1)]
    teams = d["info"].get("TeamNames"); rew = d["rewards"]
    st = [steps[-1][s].get("status") for s in (0, 1)]
    return dict(ep=eid, seed=int(d["info"]["seed"]), teams=teams, rewards=rew, statuses=st,
                shops=list(steps[-1][0]["observation"]["town"]["unlocked_shops"]), actions=acts, day=day)


def do_fetch(limit):
    eps = json.load(open(f"{D}/episodes.json"))
    idx_fn = f"{D}/index.json"
    index = json.load(open(idx_fn)) if os.path.exists(idx_fn) else {}
    todo = sorted(((v["end"], int(k)) for k, v in eps.items() if k not in index), reverse=True)[:limit]
    print("to fetch", len(todo), "already", len(index), flush=True)
    t0 = time.time(); done = [0]

    def one(item):
        end, eid = item
        try:
            d = fetch_replay(eid)
            tape = convert(eid, d, end[:10])
            del d
        except Exception as ex:
            print("fetch fail", eid, str(ex)[:80], flush=True); return
        with gzip.open(f"{D}/{eid}.json.gz", "wt") as fh:
            json.dump(tape, fh, separators=(",", ":"))
        with lock:
            index[str(eid)] = dict(teams=tape["teams"], rewards=tape["rewards"], day=tape["day"], statuses=tape["statuses"])
            done[0] += 1
            if done[0] % 25 == 0:
                json.dump(index, open(idx_fn, "w"), ensure_ascii=False)
                print(f"{done[0]}/{len(todo)} fetched, {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(int(os.environ.get("THREADS", "5"))) as ex:
        list(ex.map(one, todo))
    json.dump(index, open(idx_fn, "w"), ensure_ascii=False)
    print("fetch done", done[0], "total tapes", len(index), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        do_list()
    else:
        do_fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 100000)
