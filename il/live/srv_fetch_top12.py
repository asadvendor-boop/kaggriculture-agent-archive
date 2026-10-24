"""Server (operator, Sep 25): every episode of the top-12 leaders' CURRENT submissions -> compact tapes in data/tapes_top12
(same tape format as pull_leaders / tapes_archive; raw replays are parsed in memory and never written to disk).
ListEpisodes one submission at a time, 3 s apart (also saves data/rows_<sub>.json with results and ratings); replays with
THREADS threads (default 12), honouring 429 Retry-After.  Re-running only fetches what is missing.
Usage: srv_fetch_top12.py            env THREADS"""
import sys, os, json, gzip, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import orjson
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import pull
R = "/Users/asad.ali/kaggriculture"; OUT = f"{R}/data/tapes_top12"; os.makedirs(OUT, exist_ok=True)
SUBS = {"Boey": [56521745], "DSM": [56564674, 56571979], "M & M & P & Q": [56553720, 56570008],      # operator, Sep 26:
        "Fourth Quadrant": [56555482], "DECEM": [56558220, 56573934], "Unknown Mother-Goose": [56567212],  # today's top 10
        "Azat Akhtyamov": [56561951], "🐚seek inspiration🐚": [56551631], "Majkel1337": [56538379],       # (Azat's id checked
        "THIRD FARM CLUB": [56520754]}                                                                    #  on Kaggle: 56561951)
if os.environ.get("SUBS_JSON"):                       # Sep 26: leaders' newer submissions (live/srv_discover_subs.py)
    for _t, _ids in json.load(open(os.environ["SUBS_JSON"])).items():
        SUBS.setdefault(_t, []); SUBS[_t] += [int(i) for i in _ids if int(i) not in SUBS[_t]]
if os.environ.get("SUBS_ONLY"):                       # Sep 27 night: exactly these submissions (the operator's current top 10)
    SUBS = {_t: [int(i) for i in _ids] for _t, _ids in json.load(open(os.environ["SUBS_ONLY"])).items() if _ids}


def fetch_replay(eid):
    """One public replay (gzip transfer, redirects followed), parsed with orjson."""
    import gzip as _gz
    req = urllib.request.Request(f"https://www.kaggleusercontent.com/episodes/{eid}.json",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
    for attempt in range(12):
        try:
            raw = urllib.request.urlopen(req, timeout=300).read(); break
        except urllib.error.HTTPError as ex:             # 429: wait as long as the server asks, then retry
            if attempt == 11:
                raise
            wait = int(ex.headers.get("Retry-After") or 30) if ex.code == 429 else 10 * (attempt + 1)
            print(f"HTTP {ex.code} on {eid}, waiting {wait}s", flush=True)
            time.sleep(wait + (attempt % 4) * 7)
        except Exception:
            if attempt == 11:
                raise
            time.sleep(10 * (attempt + 1))
    return orjson.loads(_gz.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def convert(eid, d, day):                                 # = pull_leaders.convert
    steps = d["steps"]
    acts = [[(steps[t + 1][s].get("action") if t + 1 < len(steps) and isinstance(steps[t + 1][s].get("action"), dict) else None)
             for t in range(720)] for s in (0, 1)]
    return dict(ep=eid, seed=int(d["info"]["seed"]), teams=d["info"].get("TeamNames"), rewards=d["rewards"],
                statuses=[steps[-1][s].get("status") for s in (0, 1)],
                shops=list(steps[-1][0]["observation"]["town"]["unlocked_shops"]), actions=acts, day=day)


def main():
    eps = {}
    for team, subs in SUBS.items():
        for sub in subs:
            for attempt in range(4):
                try:
                    L = pull.fetch(sub); break
                except Exception as ex:
                    print("list fail", team, sub, str(ex)[:80], flush=True); time.sleep(30)
            else:
                continue
            try:
                json.dump(pull.rows_of(L, sub), open(f"{R}/data/rows_{sub}.json", "w"))
            except Exception as ex:
                print("rows fail", sub, str(ex)[:80], flush=True)
            n = 0
            for e in L.get("episodes", []):
                if e.get("state") != "COMPLETED" or len(e.get("agents", [])) != 2:
                    continue
                r = eps.setdefault(str(e["id"]), {"end": (e.get("endTime") or "")[:16], "subs": [], "teams_top": []})
                if sub not in r["subs"]:
                    r["subs"].append(sub)
                if team not in r["teams_top"]:
                    r["teams_top"].append(team)
                n += 1
            print(f"{team:22s} sub {sub}: {n} completed episodes (unique so far {len(eps)})", flush=True)
            time.sleep(3)
    json.dump(eps, open(f"{OUT}/episodes.json", "w"), ensure_ascii=False)
    ip = f"{OUT}/index.json"; index = json.load(open(ip)) if os.path.exists(ip) else {}
    todo = sorted(((v["end"], int(k)) for k, v in eps.items() if k not in index), reverse=True)
    print(f"LISTED {len(eps)} unique episodes; to fetch {len(todo)}", flush=True)
    lock = threading.Lock(); t0 = time.time(); done = [0]

    def one(item):
        end, eid = item
        try:
            d = fetch_replay(eid); t = convert(eid, d, end[:10]); del d
        except Exception as ex:
            print("fetch fail", eid, str(ex)[:80], flush=True); return
        with gzip.open(f"{OUT}/{eid}.json.gz", "wt") as fh:
            json.dump(t, fh, separators=(",", ":"))
        with lock:
            index[str(eid)] = dict(teams=t["teams"], rewards=t["rewards"], day=t["day"], statuses=t["statuses"]); done[0] += 1
            if done[0] % 100 == 0:
                json.dump(index, open(ip, "w"), ensure_ascii=False)
                print(f"{done[0]}/{len(todo)} fetched, {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(int(os.environ.get("THREADS", "12"))) as ex:
        list(ex.map(one, todo))
    json.dump(index, open(ip, "w"), ensure_ascii=False)
    print(f"TOP12 FETCH DONE {done[0]} new, {len(index)} tapes, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
