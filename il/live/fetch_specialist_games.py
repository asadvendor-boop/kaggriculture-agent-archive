"""More games of the specialists who beat us (operator-approved plan, Sep 27), for the JAX league: for each specialist
submission (default: every opponent rated >= MINR who beat one of our live submissions, from data/rows_<ours>.json;
or --subs id,id,...), list its recent finished 2-player episodes (public ListEpisodes), download each replay (public
kaggleusercontent endpoint, gzip), keep ONLY the compact tape in the leader format of kagjax/train/tapes.py (ep, seed,
teams, rewards, statuses, shops, actions, day) - the ~35 MB replay is never written to disk. Both seats of every game
are kept; teams.txt lists the specialists' team names (join into TOP_TEAMS). Resumable (existing tapes skipped).
Kaggle rate limits: at most 12 parallel downloads (--threads).  Run on the rented server, not on the Mac.
Usage: fetch_specialist_games.py [--out data/tapes_spec_more] [--per 120] [--minr 2600] [--threads 12] [--subs ...]"""
import sys, os, json, gzip, glob, argparse, collections, urllib.request, time
from concurrent.futures import ThreadPoolExecutor, as_completed
K = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, f"{K}/tools")
from pull import fetch                       # public ListEpisodes (as the live monitors use)
from harness import fetch_replay             # public replay download (gzip)
OURS = [56595847, 56589891, 56571775, 56559691]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{K}/data/tapes_spec_more"); ap.add_argument("--per", type=int, default=120)
ap.add_argument("--minr", type=float, default=2600); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--subs", default=""); ap.add_argument("--dry", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
subs = collections.OrderedDict()
if a.subs:
    for s in a.subs.split(","): subs[int(s)] = "?"
else:
    for our in OURS:
        fn = f"{K}/data/rows_{our}.json"
        if not os.path.exists(fn): continue
        for r in json.load(open(fn)):
            if r.get("res") == "L" and (r.get("opp_rating") or 0) >= a.minr and r.get("opp_sub"):
                subs[int(r["opp_sub"])] = r["opp"]
print(f"{len(subs)} specialist submissions: " + ", ".join(f"{n} ({s})" for s, n in list(subs.items())[:40]), flush=True)
jobs = []
def fetch_backoff(s):                         # Sep 27: a Mac dry run got HTTP 429 -> back off, never hammer the API
    for k in range(6):
        try:
            return fetch(s)
        except Exception as e:
            if "429" in str(e) and k < 5:
                time.sleep(30 * 2 ** k); continue
            raise
for s, name in subs.items():
    try:
        L = fetch_backoff(s)
    except Exception as e:
        print(f"  ListEpisodes {s} failed: {e}", flush=True); continue
    eps = [e for e in L.get("episodes", []) if len(e.get("agents", [])) == 2 and e.get("state") in (None, "COMPLETED", "EPISODE_STATE_SUCCEEDED", "SUCCEEDED", 2)
           and all(ag.get("reward") is not None for ag in e["agents"])]
    eps.sort(key=lambda e: e.get("endTime") or "", reverse=True)
    for e in eps[:a.per]: jobs.append((int(e["id"]), name, (e.get("endTime") or "")[:10]))
    time.sleep(3)                             # <= 20 list calls per minute
jobs = [j for j in dict((j[0], j) for j in jobs).values() if not os.path.exists(f"{a.out}/{j[0]}.json.gz")]
print(f"{len(jobs)} replays to fetch with {a.threads} threads", flush=True)
if a.dry: sys.exit(0)
def one(job):
    ep, name, day = job
    d = fetch_replay(ep); steps = d["steps"]
    acts = [[(steps[t + 1][s].get("action") if t + 1 < len(steps) and isinstance(steps[t + 1][s].get("action"), dict) else None) for t in range(720)] for s in (0, 1)]
    teams = list(d["info"].get("TeamNames") or ["?", "?"])
    st = [steps[-1][s].get("status") for s in (0, 1)]
    out = dict(ep=ep, seed=int(d["info"]["seed"]), teams=teams, rewards=d.get("rewards"), statuses=st, shops=[], actions=acts, day=day)
    json.dump(out, gzip.open(f"{a.out}/{ep}.json.gz.tmp", "wt"), separators=(",", ":")); os.replace(f"{a.out}/{ep}.json.gz.tmp", f"{a.out}/{ep}.json.gz")
    return ep, teams, st, out["rewards"], day
idx_fn = f"{a.out}/index.json"; index = json.load(open(idx_fn)) if os.path.exists(idx_fn) else {}
done = 0
with ThreadPoolExecutor(a.threads) as ex:
    for fu in as_completed([ex.submit(one, j) for j in jobs]):
        try:
            ep, teams, st, rw, day = fu.result()
            index[str(ep)] = dict(teams=teams, statuses=st, rewards=rw, day=day); done += 1
            if done % 50 == 0:
                json.dump(index, open(idx_fn, "w")); print(f"  {done}/{len(jobs)}", flush=True)
        except Exception as e:
            print(f"  failed: {str(e)[:120]}", flush=True)
json.dump(index, open(idx_fn, "w"))
open(f"{a.out}/teams.txt", "w").write("|".join(sorted(set(n for n in subs.values() if n != "?"))))
print(f"done: {done} new tapes, {len(index)} in {a.out}", flush=True)
