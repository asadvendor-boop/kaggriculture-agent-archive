"""Sep 28 evening (operator: "update the tests with the new replays of today's top leaders"): today's top-band games ->
data/tapes_today/*.json.gz (+ index.json) and the leader-style test il/live/top_worlds_fresh3_300.json (top_bench DAY=fresh3).
Hop 1: the submissions rated >= 2,700 that our live scripts met today (<subs.json> {team: [sub, ...]}, from data/rows_*.json).
Hop 2: every submission rated >= HOP2 (default 2,800) seen in hop 1's episodes (today's leaders, whom we rarely meet).
Kaggle PUBLIC endpoints only (ListEpisodes 3 s apart; replay JSON with THREADS threads, 429 Retry-After honoured) - no token.
Keeps COMPLETED 2-agent episodes that ended on/after SINCE with BOTH ratings >= MINR, never our own submissions; fetches the
MAXF best (highest lower rating) and writes both seats of the best N/2 episodes not in any other test as worlds.
Usage: srv_fetch_today.py <subs.json>        env SINCE (2026-09-28) MINR (2650) HOP2 (2800) THREADS (12) MAXF (400) N (300)"""
import sys, os, json, gzip, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
import pull
from srv_fetch_top12 import fetch_replay, convert
R = "/Users/asad.ali/kaggriculture"; OUT = f"{R}/data/" + os.environ.get("OUTD", "tapes_today"); os.makedirs(OUT, exist_ok=True)
OUTW = os.environ.get("OUTW", "fresh3")   # Sep 29: e.g. OUTD=tapes_sep29 OUTW=fresh4 SINCE=2026-09-29 (never overwrite fresh3)
SINCE = os.environ.get("SINCE", "2026-09-28"); MINR = float(os.environ.get("MINR", "2650")); HOP2 = float(os.environ.get("HOP2", "2800"))
OURS = {56631517, 56631711, 56627684, 56614508, 56595847, 56559691, 56571775, 56510358, 56641949, 56651505, 56661802}   # our submissions (never test worlds)
eps = {}; listed = set()


def list_sub(team, sub):
    L = None
    for attempt in range(int(os.environ.get("LIST_TRIES", "4"))):
        try:
            L = pull.fetch(int(sub)); break
        except Exception as ex:
            print("list fail", team, sub, str(ex)[:80], flush=True); time.sleep(int(os.environ.get("LIST_RETRY_S", "30")))
    listed.add(int(sub))
    if L is None:
        return []
    teams = {t.get("id"): (t.get("teamName") or t.get("name")) for t in L.get("teams", []) or []}
    subteam = {s.get("id"): s.get("teamId") for s in L.get("submissions", []) or []}
    seen, n = [], 0
    for e in L.get("episodes", []):
        ag = e.get("agents", [])
        if e.get("state") != "COMPLETED" or len(ag) != 2 or (e.get("endTime") or "") < SINCE:
            continue
        rt = [float(a.get("initialScore") or 0) for a in ag]; ss = [int(a.get("submissionId") or 0) for a in ag]
        names = [teams.get(subteam.get(a.get("submissionId")), "?") for a in ag]
        seen += [(names[i], ss[i], rt[i]) for i in (0, 1)]
        me = ss.index(int(sub)) if int(sub) in ss else -1     # Sep 30: MINR_MODE=opp -> only the OPPONENT's rating must reach MINR
        low = rt[1 - me] if (os.environ.get("MINR_MODE") == "opp" and me >= 0) else min(rt)   # (a new upload starts low)
        if low < MINR or set(ss) & OURS:
            continue
        eps[str(e["id"])] = dict(end=(e.get("endTime") or "")[:16], names=names, ratings=rt, subs=ss); n += 1
    print(f"{team[:24]:24s} sub {sub}: {n} top-band games since {SINCE} (unique {len(eps)})", flush=True)
    time.sleep(float(os.environ.get("LIST_SLEEP", "3")))   # Sep 30: slower listing after 429s
    return seen


SUBS = json.load(open(sys.argv[1])); hop2 = {}
for team, subs in SUBS.items():
    for sub in subs:
        for name, s, r in list_sub(team, sub):
            if r >= HOP2 and s not in OURS:
                hop2[s] = name
todo2 = {s: n for s, n in hop2.items() if s not in listed}
print(f"HOP 2: {len(todo2)} submissions rated >= {HOP2:.0f}: {sorted(set(todo2.values()))}", flush=True)
for s, name in todo2.items():
    list_sub(name, s)
json.dump(eps, open(f"{OUT}/episodes.json", "w"), ensure_ascii=False)
ip = f"{OUT}/index.json"; index = json.load(open(ip)) if os.path.exists(ip) else {}
best = sorted(eps, key=lambda k: -min(eps[k]["ratings"]))[:int(os.environ.get("MAXF", "400"))]
todo = [k for k in best if k not in index]
if os.environ.get("SKIP_KNOWN") == "1":                 # Sep 30: second pass - skip games on disk or listed by another pass
    import glob as _g
    known = {os.path.basename(f).split(".")[0] for f in _g.glob(f"{R}/data/tapes_*/*.json.gz")}
    for ej in _g.glob(f"{R}/data/tapes_*/episodes.json"):
        if not ej.startswith(OUT + "/"):
            known |= set(json.load(open(ej)))
    todo = [k for k in todo if k not in known]
print(f"LISTED {len(eps)} top-band episodes; fetching {len(todo)} of the best {len(best)}", flush=True)
lock = threading.Lock(); t0 = time.time(); done = [0]


def one(k):
    try:
        d = fetch_replay(int(k)); t = convert(int(k), d, eps[k]["end"][:10]); del d
    except Exception as ex:
        print("fetch fail", k, str(ex)[:80], flush=True); return
    with gzip.open(f"{OUT}/{k}.json.gz", "wt") as fh:
        json.dump(t, fh, separators=(",", ":"))
    with lock:
        index[k] = dict(teams=t["teams"], rewards=t["rewards"], day=t["day"], statuses=t["statuses"], ratings=eps[k]["ratings"],
                        end=eps[k]["end"]); done[0] += 1
        if done[0] % 50 == 0:
            json.dump(index, open(ip, "w"), ensure_ascii=False); print(f"{done[0]}/{len(todo)} fetched, {time.time()-t0:.0f}s", flush=True)


with ThreadPoolExecutor(int(os.environ.get("THREADS", "12"))) as ex:
    list(ex.map(one, todo))
json.dump(index, open(ip, "w"), ensure_ascii=False)
# worlds: both seats of the best N/2 fetched episodes that no other test uses
used = set()
for f in ("top_worlds_fresh_300.json", "top_worlds_fresh2_300.json", "top_worlds_fresh3_300.json", "top_worlds_llall_300.json", "top_worlds_2026-09-24_300.json"):
    try:
        used |= {int(w["ep"]) for w in json.load(open(f"{R}/il/live/{f}"))}
    except Exception:
        pass
try:
    _lw = json.load(open(f"{R}/il/live/ladder_worlds.json"))
    used |= {int(w["ep"]) for w in (_lw if isinstance(_lw, list) else _lw.get("test", [])) if isinstance(w, dict) and "ep" in w}
except Exception:
    pass
W = []
for k in sorted(index, key=lambda k: -min(index[k]["ratings"])):
    v = index[k]; rw = v.get("rewards") or []; tm = v.get("teams") or []
    if int(k) in used or len(rw) != 2 or None in rw or len(tm) != 2 or any(s != "DONE" for s in (v.get("statuses") or [])):
        continue
    for s in (0, 1):
        W.append(dict(path=f"{OUT}/{k}.json.gz", ep=int(k), seat=s, replaced=tm[s], opp=tm[1 - s], rec_margin=rw[s] - rw[1 - s],
                      rec_opp=rw[1 - s], day=v["day"], opp_rating=v["ratings"][1 - s]))
    if len(W) >= int(os.environ.get("N", "300")):
        break
if OUTW == "fresh3" and os.path.exists(f"{R}/il/live/top_worlds_fresh3_300.json") and os.environ.get("OUTD", "tapes_today") != "tapes_today":
    raise SystemExit("refusing to overwrite fresh3 from another tape dir")
json.dump(W, open(f"{R}/il/live/top_worlds_{OUTW}_300.json", "w"), ensure_ascii=False)
lo = min(min(index[str(w["ep"])]["ratings"]) for w in W) if W else 0
print(f"{OUTW.upper()} READY: {len(W)} worlds from {len(W) // 2} games (lower rating >= {lo:.0f}), {done[0]} replays fetched, {time.time()-t0:.0f}s", flush=True)
