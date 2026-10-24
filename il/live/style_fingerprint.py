"""Sep 30: style fingerprint per team (the replay-data-miner notebook's idea: action counts per player-game), richer: market buys
by item and phase, worker jobs by type, crops planted, hiring pace. Compared with DSM and with our live bot (cosine on z-scores)."""
import json, gzip, glob, random, collections, math, os, sys
R = "/Users/asad.ali/kaggriculture/data"
TOP = list(json.load(open("/Users/asad.ali/kaggriculture/il/live/top25_teams.json")).values())
DIRS = ["tapes_top25", "tapes_top30", "tapes_top30b", "tapes_zip27", "tapes_zip28", "tapes_top10d", "tapes_lead29", "tapes_today",
        "tapes_rep29", "tapes_top12_recent", "tapes_spec_more_recent"]
OURS = ["tapes_allg055", "tapes_allg", "tapes_all"]
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
def fp(acts):
    c = collections.Counter(); nh = 0
    for t, x in enumerate(acts):
        if not x: continue
        ph = "p" + str(min(2, (t // 24) // 10))
        hands = x.get("hands") or []; nh += len(hands)
        for w in [x.get("farmer")] + hands:
            if not isinstance(w, list) or not w: continue
            op = "MOVE" if w[0] in MOVES else str(w[0])
            c["job:" + op] += 1
            if op == "PLANT" and len(w) > 1: c["plant:" + str(w[1])] += 1; c[ph + ":plant:" + str(w[1])] += 1
        for o in x.get("market") or []:
            if not isinstance(o, list) or not o: continue
            k = str(o[0]) + (":" + str(o[1]) if len(o) > 1 else ""); q = o[2] if len(o) > 2 and isinstance(o[2], (int, float)) else 1
            c["mk:" + k] += q
            if str(o[0]).startswith("BUY"): c[ph + ":mk:" + k] += q
    c["workers_avg"] = nh / 720.0
    return c
random.seed(0); per = collections.defaultdict(list); seen = set()
for d in DIRS:
    for f in glob.glob(f"{R}/{d}/*.json.gz"):
        ep = os.path.basename(f).split(".")[0]
        if ep in seen: continue
        seen.add(ep); per["__files"].append(f)
files = per.pop("__files"); random.shuffle(files); cnt = collections.Counter()
for f in files:
    try: d = json.load(gzip.open(f))
    except Exception: continue
    nm = d.get("teams") or d.get("names") or []
    for s in (0, 1):
        n = nm[s] if s < len(nm) else None
        if n in TOP and cnt[n] < 120:
            per[n].append(fp(d["actions"][s])); cnt[n] += 1
    if all(cnt[t] >= 120 for t in TOP): break
for d in OURS:
    for f in glob.glob(f"{R}/{d}/*.json.gz")[:150]:
        try: x = json.load(gzip.open(f))
        except Exception: continue
        if x.get("seat") is not None and len(per["OUR BOT (ALLG/ALLG055)"]) < 150: per["OUR BOT (ALLG/ALLG055)"].append(fp(x["actions"][x["seat"]]))
keys = sorted({k for L in per.values() for c in L for k in c})
mean = {t: [sum(c.get(k, 0) for c in L) / len(L) for k in keys] for t, L in per.items() if len(L) >= 10}
mu = [sum(v[i] for v in mean.values()) / len(mean) for i in range(len(keys))]
sd = [math.sqrt(sum((v[i] - mu[i]) ** 2 for v in mean.values()) / len(mean)) or 1 for i in range(len(keys))]
z = {t: [(v[i] - mu[i]) / sd[i] for i in range(len(keys))] for t, v in mean.items()}
cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)
ref = ["DSM", "OUR BOT (ALLG/ALLG055)"]
print(f"{'team':30s} games  sim-DSM  sim-OURS  cows sheep geese  wheat tomato straw melon carrot  workers")
show = lambda t: [round(mean[t][keys.index(k)], 1) if k in keys else 0 for k in ("mk:BUY_ANIMAL:COW", "mk:BUY_ANIMAL:SHEEP", "mk:BUY_ANIMAL:GOOSE", "plant:WHEAT", "plant:TOMATO", "plant:STRAWBERRY", "plant:MELON", "plant:CARROT", "workers_avg")]
for t in sorted(mean, key=lambda t: -cos(z[t], z["DSM"])):
    print(f"{t[:29]:30s} {len(per[t]):5d}  {cos(z[t], z['DSM']):+.2f}    {cos(z[t], z[ref[1]]):+.2f}    ", show(t))
json.dump({t: dict(sim_dsm=cos(z[t], z["DSM"]), sim_ours=cos(z[t], z[ref[1]]), games=len(per[t])) for t in mean}, open("/Users/asad.ali/kaggriculture/il/live/style_fingerprint.json", "w"), ensure_ascii=False, indent=1)
