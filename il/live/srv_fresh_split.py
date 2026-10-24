"""Server (operator, Sep 25): training set il/data/<name> = symlinks to the built seats (il/data/fresh_all) of the games in
the given tape dirs, minus held-out games:
  (a) every episode of the ladder benchmark (il/live/ladder_worlds.json, both splits)
  (b) a NEW leader benchmark: leader-vs-leader games of the top-12 current submissions (data/tapes_top12) with ep % 8 == 3;
      both seats become worlds in il/live/top_worlds_fresh_300.json (top_bench.py with DAY=fresh N=300); written once.
Seats without v6 features (il/data/x6) are left out.   Usage: srv_fresh_split.py <name> <tape_dir> [...]"""
import sys, os, json, glob, random, collections
R = "/Users/asad.ali/kaggriculture"
TOP12 = {"DSM", "Boey", "M & M & P & Q", "Unknown Mother-Goose", "DECEM", "Fourth Quadrant", "Majkel1337", "mtmr_s1",
         "Vadim Vasilenko", "吃白饭的大肥鱼", "Kaggledew Valley 🏆", "THIRD FARM CLUB", "Azat Akhtyamov", "🐚seek inspiration🐚"}
TOP10_TODAY = {"Boey", "DSM", "M & M & P & Q", "Fourth Quadrant", "DECEM", "Unknown Mother-Goose", "Azat Akhtyamov",     # Sep 26
               "🐚seek inspiration🐚", "Majkel1337", "THIRD FARM CLUB"}
name, dirs = sys.argv[1], [d if d.startswith("/") else f"{R}/{d}" for d in sys.argv[2:]]
lad = {int(w["ep"]) for w in json.load(open(f"{R}/il/live/ladder_worlds.json"))}
eps = {}
for d in dirs:
    for ep, v in json.load(open(f"{d}/index.json")).items():
        if os.path.exists(f"{d}/{ep}.json.gz"):
            eps.setdefault(int(ep), (d, v))
T12 = f"{R}/data/tapes_top12"; held = set(); worlds = []
for ep, v in json.load(open(f"{T12}/index.json")).items():
    ep = int(ep); t = v.get("teams") or []; rw = v.get("rewards") or []
    if ep % 8 == 3 and len(t) == 2 and t[0] in TOP12 and t[1] in TOP12 and len(rw) == 2 and None not in rw:
        held.add(ep)
        for s in (0, 1):
            worlds.append(dict(path=f"{T12}/{ep}.json.gz", ep=ep, seat=s, replaced=t[s], opp=t[1 - s],
                               rec_margin=rw[s] - rw[1 - s], rec_opp=rw[1 - s], day=v.get("day")))
fw = f"{R}/il/live/top_worlds_fresh_300.json"
if not os.path.exists(fw):
    worlds.sort(key=lambda w: (w["ep"], w["seat"])); random.Random(17).shuffle(worlds)
    json.dump(worlds[:300], open(fw, "w"), ensure_ascii=False)
# (c) Sep 26: a second leader benchmark from the held-out games (same ep % 8 == 3 rule) between TODAY's top 10 played from
#     Sep 25 on - their newest submissions; both seats -> il/live/top_worlds_fresh2_300.json (top_bench DAY=fresh2), written once
fw2 = f"{R}/il/live/top_worlds_fresh2_300.json"
if not os.path.exists(fw2):
    w2 = [w for w in worlds if w["replaced"] in TOP10_TODAY and w["opp"] in TOP10_TODAY and (w.get("day") or "") >= "2026-09-25"]
    if len(w2) >= 60:
        w2.sort(key=lambda w: (w["ep"], w["seat"])); random.Random(17).shuffle(w2)
        json.dump(w2[:300], open(fw2, "w"), ensure_ascii=False)
        print(f"today's top-10 benchmark: {min(300, len(w2))} worlds -> {fw2}", flush=True)
    else:
        print(f"today's top-10 benchmark: only {len(w2)} worlds, not written", flush=True)
json.dump(sorted(held | lad), open(f"{R}/il/live/fresh_heldout_eps.json", "w"))    # never trained on (DSM fine-tune too)
print(f"leader benchmark: {len(held)} held-out leader-vs-leader games -> {min(300, len(worlds))} worlds ({fw})", flush=True)
out = f"{R}/il/data/{name}"; os.makedirs(out, exist_ok=True); n = collections.Counter(); by_src = collections.Counter()
for f in glob.glob(f"{R}/il/data/fresh_all/*.npz"):
    b = os.path.basename(f); ep = int(b.split("_")[0])
    if ep not in eps:
        continue
    if ep in lad:
        n["ladder game (excluded)"] += 1; continue
    if ep in held:
        n["leader benchmark (excluded)"] += 1; continue
    if not os.path.exists(f"{R}/il/data/x6/{b[:-4]}.npy"):
        n["no v6 features (left out)"] += 1; continue
    if not os.path.lexists(f"{out}/{b}"):
        os.symlink(os.path.realpath(f), f"{out}/{b}")
    n["kept"] += 1; by_src[os.path.basename(eps[ep][0])] += 1
print(f"{name}: {dict(n)}; kept by source {dict(by_src)}; files now {len(os.listdir(out))}", flush=True)
