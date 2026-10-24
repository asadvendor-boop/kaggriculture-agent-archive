"""Per-seat imitation weights by player strength (operator, Sep 26: "weight imitation data by player strength ... even a
soft push is worth it").  A seat's rating = the LATEST rating of the submission that played it (so a new submission still
climbing is judged by where it got to) when a ListEpisodes rows file names that submission, else the team's rating (the
best latest rating over its submissions seen in any rows file), else UNKNOWN (default 2500).  Weight = piecewise-linear in
the rating through (2300, 0.3) (2500, 0.5) (2800, 1.0) (3000, 1.5) (3150, 1.8), flat beyond; train6.py (RATING_W)
multiplies it into the win/loss weight and normalizes per batch, so only the ratios matter.
Usage: srv_rating_w.py <seat_dir> <out.json> [<tape_dir> ...]   (tape dirs with index.json; default: data/tapes_top12 +
data/tapes_kd/*)      env UNKNOWN, SUBS_JSON (leaders' newer submissions, as for srv_fetch_top12)"""
import sys, os, json, glob, collections
import numpy as np
R = "/Users/asad.ali/kaggriculture"
XS, YS = [2300, 2500, 2800, 3000, 3150], [0.3, 0.5, 1.0, 1.5, 1.8]
SUBS = {"Boey": [56521745], "DSM": [56564674, 56571979], "M & M & P & Q": [56553720, 56570008],      # operator, Sep 26:
        "Fourth Quadrant": [56555482], "DECEM": [56558220, 56573934], "Unknown Mother-Goose": [56567212],  # today's top 10
        "Azat Akhtyamov": [56561951], "🐚seek inspiration🐚": [56551631], "Majkel1337": [56538379],       # (Azat's id checked
        "THIRD FARM CLUB": [56520754]}                                                                    #  on Kaggle: 56561951)
OLD_SUBS = {"DSM": [56541087], "Boey": [56521745], "M & M & P & Q": [56524807, 56544748],
        "Unknown Mother-Goose": [56525914, 56538672], "DECEM": [56526629, 56540966], "Fourth Quadrant": [56521606],
        "Majkel1337": [56538157], "mtmr_s1": [56522781], "Vadim Vasilenko": [56532063, 56550228],
        "吃白饭的大肥鱼": [56506820], "Kaggledew Valley 🏆": [56478402], "THIRD FARM CLUB": [56520754, 56539374],
        "Azat Akhtyamov": [56533329], "🐚seek inspiration🐚": [56489275, 56489993]}
if os.environ.get("SUBS_JSON") and os.path.exists(os.environ["SUBS_JSON"]):
    for t, ids in json.load(open(os.environ["SUBS_JSON"])).items():
        SUBS.setdefault(t, []); SUBS[t] += [int(i) for i in ids if int(i) not in SUBS[t]]
sub_team = {s: t for t, ids in list(OLD_SUBS.items()) + list(SUBS.items()) for s in ids}      # old ids only map rows to teams
seat_dir, out = sys.argv[1], sys.argv[2]
tdirs = sys.argv[3:] or [f"{R}/data/tapes_top12"] + sorted(glob.glob(f"{R}/data/tapes_kd/*"))
UNKNOWN = float(os.environ.get("UNKNOWN", "2500"))

latest = {}                                   # sub -> (end, rating)
ep_sub = {}                                   # (ep, team) -> sub
sub_of_team = collections.defaultdict(set)


def note(sub, team, end, rating):
    if not sub or not rating:
        return
    sub = int(sub); p = latest.get(sub)
    if p is None or end > p[0]:
        latest[sub] = (end, float(rating))
    sub_of_team[team].add(sub)


for fn in glob.glob(f"{R}/data/rows_*.json"):
    try:
        me_sub = int(os.path.basename(fn)[5:-5]); rows = json.load(open(fn))
    except Exception:
        continue
    me = sub_team.get(me_sub)
    for r in rows:
        if r.get("res") not in ("W", "L", "T"):
            continue
        end = r.get("end") or ""; ep = int(r["ep"])
        if r.get("opp"):
            note(r.get("opp_sub"), r["opp"], end, r.get("opp_rating"))
            if r.get("opp_sub"):
                ep_sub[(ep, r["opp"])] = int(r["opp_sub"])
        if me:
            note(me_sub, me, end, r.get("rating"))
            ep_sub[(ep, me)] = me_sub
team_r = {t: max(latest[s][1] for s in subs) for t, subs in sub_of_team.items() if subs}
teams_of = {}
for d in tdirs:
    fn = f"{d}/index.json"
    if os.path.exists(fn):
        for ep, v in json.load(open(fn)).items():
            teams_of.setdefault(int(ep), v.get("teams") or [None, None])
W = {}; src = collections.Counter(); by_team = collections.defaultdict(list)
for f in glob.glob(f"{seat_dir}/*.npz"):
    b = os.path.basename(f)[:-4]; ep, s = (int(x) for x in b.split("_"))
    team = (teams_of.get(ep) or [None, None])[s]
    sub = ep_sub.get((ep, team))
    if sub in latest:
        r = latest[sub][1]; src["submission"] += 1
    elif team in team_r:
        r = team_r[team]; src["team"] += 1
    else:
        r = UNKNOWN; src["unknown"] += 1
    W[b] = round(float(np.interp(r, XS, YS)), 3); by_team[team].append(W[b])
json.dump(W, open(out, "w"), ensure_ascii=False)
w = np.array(list(W.values())) if W else np.zeros(1)
print(f"{len(W)} seats -> {out}; rating source {dict(src)}; weight mean {w.mean():.2f} p10 {np.percentile(w, 10):.2f} "
      f"p50 {np.percentile(w, 50):.2f} p90 {np.percentile(w, 90):.2f}", flush=True)
for t, v in sorted(by_team.items(), key=lambda kv: -len(kv[1]))[:20]:
    print(f"   {str(t)[:24]:24s} seats {len(v):5d}  weight {np.mean(v):.2f}  (team rating {team_r.get(t, float('nan')):.0f})", flush=True)
