"""Leaders' newer submissions (Sep 26): every rows file (ours from the live pollers, the leaders' from srv_fetch_top12)
records each opponent's submission id (opp_sub).  For each top-12 team, list the submissions seen as an opponent that are
newer (larger id) than the team's known submissions in srv_fetch_top12.SUBS, newest first, with their last game time and rating -> <out.json> {team: [sub, ...]}
(only submissions seen since SINCE, default 2026-09-24).   Usage: srv_discover_subs.py <out.json>"""
import sys, os, json, glob
R = "/Users/asad.ali/kaggriculture"
KNOWN = {"Boey": [56521745], "DSM": [56564674, 56571979], "M & M & P & Q": [56553720, 56570008],      # operator, Sep 26:
        "Fourth Quadrant": [56555482], "DECEM": [56558220, 56573934], "Unknown Mother-Goose": [56567212],  # today's top 10
        "Azat Akhtyamov": [56561951], "🐚seek inspiration🐚": [56551631], "Majkel1337": [56538379],       # (Azat's id checked
        "THIRD FARM CLUB": [56520754]}                                                                    #  on Kaggle: 56561951)
if os.environ.get("KNOWN_JSON"):                      # Sep 27 night: the operator's current top 10 (new teams with [])
    KNOWN = {t: [int(i) for i in ids] for t, ids in json.load(open(os.environ["KNOWN_JSON"])).items()}
SINCE = os.environ.get("SINCE", "2026-09-24")
seen = {}
for fn in glob.glob(f"{R}/data/rows_*.json"):
    try:
        rows = json.load(open(fn))
    except Exception:
        continue
    for r in rows:
        t, s, e = r.get("opp"), r.get("opp_sub"), r.get("end") or ""
        if t in KNOWN and s and int(s) > max(KNOWN[t] or [0]) and e >= SINCE:   # newer than the known current ones
            k = (t, int(s)); p = seen.get(k)
            if p is None or e > p[0]:
                seen[k] = (e, float(r.get("opp_rating") or 0), (p[2] + 1) if p else 1)
            else:
                seen[k] = (p[0], p[1], p[2] + 1)
out = {}
for (t, s), (e, rt, n) in sorted(seen.items(), key=lambda kv: kv[1][0], reverse=True):
    out.setdefault(t, []).append(s)
    print(f"{t:22s} sub {s}: last seen {e}, rating then {rt:.0f}, {n} games vs our/leader subs", flush=True)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
print(f"{sum(len(v) for v in out.values())} new leader submissions -> {sys.argv[1]}", flush=True)
