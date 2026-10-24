"""Sep 30: map Kaggle team ids -> their newest submission ids (ListEpisodes by submissionId only; the public endpoint rejects teamId).
Seeds = known submissions; every listed episode's 'submissions' carries teamId, so the top teams' newest uploads appear through
their games against the seeds. Hop 2 lists the newly found top-team submissions too. Output: live/team_subs_now.json"""
import sys, json, time
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import urllib.request
TEAMS = json.load(open(sys.argv[1]))                       # {"teamId": "name", ...} in rank order
SEEDS = [int(s) for s in json.load(open(sys.argv[2]))]
def lst(sub):
    for a in range(3):
        try:
            req = urllib.request.Request("https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
                  data=json.dumps({"submissionId": sub}).encode(), headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as ex:
            print("fail", sub, str(ex)[:60], flush=True); time.sleep(20)
    return {}
found, listed, last = {}, set(), {}
def scan(sub):
    L = lst(sub); listed.add(sub); time.sleep(3)
    for s in L.get("submissions", []) or []:
        t = str(s.get("teamId")); sid = int(s.get("id") or 0)
        if t in TEAMS and sid:
            found.setdefault(t, set()).add(sid)
    for e in L.get("episodes", []) or []:
        for a in e.get("agents", []) or []:
            sid = int(a.get("submissionId") or 0); last[sid] = max(last.get(sid, ""), (e.get("endTime") or "")[:16])
for s in SEEDS:
    scan(s)
for rnd in range(2):                                        # hop: list the newest found submissions of each team
    todo = [max(v) for v in found.values() if max(v) not in listed]
    print(f"hop {rnd + 1}: {len(todo)} new submissions to list", flush=True)
    for s in todo:
        scan(s)
out = {}
for t, name in TEAMS.items():
    subs = sorted(found.get(t, []), reverse=True)[:2]
    out[name] = subs
    print(f"{name[:26]:28s} team {t}: newest subs {subs}  last game {[last.get(s, '-') for s in subs]}", flush=True)
json.dump(out, open("/Users/asad.ali/kaggriculture/il/live/team_subs_now.json", "w"), ensure_ascii=False, indent=1)
