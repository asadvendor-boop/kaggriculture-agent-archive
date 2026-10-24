"""Server (operator, Sep 25: "if we need to reach the top, we must imitate DSM best"): il/data/<name> = symlinks to every
DSM seat (team id 10) we have - il/data/fresh_all (top-12 current submissions + daily top games) and il/data/union (older
games) - that has v6 features, minus held-out games (il/live/fresh_heldout_eps.json: ladder + fresh leader benchmark).
Usage: srv_build_dsm.py <name>"""
import sys, os, glob, json, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
R = "/Users/asad.ali/kaggriculture"; TEAM = int(os.environ.get("TEAM", "10")); name = sys.argv[1]      # Sep 26: TEAM=0 -> Boey
held = set(json.load(open(f"{R}/il/live/fresh_heldout_eps.json")))
files = {}
for d in ("fresh_all", "union"):
    for f in glob.glob(f"{R}/il/data/{d}/*.npz"):
        files.setdefault(os.path.basename(f), f)
def team(f):
    try:
        return int(np.load(f)["meta"][6])
    except Exception:
        return -1
names = sorted(files)
with ThreadPoolExecutor(8) as ex:
    teams = list(ex.map(lambda b: team(files[b]), names))
out = f"{R}/il/data/{name}"; os.makedirs(out, exist_ok=True); n = collections.Counter()
for b, t in zip(names, teams):
    if t != TEAM:
        continue
    if int(b.split("_")[0]) in held:
        n["held out"] += 1; continue
    if not os.path.exists(f"{R}/il/data/x6/{b[:-4]}.npy"):
        n["no v6 features"] += 1; continue
    if not os.path.lexists(f"{out}/{b}"):
        os.symlink(os.path.realpath(files[b]), f"{out}/{b}")
    n["kept"] += 1; n["from " + files[b].split("/")[-2]] += 1
print(f"{name}: team {TEAM} seats {dict(n)}", flush=True)
