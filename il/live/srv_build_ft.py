"""Server: fine-tune set il/data/<name> = symlinks to the seats of the given team ids in the given seat dirs that have v6
features and extended-menu labels (il/data/m8), minus held-out games (il/live/fresh_heldout_eps.json).
Usage: srv_build_ft.py <name> <team_ids comma> <seat_dir> [...]"""
import sys, os, glob, json, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
R = "/Users/asad.ali/kaggriculture"; name, teams = sys.argv[1], {int(x) for x in sys.argv[2].split(",")}
held = set(json.load(open(f"{R}/il/live/fresh_heldout_eps.json")))
files = {}
for d in sys.argv[3:]:
    for f in glob.glob(f"{d}/*.npz"):
        files.setdefault(os.path.basename(f), f)
def team(f):
    try:
        return int(np.load(f)["meta"][6])
    except Exception:
        return -1
names = sorted(files)
with ThreadPoolExecutor(8) as ex:
    tm = list(ex.map(lambda b: team(files[b]), names))
out = f"{R}/il/data/{name}"; os.makedirs(out, exist_ok=True); n = collections.Counter()
for b, t in zip(names, tm):
    if t not in teams:
        continue
    if int(b.split("_")[0]) in held:
        n["held out"] += 1; continue
    if not (os.path.exists(f"{R}/il/data/x6/{b[:-4]}.npy") and os.path.exists(f"{R}/il/data/m8/{b[:-4]}.npy")):
        n["missing features/labels"] += 1; continue
    if not os.path.lexists(f"{out}/{b}"):
        os.symlink(os.path.realpath(files[b]), f"{out}/{b}")
    n[f"team {t}"] += 1
print(f"{name}: {dict(n)}", flush=True)
