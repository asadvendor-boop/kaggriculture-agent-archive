"""Parallel version of tools/tape_sub.py (same tape format): tape_sub_par.py <sub> <out_dir>  (THREADS env, default 8)."""
import sys, os, json, gzip
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import pull_leaders as PL                         # fetch_replay with Retry-After backoff
R = "/Users/asad.ali/kaggriculture"
sub, out = int(sys.argv[1]), sys.argv[2]; os.makedirs(out, exist_ok=True)
rows = [r for r in json.load(open(f"{R}/data/rows_{sub}.json")) if not os.path.exists(f"{out}/{r['ep']}.json.gz")]
def one(r):
    try:
        d = PL.fetch_replay(r["ep"]); steps = d["steps"]
    except Exception as ex:
        return f"fail {r['ep']} {str(ex)[:60]}"
    acts = [[(steps[t + 1][s].get("action") if t + 1 < len(steps) and isinstance(steps[t + 1][s].get("action"), dict) else None) for t in range(720)] for s in (0, 1)]
    json.dump(dict(ep=r["ep"], seed=int(d["info"]["seed"]), names=d["info"].get("TeamNames"), rewards=d["rewards"], seat=r["seat"], res=r["res"],
                   opp=r["opp"], opp_rating=r["opp_rating"], actions=acts), gzip.open(f"{out}/{r['ep']}.json.gz", "wt"), separators=(",", ":"))
    return None
with ThreadPoolExecutor(int(os.environ.get("THREADS", "8"))) as ex:
    errs = [e for e in ex.map(one, rows) if e]
print(f"sub {sub}: taped {len(rows) - len(errs)} of {len(rows)} missing; failures {len(errs)}", flush=True)
