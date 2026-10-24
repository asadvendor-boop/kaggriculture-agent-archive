"""How much of each team's CURRENT market play (top-12 current submissions) each output menu can represent:
old menu (features.market_labels via leader_study) - share of market turns fully representable, and why not.
Usage: menu_cover.py <team> [...]     env N (60)"""
import sys, os, json, collections
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from leader_study import study
R = "/Users/asad.ali/kaggriculture"
if __name__ == "__main__":
    idx = json.load(open(f"{R}/data/tapes_top12/index.json")); N = int(os.environ.get("N", "60"))
    for team in sys.argv[1:]:
        items = [(f"{R}/data/tapes_top12/{e}.json.gz", v["teams"].index(team)) for e, v in sorted(idx.items(), reverse=True) if team in (v.get("teams") or [])][:N]
        with mp.get_context("spawn").Pool(8) as pool:
            res = pool.map(study, items)
        rep = collections.Counter(); why = collections.Counter()
        for r, w, _ in res:
            rep.update(r); why.update(w)
        print(f"{team:22s} {len(items)} games: old menu represents {1 - rep['unrep'] / max(1, rep['turns']):.1%} of market turns; not: " +
              ", ".join(f"{k} {v / max(1, rep['turns']):.1%}" for k, v in why.most_common(4)), flush=True)
