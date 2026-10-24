"""Sep 28 (operator: "check the latest live losses and see the recommended package wins them"): the newest N live losses of
both active scripts, replayed with the opponent's recorded moves, played by the EXACT package files (numpy runtime, rules
baked in): the candidate package and, as a sanity check, the live NEURAL_20MW8_SE package (it must reproduce its own
live losses exactly). One game per process.   Usage: latest_losses_pkg.py <candidate main.py> <live main.py> [N]"""
import sys, os, json, glob, gzip
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
K = "/Users/asad.ali/kaggriculture"


def play(args):
    pkg, w = args
    from live_rollout import run_scripted
    r = run_scripted(dict(us=pkg, games=[(0, w["seat"], "TAPE:" + w["path"])]))[0]
    return r["margin"], r["bank"], r["opp_bank"]


if __name__ == "__main__":
    cand, live = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    W = json.load(open(f"{K}/il/live/top_worlds_livelosses_300.json"))
    W = sorted(W, key=lambda w: -int(w["ep"]))[:n]
    rows = {}
    for d in ("data/rows_56614508.json", "data/rows_56595847.json"):
        for r in json.load(open(f"{K}/{d}")):
            rows[int(r["ep"])] = dict(r, script="20MW8" if "56614508" in d else "20MW5")
    jobs = [(p, w) for w in W for p in (cand, live)]
    with mp.get_context("spawn").Pool(min(len(jobs), 24)) as pool:
        res = pool.map(play, jobs)
    wins_c = wins_l = 0
    print(f"{'episode':>10} {'script':6} {'opponent':24} {'rating':>6} | {'live margin':>11} | {'live pkg replay':>15} | {'PICK replay':>11} (our bank / opp bank)")
    for i, w in enumerate(W):
        (mc, bc, oc), (ml, bl, ol) = res[2 * i], res[2 * i + 1]
        r = rows.get(int(w["ep"]), {})
        wins_c += mc > 0; wins_l += ml > 0
        print(f"{int(w['ep']):>10} {r.get('script', '?'):6} {str(w.get('opp', r.get('opp', '?')))[:24]:24} {float(r.get('opp_rating') or 0):6.0f} | "
              f"{float(r.get('margin', w.get('rec_margin', 0)) or 0):>+11,.0f} | {ml:>+15,.0f} | {mc:>+11,.0f} ({bc:,.0f} / {oc:,.0f}) {'WIN' if mc > 0 else 'loss'}", flush=True)
    print(f"latest {len(W)} live losses: live package replay wins {wins_l}, PICK wins {wins_c}", flush=True)
