"""Closed-loop test of a rule-layer setting against REACTIVE opponents (Sep 28): tape rulers let a recorded opponent
collapse when our change moves its cash (its later recorded buys fail, SELL_ORDER=base showed +$50-117k "gains" in 12 of
54 live-loss replays). Here the opponent plays for itself: a scripted bot (harness name, e.g. M7) or an exported
package main.py (e.g. the live NEURAL_20MW8_SE, which carries its own baked-in rules). Fresh engine worlds, both seats,
our network with the env rules (SE_GUARD, SELL_ORDER, ...). Rows -> il/logs/h2h_<tag>.json; compare two tags with
live/pair_cmp.py (same seeds and seats).
Usage: NW=12 SE_GUARD=1 SELL_ORDER=base python live/h2h_rules.py <ckpt> <opponent> <n_seeds> <tag> [seed0]"""
import sys, os, json
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")

if __name__ == "__main__":
    ck, opp, n, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    s0 = int(sys.argv[5]) if len(sys.argv) > 5 else 900000
    from live_rollout import run_job
    games = [(s0 + i, seat, opp) for i in range(n) for seat in (0, 1)]
    nw = int(os.environ.get("NW", "12"))
    jobs = [dict(weights=ck, team=int(os.environ.get("TEAM", "10")), tau=float(os.environ.get("TAU", "0.5")), T=0.2, ks=10.0,
                 greedy=True, seed=k, games=games[k::nw]) for k in range(nw)]
    with mp.get_context("spawn").Pool(nw) as pool:
        res = sum(pool.map(run_job, jobs), [])
    rows = [dict(ep=int(r["seed"]) * 10 + int(r["seat"]), seat=int(r["seat"]), opp=os.path.basename(opp), margin=r["margin"], bank=r["bank"],
                 opp_bank=r["opp_bank"], rec_margin=0.0, rec_opp=r["opp_bank"]) for r in res]
    rows.sort(key=lambda r: (r["ep"], r["seat"]))
    fn = f"/Users/asad.ali/kaggriculture/il/logs/h2h_{tag}.json"
    json.dump(rows, open(fn, "w"))
    import numpy as np
    m = np.array([r["margin"] for r in rows])
    print(f"h2h {tag}: n {len(m)} W {(m > 0).sum()}-{(m < 0).sum()} mean margin {m.mean():+,.0f} (se {m.std() / np.sqrt(len(m)):,.0f})", flush=True)
