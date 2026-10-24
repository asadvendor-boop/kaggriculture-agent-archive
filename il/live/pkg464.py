"""Sep 29: a PACKAGE (numpy runtime, CPU, exactly what Kaggle runs) on the 464 ladder worlds (opponent on tape), split over workers.
Usage: pkg464.py <package main.py> <tag> <worker> <workers>  -> il/logs/pkg464_<tag>_w<k>.json ;  pkg464.py --merge <tag> <workers>
Rows carry ep/opp/opp_rating like bench_multi's, so live/pair_cmp.py compares two tags directly."""
import sys, os, json
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
L = "/Users/asad.ali/kaggriculture/il/logs"
ref = {r["ep"]: r for r in json.load(open(f"{L}/ladder_ilsp_v13_p3_t10_r3_tau0.5_soh_st2_wg_test.json"))}
if sys.argv[1] == "--merge":
    tag, n = sys.argv[2], int(sys.argv[3]); rows = []
    for k in range(n): rows += json.load(open(f"{L}/pkg464_{tag}_w{k}.json"))
    json.dump(sorted(rows, key=lambda r: r["ep"]), open(f"{L}/pkg464_{tag}.json", "w")); print("merged", len(rows)); sys.exit()
from live_rollout import run_scripted
pkg, tag, k, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
W = [w for w in json.load(open("/Users/asad.ali/kaggriculture/il/live/ladder_worlds.json")) if w["ep"] in ref][k::n]
res = run_scripted(dict(us=pkg, games=[(w["seed"], w["seat"], "TAPE:" + w["path"]) for w in W]))
rows = [dict(ep=w["ep"], opp=ref[w["ep"]]["opp"], opp_rating=ref[w["ep"]]["opp_rating"], margin=x["margin"], bank=x.get("bank"), opp_bank=x.get("opp_bank"))
        for w, x in zip(W, res)]
json.dump(rows, open(f"{L}/pkg464_{tag}_w{k}.json", "w")); print(tag, k, "done", len(rows))
