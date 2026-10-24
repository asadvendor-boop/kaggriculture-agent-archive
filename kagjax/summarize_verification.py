"""Collect the operator's verification items 1-4 into results/verification_summary.json (+ print a table).
Inputs (copied back from the server into results/server/): parity_srv_*.json, replay_gpu_*.json,
rng_stats_gpu_*.json, batchindep_*.json; Mac-side results in results/.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
SRV = os.path.join(R, "server")


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    out = {}
    # 1. full recorded-game parity (full observe() dicts of both seats, every step)
    games = exact = tape = 0
    fails = []
    parts = []
    for p in sorted(glob.glob(os.path.join(SRV, "parity_srv_*.json"))) + sorted(glob.glob(os.path.join(R, "parity_*.json"))):
        d = jl(p)
        if not d or "summary" not in d:
            continue
        s = d["summary"]
        parts.append({"file": os.path.basename(p), "games": s["games"], "exact": s["exact_all_steps"],
                      "cfg": s.get("cfg", "seq"), "start": s.get("start"), "dir": s.get("dir")})
        if os.path.basename(p).startswith("parity_srv_"):
            games += s["games"]
            exact += s["exact_all_steps"]
            tape += s["tape_reward_equal"]
            for r in d["results"]:
                if not (r["mismatch"] is None and r["reward_equal"]):
                    fails.append({"game": r["game"], "first_bad_step": r["exact_steps"],
                                  "fields": (r["mismatch"] or {}).get("diff", [])[:5]})
    out["1_full_parity_server"] = {"games": games, "exact_every_step": exact, "final_reward_equals_kaggle_tape": tape,
                                   "divergences": fails, "parts": parts}
    # 2. GPU = C++ (replays of datasets with C++ digests)
    rep = []
    for p in sorted(glob.glob(os.path.join(SRV, "replay_gpu_*.json"))) + sorted(glob.glob(os.path.join(R, "replay_cpu_*.json"))):
        d = jl(p)
        if d:
            rep.append({k: d[k] for k in ("data", "device", "cfg", "games", "games_all_steps_identical_to_cpp",
                                          "final_money_equal_cpp", "final_money_equal_kaggle_tape", "wall_s")}
                       | {"mismatches": d.get("mismatches", [])[:10]})
    out["2_device_replays_vs_cpp"] = rep
    # 3. training randomness
    d = jl(os.path.join(SRV, "rng_stats_gpu_4096.json")) or jl(os.path.join(R, "rng_stats_cpu_4096.json"))
    if d:
        keep = {k: d[k] for k in d if k not in ("shops", "weeds_per_day")}
        keep["weeds_per_day_tests"] = {k: d["weeds_per_day"][k] for k in ("per_day_mannwhitney_p_min",
                                                                          "fisher_combined_p")}
        keep["shops_min_homog_p"] = min(s["homog_p"] for s in d["shops"])
        keep["shops_min_uniform_p"] = min(min(s["uniform_p_exact"], s["uniform_p_train"]) for s in d["shops"])
        out["3_training_randomness"] = keep
    # 4. batch independence
    bi = []
    for p in sorted(glob.glob(os.path.join(SRV, "batchindep_*.json"))) + sorted(glob.glob(os.path.join(R, "batchindep_*.json"))):
        d = jl(p)
        if d:
            bi.append(d)
    out["4_batch_independence"] = bi
    json.dump(out, open(os.path.join(R, "verification_summary.json"), "w"), indent=1)
    print(json.dumps({"1": {k: v for k, v in out["1_full_parity_server"].items() if k != "parts"},
                      "2": [(r["cfg"], os.path.basename(r["data"]), r["games"], r["games_all_steps_identical_to_cpp"],
                             r["final_money_equal_kaggle_tape"], r["device"]) for r in rep],
                      "4": [(os.path.basename(b["data"]), b["device"], b["cfg"], b["sizes"],
                             b["identical_across_batch_sizes"], b["identical_to_cpp"]) for b in bi]}, indent=1))
    if "3_training_randomness" in out:
        t = out["3_training_randomness"]
        print("3:", {k: t[k] for k in ("final_money_seat0", "weeds_per_game", "weed_rate_exact", "weed_rate_train",
                                       "weed_rate_two_sample_p", "shop_pair_01_homog_p", "shops_min_homog_p",
                                       "weeds_per_day_tests")})


if __name__ == "__main__":
    main()
