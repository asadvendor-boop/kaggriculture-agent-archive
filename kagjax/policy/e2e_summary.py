"""Aggregate the end-to-end results (e2e_compare / e2e_verify outputs) into results_e2e_summary.json."""
import json, os, glob
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(D, "e2e_out")
summ = {}
for name in ("rl2_hi", "g20_hi", "vs_hi", "cpu_rl2_hi", "cpu_g20_hi", "cpu_vs_hi"):
    p = os.path.join(O, f"cmp_{name}.json")
    if not os.path.exists(p):
        continue
    c = json.load(open(p))["vs_reference"]
    allg = c["all"]
    jb = np.array([g["jax_banks"] for g in allg]); rb = np.array([g["ref_banks"] for g in allg])
    v = os.path.join(O, f"verify_{name}.json")
    ver = json.load(open(v)) if os.path.exists(v) else []
    gaps = [x["smallest_gap"] for x in ver if x.get("smallest_gap") is not None]
    summ[name] = dict(games=c["games"], identical=c["identical"], diverged=c["games"] - c["identical"],
                      divergence_steps=sorted(g["first_divergence"]["step"] for g in c["diverged"] if g["first_divergence"]),
                      divergence_kinds=[m["kind"] for x in ver for m in x.get("margins", [])],
                      largest_flipped_margin=max(gaps) if gaps else None,
                      mean_bank_jax=jb.mean(0).tolist(), mean_bank_ref=rb.mean(0).tolist(),
                      mean_bank_diff=(jb - rb).mean(0).tolist(), sd_bank_diff=(jb - rb).std(0).tolist(),
                      identical_games_banks_equal=all(g["jax_banks"] == g["ref_banks"] for g in allg if g["identical"]))
for name in ("rl2_tf32", "g20_tf32"):
    p = os.path.join(O, f"cmp_{name}.json")
    if os.path.exists(p):
        c = json.load(open(p))
        v = c["vs_fp32"]
        summ[name] = {k: v[k] for k in v if k != "all"}
        summ[name]["identical_to_reference"] = c["vs_reference"]["identical"]
        steps = sorted(x["first_divergence"]["step"] for x in c["vs_reference"]["diverged"] if x["first_divergence"])
        summ[name]["first_divergence_vs_reference_median_step"] = float(np.median(steps)) if steps else None
x = os.path.join(D, "results_server", "xplat_mac.npz"); y = os.path.join(D, "results_server", "xplat_linux.npz")
if os.path.exists(x) and os.path.exists(y):
    a, b = np.load(x), np.load(y)
    summ["numpy_runtime_mac_vs_linux"] = dict(states=int(len(a["keys"])), max_abs_logit_diff_ops=float(np.abs(a["op"] - b["op"]).max()),
                                              max_abs_logit_diff_market=float(np.abs(a["mk"] - b["mk"]).max()),
                                              frac_identical_ops=float((a["op"] == b["op"]).mean()), frac_identical_market=float((a["mk"] == b["mk"]).mean()))
json.dump(summ, open(os.path.join(D, "results_e2e_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
