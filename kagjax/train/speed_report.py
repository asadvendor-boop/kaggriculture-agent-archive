"""Summarise the update-speed measurements (speed_chain.sh outputs) into tables + the dedicated-A100 projection.
usage: python speed_report.py [kagjax/logs_server/D/speed]   (prints; also writes speed_summary.json there)"""
import os, sys, json, glob, re
import numpy as np

S = sys.argv[1] if len(sys.argv) > 1 else "/Users/asad.ali/kaggriculture/kagjax/logs_server/D/speed"
DEDICATED_TF32 = 125.0          # TF32 TFLOP/s of an 8192^2 matmul on a dedicated A100 (conservative; peak 156)


def rows(pattern):
    out = []
    for fn in sorted(glob.glob(os.path.join(S, pattern))):
        for l in open(fn):
            l = l.strip()
            if l.startswith("{"):
                d = json.loads(l); d["_file"] = os.path.basename(fn); out.append(d)
    return out


def main():
    summ = {}
    # ---- contention probes
    probes = [r for r in rows("bench_*.jsonl") if r.get("tf32_tflops")]
    if probes:
        t = [r["tf32_tflops"] for r in probes]
        summ["probe_tf32_tflops"] = dict(min=min(t), max=max(t), mean=float(np.mean(t)))
        print(f"shared-GPU TF32 matmul probe: {min(t)}-{max(t)} TFLOP/s (mean {np.mean(t):.0f}); dedicated A100 ~{DEDICATED_TF32:.0f}")
    fac = DEDICATED_TF32 / summ["probe_tf32_tflops"]["mean"] if probes else 1.0
    # ---- update throughput
    up = [r for r in rows("bench_update_*.jsonl") if r.get("bench") == "update"]
    print("\nPPO update, 92,160 samples (128 games x 720 steps), 20M policy unless [rf_w_dsm_s0] (10M); x gate = vs that model's gate run:")
    print(f"{'variant':40s} {'samples/s':>9s} {'x gate':>6s} {'sample-pass/s':>13s} {'temp GiB':>8s} {'LR0: |r-1|>1e-2 / >5e-2 / max (units)':>38s} {'market >5e-2':>12s} {'clip0':>6s}")
    g0s = {}                                                  # gate rate per model (x gate column)
    for r in up:
        if r["variant"] == "gate_E2_MB256":
            g0s.setdefault(r.get("model", "g20_w_dsm_s0"), []).append(r["samples_per_s"])
    g0s = {m: float(np.mean(v)) for m, v in g0s.items()}
    by = {}
    HEADV = ("E1_MB4096_ref25_head", "E2_MB4096_ref25_head", "E1_MB4096_head")
    for r in up:
        name = r["variant"]
        if name.startswith(HEADV) and not name.endswith(("_m1024", "_m2048")) and r.get("micro", 512) != 512:
            name += f"@m{r['micro']}"                       # (the first runs used MICRO 1024; MICRO 512 since 22:52)
        if r.get("model", "g20_w_dsm_s0") != "g20_w_dsm_s0":
            name += f" [{r['model']}]"
        by.setdefault(name, []).append(r)
    summ["update"] = {}
    for v, rs in by.items():
        sp = float(np.mean([r["samples_per_s"] for r in rs]))
        r = rs[-1]
        summ["update"][v] = dict(samples_per_s=sp, runs=[x["samples_per_s"] for x in rs], passes_per_s=r["sample_passes_per_s"],
                                 temp_gib=r.get("temp_gib"), r0_ru_frac1e2=r.get("r0_ru_frac1e2"), r0_ru_frac5e2=r.get("r0_ru_frac5e2"),
                                 r0_ru_maxdev=r.get("r0_ru_maxdev"), r0_rm_frac5e2=r.get("r0_rm_frac5e2"), r0_clipfrac=r.get("r0_clipfrac"),
                                 finite=r.get("finite"))
        g0 = g0s.get(r.get("model", "g20_w_dsm_s0"))
        print(f"{v:40s} {sp:9.0f} {sp / g0 if g0 else 0:6.2f} {r['sample_passes_per_s']:13.0f} {r.get('temp_gib', 0) or 0:8.2f} "
              f"{r.get('r0_ru_frac1e2', 0):12.4f} {r.get('r0_ru_frac5e2', 0):8.4f} {r.get('r0_ru_maxdev', 0):8.3f}   "
              f"{r.get('r0_rm_frac5e2', 0):12.4f} {r.get('r0_clipfrac', 0):6.3f}" + (f"   runs {[x['samples_per_s'] for x in rs]}" if len(rs) > 1 else ""))
    for r in rows("bench_update_*.jsonl"):
        if r.get("bench") == "update_error":
            print(f"{r['variant'] + ' [' + r.get('model', '') + ']':40s} FAILED: {r['err'][:110]}")
    rr = [r for r in up if r.get("probe_before")]
    if rr:
        print("\nback-to-back repeat with a TF32 matmul probe around each timed update (dedicated ~ samples/s x "
              f"{DEDICATED_TF32:.0f} / probe):")
        summ["update_probed"] = {}
        for r in rr:
            pr = 0.5 * (r["probe_before"] + r["probe_after"])
            name = r["variant"] + ("" if r.get("model", "g20_w_dsm_s0") == "g20_w_dsm_s0" else f" [{r['model']}]")
            summ["update_probed"][name] = dict(samples_per_s=r["samples_per_s"], probe=pr, dedicated=r["samples_per_s"] * DEDICATED_TF32 / pr)
            print(f"  {name:40s} {r['samples_per_s']:6d}/s at probe {r['probe_before']:5.1f}/{r['probe_after']:5.1f} TFLOP/s "
                  f"-> dedicated ~{r['samples_per_s'] * DEDICATED_TF32 / pr:6.0f}/s")
    # ---- rollout
    ro = [r for r in rows("bench_rollout_*.jsonl") if r.get("bench") == "rollout"]
    rprobe = {}
    for r in rows("bench_rollout_*.jsonl"):
        if r.get("tf32_tflops"):
            rprobe.setdefault(r["_file"], []).append(r["tf32_tflops"])
    summ["rollout"] = {}
    if ro:
        print("\nRollout, B games per lockstep step (both seats: learner forward + opponent forwards + engine + rules):")
        for r in ro:
            mdl = r.get("model", "g20_w_dsm_s0")
            pr = float(np.mean(rprobe.get(r["_file"], [0]))) or 1.0
            print(f"  {mdl:14s} {r['mode']:15s} B={r['B']:5d}: {r['ms_per_step']:6.1f} ms/step = {r['game_steps_per_s']:6d} game-steps/s "
                  f"(probe {pr:4.1f} -> dedicated ~{r['game_steps_per_s'] * DEDICATED_TF32 / pr:6.0f}; peak {r.get('proc_peak_gib', '?')} GiB; "
                  f"{r['_file']}{'; rounds ' + str(r['ms_rounds']) if r.get('ms_rounds') else ''})")
            key = f"{r['mode']}@{r['B']}" + ("" if mdl == "g20_w_dsm_s0" else f" [{mdl}]")
            summ["rollout"].setdefault(key, []).append(r["game_steps_per_s"])
        for r in rows("bench_rollout_*.jsonl"):
            if r.get("bench") == "rollout_error":
                print(f"  {r.get('model', ''):14s} {r['mode']:15s} B={r['B']:5d}: FAILED {r['err'][:120]}")
    # ---- KL subsample
    kl = [r for r in rows("exp/exp_kl.jsonl") if r.get("exp") == "kl"]
    if kl:
        k = kl[-1]; summ["kl"] = k
        print(f"\nKL anchor subsample on {k['samples']} samples (full KL units {k['klu_full']:.4f}, market {k['klm_full']:.5f}):")
        for f in ("f0.5", "f0.25"):
            x = k[f]
            print(f"  {f}: mean of 2,000 draws / full = {x['klu_mean'] / k['klu_full']:.4f} (units, bias {x['klu_bias_in_se']:+.2f} se), "
                  f"{x['klm_mean'] / k['klm_full']:.4f} (market, bias {x['klm_bias_in_se']:+.2f} se); per-draw rel sd {x['klu_rel_sd_per_draw']:.3f} / {x['klm_rel_sd_per_draw']:.3f}")
        g = k["grad"]
        print(f"  gradient of beta*KL, f=0.25: per-draw cosine {min(g['per_draw_cos'])}-{max(g['per_draw_cos'])}, mean of 8 draws cosine "
              f"{g['mean8_cos']:.4f} rel err {g['mean8_rel_err']:.3f} (per draw {np.mean(g['per_draw_rel_err']):.3f})")
    # ---- value options
    va = [r for r in rows("exp/exp_value.jsonl") if r.get("exp") == "value"]
    if va:
        print("\nValue options on the same 4 rollouts (policy frozen): explained variance of the MC return BEFORE each update")
        print(f"{'option':20s} {'r0 (initial)':>12s} {'r1':>7s} {'r2':>7s} {'r3':>7s} {'held-out mean':>13s} {'adv sd r1-3':>11s} {'s/update':>8s}")
        summ["value"] = {}
        opts = {}
        for r in va:
            opts.setdefault(r["option"], {})[r["rollout"]] = r
        for o, rr in opts.items():
            ev = [rr[i]["ev_mc"] if i in rr else float("nan") for i in range(4)]
            ho = float(np.nanmean(ev[1:]))
            adv = float(np.mean([rr[i]["adv_std"] for i in range(1, 4) if i in rr]))
            sec = float(np.mean([rr[i]["seconds"] for i in range(1, 4) if i in rr]))
            summ["value"][o] = dict(ev_mc=ev, heldout_ev=ho, adv_std=adv, seconds=sec, ev_gae=[rr[i]["ev_gae"] if i in rr else None for i in range(4)])
            print(f"{o:20s} {ev[0]:12.3f} {ev[1]:7.3f} {ev[2]:7.3f} {ev[3]:7.3f} {ho:13.3f} {adv:11.4f} {sec:8.1f}")
    # ---- sanity runs
    print("\n3-iteration sanity runs from g20_w_dsm_s0 (128 games; eval: 128 fixed worlds, greedy fp32, paired vs iteration 0):")
    summ["sanity"] = {}
    for d in sorted(glob.glob(os.path.join(S, "san_*"))):
        n = os.path.basename(d)
        st = [json.loads(l) for l in open(os.path.join(d, "stats.jsonl"))] if os.path.exists(os.path.join(d, "stats.jsonl")) else []
        ev = [l.strip() for l in open(os.path.join(d, "log.txt"))] if os.path.exists(os.path.join(d, "log.txt")) else []
        ev = [l for l in ev if l.startswith("[eval it 3]")]
        its = [dict(it=s["it"], klu=round(s["klu"], 4), klm=round(s["klm"], 4), clip=round(s["clipfrac"], 3), ent=round(s["ent"], 3),
                    l_v=round(s["l_v"], 4), ev_mc=round(s.get("ev_mc", float("nan")), 3), finite=s["finite"], t_roll=round(s["t_roll"]),
                    t_upd=round(s["t_upd"])) for s in st]
        summ["sanity"][n] = dict(iterations=its, eval3=ev[-1] if ev else None)
        print(f"  {n}:")
        for x in its:
            print(f"    it {x['it']}: KL u/m {x['klu']}/{x['klm']} clip {x['clip']} ent {x['ent']} v {x['l_v']} EV {x['ev_mc']} finite {x['finite']} "
                  f"rollout {x['t_roll']}s update {x['t_upd']}s")
        if ev:
            print("    " + re.sub(r"^\[eval it 3\] \d+s \| ", "", ev[-1]))
    # ---- projection
    if summ.get("update") and summ.get("rollout"):
        print(f"\nProjection to a dedicated A100 (x{fac:.2f} from the matmul probe; 10M ~x1.9 of the 20M by FLOPs):")
        for rk in ("map_K4@1024", "groups_G1@1024", "map_K4_oppbf16@1024", "map_K4@2048"):
            if rk not in summ["rollout"]:
                continue
            R = float(np.mean(summ["rollout"][rk]))
            print(f" rollout {rk}: {R:.0f} game-steps/s shared -> ~{R * fac:.0f} dedicated")
            for v in ("gate_E2_MB256", "E1_MB4096_ref25", "E1_MB4096_ref25_head", "E1_MB4096_ref25_head_bf16grad", "E1_MB4096_ref25_head_bf16"):
                if v in summ["update"]:
                    U = summ["update"][v]["samples_per_s"]
                    e2e = 1.0 / (1.0 / R + 1.0 / U)
                    print(f"  {v:28s}: shared update {U:6.0f}/s, end-to-end {e2e:6.0f} game-steps/s -> dedicated ~{U * fac:6.0f}/s, "
                          f"end-to-end ~{e2e * fac:6.0f}/s (20M)")
    json.dump(summ, open(os.path.join(S, "speed_summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
