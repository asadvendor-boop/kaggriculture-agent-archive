"""Collect the pre-training gate results (gate_gpu.json, gate_cpu_rules.json, gate_cpu_roundtrip.json, e2e_cpu.out,
gate5 run stats) into $GATE_DIR/gate_report.json and print a short verdict per item."""
import os, sys, json, glob

GD = sys.argv[1] if len(sys.argv) > 1 else "/Users/asad.ali/kaggriculture/kagjax/logs_server/D/gate"
D = os.path.dirname(GD)


def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"missing": repr(e)}


def main():
    rep = {"gpu": load(os.path.join(GD, "gate_gpu.json")), "rules": load(os.path.join(GD, "gate_cpu_rules.json")),
           "rules_se_window": load(os.path.join(GD, "gate_cpu_se.json")), "roundtrip": load(os.path.join(GD, "gate_cpu_roundtrip.json"))}
    hp = os.path.join(GD, "lr0_highest", "stats.jsonl")
    if os.path.exists(hp):
        rep["lr0_full_fp32"] = [json.loads(l) for l in open(hp)]
    e2e = [l.strip() for l in open(os.path.join(D, "e2e_cpu.out"))] if os.path.exists(os.path.join(D, "e2e_cpu.out")) else []
    rep["e2e"] = [l for l in e2e if l.startswith("[game") or l.startswith("E2E") or l.startswith("[jax]")]
    g5 = os.path.join(D, "gate5")
    if os.path.exists(os.path.join(g5, "stats.jsonl")):
        rep["learning"] = dict(stats=[json.loads(l) for l in open(os.path.join(g5, "stats.jsonl"))],
                               evals=[json.loads(l) for l in open(os.path.join(g5, "eval.jsonl"))] if os.path.exists(os.path.join(g5, "eval.jsonl")) else [],
                               log_tail=[l.strip()[:300] for l in open(os.path.join(g5, "log.txt"))][-12:])
    json.dump(rep, open(os.path.join(GD, "gate_report.json"), "w"), indent=1, default=str)
    g = rep["gpu"]
    for k in ("item1_behavior", "item1_recompute"):
        if k in g:
            it = g[k]["per_iteration"]
            print(f"[1 {k}] bit-unchanged params: {g[k]['bit_unchanged']}; max|ratio-1| units {max(x['ru_maxdev'] for x in it):.2e} "
                  f"qty {max(x['rq_maxdev'] for x in it):.2e} market {max(x['rm_maxdev'] for x in it):.2e}; clipfrac {max(x['clipfrac'] for x in it)}; "
                  f"KL u/m {max(abs(x['klu']) for x in it):.1e}/{max(abs(x['klm']) for x in it):.1e}; stored-vs-recomputed "
                  f"{max(x['old_diff_lpu'] for x in it):.1e}/{max(x['old_diff_lpq'] for x in it):.1e}/{max(x['old_diff_lpm'] for x in it):.1e}")
    if "item3" in g:
        print(f"[3] {json.dumps(g['item3'])}")
    for x in rep.get("lr0_full_fp32", []):
        print(f"[1 full fp32] max|ratio-1| units {x['ru_maxdev']:.2e} qty {x['rq_maxdev']:.2e} market {x['rm_maxdev']:.2e}; "
              f"stored-vs-recomputed {x.get('old_diff_lpu', 0):.1e}/{x.get('old_diff_lpq', 0):.1e}/{x.get('old_diff_lpm', 0):.1e}; clip {x['clipfrac']}")
    se = rep["rules_se_window"]
    if "states_days_9_to_14" in se:
        print("[2 se window] " + "; ".join(f"{c}: {v['states']} states, changed {v['rule_changed_action']}, mismatches {v['mismatches']}"
                                          for c, v in se["states_days_9_to_14"].items()))
    r = rep["rules"]
    if "real_states" in r:
        for k in ("real_states", "real_states_market_padded_to_10"):
            print(f"[2 {k}] " + "; ".join(f"{c}: {v['states']} states, changed {v['rule_changed_action']}, mismatches {v['mismatches']}" for c, v in r[k].items()))
    if "decisions" in rep["roundtrip"]:
        rt = rep["roundtrip"]
        print(f"[4] package vs JAX greedy decisions {rt['identical']}/{rt['decisions']}; live_rollout load: {rt.get('live_rollout_net')}; BCAgent ok {rt.get('bcagent_action_ok')}")
    for l in rep["e2e"]:
        print("[e2e]", l[:200])
    if "learning" in rep:
        for s in rep["learning"]["stats"]:
            print(f"[5] it {s['it']}: klu {s['klu']:.4f} klm {s['klm']:.4f} clip {s['clipfrac']:.3f} ent {s['ent']:.3f} v {s['l_v']:.4f} finite {s['finite']} "
                  f"t_roll {s['t_roll']:.0f}s t_upd {s['t_upd']:.0f}s")
        for e in rep["learning"]["evals"]:
            print("[5 eval]", json.dumps(e))


if __name__ == "__main__":
    main()
