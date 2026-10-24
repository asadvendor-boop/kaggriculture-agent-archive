"""Compact status of sprint runs: progress, per-iteration win rates by opponent kind, evaluations, exports.
usage: python kagjax/train/sprint_status.py <run_dir> [<run_dir> ...]"""
import os, sys, json, re


def show(d):
    name = os.path.basename(d.rstrip("/"))
    if not os.path.isdir(d):
        print(f"== {name}: not started"); return
    log = open(os.path.join(d, "log.txt")).read().splitlines() if os.path.exists(os.path.join(d, "log.txt")) else []
    its = [l for l in log if re.match(r"^it \d+ ", l)]
    st = [json.loads(l) for l in open(os.path.join(d, "stats.jsonl"))] if os.path.exists(os.path.join(d, "stats.jsonl")) else []
    fin = json.load(open(os.path.join(d, "FINAL.json"))) if os.path.exists(os.path.join(d, "FINAL.json")) else None
    print(f"== {name}: {len(st)} iterations, {st[-1]['samples'] / 1e6:,.0f}M samples" if st else f"== {name}: starting",
          "| FINISHED " + fin["reason"] if fin else "")
    if its:
        m = re.search(r"\(([\d.]+)s: rollout ([\d.]+) update ([\d.]+); ([\d,]+) samples/s\)", its[-1])
        if m:
            print(f"   last iteration {float(m.group(1)):.0f}s (rollout {m.group(2)} update {m.group(3)}), {m.group(4)} samples/s", end="")
        m2 = re.search(r"mem peak ([\d.]+) GiB", its[-1])
        print(f", peak {m2.group(1)} GiB" if m2 else "")
    if st:
        s = st[-1]
        print(f"   last: KL u/m {s['klu']:.4f}/{s['klm']:.4f} beta {s.get('beta', 0):.4f} ent {s['ent']:.3f} (bonus x{s.get('ent_coef', 0):.4f}) "
              f"clip {s['clipfrac']:.3f} akl {s['akl']:.4f} EV {s.get('ev_mc', 0):.2f} winb {s.get('winb', 0):.2f} finite {s['finite']:.0f}"
              + (f" dp-diff {s['dp_replica_diff']:.2g}" if "dp_replica_diff" in s else ""))
    gj = os.path.join(d, "games.jsonl")
    if os.path.exists(gj):
        rows = [json.loads(l) for l in open(gj)]
        kinds = sorted({k for r in rows for k in r if k not in ("it", "samples")})
        print("   training W by opponent (iterations):  " + "  ".join(f"{k:>7s}" for k in kinds))
        for r in rows[-12:]:
            print(f"   it {r['it']:4d} {r['samples'] / 1e6:7,.0f}M                  " +
                  "  ".join(f"{100 * r[k]['w']:6.0f}%" if k in r else "      -" for k in kinds))
    ev = [l for l in log if l.startswith("[eval")]
    for l in ev[-8:]:
        print("   " + l[:230])
    idx = os.path.join(d, "pt", "index.json")
    if os.path.exists(idx):
        e = json.load(open(idx))
        print("   exports: " + ", ".join(f"it{x['it']} ({x['samples'] / 1e6:,.0f}M{', kept' if x['keep'] else ''})" for x in e[-8:]))
    for l in [l for l in log if l.startswith(("[STOP]", "[stop]", "[dp]"))][-3:]:
        print("   " + l[:200])


if __name__ == "__main__":
    for d in sys.argv[1:]:
        show(d)
