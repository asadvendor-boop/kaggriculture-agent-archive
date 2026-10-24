"""Per-iteration table of a trainer run (stats.jsonl + log.txt + eval.jsonl) and its checkpoints.
usage: python run_report.py <run_dir> [--json out.json]"""
import os, sys, json, re, glob


def main():
    d = sys.argv[1]
    st = [json.loads(l) for l in open(os.path.join(d, "stats.jsonl"))] if os.path.exists(os.path.join(d, "stats.jsonl")) else []
    ev = [json.loads(l) for l in open(os.path.join(d, "eval.jsonl"))] if os.path.exists(os.path.join(d, "eval.jsonl")) else []
    log = [l.rstrip("\n") for l in open(os.path.join(d, "log.txt"))] if os.path.exists(os.path.join(d, "log.txt")) else []
    games = {}                                                   # it -> "IL@3 +3,454 W75%(32) ..." (training games by opponent)
    for l in log:
        m = re.match(r"^it (\d+) .*?\) \| (.*?) \| loss", l)
        if m:
            games[int(m.group(1))] = m.group(2)
    evals = {}
    for l in log:
        m = re.match(r"^\[eval it (\d+)\] \d+s \| (.*)$", l)
        if m:
            evals[int(m.group(1))] = m.group(2)
    print(f"{'it':>3s} {'roll s':>6s} {'upd s':>6s} {'KL units':>9s} {'KL mkt':>8s} {'clip':>6s} {'ent':>6s} {'v loss':>7s} {'EV':>6s} {'fin':>3s}  training games (margin, win rate, n)")
    rows = []
    for s in st:
        it = s["it"]
        row = dict(it=it, t_roll=round(s["t_roll"], 1), t_upd=round(s["t_upd"], 1), klu=s["klu"], klm=s["klm"], clipfrac=s["clipfrac"],
                   ent=s["ent"], l_v=s["l_v"], ev=s.get("ev_mc"), finite=s["finite"], games=games.get(it, ""))
        rows.append(row)
        evs = f"{row['ev']:6.3f}" if row["ev"] is not None else "     -"
        print(f"{it:3d} {row['t_roll']:6.0f} {row['t_upd']:6.0f} {row['klu']:9.5f} {row['klm']:8.5f} {row['clipfrac']:6.3f} {row['ent']:6.3f} "
              f"{row['l_v']:7.4f} {evs} {int(row['finite']):3d}  {row['games']}")
    if evals:
        print("\ngreedy evaluation (fp32, fixed worlds; paired vs iteration 0):")
        for it in sorted(evals):
            print(f"  it {it:3d}: {evals[it]}")
    cks = sorted(glob.glob(os.path.join(d, "pt", "*.pt"))) + [p for p in (os.path.join(d, "best.pt"), os.path.join(d, "final.pt")) if os.path.exists(p)]
    if cks:
        print("\ncheckpoints (ppo_live .pt):")
        for p in cks:
            print(f"  {p}  ({os.path.getsize(p) / 1e6:.0f} MB)")
    if os.path.exists(os.path.join(d, "DONE")):
        print(f"\nDONE: {open(os.path.join(d, 'DONE')).read().strip()}")
    if "--json" in sys.argv:
        json.dump(dict(iterations=rows, evals=evals, eval_json=ev, checkpoints=cks), open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
