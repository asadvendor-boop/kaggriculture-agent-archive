"""Print the start state summary and actions of statefuzz failures (usage: inspect_fail.py <json> [family])."""
import json, sys
d = json.load(open(sys.argv[1]))
fam = sys.argv[2] if len(sys.argv) > 2 else None
for f in d["fails"]:
    if fam and f["family"] != fam:
        continue
    print("FAIL", f["cfg"], f["family"], "t", f["t"], "k", f["step_in_scenario"], "seat", f["seat"], "seed", f["seed"], f["diff"][:6])
    if not f.get("start_obs"):
        continue
    o0, o1 = f["start_obs"]
    for p, o in enumerate((o0, o1)):
        fm = o["farms"][p]
        print("  seat", p, "money", fm["money"], "units", 1 + len(fm["hands"]), "seeds", o["private"]["seeds"],
              "shed", {k: v for k, v in o["private"]["shed"].items() if v})
    print("  market inv", o0["market"]["inventory"])
    for k, (a0, a1) in enumerate(f["actions"]):
        for s, a in ((0, a0), (1, a1)):
            print("  step", k, "seat", s, "market", a.get("market") if isinstance(a, dict) else a)
