# Where does the BC agent lose to the leader it imitates?  Per game: hires, land, animals, seeds, production, sales,
# plant deaths, animal escapes and failed unit actions, BC (in the leader's seat, world pinned) vs the recorded leader.
import sys, json, gzip, collections, os, glob
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import torch; torch.set_num_threads(1)
from harness import kagsim_trace, PASS, ROOT
import topruler
from agent import BCAgent

def run(path, seat, agent=None):
    d = json.load(gzip.open(path, "rt")); th = 1 - seat
    shops, e0, exact = topruler.recorded_world(path, d)
    A = d["actions"]
    g = kagsim_trace.Game(d["seed"], 720, True); g.set_forced_shops(shops)
    if th == 1: g.set_weed_prefix(1, e0)
    S = collections.Counter(); prod = collections.Counter(); rev = collections.Counter(); buy = collections.Counter()
    for t in range(720):
        o = g.observe(seat)
        a = agent(o) if agent else (A[seat][t] if isinstance(A[seat][t], dict) else dict(PASS))
        now = [None, None]; now[seat] = a; now[th] = A[th][t] if isinstance(A[th][t], dict) else dict(PASS)
        g.step(now[0], now[1]); ts = g.trace_step(g.trace_len() - 1)
        for r in ts["unit_receipts"]:
            if r["seat"] != seat: continue
            S["unit_steps"] += 1
            if not r["executed"] and r["op"] != "PASS": S["failed"] += 1
            if r["op"] == "PASS": S["pass"] += 1
            if r["op"] == "HARVEST" and r["executed"]: prod[r.get("eff_item")] += r.get("eff_n", 0)
        for rc in ts.get("receipts", []):
            if rc["seat"] != seat or not rc["executed"]: continue
            dm = (rc.get("money_after") or 0) - (rc.get("money_before") or 0)
            if rc["op"] == "SELL": rev[rc["item"]] += dm
            elif rc["op"] == "HIRE": S["hire_cost"] += -dm; S["hires"] += rc["executed"]
            elif rc["op"] == "BUY_LAND": S["land"] += 1; S["land_day_sum"] += t // 24
            else: buy[f"{rc['op'][4:]}:{rc.get('item')}"] += rc["executed"]
        for e in ts.get("escapes", []) or []:
            if (e.get("seat") if isinstance(e, dict) else None) == seat: S["escapes"] += 1
        if t % 24 == 23:
            f = g.observe(seat)["farms"][seat]
            S["weeds_eod"] += sum(1 for row in f["tiles"] for tl in row if isinstance(tl, dict) and tl.get("kind") == "WEED")
        if g.done: break
    return g.reward(seat), g.reward(th), S, prod, rev, buy

ckpt, n, tau = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
c = torch.load(ckpt, map_location="cpu"); files = sorted(glob.glob(f"{ROOT}/il/data/dsm/*.npz"))
agg = {k: collections.Counter() for k in ("bcS", "dsS", "bcP", "dsP", "bcR", "dsR", "bcB", "dsB")}
for i in c["files_val"][:n]:
    ep, seat = os.path.basename(files[i]).split(".")[0].split("_"); seat = int(seat)
    p = next(f"{ROOT}/{d_}/{ep}.json.gz" for d_ in ("data/tapes_0922", "data/tapes_archive") if os.path.exists(f"{ROOT}/{d_}/{ep}.json.gz"))
    b = run(p, seat, BCAgent(ckpt, mk_tau=tau)); r = run(p, seat)
    for k, x in zip(("bcS", "bcP", "bcR", "bcB"), b[2:]): agg[k].update(x)
    for k, x in zip(("dsS", "dsP", "dsR", "dsB"), r[2:]): agg[k].update(x)
    print(f"{ep} BC {b[0]:,.0f} vs DSM {r[0]:,.0f}", flush=True)
N = n
def row(name, a, b): print(f"  {name:28s} BC {a/N:>10,.1f}   DSM {b/N:>10,.1f}")
print("per game averages:")
for k in ("hires", "hire_cost", "land", "failed", "pass", "unit_steps", "escapes", "weeds_eod"): row(k, agg["bcS"][k], agg["dsS"][k])
print("  land day (avg)", round(agg["bcS"]["land_day_sum"] / max(1, agg["bcS"]["land"]), 1), "vs", round(agg["dsS"]["land_day_sum"] / max(1, agg["dsS"]["land"]), 1))
for it in sorted(set(agg["bcP"]) | set(agg["dsP"])): row("produced " + str(it), agg["bcP"][it], agg["dsP"][it])
for it in sorted(set(agg["bcR"]) | set(agg["dsR"])): row("revenue " + it, agg["bcR"][it], agg["dsR"][it])
for it in sorted(set(agg["bcB"]) | set(agg["dsB"])): row("bought " + it, agg["bcB"][it], agg["dsB"][it])
