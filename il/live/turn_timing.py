"""Per-turn timing of a package (Sep 26, before a 20M upload): Kaggle gives 720 turns, actTimeout 1 s plus a 60 s overage
pool, on a 1.6 vCPU machine.  Plays full games in kagsim vs an opponent and reports the load time, mean, p50/p90/p99/max
per-turn time and the seconds over 1 s (what would eat the overage).  Run with OMP/OPENBLAS/MKL_NUM_THREADS=1.
Usage: turn_timing.py <package main.py> [opp=V94] [seed ...]"""
import sys, time
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
import harness
from harness import mk
pkg = sys.argv[1]; opp = sys.argv[2] if len(sys.argv) > 2 else "V94"; seeds = [int(x) for x in sys.argv[3:]] or [7]
for seed in seeds:
    for seat in (0, 1):
        t0 = time.perf_counter(); us = mk(pkg)(); load = time.perf_counter() - t0
        g = harness.kagsim_trace.Game(seed, 720, False); ag = [None, None]; ag[seat] = us; ag[1 - seat] = mk(opp)()
        ts = []
        for t in range(720):
            o = (g.observe(0), g.observe(1)); acts = [None, None]
            for s in (0, 1):
                t1 = time.perf_counter()
                acts[s] = ag[s](o[s])
                if s == seat: ts.append(time.perf_counter() - t1)
            g.step(acts[0], acts[1])
            if g.done: break
        q = sorted(ts); n = len(q)
        over = sum(max(0.0, x - 1.0) for x in ts)
        print(f"seed {seed} seat {seat}: load {load:.1f}s | {n} turns mean {sum(q)/n*1000:.0f} ms p50 {q[n//2]*1000:.0f} "
              f"p90 {q[int(n*.9)]*1000:.0f} p99 {q[int(n*.99)]*1000:.0f} max {q[-1]*1000:.0f} ms | first {ts[0]*1000:.0f} ms | "
              f"over-1s total {over:.1f}s | score {g.reward(seat):,.0f} vs {g.reward(1-seat):,.0f}", flush=True)
