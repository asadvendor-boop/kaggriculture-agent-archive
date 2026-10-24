"""Dry run: the fine-tuned checkpoint plays one full game through il/live_rollout.run_job (the rollout path of every
benchmark: Net(width, blocks, n_teams, g_in, n_aux, ar, menu) from the ckpt dict, v6 features, rule layer from env =
the ST2 package's rules) against the public V94 bot; the base v13_p3 plays the same world for comparison."""
import sys, os, time
os.environ.update(SE_GUARD="1", SELL_ORDER="held", STRADDLE="2", MIDNIGHT="1", FINALFIX="1", HOMING="1")
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools")
from live_rollout import run_job
for ck in sys.argv[1:]:
    t0 = time.time()
    r = run_job(dict(weights=ck, team=10, tau=0.5, T=1.0, ks=1.0, greedy=True, seed=0, games=[(900001, 0, "V94")]))[0]
    print(f"{os.path.basename(ck)}: vs V94 seed 900001 seat 0 bank {r['bank']:,.0f} opp {r['opp_bank']:,.0f} margin {r['margin']:+,.0f} ({time.time()-t0:.0f} s)", flush=True)
