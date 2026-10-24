"""Sep 26: speed of one RL rollout worker (6 lockstep games, sampling like PPO) with the policy forward on the CPU vs the
GPU (ROLLOUT_DEVICE).  Usage: gpu_speed_test.py <cpu|cuda>"""
import sys, os, time, json
os.environ["ROLLOUT_DEVICE"] = sys.argv[1]
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import live_rollout as LR
W = [w for w in json.load(open("/Users/asad.ali/kaggriculture/il/live/ladder_worlds.json")) if w["ep"] % 2 == 0][:6]
job = dict(weights="/Users/asad.ali/kaggriculture/il/ckpt/rl2_it24.pt", team=10, tau=0.5, T=0.2, ks=10.0, greedy=False, seed=7, winb=0.0,
           games=[(w["seed"], w["seat"], "TAPE:" + w["path"]) for w in W])
t0 = time.time(); R = LR.run_job(job); dt = time.time() - t0
print(f"{sys.argv[1]}: 6 games in {dt:.0f}s ({dt/6:.1f}s/game), mean margin {sum(r['margin'] for r in R)/6:+,.0f}", flush=True)
