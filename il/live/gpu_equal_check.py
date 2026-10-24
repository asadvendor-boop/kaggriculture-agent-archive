import sys, time, os, json
t0 = time.time()
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il/live")
import torch
import live_rollout as LR
N = int(os.environ.get("NG", "50"))
W = json.load(open("/Users/asad.ali/kaggriculture/il/live/top_worlds_fresh3_300.json"))[:N]
job = dict(weights="/Users/asad.ali/kaggriculture/il/ckpt/ilsp_v13_p3.pt", team=10, tau=0.5, T=0.2, ks=10.0, greedy=True, seed=0,
           games=[(0, w["seat"], "TAPE:" + w["path"]) for w in W])
print("device", os.environ.get("ROLLOUT_DEVICE"), "LOCKSTEP", os.environ.get("LOCKSTEP"), "games", N, "cudnn tf32", torch.backends.cudnn.allow_tf32, flush=True)
t = time.time(); res = LR.run_job(job); dt = time.time() - t
base = {(r["ep"], r["seat"]): r["margin"] for r in json.load(open("/Users/asad.ali/kaggriculture/il/logs/top_fresh3_ilsp_v13_p3_t10_soh.json"))}
by = {(r["opp"][5:], r["seat"]): r["margin"] for r in res}
same = diff = 0; dd = []
for w in W:
    m = by[(w["path"], w["seat"])]; b = base[(w["ep"], w["seat"])]
    if abs(m - b) < 1: same += 1
    else: diff += 1; dd.append(round(m - b))
print(f"GPU TIME {dt:.1f} s for {N} games ({dt / N:.1f} s/game); identical to the CPU baseline in {same}/{N}, different {diff}: {dd[:12]}", flush=True)
