# Per-head market thresholds by frequency matching on validation games (teacher-forced leader states):
# tau_h = the P(order) quantile at which the model orders exactly as often as the leader did.
import sys, glob, json, numpy as np, torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il"); torch.set_num_threads(2)
import features as F
from model import Net, split_heads
ckpt, out = sys.argv[1], sys.argv[2]
c = torch.load(ckpt, map_location="cpu", weights_only=False); net = Net(width=c["width"]); net.load_state_dict(c["state"]); net.eval()
files = sorted(glob.glob("/Users/asad.ali/kaggriculture/il/data/dsm/*.npz"))
P = [[] for _ in F.MARKET_HEADS]; Y = [[] for _ in F.MARKET_HEADS]
for i in c["files_val"]:
    z = np.load(files[i])
    planes = torch.from_numpy(z["planes"].astype(np.float32) / 255.0); g = torch.from_numpy(z["glob"].astype(np.float32))
    pos = torch.from_numpy(z["pos"].astype(np.int64)); prev = torch.from_numpy(z["prev_op"].astype(np.int64))
    oh = torch.nn.functional.one_hot(prev.clamp(min=0), F.N_OPS).float() * (prev >= 0).float()[..., None]
    uin = torch.cat([torch.from_numpy(z["ufeat"].astype(np.float32)), oh], 2)
    with torch.no_grad(): _, _, lm, _ = net(planes, g, pos, uin)
    for j, h in enumerate(split_heads(lm)):
        P[j].append(1 - torch.softmax(h, -1)[:, 0].numpy()); Y[j].append(z["market"][:, j] > 0)
taus = {}
for j, (name, _) in enumerate(F.MARKET_HEADS):
    p = np.concatenate(P[j]); y = np.concatenate(Y[j]); rate = y.mean()
    tau = float(np.quantile(p, 1 - rate)) if rate > 0 else 0.99
    tau = float(np.clip(tau, 0.05, 0.9)); taus[name] = tau
    fire = p > tau
    print(f"{name:22s} DSM rate {rate:.4f} tau {tau:.3f} | recall {(fire & y).sum()/max(1,y.sum()):.2f} precision {(fire & y).sum()/max(1,fire.sum()):.2f}")
json.dump(taus, open(out, "w"), indent=1)
