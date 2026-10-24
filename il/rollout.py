"""Batched rollouts for the imitation / RL agent: K C++ games in lockstep, one batched forward pass per step on the GPU.

Worlds: a leader game (tape) whose opponent plays its recorded actions while our policy plays the leader's seat, with
the world pinned to the recording (shops, weed stream) - the same setup as tools/topruler.run_tape.
Decoding matches agent.BCAgent (legal-masked units; market heads by threshold) when greedy; when sampling, unit ops
are drawn from the masked softmax (temperature) and each market head orders with probability sharpen(P(order)) around
the threshold (so the mode equals the greedy decision).  Returns final banks and, if record=True, the per-step
tensors needed for policy-gradient updates.
"""
import sys, json, gzip, time
import numpy as np
import torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from harness import kagsim_trace, PASS
import topruler
import features as F
from model import Net, split_heads, HEAD_SIZES
from agent import legal_mask

MARKET_OFF = np.cumsum([0] + HEAD_SIZES)


def load_world(path):
    d = json.load(gzip.open(path, "rt"))
    shops, e0, exact = topruler.recorded_world(path, d)
    return d, shops, e0


class World:
    def __init__(self, path, seat):
        self.d, self.shops, self.e0 = load_world(path)
        self.seat, self.th = seat, 1 - seat
        self.A = self.d["actions"]
        self.g = kagsim_trace.Game(self.d["seed"], 720, False); self.g.set_forced_shops(self.shops)
        if self.th == 1:
            self.g.set_weed_prefix(1, self.e0)
        self.prev = None
        self.done = False

    def opp_action(self, t):
        a = self.A[self.th][t] if t < len(self.A[self.th]) else None
        return a if isinstance(a, dict) else dict(PASS)


def sharpen(p, tau, k):
    """Order probability for a market head: logistic in logit(p) - logit(tau) with gain k."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = k * (np.log(p / (1 - p)) - np.log(tau / (1 - tau)))
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def run(net, jobs, dev, tau=0.35, greedy=True, temp=1.0, k_sharp=4.0, rng=None, record=False, team=0):
    """jobs: list of (tape_path, leader_seat).  Returns list of dicts (bank, opp_bank, rec_leader, rec_opp[, traj])."""
    rng = rng or np.random.RandomState(0)
    W = [World(p, s) for p, s in jobs]
    K = len(W)
    traj = [dict(planes=[], glob=[], pos=[], uin=[], nu=[], uact=[], qact=[], mact=[], mfire=[]) for _ in range(K)] if record else None
    for t in range(720):
        live = [i for i in range(K) if not W[i].done]
        if not live:
            break
        obs = [W[i].g.observe(W[i].seat) for i in live]
        enc = [F.encode_obs(o, W[i].prev) for o, i in zip(obs, live)]
        n = len(live)
        planes = torch.from_numpy(np.stack([e[0] for e in enc]).astype(np.float32) / 255.0).to(dev)
        glob = torch.from_numpy(np.stack([e[1] for e in enc])).to(dev)
        pos = np.zeros((n, F.MAX_UNITS, 2), np.int64); uin = np.zeros((n, F.MAX_UNITS, F.U), np.float32); nu = []
        for j, e in enumerate(enc):
            m = len(e[2]); nu.append(m); pos[j, :m] = e[2]; uin[j, :m] = e[3]
        tt = torch.full((n,), int(team), dtype=torch.long, device=dev) if getattr(net, "n_teams", 0) else None
        lop, lq, lm, _ = net(planes, glob, torch.from_numpy(pos).to(dev), torch.from_numpy(uin).to(dev), tt)
        lop = lop.float().cpu().numpy(); lq = lq.float().cpu().numpy(); lm = lm.float().cpu().numpy()
        for j, i in enumerate(live):
            w = W[i]; o = obs[j]
            shed = o["private"].get("shed", {}) or {}; invs = o["private"].get("inventories") or [{}]
            cmds, ops, qs = [], [], []
            for k in range(nu[j]):
                allowed = legal_mask(o, k, pos[j, k])
                z = np.where(allowed, lop[j, k].astype(np.float64), -1e9)
                if greedy:
                    oc = int(np.argmax(z))
                else:
                    pz = np.exp((z - z.max()) / temp); pz /= pz.sum(); oc = int(rng.choice(len(pz), p=pz))
                name = F.OPS[oc]; qc = 0
                if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                    qc = int(np.argmax(lq[j, k])); it = name.split("_", 1)[1]
                    have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                    qc = max(0, min(qc, int(have) - 1))
                cmds.append(F.unit_command(oc, qc)); ops.append(oc); qs.append(qc)
            w.prev = ops
            classes, fires = [], []
            for h in range(F.N_HEADS):
                zz = lm[j, MARKET_OFF[h]:MARKET_OFF[h + 1]].astype(np.float64)
                p = np.exp(zz - zz.max()); p /= p.sum()
                th = tau.get(F.MARKET_HEADS[h][0], tau.get("_default", 0.6)) if isinstance(tau, dict) else tau
                if greedy:
                    fire = 1.0 - p[0] > th
                else:
                    fire = rng.random_sample() < sharpen(1.0 - p[0], th, k_sharp)
                fires.append(int(fire))
                classes.append(int(1 + np.argmax(p[1:])) if fire else 0)
            a = {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": F.decode_market(classes, o)}
            if record:
                tr = traj[i]
                tr["planes"].append(enc[j][0]); tr["glob"].append(enc[j][1]); tr["pos"].append(pos[j].astype(np.int8))
                tr["uin"].append(uin[j].astype(np.float16)); tr["nu"].append(nu[j])
                ua = np.full(F.MAX_UNITS, -1, np.int16); ua[:nu[j]] = ops; tr["uact"].append(ua)
                qa = np.full(F.MAX_UNITS, -1, np.int16); qa[:nu[j]] = qs; tr["qact"].append(qa)
                tr["mact"].append(np.array(classes, np.int16)); tr["mfire"].append(np.array(fires, np.int8))
            now = [None, None]; now[w.seat] = a; now[w.th] = w.opp_action(t)
            w.g.step(now[0], now[1])
            if w.g.done:
                w.done = True
    out = []
    for i, w in enumerate(W):
        rec = w.d["rewards"]
        r = dict(bank=w.g.reward(w.seat), opp=w.g.reward(w.th), rec_leader=rec[w.seat], rec_opp=rec[w.th], path=jobs[i][0], seat=w.seat)
        if record:
            r["traj"] = {k: np.stack(v) if k != "nu" else np.array(v) for k, v in traj[i].items()}
        out.append(r)
    return out


def load_net(ckpt, dev):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"), n_aux=c.get("n_aux", 0), ar=c.get("ar", 0)); net.load_state_dict(c["state"]); net.eval().to(dev)
    return net, c


if __name__ == "__main__":
    import glob, os
    ckpt, n = sys.argv[1], int(sys.argv[2]); tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    dev = torch.device("mps")
    net, c = load_net(ckpt, dev)
    files = sorted(glob.glob("/Users/asad.ali/kaggriculture/il/data/dsm/*.npz"))
    jobs = []
    for i in c["files_val"][:n]:
        ep, seat = os.path.basename(files[i]).split(".")[0].split("_")
        p = next(f"/Users/asad.ali/kaggriculture/{d_}/{ep}.json.gz" for d_ in ("data/tapes_0922", "data/tapes_archive")
                 if os.path.exists(f"/Users/asad.ali/kaggriculture/{d_}/{ep}.json.gz"))
        jobs.append((p, int(seat)))
    t0 = time.time(); res = run(net, jobs, dev, tau=tau)
    b = np.array([r["bank"] for r in res]); L = np.array([r["rec_leader"] for r in res])
    print(f"{len(res)} games in {time.time()-t0:.1f}s: bank {b.mean():,.0f} vs DSM {L.mean():,.0f} (ratio {b.mean()/L.mean():.2f}); "
          f"wins vs taped opponent {sum(r['bank'] > r['opp'] for r in res)}/{len(res)}", flush=True)
