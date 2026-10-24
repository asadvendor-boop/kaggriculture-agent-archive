"""PPO fine-tuning of the imitation policy in the C++ engine.

Worlds: a top leader's recorded game; the leader's seat plays its recorded actions (tape), world pinned (shops, weed
stream); our policy plays the other seat.  Reward per step = change of (our money - rival money) / 2e4, so the
undiscounted return is the final margin (gamma = 1, GAE lambda).
Policy (stochastic, matches greedy decoding at its mode): unit ops ~ softmax(masked logits / T); PICKUP/PLACE qty ~
softmax(qty logits / T); market head h orders with p = sigmoid(K (logit(P_h(order)) - logit(tau))), class = argmax of
the non-zero classes.  Loss = per-decision clipped surrogate (unit ops, qty, market fires, same step advantage)
+ value loss + beta * KL(pi || pi_IL) (IL policy frozen).  Greedy evaluation on held-out leader games every EVAL its.

Usage: ppo.py <init_ckpt> <out_ckpt> <iterations>   env: K (envs/iter, 48), TEAM (style id), LR (2e-5), BETA (0.02),
       T (0.5), KSHARP (4), TAU (0.35), EVAL (5), EVAL_N (48)
"""
import sys, os, json, glob, time, random
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
from harness import PASS
import features as F
from model import Net, split_heads, HEAD_SIZES
from agent import legal_mask
from rollout import World, sharpen, MARKET_OFF

dev = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "mps"))   # Mac: mps (unchanged); server: cuda
K = int(os.environ.get("K", "48")); TEAM = int(os.environ.get("TEAM", "0")); LR = float(os.environ.get("LR", "2e-5"))
BETA = float(os.environ.get("BETA", "0.02")); T = float(os.environ.get("T", "0.5")); KS = float(os.environ.get("KSHARP", "4"))
TAU = float(os.environ.get("TAU", "0.35")); EVAL = int(os.environ.get("EVAL", "5")); EVAL_N = int(os.environ.get("EVAL_N", "48"))
LAM, CLIP, EPOCHS, MB = 0.95, 0.2, int(os.environ.get("EPOCHS", "3")), int(os.environ.get("MB", "2048"))
ACCEPT = int(os.environ.get("ACCEPT", "0"))     # accept an update only if greedy play on a fixed acceptance set does not get worse
VWARM = int(os.environ.get("VWARM", "0"))       # first VWARM iterations: value head only (policy frozen)
LOGIT_TAU = float(np.log(TAU / (1 - TAU)))
R = "/Users/asad.ali/kaggriculture"
LEADERS = {"Boey", "M & M & P & Q", "nah id win", "DECEM", "Unknown Mother-Goose", "Vadim Vasilenko", "吃白饭的大肥鱼",
           "THIRD FARM CLUB", "Kaggledew Valley 🏆", "mtmr_s1", "DSM"}


def world_jobs():
    """(tape path, our seat) for every tape where the other seat is a leader; split train / eval by ep % 37."""
    tr, ev = [], []
    for d_ in ("data/tapes_pull", "data/tapes_0922", "data/tapes_archive"):
        fn = f"{R}/{d_}/index.json"
        if not os.path.exists(fn):
            continue
        for ep, m in json.load(open(fn)).items():
            teams = m.get("teams") or [None, None]; rw = m.get("rewards") or [None, None]
            if rw[0] is None or rw[1] is None or not os.path.exists(f"{R}/{d_}/{ep}.json.gz"):
                continue
            for ls in (0, 1):
                if teams[ls] in LEADERS:
                    (ev if int(ep) % 37 == 0 else tr).append((f"{R}/{d_}/{ep}.json.gz", 1 - ls))
    return tr, ev


def load(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"), n_aux=c.get("n_aux", 0), ar=c.get("ar", 0), menu=c.get("menu", 0)); net.load_state_dict(c["state"])
    return net.to(dev), c


def forward(net, planes, glob, pos, uin):
    tt = torch.full((planes.shape[0],), TEAM, dtype=torch.long, device=dev) if net.n_teams else None
    return net(planes, glob, pos, uin, tt)


@torch.no_grad()
def collect(net, jobs, rng, greedy=False):
    W = [World(p, s) for p, s in jobs]
    n_env = len(W)
    buf = [dict(planes=[], glob=[], pos=[], uin=[], umask=[], uact=[], qact=[], mfire=[], mclass=[], mdiff=[]) for _ in W]
    for t in range(720):
        live = [i for i in range(n_env) if not W[i].done]
        if not live:
            break
        obs = [W[i].g.observe(W[i].seat) for i in live]
        enc = [F.encode_obs(o, W[i].prev) for o, i in zip(obs, live)]
        n = len(live)
        pos = np.zeros((n, F.MAX_UNITS, 2), np.int64); uin = np.zeros((n, F.MAX_UNITS, F.U), np.float32); nu = []
        for j, e in enumerate(enc):
            m = len(e[2]); nu.append(m); pos[j, :m] = e[2]; uin[j, :m] = e[3]
        pl = torch.from_numpy(np.stack([e[0] for e in enc]).astype(np.float32) / 255.0).to(dev)
        gl = torch.from_numpy(np.stack([e[1] for e in enc])).to(dev)
        lop, lq, lm, _ = forward(net, pl, gl, torch.from_numpy(pos).to(dev), torch.from_numpy(uin).to(dev))
        lop = lop.float().cpu().numpy(); lq = lq.float().cpu().numpy(); lm = lm.float().cpu().numpy()
        for j, i in enumerate(live):
            w = W[i]; o = obs[j]
            shed = o["private"].get("shed", {}) or {}; invs = o["private"].get("inventories") or [{}]
            umask = np.zeros((F.MAX_UNITS, F.N_OPS), bool); ua = np.full(F.MAX_UNITS, -1, np.int16); qa = np.full(F.MAX_UNITS, -1, np.int16)
            cmds, ops = [], []
            for k in range(nu[j]):
                allowed = legal_mask(o, k, pos[j, k]); umask[k] = allowed
                z = np.where(allowed, lop[j, k].astype(np.float64), -1e9)
                if greedy:
                    oc = int(np.argmax(z))
                else:
                    pz = np.exp((z - z.max()) / T); pz /= pz.sum(); oc = int(rng.choice(len(pz), p=pz))
                name = F.OPS[oc]; qc = 0
                if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                    zq = lq[j, k].astype(np.float64)
                    if greedy:
                        qc = int(np.argmax(zq))
                    else:
                        pq = np.exp((zq - zq.max()) / T); pq /= pq.sum(); qc = int(rng.choice(len(pq), p=pq))
                    qa[k] = qc
                    it = name.split("_", 1)[1]
                    have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                    qc = max(0, min(qc, int(have) - 1))
                cmds.append(F.unit_command(oc, qc)); ops.append(oc); ua[k] = oc
            w.prev = ops
            classes, fires = [], np.zeros(F.N_HEADS, np.int8)
            for h in range(F.N_HEADS):
                zz = lm[j, MARKET_OFF[h]:MARKET_OFF[h + 1]].astype(np.float64)
                p = np.exp(zz - zz.max()); p /= p.sum()
                fire = (1.0 - p[0] > TAU) if greedy else (rng.random_sample() < sharpen(1.0 - p[0], TAU, KS))
                fires[h] = int(fire)
                classes.append(int(1 + np.argmax(p[1:])) if fire else 0)
            a = {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": F.decode_market(classes, o)}
            me = o["farms"][w.seat]["money"] - o["farms"][w.th]["money"]
            if not greedy:
                b = buf[i]
                b["planes"].append(enc[j][0]); b["glob"].append(enc[j][1]); b["pos"].append(pos[j].astype(np.int8))
                b["uin"].append(uin[j].astype(np.float16)); b["umask"].append(umask); b["uact"].append(ua); b["qact"].append(qa)
                b["mfire"].append(fires); b["mclass"].append(np.array(classes, np.int16)); b["mdiff"].append(me)
            now = [None, None]; now[w.seat] = a; now[w.th] = w.opp_action(t)
            w.g.step(now[0], now[1])
            if w.g.done:
                w.done = True
    res = []
    for i, w in enumerate(W):
        final = w.g.reward(w.seat) - w.g.reward(w.th)
        r = dict(bank=w.g.reward(w.seat), opp=w.g.reward(w.th), margin=final)
        if not greedy:
            b = buf[i]
            md = np.array(b["mdiff"] + [final], np.float64)
            r["traj"] = {k: np.stack(v) for k, v in b.items() if k != "mdiff"}
            r["traj"]["rew"] = ((md[1:] - md[:-1]) / 2e4).astype(np.float32)
        res.append(r)
    return res


def logp_terms(net, D, idx):
    """Per-decision log-probs (and probabilities for KL) of the recorded actions under net (temperature T)."""
    planes = D["planes"][idx].float() / 255.0
    lop, lq, lm, v = forward(net, planes, D["glob"][idx], D["pos"][idx].long(), D["uin"][idx].float())
    umask = D["umask"][idx]; ua = D["uact"][idx].long(); qa = D["qact"][idx].long(); fire = D["mfire"][idx].float()
    zu = torch.where(umask, lop / T, torch.full_like(lop, -1e9))
    lpu_all = torch.log_softmax(zu, -1)
    valid = ua >= 0
    lpu = torch.gather(lpu_all, 2, ua.clamp(min=0)[..., None]).squeeze(-1)
    lpq_all = torch.log_softmax(lq / T, -1)
    qvalid = qa >= 0
    lpq = torch.gather(lpq_all, 2, qa.clamp(min=0)[..., None]).squeeze(-1)
    heads = split_heads(lm, getattr(net, "head_sizes", None))
    pnz = torch.stack([1 - torch.softmax(h, -1)[:, 0] for h in heads], 1).clamp(1e-6, 1 - 1e-6)
    pf = torch.sigmoid(KS * (torch.log(pnz / (1 - pnz)) - LOGIT_TAU)).clamp(1e-6, 1 - 1e-6)
    lpm = fire * torch.log(pf) + (1 - fire) * torch.log(1 - pf)
    return dict(lpu=lpu, valid=valid, lpu_all=lpu_all, umask=umask, lpq=lpq, qvalid=qvalid, lpq_all=lpq_all,
                lpm=lpm, pf=pf, v=v)


def main():
    init, out, ITERS = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rng = np.random.RandomState(0); random.seed(0)
    net, c = load(init)
    ref, _ = load(init); ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    vnet, _ = load(init); vopt = torch.optim.Adam(vnet.parameters(), lr=float(os.environ.get('VLR', '1e-4')))
    tr, _ = world_jobs()
    ev = [tuple(x) for x in json.load(open(f'{R}/il/eval_worlds.json'))][:EVAL_N]
    evs = set(ev); tr = [j for j in tr if j not in evs]
    acc_set = []
    if ACCEPT:
        acc_set = random.Random(7).sample(sorted(tr), 48); accs = set(acc_set); tr = [j for j in tr if j not in accs]
    print(f"worlds: train {len(tr)}, eval {len(ev)}; K {K}, team {TEAM}, lr {LR}, beta {BETA}, T {T}", flush=True)

    def evaluate(tag):
        net.eval()
        res = []
        for b in range(0, len(ev), 48):
            res += collect(net, ev[b:b + 48], rng, greedy=True)
        m = np.array([r["margin"] for r in res]); bank = np.array([r["bank"] for r in res])
        print(f"[eval {tag}] {len(res)} games vs leader tapes: wins {(m > 0).sum()}/{len(m)}, mean margin {m.mean():+,.0f}, mean bank {bank.mean():,.0f}", flush=True)
        net.train()
        return (m > 0).mean() + m.mean() / 1e6

    best = evaluate("init")
    import copy
    def acc_score():
        net.eval(); res = []
        for b in range(0, len(acc_set), 48):
            res += collect(net, acc_set[b:b + 48], rng, greedy=True)
        net.train(); m = np.array([r['margin'] for r in res])
        return float(m.mean()), int((m > 0).sum())
    if ACCEPT:
        acc_best, acc_w = acc_score(); acc_state = copy.deepcopy(net.state_dict()); lr_now = LR
        print(f'[accept set] init margin {acc_best:+,.0f} wins {acc_w}/48', flush=True)
    torch.save({"state": net.state_dict(), "width": c["width"], "n_teams": c.get("n_teams", 0), "team": TEAM, "it": 0, "score": best}, out)
    t0 = time.time()
    for it in range(1, ITERS + 1):
        jobs = random.sample(tr, K)
        net.eval()
        res = collect(net, jobs, rng)
        net.train()
        cat = lambda k: np.concatenate([r["traj"][k] for r in res])
        D = {"planes": torch.from_numpy(cat("planes")).to(dev), "glob": torch.from_numpy(cat("glob")).to(dev),
             "pos": torch.from_numpy(cat("pos")).to(dev), "uin": torch.from_numpy(cat("uin")).to(dev),
             "umask": torch.from_numpy(cat("umask")).to(dev), "uact": torch.from_numpy(cat("uact")).to(dev),
             "qact": torch.from_numpy(cat("qact")).to(dev), "mfire": torch.from_numpy(cat("mfire")).to(dev)}
        N = D["planes"].shape[0]
        # old log-probs, values, reference probabilities (no grad)
        with torch.no_grad():
            old = {k: [] for k in ("lpu", "lpq", "lpm", "v")}; refp = {k: [] for k in ("lpu_all", "lpq_all", "pf")}
            for b in range(0, N, MB):
                idx = torch.arange(b, min(N, b + MB), device=dev)
                o_ = logp_terms(net, D, idx)
                o_["v"] = forward(vnet, D["planes"][idx].float() / 255.0, D["glob"][idx], D["pos"][idx].long(), D["uin"][idx].float())[3]
                for k in old: old[k].append(o_[k])
                r_ = logp_terms(ref, D, idx)
                for k in refp: refp[k].append(r_[k])
            old = {k: torch.cat(v) for k, v in old.items()}; refp = {k: torch.cat(v) for k, v in refp.items()}
        # GAE per trajectory (gamma = 1)
        vals = old["v"].cpu().numpy(); adv = np.zeros(N, np.float32); ret = np.zeros(N, np.float32); s = 0
        for r in res:
            rew = r["traj"]["rew"]; L = len(rew); v = vals[s:s + L]; last = 0.0
            for t in range(L - 1, -1, -1):
                nv = v[t + 1] if t + 1 < L else 0.0
                delta = rew[t] + nv - v[t]
                last = delta + LAM * last
                adv[s + t] = last
            ret[s:s + L] = adv[s:s + L] + v; s += L
        A = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8)).to(dev); RET = torch.from_numpy(ret).to(dev)
        stats = np.zeros(5); nb = 0
        for ep in range(EPOCHS):
            perm = torch.randperm(N, device=dev)
            for b in range(0, N, MB):
                idx = perm[b:b + MB]
                cur = logp_terms(net, D, idx)
                a = A[idx]
                ru = torch.exp(cur["lpu"] - old["lpu"][idx]); vm = cur["valid"]
                su = torch.min(ru * a[:, None], torch.clamp(ru, 1 - CLIP, 1 + CLIP) * a[:, None])
                l_u = -(su[vm]).mean()
                rq = torch.exp(cur["lpq"] - old["lpq"][idx]); qm = cur["qvalid"]
                sq = torch.min(rq * a[:, None], torch.clamp(rq, 1 - CLIP, 1 + CLIP) * a[:, None])
                l_q = -(sq[qm]).mean() if qm.any() else torch.zeros((), device=dev)
                rm = torch.exp(cur["lpm"] - old["lpm"][idx])
                sm = torch.min(rm * a[:, None], torch.clamp(rm, 1 - CLIP, 1 + CLIP) * a[:, None])
                l_m = -sm.mean()
                # KL(pi || pi_ref) on unit ops (valid units) and market fires
                pu = torch.exp(cur["lpu_all"])
                klu = (pu * (cur["lpu_all"] - refp["lpu_all"][idx])).masked_fill(~cur["umask"], 0).sum(-1)[vm].mean()
                pf, pr = cur["pf"], refp["pf"][idx]
                klm = (pf * torch.log(pf / pr) + (1 - pf) * torch.log((1 - pf) / (1 - pr))).mean()
                vcur = forward(vnet, D["planes"][idx].float() / 255.0, D["glob"][idx], D["pos"][idx].long(), D["uin"][idx].float())[3]
                l_v = Fn.mse_loss(vcur, RET[idx])
                vopt.zero_grad(); l_v.backward(); torch.nn.utils.clip_grad_norm_(vnet.parameters(), 1.0); vopt.step()
                if it > VWARM:
                    loss = l_u + 0.5 * l_q + l_m + BETA * (klu + klm)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                stats += [l_u.item(), l_m.item(), klu.item(), klm.item(), l_v.item()]; nb += 1
        m = np.array([r["margin"] for r in res])
        print(f"it {it} {time.time()-t0:.0f}s | sampled games: wins {(m > 0).sum()}/{K}, margin {m.mean():+,.0f} | "
              f"loss u/m {stats[0]/nb:+.4f}/{stats[1]/nb:+.4f} kl u/m {stats[2]/nb:.4f}/{stats[3]/nb:.4f} v {stats[4]/nb:.4f}", flush=True)
        del D, old, refp
        if ACCEPT and it > VWARM:
            sc_m, sc_w = acc_score()
            if sc_m >= acc_best:
                acc_best = sc_m; acc_state = copy.deepcopy(net.state_dict())
                print(f'  accepted: acceptance margin {sc_m:+,.0f} wins {sc_w}/48', flush=True)
            else:
                net.load_state_dict(acc_state); lr_now = max(2e-6, lr_now * 0.7)
                for gp in opt.param_groups: gp['lr'] = lr_now
                opt.state.clear()
                print(f'  rejected: acceptance margin {sc_m:+,.0f} < {acc_best:+,.0f}; reverted, lr -> {lr_now:.1e}', flush=True)
        if it % EVAL == 0 or it == ITERS:
            sc = evaluate(f"it {it}")
            torch.save({"state": net.state_dict(), "width": c["width"], "n_teams": c.get("n_teams", 0), "team": TEAM, "it": it, "score": sc},
                       out.replace(".pt", "_last.pt"))
            if sc > best:
                best = sc
                torch.save({"state": net.state_dict(), "width": c["width"], "n_teams": c.get("n_teams", 0), "team": TEAM, "it": it, "score": sc}, out)
                print(f"  new best -> {out}", flush=True)


if __name__ == "__main__":
    main()
