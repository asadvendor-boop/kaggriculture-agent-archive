"""Self-imitation fine-tuning (policy improvement by selection): per round, play W training worlds greedily (baseline)
and M sampled games each; keep each world's best sample if it beats the greedy margin by DELTA; clone the kept games
(unit ops, qty, market classes) with KL regularisation to the frozen IL policy.  Greedy evaluation on the frozen
eval worlds every EVAL rounds; best checkpoint kept.
Usage: selfimit.py <init_ckpt> <out_ckpt> <rounds>  env: TEAM, TAU, T, KSHARP, W (32), M (3), DELTA (1000), LR (2e-5),
       STEPS (per round, 150), BETA (0.5), EVAL (4), EVAL_N (48)
"""
import sys, os, json, random, time
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import ppo
import features as F
from model import split_heads

W_ = int(os.environ.get("W", "32")); M_ = int(os.environ.get("M", "3")); DELTA = float(os.environ.get("DELTA", "1000"))
LR = float(os.environ.get("LR", "2e-5")); STEPS = int(os.environ.get("STEPS", "150")); BETA = float(os.environ.get("BETA", "0.5"))
EVAL = int(os.environ.get("EVAL", "4")); EVAL_N = int(os.environ.get("EVAL_N", "48")); BS = 1024
dev = ppo.dev


def main():
    init, out, ROUNDS = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rng = np.random.RandomState(1); random.seed(1)
    net, c = ppo.load(init); ref, _ = ppo.load(init); ref.eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    tr, _ = ppo.world_jobs()
    ev = [tuple(x) for x in json.load(open(f"{ppo.R}/il/eval_worlds.json"))][:EVAL_N]
    evs = set(ev); tr = [j for j in tr if j not in evs]
    acc_set = random.Random(7).sample(sorted(tr), 48); accs = set(acc_set); tr = [j for j in tr if j not in accs]
    import copy
    def acc_score():
        net.eval(); res = ppo.collect(net, acc_set, rng, greedy=True); m = np.array([r['margin'] for r in res])
        return float(m.mean()), int((m > 0).sum())
    print(f"self-imitation: train worlds {len(tr)}, eval {len(ev)}, W {W_} M {M_} delta {DELTA} lr {LR} team {ppo.TEAM} tau {ppo.TAU} T {ppo.T}", flush=True)

    def evaluate(tag):
        net.eval(); res = []
        for b in range(0, len(ev), 48):
            res += ppo.collect(net, ev[b:b + 48], rng, greedy=True)
        m = np.array([r["margin"] for r in res]); bank = np.array([r["bank"] for r in res])
        print(f"[eval {tag}] wins {(m > 0).sum()}/{len(m)}, mean margin {m.mean():+,.0f}, mean bank {bank.mean():,.0f}", flush=True)
        return (m > 0).mean() + m.mean() / 1e6

    best = evaluate("init")
    acc_best, acc_w = acc_score(); acc_state = copy.deepcopy(net.state_dict()); lr_now = LR
    print(f'[accept set] init margin {acc_best:+,.0f} wins {acc_w}/48', flush=True)
    torch.save({"state": net.state_dict(), "width": c["width"], "n_teams": c.get("n_teams", 0), "team": ppo.TEAM, "round": 0, "score": best}, out)
    t0 = time.time()
    for rd in range(1, ROUNDS + 1):
        worlds = random.sample(tr, W_)
        net.eval()
        g = ppo.collect(net, worlds, rng, greedy=True)
        samples = ppo.collect(net, [w for w in worlds for _ in range(M_)], rng, greedy=False)
        kept = []; gains = []
        for i in range(W_):
            cand = samples[i * M_:(i + 1) * M_]
            bi = int(np.argmax([r["margin"] for r in cand]))
            gain = cand[bi]["margin"] - g[i]["margin"]
            gains.append(gain)
            if gain > DELTA:
                kept.append(cand[bi]["traj"])
        if kept:
            cat = lambda k: np.concatenate([t[k] for t in kept])
            D = {k: torch.from_numpy(cat(k)).to(dev) for k in ("planes", "glob", "pos", "uin", "umask", "uact", "qact", "mclass")}
            N = D["planes"].shape[0]; net.train(); agg = np.zeros(3)
            for s in range(STEPS):
                idx = torch.randint(0, N, (min(BS, N),), device=dev)
                planes = D["planes"][idx].float() / 255.0
                lop, lq, lm, _ = ppo.forward(net, planes, D["glob"][idx], D["pos"][idx].long(), D["uin"][idx].float())
                with torch.no_grad():
                    rop, rq, rm, _ = ppo.forward(ref, planes, D["glob"][idx], D["pos"][idx].long(), D["uin"][idx].float())
                um = D["umask"][idx]; ua = D["uact"][idx].long(); valid = ua >= 0
                zl = torch.where(um, lop, torch.full_like(lop, -1e9)); zr = torch.where(um, rop, torch.full_like(rop, -1e9))
                l_u = Fn.cross_entropy(zl[valid], ua[valid])
                qa = D["qact"][idx].long(); qv = qa >= 0
                l_q = Fn.cross_entropy(lq[qv], qa[qv]) if qv.any() else lop.sum() * 0
                mc = D["mclass"][idx].long()
                heads = split_heads(lm); rheads = split_heads(rm)
                l_m = sum(Fn.cross_entropy(h, mc[:, j]) for j, h in enumerate(heads)) / len(heads)
                lpn, lpr = torch.log_softmax(zl[valid], -1), torch.log_softmax(zr[valid], -1)
                kl = (lpn.exp() * (lpn - lpr)).sum(-1).mean()
                klm = sum((torch.softmax(h, -1) * (torch.log_softmax(h, -1) - torch.log_softmax(r, -1))).sum(-1).mean()
                          for h, r in zip(heads, rheads)) / len(heads)
                loss = l_u + 0.5 * l_q + l_m + BETA * (kl + klm)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                agg += [l_u.item(), l_m.item(), (kl + klm).item()]
            del D
            agg /= STEPS
            sc_m, sc_w = acc_score()
            if sc_m >= acc_best:
                acc_best = sc_m; acc_state = copy.deepcopy(net.state_dict()); verdict = f'accepted ({sc_m:+,.0f}, {sc_w}/48)'
            else:
                net.load_state_dict(acc_state); lr_now = max(2e-6, lr_now * 0.7)
                for gp in opt.param_groups: gp['lr'] = lr_now
                opt.state.clear(); verdict = f'rejected ({sc_m:+,.0f} < {acc_best:+,.0f}), lr {lr_now:.1e}'
        gm = np.array([r["margin"] for r in g])
        print(f"round {rd} {time.time()-t0:.0f}s | greedy train margin {gm.mean():+,.0f} wins {(gm > 0).sum()}/{W_} | best-sample gain mean {np.mean(gains):+,.0f}, kept {len(kept)}/{W_}"
              + (f" | loss u/m {agg[0]:.3f}/{agg[1]:.3f} kl {agg[2]:.4f} | {verdict}" if kept else ""), flush=True)
        if rd % EVAL == 0 or rd == ROUNDS:
            sc = evaluate(f"round {rd}")
            state = {"state": net.state_dict(), "width": c["width"], "n_teams": c.get("n_teams", 0), "team": ppo.TEAM, "round": rd, "score": sc}
            torch.save(state, out.replace(".pt", "_last.pt"))
            if sc > best:
                best = sc; torch.save(state, out); print(f"  new best -> {out}", flush=True)


if __name__ == "__main__":
    main()
