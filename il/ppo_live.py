"""PPO against LIVE opponents on fresh worlds (scripted bots + frozen snapshots of our own policy), rollouts in a pool
of CPU worker processes (live_rollout.run_job), PPO update on the GPU (same loss as ppo.py: near-greedy sampling,
per-decision clipped surrogate, KL to the frozen IL policy, separate value network).

Usage: ppo_live.py <init_ckpt> <out_ckpt> <iterations>
Env: TEAM, TAU, T, KSHARP, LR, BETA, EPOCHS (as ppo.py); NW (workers, 8), GPW (games per worker, 6),
     OPPS (comma list, name=weight; SELF = frozen snapshots of this run), EVAL (iterations between evaluations, 5),
     EVAL_N (games per eval opponent, 100), EVAL_OPPS (comma list), REF (KL anchor checkpoint, default init).
LOSSES (in OPPS): a live game our submission lost (LOSS_SUBS, default 56510358; tapes in data/tapes_neural), replayed
with the world pinned and the opponent's recorded actions; even episode ids train, odd ones are held out and reported
at every evaluation.
LADDER (in OPPS): a live ladder game of any of our submissions (il/live/ladder_worlds.json, from il/live/ladder.py
list), pinned, opponent on tape; even episode ids train, odd ones are held out (EVAL_LADDER of them evaluated).
LEADERS (in OPPS): imitation clone of a top team (LEAD_CKPT, style id drawn from LEAD_TEAMS), a reactive opponent.
PREFIX_P: share of training games that start with a scripted opening: PREFIX_BOT (default HighWool) plays our seat for
PREFIX_DAYS (range, default 2,14) days, then the network takes over (only its own decisions are trained).
REF_LAG=K: "delayed moving teacher" (Orbit Wars 5th place) - every K iterations the KL reference becomes the policy
snapshot from K iterations earlier (the anchor improves with the learner but lags it); 0 = fixed REF.
WINB: terminal win/loss bonus added to the reward (reward unit = $20k of margin), e.g. 0.3.
PFSP=1: opponent kinds we lose to more often are drawn more often (weight x (0.25 + recent loss rate)).
Checkpoints carry the value network ("vstate"); a run started from one skips the value warm-up unless VWARM is set.
Evaluation: greedy on fixed seeds 251000.. vs each EVAL_OPPS opponent, paired against iteration 0.
"""
import os, sys, json, time, random, copy
import numpy as np
import torch
import torch.nn.functional as Fn
import multiprocessing as mp
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import ppo
from live_rollout import run_job

R = "/Users/asad.ali/kaggriculture"
NW = int(os.environ.get("NW", "8")); GPW = int(os.environ.get("GPW", "6"))
EVAL = int(os.environ.get("EVAL", "5")); EVAL_N = int(os.environ.get("EVAL_N", "100"))
PKG = lambda n: f"{R}/package/{n}/main.py"
OPPS_DEF = f"{PKG('STACK1_RACE4')}=3,{PKG('STACK1_RACE')}=1.5,V94=1.5,V54=1,V57=1,SELF=2"
EVAL_OPPS = os.environ.get("EVAL_OPPS", f"{PKG('STACK1_RACE4')},V94").split(",")
dev = ppo.dev
WINB = float(os.environ.get("WINB", "0")); PFSP = int(os.environ.get("PFSP", "0"))
LEAD_CKPT = os.environ.get("LEAD_CKPT", f"{R}/il/ckpt/bc_lead_v3d.pt")
LEAD_TEAMS = [int(x) for x in os.environ.get("LEAD_TEAMS", "0,1,2,3,4,5,6,7,8,9,10").split(",")]
EVAL_LADDER = int(os.environ.get("EVAL_LADDER", "0"))
PRE_P = float(os.environ.get("PREFIX_P", "0")); PRE_BOT = os.environ.get("PREFIX_BOT", "HighWool")
PRE_D = [int(x) for x in os.environ.get("PREFIX_DAYS", "2,14").split(",")]


def top_worlds():
    """[(tape, our_seat, seed, 0)] from the daily top games where the OTHER seat is a current top-10 team (recorded,
    pinned); the ladder benchmark's held-out games come from other sources."""
    LEAD = os.environ.get("TOP_TEAMS", "Boey|M & M & P & Q|nah id win|DECEM|Unknown Mother-Goose|Vadim Vasilenko|吃白饭的大肥鱼|"
                          "THIRD FARM CLUB|Kaggledew Valley 🏆|mtmr_s1").split("|")           # server: the current top 12
    held = set(json.load(open(f"{R}/il/live/fresh_heldout_eps.json"))) if os.path.exists(f"{R}/il/live/fresh_heldout_eps.json") else set()
    out = []
    for d_ in os.environ.get("TOP_DIRS", "data/tapes_daily,data/tapes_pull").split(","):        # server: fresh leader tapes
        for ep, m in json.load(open(f"{R}/{d_}/index.json")).items():
            if int(ep) in held:                                                                # never train on benchmark games
                continue
            t = m.get("teams") or []
            for s in (0, 1):
                if len(t) == 2 and t[1 - s] in LEAD and os.path.exists(f"{R}/{d_}/{ep}.json.gz"):
                    out.append((f"{R}/{d_}/{ep}.json.gz", s, None, 0))
    return out


def ladder_worlds(train=True):
    """[(tape_path, our_seat, seed, recorded margin)] of ladder games; even episode ids = training, odd = held out."""
    fn = f"{R}/il/live/ladder_worlds.json"
    if not os.path.exists(fn):
        return []
    return [(w["path"], w["seat"], w["seed"], w["rec_margin"]) for w in json.load(open(fn)) if (w["ep"] % 2 == 0) == train]


def loss_worlds(train=True):
    """[(tape_path, our_seat, seed)] of live games we lost; even episode ids = training, odd = held out."""
    import gzip
    out = []
    for sub in os.environ.get("LOSS_SUBS", "56510358").split(","):
        fn = f"{R}/data/rows_{sub}.json"
        if not os.path.exists(fn):
            continue
        tdir = os.environ.get("LOSS_DIR", f"{R}/data/tapes_neural")
        try:
            rows = json.load(open(fn))
        except Exception:                      # the poller may be rewriting the file
            rows = _ROWS.get(sub, [])
        _ROWS[sub] = rows
        for r in rows:
            p = f"{tdir}/{r['ep']}.json.gz"
            if r["res"] == "L" and os.path.exists(p) and (r["ep"] % 2 == 0) == train:
                if p not in _LW:
                    d = json.load(gzip.open(p, "rt")); _LW[p] = (p, d["seat"], int(d["seed"]), r["margin"])
                out.append(_LW[p])
    return out


_LW, _ROWS = {}, {}


def parse_opps(s):
    out = []
    for part in s.split(","):
        name, w = part.rsplit("=", 1) if "=" in part else (part, "1")
        out.append((name, float(w)))
    return out


def main():
    init, out, ITERS = sys.argv[1], sys.argv[2], int(sys.argv[3])
    opps = parse_opps(os.environ.get("OPPS", OPPS_DEF))
    names, weights = [o[0] for o in opps], np.array([o[1] for o in opps]); weights = weights / weights.sum()
    rng = np.random.RandomState(5); random.seed(5)
    net, c = ppo.load(init); ref, _ = ppo.load(os.environ.get("REF", init)); ref.eval(); vnet, _ = ppo.load(init)
    vwarm = ppo.VWARM
    if "vstate" in c:
        vnet.load_state_dict(c["vstate"]); vwarm = int(os.environ.get("VWARM", "0"))
        print("value network loaded from init; warm-up", vwarm, flush=True)
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=ppo.LR)
    vopt = torch.optim.Adam(vnet.parameters(), lr=float(os.environ.get("VLR", "1e-4")))
    if "opt" in c and "vopt" in c:                # Adam statistics carried across restarts
        opt.load_state_dict(c["opt"]); vopt.load_state_dict(c["vopt"])
        for g_ in opt.param_groups: g_["lr"] = ppo.LR
        print("optimizer state loaded from init", flush=True)
    snap_dir = f"{R}/il/ckpt/selfplay_{os.path.basename(out).replace('.pt', '')}"; os.makedirs(snap_dir, exist_ok=True)
    cur_w = f"{snap_dir}/current.pt"
    meta = {"width": c["width"], "blocks": c.get("blocks", 5), "n_teams": c.get("n_teams", 0), "team": ppo.TEAM,
            "g_in": c.get("g_in"), "n_aux": c.get("n_aux", 0), "feat": c.get("feat", 5), "menu": c.get("menu", 0), "ar": c.get("ar", 0)}      # v6 checkpoints stay v6

    def save(path, extra=None):
        d = dict(meta); d["state"] = net.state_dict(); d["vstate"] = vnet.state_dict()
        if extra is not None:                       # evaluation checkpoints also carry the optimizer state
            d["opt"] = opt.state_dict(); d["vopt"] = vopt.state_dict()
        d.update(extra or {}); torch.save(d, path)

    save(f"{snap_dir}/snap_0.pt"); snaps = [f"{snap_dir}/snap_0.pt"]
    pool = mp.get_context("spawn").Pool(NW, maxtasksperchild=int(os.environ.get("MAXTASKS", "6")))   # fresh workers: bot leaks

    def rollouts(specs, greedy, seed0):
        jobs = []
        for w in range(NW):
            games = specs[w::NW]
            if games:
                jobs.append(dict(weights=cur_w, team=ppo.TEAM, tau=ppo.TAU, T=ppo.T, ks=ppo.KS, greedy=greedy, seed=seed0 + w, games=games,
                                 winb=0.0 if greedy else WINB))
        res = []
        for r in pool.map(run_job, jobs):
            res += r
        return res

    def evaluate(tag, base=None):
        save(cur_w)
        specs = [(251000 + i, (251000 + i) % 2, o) for o in EVAL_OPPS for i in range(EVAL_N)]
        res = rollouts(specs, True, 777)
        by = {}
        for r in res:
            by.setdefault(r["opp"], {})[r["seed"]] = r["margin"]
        line = f"[eval {tag}]"; score = 0.0; nsc = len(EVAL_OPPS)
        if LAD_TE:
            lr_ = rollouts([(sd, st, "TAPE:" + p_) for p_, st, sd, _ in LAD_TE], True, 779)
            lb = {r["opp"]: r["margin"] for r in lr_}; by["LADDER"] = lb
            mm = np.array([lb[k] for k in sorted(lb)])
            line += f" | ladder held-out: W {(mm > 0).sum()}/{len(mm)} margin {mm.mean():+,.0f}"
            if base is not None and "LADDER" in base:
                d = mm - np.array([base["LADDER"][k] for k in sorted(lb)])
                line += f" (paired {d.mean():+,.0f} se {d.std() / np.sqrt(len(d)):,.0f})"
            score += (mm > 0).mean(); nsc += 1
        for o in EVAL_OPPS:
            m = np.array([by[o][s] for s in sorted(by[o])])
            short = (os.path.basename(o.split(":")[1]).replace(".pt", "") + ":" + o.split(":")[2]) if o.startswith("NN:") else o.split("/")[-2] if "/" in o else o
            line += f" | vs {short}: W {(m > 0).sum()}/{len(m)} margin {m.mean():+,.0f}"
            if base is not None:
                d = m - np.array([base[o][s] for s in sorted(by[o])])
                line += f" (paired {d.mean():+,.0f} se {d.std() / np.sqrt(len(d)):,.0f})"
            score += (m > 0).mean()
        for split in (True, False):
            lw = loss_worlds(split)
            if lw:
                lr_ = rollouts([(sd, st, "TAPE:" + p_) for p_, st, sd, _ in lw], True, 778)
                mm = np.array([r["margin"] for r in lr_]); live = np.array([x[3] for x in lw])
                line += f" | live losses ({'train' if split else 'held-out'}) now W {(mm > 0).sum()}/{len(mm)} margin {mm.mean():+,.0f} (live {live.mean():+,.0f})"
        print(line + f" | score (mean win rate) {score / nsc:.3f}", flush=True)
        return by, score / nsc

    LAD_TR = ladder_worlds(True); te = ladder_worlds(False)
    TOP_W = top_worlds() if "TOP" in names else []
    LAD_TE = te[::max(1, len(te) // EVAL_LADDER)][:EVAL_LADDER] if EVAL_LADDER else []
    print(f"ladder worlds: {len(LAD_TR)} train, {len(LAD_TE)} held-out evaluated; top-game worlds {len(TOP_W)}; WINB {WINB} PFSP {PFSP}", flush=True)
    hist = {n: [] for n in names}

    TOP_SET = {x[0] for x in TOP_W}

    def kind_of(spec):
        if spec.startswith("TAPE:"):              # (Sep 26 fix: TOP tapes from TOP_DIRS such as tapes_top12 were counted as LADDER)
            return "TOP" if spec[5:] in TOP_SET else "LOSSES" if any(spec[5:] == x[0] for x in loss_worlds(True)) else "LADDER"
        if spec.startswith("NN:"):
            return "SELF" if "/selfplay_" in spec else "LEADERS"
        return spec

    base, best = evaluate("init")
    save(out, {"it": 0, "score": best})
    t0 = time.time()
    for it in range(1, int(ITERS) + 1):
        save(cur_w)
        specs = []; ew = weights.copy()
        if PFSP:
            ew = ew * np.array([0.25 + (np.mean(hist[n][-64:]) if hist[n] else 0.5) for n in names]); ew = ew / ew.sum()
        for k in range(NW * GPW):
            o = names[rng.choice(len(names), p=ew)]
            if o in ("LOSSES", "LADDER", "TOP"):
                lw = loss_worlds(True) if o == "LOSSES" else LAD_TR if o == "LADDER" else TOP_W
                if lw:
                    p_, seat_, seed_, _ = lw[rng.randint(len(lw))]
                    specs.append((seed_ if seed_ is not None else 0, seat_, "TAPE:" + p_)); continue
                o = names[0]
            if o == "SELF":
                o = f"NN:{random.choice(snaps)}:{ppo.TEAM}:{ppo.TAU}"
            elif o == "LEADERS":
                o = f"NN:{LEAD_CKPT}:{random.choice(LEAD_TEAMS)}:{ppo.TAU}"
            seed = int(rng.randint(300000, 2_000_000))
            specs.append((seed, seed % 2, o))
        if PRE_P > 0:
            specs = [sp + ((PRE_BOT, 24 * int(rng.randint(PRE_D[0], PRE_D[1] + 1))),) if rng.random_sample() < PRE_P else sp for sp in specs]
        res_all = rollouts(specs, False, 1000 * it)
        res = [r for r in res_all if r.get("traj") is not None]
        net.train()
        cat = lambda k: np.concatenate([r["traj"][k] for r in res])
        D = {"planes": torch.from_numpy(cat("planes")).to(dev), "glob": torch.from_numpy(cat("glob")).to(dev),
             "pos": torch.from_numpy(cat("pos")).to(dev), "uin": torch.from_numpy(cat("uin")).to(dev),
             "umask": torch.from_numpy(cat("umask")).to(dev), "uact": torch.from_numpy(cat("uact")).to(dev),
             "qact": torch.from_numpy(cat("qact")).to(dev), "mfire": torch.from_numpy(cat("mfire")).to(dev)}
        N = D["planes"].shape[0]; MB = ppo.MB
        with torch.no_grad():
            old = {k: [] for k in ("lpu", "lpq", "lpm", "v")}; refp = {k: [] for k in ("lpu_all", "lpq_all", "pf")}
            for b in range(0, N, MB):
                idx = torch.arange(b, min(N, b + MB), device=dev)
                o_ = ppo.logp_terms(net, D, idx)
                o_["v"] = ppo.forward(vnet, D["planes"][idx].float() / 255.0, D["glob"][idx], D["pos"][idx].long(), D["uin"][idx].float())[3]
                for k in old: old[k].append(o_[k])
                r_ = ppo.logp_terms(ref, D, idx)
                for k in refp: refp[k].append(r_[k])
            old = {k: torch.cat(v) for k, v in old.items()}; refp = {k: torch.cat(v) for k, v in refp.items()}
        vals = old["v"].cpu().numpy(); adv = np.zeros(N, np.float32); ret = np.zeros(N, np.float32); s = 0
        for r in res:
            rew = r["traj"]["rew"]; L = len(rew); v = vals[s:s + L]; last = 0.0
            for t in range(L - 1, -1, -1):
                nv = v[t + 1] if t + 1 < L else 0.0
                last = rew[t] + nv - v[t] + ppo.LAM * last
                adv[s + t] = last
            ret[s:s + L] = adv[s:s + L] + v; s += L
        A = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8)).to(dev); RET = torch.from_numpy(ret).to(dev)
        stats = np.zeros(5); nb = 0
        for ep in range(ppo.EPOCHS):
            perm = torch.randperm(N, device=dev)
            for b in range(0, N, MB):
                idx = perm[b:b + MB]; cur = ppo.logp_terms(net, D, idx); a = A[idx]
                ru = torch.exp(cur["lpu"] - old["lpu"][idx]); vm = cur["valid"]
                l_u = -(torch.min(ru * a[:, None], torch.clamp(ru, 1 - ppo.CLIP, 1 + ppo.CLIP) * a[:, None])[vm]).mean()
                rq = torch.exp(cur["lpq"] - old["lpq"][idx]); qm = cur["qvalid"]
                l_q = -(torch.min(rq * a[:, None], torch.clamp(rq, 1 - ppo.CLIP, 1 + ppo.CLIP) * a[:, None])[qm]).mean() if qm.any() else torch.zeros((), device=dev)
                rm = torch.exp(cur["lpm"] - old["lpm"][idx])
                l_m = -torch.min(rm * a[:, None], torch.clamp(rm, 1 - ppo.CLIP, 1 + ppo.CLIP) * a[:, None]).mean()
                pu = torch.exp(cur["lpu_all"])
                klu = (pu * (cur["lpu_all"] - refp["lpu_all"][idx])).masked_fill(~cur["umask"], 0).sum(-1)[vm].mean()
                pf, pr = cur["pf"], refp["pf"][idx]
                klm = (pf * torch.log(pf / pr) + (1 - pf) * torch.log((1 - pf) / (1 - pr))).mean()
                vcur = ppo.forward(vnet, D["planes"][idx].float() / 255.0, D["glob"][idx], D["pos"][idx].long(), D["uin"][idx].float())[3]
                l_v = Fn.mse_loss(vcur, RET[idx])
                vopt.zero_grad(); l_v.backward(); torch.nn.utils.clip_grad_norm_(vnet.parameters(), 1.0); vopt.step()
                if it > vwarm:
                    loss = l_u + 0.5 * l_q + l_m + ppo.BETA * (klu + klm)
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                stats += [l_u.item(), l_m.item(), klu.item(), klm.item(), l_v.item()]; nb += 1
        del D, old, refp, A, RET
        if dev.type == "mps":                        # hand the update's GPU buffers back (memory pressure)
            torch.mps.empty_cache()
        per = {}
        for r in res_all:
            kd = kind_of(r["opp"])
            if kd in hist and not r.get("pre"):
                hist[kd].append(float(r["margin"] <= 0))
            k = kd.split("/")[-2] if "/" in kd else kd
            k = "OPENING" if r.get("pre") else k
            per.setdefault(k, []).append(r["margin"])
        print(f"it {it} {time.time()-t0:.0f}s | " + " ".join(f"{k} {np.mean(v):+,.0f} W{np.mean(np.array(v) > 0):.0%}({len(v)})" for k, v in sorted(per.items()))
              + f" | kl u/m {stats[2]/nb:.4f}/{stats[3]/nb:.4f} v {stats[4]/nb:.4f}", flush=True)
        if it % 5 == 0:
            p = f"{snap_dir}/snap_{it}.pt"; save(p); snaps.append(p)
        lag = int(os.environ.get("REF_LAG", "0"))
        if lag and it % lag == 0:
            if it >= 2 * lag:                       # teacher <- the policy as it was `lag` iterations ago
                c_ = torch.load(f"{snap_dir}/lag_{it - lag}.pt", map_location="cpu", weights_only=False)
                ref.load_state_dict(c_["state"]); ref.eval()
                print(f"  KL teacher moved to the policy of iteration {it - lag}", flush=True)
            save(f"{snap_dir}/lag_{it}.pt")
        if it % EVAL == 0:
            _, sc = evaluate(f"it {it}", base)
            save(out.replace(".pt", f"_it{it}.pt"), {"it": it, "score": sc})
            if sc > best:
                best = sc; save(out, {"it": it, "score": sc}); print(f"  new best (mean eval margin {sc:+,.0f}) -> {out}", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
