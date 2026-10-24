"""Worker-side rollouts against LIVE opponents (scripted bots or frozen neural snapshots) on fresh worlds.
Runs in a multiprocessing worker on CPU: loads the policy weights from a file, plays a batch of games in lockstep
(one batched CPU forward per step), records trajectories for PPO (same fields as ppo.collect) or just scores.

job = dict(weights=path, team=int, tau=float, T=float, ks=float, greedy=bool, seed=int,
           games=[(seed, our_seat, opponent_spec), ...])
A game entry may carry a 4th element (bot_spec, n_steps): that scripted bot plays OUR seat for the first n_steps
(a "scripted opening"), then the network takes over; only the network's own decisions are recorded.
opponent_spec: harness agent name / path (scripted), "NN:<ckpt>:<team>:<tau>" for a frozen neural policy, or
"TAPE:<tape.json.gz>" = the opponent's recorded actions of a live game with the world pinned to that game (seed from
the tape, our_seat = the seat we played in it); used to train on live losses.
"""
import sys, os
import numpy as np
sys.path.insert(0, "/Users/asad.ali/kaggriculture/tools"); sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")

_NETS = {}
ROLLOUT_DEVICE = os.environ.get("ROLLOUT_DEVICE", "cpu")   # Sep 26: "cuda" = our policy's forward on the GPU (RL rollouts
                                                              # only; benchmarks keep the CPU default = the Kaggle runtime)
MIDNIGHT = int(os.environ.get("MIDNIGHT", "1"))       # rule layer: sell the shed's overflow before the midnight drop
FINALFIX = int(os.environ.get("FINALFIX", "1"))       # rule layer: last turn drops at the shed and sells everything
HOMING = int(os.environ.get("HOMING", "1"))           # rule layer: last day, loaded units reach the shed by the final turn


def _net(path):
    import torch
    torch.set_num_threads(1)
    from model import Net
    key = (path, os.path.getmtime(path))
    if key not in _NETS:
        c = torch.load(path, map_location="cpu", weights_only=False)
        n = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"), n_aux=c.get("n_aux", 0), ar=c.get("ar", 0), menu=c.get("menu", 0)); n.load_state_dict(c["state"]); n.eval()
        _NETS.clear() if len(_NETS) > 6 else None
        _NETS[key] = n
    return _NETS[key]


class _Tape:
    """Recorded actions of the other seat of a taped game, one per call."""
    def __init__(self, acts):
        self.A = acts; self.t = 0

    def __call__(self, obs):
        from harness import PASS
        a = self.A[self.t] if self.t < len(self.A) else None; self.t += 1
        return a if isinstance(a, dict) else dict(PASS)


def _game(seed, seat, spec):
    """(engine game, opponent callable) for one job entry."""
    from harness import kagsim_trace
    if spec.startswith("TAPE:"):
        from rollout import load_world
        d, shops, e0 = load_world(spec[5:])
        g = kagsim_trace.Game(int(d["seed"]), 720, False); g.set_forced_shops(shops)
        if 1 - seat == 1:
            g.set_weed_prefix(1, e0)
        return g, _Tape(d["actions"][1 - seat])
    return kagsim_trace.Game(int(seed), 720, False), _opponent(spec)


def _opponent(spec):
    from harness import mk
    if spec.startswith("NN:"):
        from agent import BCAgent
        _, ck, team, tau = spec.split(":")
        return BCAgent(ck, mk_tau=float(tau), team=int(team))
    return mk(spec)()


def run_job(job):
    """Plays job["games"] in lockstep batches of at most LOCKSTEP (10) games: every scripted opponent is a full copy of
    its bot (tens of MB each), so a worker must not hold dozens of them at once (memory pressure)."""
    import gc
    B = int(os.environ.get("LOCKSTEP", "10"))
    if len(job["games"]) > B:
        out = []
        for i in range(0, len(job["games"]), B):
            out += _run_batch(dict(job, games=job["games"][i:i + B], seed=int(job["seed"]) * 1000 + i))
            gc.collect()
        return out
    return _run_batch(job)


def _run_batch(job):
    import torch
    from harness import kagsim_trace, PASS
    import features as F
    from agent import legal_mask
    from rollout import sharpen, MARKET_OFF
    torch.set_num_threads(1)
    F.FINAL_FIX = FINALFIX; F.HOMING = HOMING; F.SELL_ORDER = os.environ.get("SELL_ORDER", "canonical")
    F.FINAL_ORDER = os.environ.get("FINAL_ORDER", "qty")
    F.SE_GUARD = int(os.environ.get("SE_GUARD", "0")); F.SE_HOUR = int(os.environ.get("SE_HOUR", "12"))      # Sep 26 plan rules
    F.SE_LAST_DAY = int(os.environ.get("SE_LAST_DAY", "13")); F.SE_RESERVE = int(os.environ.get("SE_RESERVE", "300"))
    F.SE_SHOPS = int(os.environ.get("SE_SHOPS", "0"))   # Sep 28: SE only in crop-demand worlds
    F.SHEEP_YARN = int(os.environ.get("SHEEP_YARN", "0")); F.SHEEP_BASE = int(os.environ.get("SHEEP_BASE", "3"))   # Sep 28 W1
    F.SHEEP_MAX = int(os.environ.get("SHEEP_MAX", "18")); F.SHEEP_LAST_DAY = int(os.environ.get("SHEEP_LAST_DAY", "20"))
    F.TOMATO_SHOPS = int(os.environ.get("TOMATO_SHOPS", "0")); F.TOMATO_PER_SHOP = int(os.environ.get("TOMATO_PER_SHOP", "8"))  # W6
    F.TOMATO_LAST_DAY = int(os.environ.get("TOMATO_LAST_DAY", "18")); F.TOMATO_FROM_DAY = int(os.environ.get("TOMATO_FROM_DAY", "2"))
    F.COW_MILK = int(os.environ.get("COW_MILK", "0")); F.COW_BASE = int(os.environ.get("COW_BASE", "5"))   # W7
    F.COW_MAX = int(os.environ.get("COW_MAX", "15")); F.COW_LAST_DAY = int(os.environ.get("COW_LAST_DAY", "18"))
    F.STRAW_UP = int(os.environ.get("STRAW_UP", "0")); F.STRAW_MIN = int(os.environ.get("STRAW_MIN", "150"))   # TOPVSTOP_A
    F.STRAW2 = int(os.environ.get("STRAW2", "0")); F.STRAW2_P = int(os.environ.get("STRAW2_P", "120"))   # Sep 28 night: demand rules
    F.STRAW2_FROM = int(os.environ.get("STRAW2_FROM", "9")); F.STRAW2_LAST = int(os.environ.get("STRAW2_LAST", "13"))
    F.STRAW2_CUT = int(os.environ.get("STRAW2_CUT", "0")); F.STRAW2_FUT = int(os.environ.get("STRAW2_FUT", "100"))
    F.STRAW2_PER = int(os.environ.get("STRAW2_PER", "185")); F.STRAW2_SHARE = int(os.environ.get("STRAW2_SHARE", "0"))
    F.COW2 = int(os.environ.get("COW2", "0")); F.COW2_SHOPS = int(os.environ.get("COW2_SHOPS", "2"))
    F.COW2_FLOOR = int(os.environ.get("COW2_FLOOR", "190")); F.COW2_LAST_DAY = int(os.environ.get("COW2_LAST_DAY", "10"))
    F.COW2_GAP = int(os.environ.get("COW2_GAP", "1")); F.COW2_MAX = int(os.environ.get("COW2_MAX", "14"))
    F.COW2_SI = int(os.environ.get("COW2_SI", "0")); F.COW2_FROM = int(os.environ.get("COW2_FROM", "7"))
    F.SHEEP_FLOOR = int(os.environ.get("SHEEP_FLOOR", "0"))   # Sep 28 night: fixed wool floor for SHEEP_YARN
    F.STRADDLE_PREBUY = int(os.environ.get("STRADDLE_PREBUY", "0")); F.STRADDLE_PREBUY_MIN = int(os.environ.get("STRADDLE_PREBUY_MIN", "25"))
    F.SHEEP_EARLY = int(os.environ.get("SHEEP_EARLY", "0")); F.SHEEP_MIRROR = int(os.environ.get("SHEEP_MIRROR", "0"))   # wool study
    F.SHEEP_MFLOOR = int(os.environ.get("SHEEP_MFLOOR", "0")); F.SHEEP_MLAST = int(os.environ.get("SHEEP_MLAST", "0"))
    F.SHEEP_MGAP = int(os.environ.get("SHEEP_MGAP", "0")); F.SHEEP_MINY = int(os.environ.get("SHEEP_MINY", "0"))
    F.SHEEP_KEEP = int(os.environ.get("SHEEP_KEEP", "0")); F.SHEEP_HOLD = int(os.environ.get("SHEEP_HOLD", "0"))
    F.GOOSE_EGG = int(os.environ.get("GOOSE_EGG", "0")); F.GOOSE_BASE = int(os.environ.get("GOOSE_BASE", "5"))   # Sep 29 eggs (W12)
    F.GOOSE_MAX = int(os.environ.get("GOOSE_MAX", "13")); F.GOOSE_FROM_DAY = int(os.environ.get("GOOSE_FROM_DAY", "10"))
    F.GOOSE_LAST_DAY = int(os.environ.get("GOOSE_LAST_DAY", "13")); F.GOOSE_FLOOR = int(os.environ.get("GOOSE_FLOOR", "45"))
    F.GOOSE_RIVAL = int(os.environ.get("GOOSE_RIVAL", "0")); F.GOOSE_YARN = int(os.environ.get("GOOSE_YARN", "2")); F.GOOSE_HOUR = int(os.environ.get("GOOSE_HOUR", "16"))
    F.GOOSE_GAP = int(os.environ.get("GOOSE_GAP", "3")); F.GOOSE_BATCH = int(os.environ.get("GOOSE_BATCH", "6"))
    F.SHEEP_CAP = int(os.environ.get("SHEEP_CAP", "0"))
    F.SEED_BUDGET = int(os.environ.get("SEED_BUDGET", "0"))
    F.WASH = int(os.environ.get("WASH", "0")); F.WASH_MODE = os.environ.get("WASH_MODE", "sells")
    F.FEED_GUARD = int(os.environ.get("FEED_GUARD", "0")); F.FEED_SLACK = int(os.environ.get("FEED_SLACK", "1"))   # Sep 28 R2
    F.COW_CARE = int(os.environ.get("COW_CARE", "0")); F.CARE_MIN = int(os.environ.get("CARE_MIN", "100"))   # Sep 29 milk study
    F.CARE_HOUR = int(os.environ.get("CARE_HOUR", "15")); F.CARE_PEN = int(os.environ.get("CARE_PEN", "4"))
    F.CARE_ESC = int(os.environ.get("CARE_ESC", "0")); F.CARE_ESC_MIN = int(os.environ.get("CARE_ESC_MIN", "40"))
    F.CARE_ESC_DAYS = int(os.environ.get("CARE_ESC_DAYS", "4")); F.CARE_DEDUP = int(os.environ.get("CARE_DEDUP", "0"))
    F.LAND2_DAY = int(os.environ.get("LAND2_DAY", "0")); F.LAND2_HOUR = int(os.environ.get("LAND2_HOUR", "5"))   # Sep 28: DSM's
    F.NO_LAND3 = int(os.environ.get("NO_LAND3", "0")); F.EARLY_COWS = int(os.environ.get("EARLY_COWS", "0"))     # Sep 26 plan
    F.EARLY_COW_HOUR = int(os.environ.get("EARLY_COW_HOUR", "6")); F.EARLY_COW_BASE = int(os.environ.get("EARLY_COW_BASE", "2"))
    F.MELON_WAVE = int(os.environ.get("MELON_WAVE", "0")); F.MELON_DAY = int(os.environ.get("MELON_DAY", "6"))
    F.LAND_GUARD = int(os.environ.get("LAND_GUARD", "0")); F.MELON_WHEAT_MIN = int(os.environ.get("MELON_WHEAT_MIN", "0"))   # Sep 28 evening: rule buys keep the next land's price
    F.STRADDLE = int(os.environ.get("STRADDLE", "0"))       # Sep 28: wheat drain straddle, applied last (0 = off)
    F.STRADDLE_DAY = int(os.environ.get("STRADDLE_DAY", "12")); F.STRADDLE_CAP = int(os.environ.get("STRADDLE_CAP", "100"))
    net = _net(job["weights"]); team = job["team"]
    dev = ROLLOUT_DEVICE
    if dev != "cpu":                                 # a GPU copy of the same weights, cached per worker process
        torch.backends.cudnn.allow_tf32 = False       # full fp32 math, as on the CPU / in the Kaggle runtime
        torch.backends.cuda.matmul.allow_tf32 = False
        key = (job["weights"], os.path.getmtime(job["weights"]), dev)
        if key not in _NETS:
            import copy
            _NETS[key] = copy.deepcopy(net).to(dev).eval()
        net = _NETS[key]
    if getattr(net, "g_in", F.G) != F.G:
        import features6 as FE                       # v6 policy: v5 features + "if nobody acts" future features
    else:
        FE = F
    if getattr(net, "menu", 0) == 8:                 # extended fixed menu (features8)
        import features8 as MK
        MOFF = MK.MARKET_OFF
    else:
        MK = F; MOFF = MARKET_OFF
    MNAMES = [h for h, _ in MK.MARKET_HEADS]; NH = len(MNAMES)
    tau = job["tau"]; T = job["T"]; KS = job["ks"]; greedy = job["greedy"]
    rng = np.random.RandomState(job["seed"])
    G = []
    from harness import mk
    for entry in job["games"]:
        seed, seat, spec = entry[:3]; pre = entry[3] if len(entry) > 3 and entry[3] else None
        g_, opp_ = _game(seed, seat, spec)
        G.append(dict(g=g_, seat=seat, opp=opp_, prev=None, done=False, pre_bot=mk(pre[0])() if pre else None, pre_n=int(pre[1]) if pre else 0,
                      buf=dict(planes=[], glob=[], pos=[], uin=[], umask=[], uact=[], qact=[], mfire=[], mclass=[], mdiff=[])))
    for t in range(720):
        for x in G:                                  # scripted opening: the bot plays our seat, nothing recorded
            if not x["done"] and t < x["pre_n"]:
                o = x["g"].observe(x["seat"]); th = 1 - x["seat"]
                try:
                    a = x["pre_bot"](o)
                except Exception:
                    a = dict(PASS)
                try:
                    b_ = x["opp"](x["g"].observe(th))
                except Exception:
                    b_ = dict(PASS)
                if isinstance(a, dict):
                    x["prev"] = [F.unit_op_label(u)[0] for u in [a.get("farmer")] + list(a.get("hands") or [])]
                now = [None, None]; now[x["seat"]] = a; now[th] = b_; x["g"].step(now[0], now[1])
                x["done"] = bool(x["g"].done)
        live = [x for x in G if not x["done"] and t >= x["pre_n"]]
        if not live:
            if all(x["done"] for x in G):
                break
            continue
        obs = [x["g"].observe(x["seat"]) for x in live]
        if F.STRADDLE:                               # the policy + rules see the farm as if no straddle were open; the real
            real = obs                               # observation (tagged per game: straddle state per game and seat) is
            for o_, x in zip(real, live):            # kept for the straddle's own legs
                o_["_seed_tag"] = id(x)
            obs = [F.straddle_obs(o_) for o_ in real]
        elif getattr(F, "SHEEP_KEEP", 0) or getattr(F, "SHEEP_MGAP", 0) or getattr(F, "SHEEP_HOLD", 0):   # per-game sheep state: tag
            for o_, x in zip(obs, live):
                o_["_seed_tag"] = id(x)
        enc = [FE.encode_obs(o, x["prev"]) for o, x in zip(obs, live)]
        n = len(live)
        pos = np.zeros((n, F.MAX_UNITS, 2), np.int64); uin = np.zeros((n, F.MAX_UNITS, F.U), np.float32); nu = []
        for j, e in enumerate(enc):
            m = len(e[2]); nu.append(m); pos[j, :m] = e[2]; uin[j, :m] = e[3]
        with torch.no_grad():
            pl = torch.from_numpy(np.stack([e[0] for e in enc]).astype(np.float32) / 255.0)  # noqa
            gl = torch.from_numpy(np.stack([e[1] for e in enc]))
            tt = torch.full((n,), team, dtype=torch.long) if net.n_teams else None
            if dev == "cpu":
                outs = net(pl, gl, torch.from_numpy(pos), torch.from_numpy(uin), tt, ret_hm=True)
            else:
                outs = net(pl.to(dev), gl.to(dev), torch.from_numpy(pos).to(dev), torch.from_numpy(uin).to(dev),
                           tt.to(dev) if tt is not None else None, ret_hm=True)
            lop, lq, lm, hm_ = outs[0], outs[1], outs[2], outs[5]
            use_ar = getattr(net, "ar", 0) and not int(os.environ.get("AR_OFF", "0"))     # AR_OFF=1: use the old heads
            ar_seq = net.ar_greedy(hm_, float(tau if not isinstance(tau, dict) else tau.get("_default", 0.5))).cpu().numpy() if use_ar else None
        lop = lop.cpu().numpy(); lq = lq.cpu().numpy(); lm = lm.cpu().numpy()
        for j, x in enumerate(live):
            o = obs[j]
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
            x["prev"] = ops
            classes, fires = [], np.zeros(NH, np.int8)
            for h in range(NH):
                zz = lm[j, MOFF[h]:MOFF[h + 1]].astype(np.float64)
                p = np.exp(zz - zz.max()); p /= p.sum()
                th_ = tau.get(MNAMES[h], tau.get("_default", 0.6)) if isinstance(tau, dict) else tau   # per-head thresholds
                fire = (1.0 - p[0] > th_) if greedy else (rng.random_sample() < sharpen(1.0 - p[0], th_, KS))
                fires[h] = int(fire); classes.append(int(1 + np.argmax(p[1:])) if fire else 0)
            if ar_seq is not None:                   # order-by-order head: its own market sequence
                import features_ar as FA
                mk_orders = FA.decode_seq(ar_seq[j, :, 0], ar_seq[j, :, 1], ar_seq[j, :, 2])
            else:
                mk_orders = MK.decode_market(classes, o)
            a = {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": mk_orders}
            if MIDNIGHT:
                a = F.midnight_fix(o, a)
            if F.STRADDLE:                           # very last: the straddle's legs, on the real observation
                a = F.straddle_fix(real[j], a)
            th = 1 - x["seat"]
            try:
                b_ = x["opp"](x["g"].observe(th))
            except Exception:
                b_ = dict(PASS)
            if not greedy:
                b = x["buf"]
                b["planes"].append(enc[j][0]); b["glob"].append(enc[j][1]); b["pos"].append(pos[j].astype(np.int8))
                b["uin"].append(uin[j].astype(np.float16)); b["umask"].append(umask); b["uact"].append(ua); b["qact"].append(qa)
                b["mfire"].append(fires); b["mclass"].append(np.array(classes, np.int16))
                b["mdiff"].append(o["farms"][x["seat"]]["money"] - o["farms"][th]["money"])
            now = [None, None]; now[x["seat"]] = a; now[th] = b_
            x["g"].step(now[0], now[1])
            if x["g"].done:
                x["done"] = True
    out = []
    for entry, x in zip(job["games"], G):
        seed, seat, spec = entry[:3]
        final = x["g"].reward(x["seat"]) - x["g"].reward(1 - x["seat"])
        r = dict(seed=seed, seat=seat, opp=spec, bank=x["g"].reward(x["seat"]), opp_bank=x["g"].reward(1 - x["seat"]), margin=final, pre=x["pre_n"])
        if not greedy and not x["buf"]["planes"]:
            r["traj"] = None
        elif not greedy:
            b = x["buf"]; md = np.array(b["mdiff"] + [final], np.float64)
            r["traj"] = {k: np.stack(v) for k, v in b.items() if k != "mdiff"}
            r["traj"]["rew"] = ((md[1:] - md[:-1]) / 2e4).astype(np.float32)
            r["traj"]["rew"][-1] += job.get("winb", 0.0) * np.sign(final)      # win/loss bonus: the ladder counts wins
        out.append(r)
    return out


def run_scripted(job):
    """A scripted bot (harness name/path) in our seat: job = dict(us=spec, games=[(seed, our_seat, opponent_spec)]).
    Returns the same result dicts as run_job (greedy, no trajectories)."""
    from harness import mk, PASS
    out = []
    make = mk(job["us"])
    for seed, seat, spec in job["games"]:
        g, opp = _game(seed, seat, spec); ag = make(); th = 1 - seat
        for t in range(720):
            try:
                a = ag(g.observe(seat))
            except Exception:
                a = dict(PASS)
            try:
                b = opp(g.observe(th))
            except Exception:
                b = dict(PASS)
            now = [None, None]; now[seat] = a; now[th] = b; g.step(now[0], now[1])
            if g.done:
                break
        out.append(dict(seed=seed, seat=seat, opp=spec, bank=g.reward(seat), opp_bank=g.reward(th), margin=g.reward(seat) - g.reward(th)))
        del g, opp, ag
        if len(out) % 5 == 0:
            import gc; gc.collect()
    return out
