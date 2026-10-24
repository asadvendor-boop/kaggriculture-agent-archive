"""ORACLE CEILING of decision-time look-ahead for end-of-day selling (Sep 26 2026; operator: "is look-ahead worth building?").

At hour 23 of days 1..28 (step 24d+23) the greedy policy's action a - exactly the benchmark path of
live_rollout._run_batch (features6 encoder, one batched forward, legal masks, argmax, quantity clamp, market heads fire
when 1 - P(class 0) > tau, decode_market, then the features.midnight_fix rule layer with MIDNIGHT/FINALFIX/HOMING) -
is compared with market-only alternatives for that one step:
  A0 = a unchanged
  A1 = hold: a without its SELL orders, then features.midnight_fix again (the forced midnight overflow sale stays)
  A2 = sell all: SELL <item> <stock> for every product in our shed at market time (shed + this turn's drops - pickups),
       canonical order, then a's other orders (10-order cap: the lowest-revenue sells are dropped first), midnight_fix
  A3 = sell half (ALTS=0123): SELL <item> ceil(stock / 2), same layout
Each alternative is played to the END in the exact C++ engine: the engine has no state clone, so a branch is a fresh game
(forced shops + weed prefix, as live_rollout._game) replayed to step t with the recorded actions of both seats; then our
seat is played by the policy and the opponent by its TAPE; value = final margin.  The best alternative is applied in the
main line (ties keep A0) and the game continues to the next decision.  A0's value is the current main line's final margin
(the main line IS the last chosen branch, so nothing is recomputed for it).  All branches of one decision (all
alternatives of all games of a job) run in lockstep with ONE batched forward per step.

ORACLE: the tape reveals the opponent's future (and never reacts to us), so this is an UPPER BOUND for this decision
family; a live version would need an opponent model.

Control = the first rollout of every game (A0 at every decision); it must reproduce the stored benchmark margins
(il/logs/ladder_<tag>_test.json).  Forward numerics (measured Sep 26 on the 20M model): on the server's x86 CPU a batch
of 1-3 rows gives bitwise different logits from a batch of 5+ (5, 6, 8, 12 identical per row, any order); the benchmark
ran lockstep batches of 6 (bench_multi LOCKSTEP=6), so every forward is padded to >= PADMIN = 6 rows with copies of row 0.
(The Mac's arm64 CPU: 1 row differs, 2+ identical.  The GPU: every batch size differs, so LA_DEVICE=cuda runs use
PADFIX = a fixed row count for every forward (one kernel shape; rows independent of position and companions) - the GPU
control is then self-consistent but may differ from the stored CPU margins where a near-tie flips.)

Usage:
  lookahead_ceiling.py run <ckpt>:<team>:<tau> <tag>
      env NW (workers, <= 16), CHUNK (games per lockstep job, 6), LA_DEVICE (cpu | cuda), ALTS (012), SUBSET (2400 =
      opp_rating >= 2400 | rest | all = 2400+ first), LIMIT (first n worlds), MODE (lookahead | control), CHECK_A0 (1 =
      also re-roll A0 at every decision and require it to equal the main line), DAYS (e.g. 20-28), SHARD (i/n: every
      n-th world from i; NW=1 runs the jobs in the process itself), PADMIN (6), PADFIX (0; GPU only)
      -> il/logs/lookahead_<tag>.json (ladder-benchmark rows + control + choices) and ..._control.json
  lookahead_ceiling.py report <tag or glob, e.g. 'g20_s012_sh*_srv'> [stored_benchmark.json]
  lookahead_ceiling.py combine <tag glob days A> <tag glob days B>   (approx. full ceiling = sum of disjoint day ranges)
"""
import sys, os, json, time, collections
import numpy as np

R = "/Users/asad.ali/kaggriculture"
sys.path.insert(0, f"{R}/tools"); sys.path.insert(0, f"{R}/il"); sys.path.insert(0, f"{R}/il/live")
CANON = ["FERTILIZER", "WHEAT", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO", "MELON"]   # decode_market's sell order


def _days():
    s = os.environ.get("DAYS", "1-28")
    a, b = (s.split("-") + [s])[:2]
    return list(range(int(a), int(b) + 1))


# ---- policy: exact replica of live_rollout._run_batch, greedy path ------------------------------------------------
class Policy:
    def __init__(self, weights, team, tau, dev="cpu"):
        import torch
        torch.set_num_threads(1)
        import features as F
        import live_rollout as LR
        from rollout import MARKET_OFF
        from agent import legal_mask
        F.FINAL_FIX = LR.FINALFIX; F.HOMING = LR.HOMING; F.SELL_ORDER = os.environ.get("SELL_ORDER", "canonical")
        F.SE_GUARD = int(os.environ.get("SE_GUARD", "0")); F.SE_HOUR = int(os.environ.get("SE_HOUR", "12"))
        F.SE_LAST_DAY = int(os.environ.get("SE_LAST_DAY", "13")); F.SE_RESERVE = int(os.environ.get("SE_RESERVE", "300"))
        F.SHEEP_CAP = int(os.environ.get("SHEEP_CAP", "0"))
        F.SEED_BUDGET = int(os.environ.get("SEED_BUDGET", "0"))
        F.WASH = int(os.environ.get("WASH", "0")); F.WASH_MODE = os.environ.get("WASH_MODE", "sells")
        self.F, self.LR, self.torch, self.legal_mask = F, LR, torch, legal_mask
        net = LR._net(weights)
        if dev != "cpu":
            import copy
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
            net = copy.deepcopy(net).to(dev).eval()
        assert not getattr(net, "ar", 0) and getattr(net, "menu", 0) != 8, "replica covers the menu-0, non-AR benchmark path"
        if getattr(net, "g_in", F.G) != F.G:
            import features6 as FE
        else:
            FE = F
        self.net, self.FE, self.dev, self.team, self.tau = net, FE, dev, int(team), float(tau)
        self.MOFF = MARKET_OFF; self.NH = len(F.MARKET_HEADS)
        self.padmin = int(os.environ.get("PADMIN", "6"))   # rows per forward at least (copies of row 0), see module doc
        self.padfix = int(os.environ.get("PADFIX", "0"))   # > 0: EVERY forward has exactly PADFIX rows (GPU: one kernel shape)
        self.rows = 0; self.calls = 0; self.t_fwd = 0.0; self.t_all = 0.0

    def act(self, obs, prevs):
        """[(action dict after the rule layer, network op classes)] for a batch of our observations."""
        F, torch, net, dev = self.F, self.torch, self.net, self.dev
        t0 = time.time()
        enc = [self.FE.encode_obs(o, p) for o, p in zip(obs, prevs)]
        n = len(enc)
        pos = np.zeros((n, F.MAX_UNITS, 2), np.int64); uin = np.zeros((n, F.MAX_UNITS, F.U), np.float32); nu = []
        for j, e in enumerate(enc):
            m = len(e[2]); nu.append(m); pos[j, :m] = e[2]; uin[j, :m] = e[3]
        planes = np.stack([e[0] for e in enc]); glob = np.stack([e[1] for e in enc])
        t1 = time.time()
        B = self.padfix or max(n, self.padmin)          # rows per forward call; padding rows are copies (discarded)
        parts = []
        for s0 in range(0, n, B):
            e0 = min(n, s0 + B)
            ix = np.r_[np.arange(s0, e0), np.full(B - (e0 - s0), s0, np.int64)]
            with torch.no_grad():
                pl = torch.from_numpy(planes[ix].astype(np.float32) / 255.0)
                gl = torch.from_numpy(glob[ix])
                tt = torch.full((B,), self.team, dtype=torch.long) if net.n_teams else None
                if dev == "cpu":
                    outs = net(pl, gl, torch.from_numpy(pos[ix]), torch.from_numpy(uin[ix]), tt, ret_hm=True)
                else:
                    outs = net(pl.to(dev), gl.to(dev), torch.from_numpy(pos[ix]).to(dev), torch.from_numpy(uin[ix]).to(dev),
                               tt.to(dev) if tt is not None else None, ret_hm=True)
                parts.append([x[:e0 - s0].cpu().numpy() for x in outs[:3]])
        lop, lq, lm = (np.concatenate([p[i] for p in parts]) for i in range(3))
        self.t_fwd += time.time() - t1
        out = []
        for j in range(n):
            o = obs[j]
            shed = o["private"].get("shed", {}) or {}; invs = o["private"].get("inventories") or [{}]
            cmds, ops = [], []
            for k in range(nu[j]):
                allowed = self.legal_mask(o, k, pos[j, k])
                z = np.where(allowed, lop[j, k].astype(np.float64), -1e9)
                oc = int(np.argmax(z))
                name = F.OPS[oc]; qc = 0
                if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                    zq = lq[j, k].astype(np.float64)
                    qc = int(np.argmax(zq))
                    it = name.split("_", 1)[1]
                    have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                    qc = max(0, min(qc, int(have) - 1))
                cmds.append(F.unit_command(oc, qc)); ops.append(oc)
            classes = []
            for h in range(self.NH):
                zz = lm[j, self.MOFF[h]:self.MOFF[h + 1]].astype(np.float64)
                p = np.exp(zz - zz.max()); p /= p.sum()
                fire = (1.0 - p[0] > self.tau)
                classes.append(int(1 + np.argmax(p[1:])) if fire else 0)
            a = {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": F.decode_market(classes, o)}
            if self.LR.MIDNIGHT:
                a = F.midnight_fix(o, a)
            out.append((a, ops))
        self.rows += n; self.calls += 1; self.t_all += time.time() - t0
        return out


# ---- worlds, replay, lockstep rollouts ------------------------------------------------------------------------------
_W = {}


def world(w):
    """(tape dict, forced shops, weed prefix) of a ladder world, cached per process (live_rollout._game's inputs)."""
    if w["path"] not in _W:
        from rollout import load_world
        if len(_W) > 40:
            _W.clear()
        _W[w["path"]] = load_world(w["path"])
    return _W[w["path"]]


def fresh_game(w):
    from harness import kagsim_trace
    d, shops, e0 = world(w)
    g = kagsim_trace.Game(int(d["seed"]), 720, False); g.set_forced_shops(shops)
    if 1 - w["seat"] == 1:
        g.set_weed_prefix(1, e0)
    return g


def opp_act(w, t):
    """The tape's action at step t (live_rollout._Tape: one recorded action per call, PASS when missing)."""
    from harness import PASS
    A = world(w)[0]["actions"][1 - w["seat"]]
    a = A[t] if t < len(A) else None
    return a if isinstance(a, dict) else dict(PASS)


def step(g, seat, a, b):
    now = [None, None]; now[seat] = a; now[1 - seat] = b; g.step(now[0], now[1])


def replay(w, acts, t, observe=False):
    """Fresh game at the start of step t (steps 0..t-1 replayed with our recorded actions and the tape).  The reference
    loop observes both seats every step; the C++ Game::observe is const and replays with / without those calls give the
    same margins (checked Sep 26), so the replay skips them (observe=True mirrors them)."""
    g = fresh_game(w); seat = w["seat"]
    for s in range(t):
        if observe:
            g.observe(seat); g.observe(1 - seat)
        step(g, seat, acts[s], opp_act(w, s))
    return g


class Branch:
    __slots__ = ("w", "g", "acts", "ops", "prev", "k", "value", "bank", "opp_bank")

    def __init__(self, w, g, acts, ops, prev, k=0):
        self.w, self.g, self.acts, self.ops, self.prev, self.k = w, g, acts, ops, prev, k
        self.value = self.bank = self.opp_bank = None


def rollout(P, branches):
    """Play every branch to the end in lockstep (one batched forward per step): policy for us, tape for the opponent."""
    live = [b for b in branches if not b.g.done]
    while live:
        obs = [b.g.observe(b.w["seat"]) for b in live]           # (the opponent's observe is skipped: const, tape ignores it)
        res = P.act(obs, [b.prev for b in live])
        for b, o, (a, ops) in zip(live, obs, res):
            t = len(b.acts)
            if int(o["step"]) != t:
                raise RuntimeError(f"step mismatch {o['step']} != {t}")
            b.acts.append(a); b.ops.append(ops); b.prev = ops
            step(b.g, b.w["seat"], a, opp_act(b.w, t))
        live = [b for b in live if not b.g.done]
    for b in branches:
        if b.g is not None:
            b.bank = b.g.reward(b.w["seat"]); b.opp_bank = b.g.reward(1 - b.w["seat"]); b.value = b.bank - b.opp_bank
            b.g = None


# ---- the decision's alternatives --------------------------------------------------------------------------------------
def _stock_at_market(obs, a):
    """Our shed's products when the market runs this turn: shed + this turn's DROPs - PICKUPs (units act first)."""
    priv = obs["private"]; invs = priv.get("inventories") or []
    stock = {k: int(v) for k, v in (priv.get("shed") or {}).items()}
    units = [a.get("farmer") or ["PASS"]] + list(a.get("hands") or [])
    for i, c in enumerate(units):
        if not c:
            continue
        if c[0] == "PICKUP" and len(c) > 2:
            stock[c[1]] = max(0, stock.get(c[1], 0) - int(c[2]))
        elif c[0] == "DROP" and i < len(invs):
            for k, v in invs[i].items():
                stock[k] = stock.get(k, 0) + int(v)
    return stock


def alternatives(F, obs, a0, which="012"):
    """[(k, action)] for k in `which` minus 0, each distinct from a0 and from the ones before it."""
    market = [list(o) for o in (a0.get("market") or []) if isinstance(o, (list, tuple))]
    rest = [o for o in market if not (o and o[0] == "SELL")]
    stock = _stock_at_market(obs, a0); prices = obs["market"]["prices"]
    room = max(0, 10 - len(rest))

    def sells(frac):
        s = [["SELL", p, (stock[p] + 1) // 2 if frac == 0.5 else stock[p]] for p in CANON if stock.get(p, 0) > 0]
        if len(s) > room:                               # keep the highest-revenue sells, canonical order
            keep = sorted(range(len(s)), key=lambda i: -prices.get(s[i][1], 0) * s[i][2])[:room]
            s = [s[i] for i in sorted(keep)]
        return s

    out, seen = [], {json.dumps(a0, sort_keys=True)}
    for k in which:
        if k == "0":
            continue
        if k == "1":
            a = F.midnight_fix(obs, dict(a0, market=[list(o) for o in rest]))
        elif k == "2":
            a = F.midnight_fix(obs, dict(a0, market=sells(1.0) + [list(o) for o in rest]))
        elif k == "3":
            a = F.midnight_fix(obs, dict(a0, market=sells(0.5) + [list(o) for o in rest]))
        else:
            raise ValueError(k)
        key = json.dumps(a, sort_keys=True)
        if key not in seen:
            seen.add(key); out.append((int(k), a))
    return out


# ---- one job = CHUNK games in lockstep --------------------------------------------------------------------------------
_P = {}


def run_job(job):
    import torch
    torch.set_num_threads(1)
    key = (job["weights"], job["team"], job["tau"], job["dev"])
    if key not in _P:
        _P.clear(); _P[key] = Policy(job["weights"], job["team"], job["tau"], job["dev"])
    P = _P[key]; P.rows = P.calls = 0; P.t_fwd = P.t_all = 0.0
    W = job["worlds"]; which = job["alts"]; check = job.get("check_a0", 0)
    t0 = time.time()
    main = [Branch(w, fresh_game(w), [], [], None) for w in W]
    rollout(P, main)
    ctrl = [(b.value, b.bank, b.opp_bank) for b in main]
    t_ctrl = time.time() - t0
    dec = [[] for _ in W]; n_branch = 0; bad = []
    if job["mode"] == "lookahead":
        for d in job["days"]:
            t = 24 * d + 23
            new = []
            for i, (w, m) in enumerate(zip(W, main)):
                if len(m.acts) <= t:
                    continue
                g = replay(w, m.acts, t)
                o = g.observe(w["seat"])
                a0 = m.acts[t]
                cand = alternatives(P.F, o, a0, which)
                if check:
                    cand = [(0, a0)] + cand
                for j, (k, a) in enumerate(cand):
                    gg = g if j == 0 else replay(w, m.acts, t)
                    step(gg, w["seat"], a, opp_act(w, t))
                    new.append((i, Branch(w, gg, m.acts[:t] + [a], m.ops[:t + 1], m.ops[t], k)))
                if not cand:
                    g = None
            n_branch += len(new)
            rollout(P, [b for _, b in new])
            for i in range(len(W)):
                mine = [b for ii, b in new if ii == i]
                if check:
                    for b in mine:
                        if b.k == 0 and b.value != main[i].value:
                            bad.append((W[i]["ep"], d, main[i].value, b.value))
                    mine = [b for b in mine if b.k != 0]
                if not mine:
                    continue
                vals = {0: main[i].value}; vals.update({b.k: b.value for b in mine})
                best = max(vals, key=lambda k: (vals[k], -k))     # ties keep the smaller index (A0 first)
                dec[i].append(dict(d=d, k=best, v={str(k): v for k, v in vals.items()}))
                if best != 0:
                    main[i] = next(b for b in mine if b.k == best)
    dt = time.time() - t0
    out = []
    for i, w in enumerate(W):
        m = main[i]
        out.append(dict(ep=w["ep"], opp=w["opp"], opp_rating=w["opp_rating"], margin=m.value, bank=m.bank, opp_bank=m.opp_bank,
                        rec_margin=w["rec_margin"], src=w.get("src"), ctrl_margin=ctrl[i][0], ctrl_bank=ctrl[i][1],
                        ctrl_opp_bank=ctrl[i][2], dec=dec[i], secs=dt / len(W), ctrl_secs=t_ctrl / len(W),
                        job_rows=P.rows, job_calls=P.calls, job_fwd=P.t_fwd, job_pol=P.t_all, job_secs=dt, job_n=len(W),
                        job_branches=n_branch, a0_mismatch=[x for x in bad if x[0] == w["ep"]]))
    return out


# ---- driver -------------------------------------------------------------------------------------------------------------
def select_worlds():
    from ladder import worlds
    W = worlds("test"); sub = os.environ.get("SUBSET", "2400")
    hi = [w for w in W if w["opp_rating"] >= 2400]; lo = [w for w in W if w["opp_rating"] < 2400]
    W = hi if sub == "2400" else lo if sub == "rest" else hi + lo if sub == "all" else W
    if os.environ.get("EPS"):
        eps = {int(x) for x in os.environ["EPS"].split(",")}
        W = [w for w in W if w["ep"] in eps]
    if os.environ.get("LIMIT"):
        W = W[:int(os.environ["LIMIT"])]
    if os.environ.get("SHARD"):                         # "i/n": every n-th world from i (independent processes, one file each)
        i, n = (int(x) for x in os.environ["SHARD"].split("/"))
        W = W[i::n]
    return W


def _atomic_dump(obj, fn):
    tmp = fn + ".tmp"
    json.dump(obj, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, fn)


def main_run(spec, tag):
    import multiprocessing as mp
    parts = spec.split(":"); ck = parts[0]
    team = int(parts[1]) if len(parts) > 1 else 10; tau = float(parts[2]) if len(parts) > 2 else 0.5
    W = select_worlds()
    fn = f"{R}/il/logs/lookahead_{tag}.json"
    done = json.load(open(fn)) if os.path.exists(fn) else []
    have = {r["ep"] for r in done}
    todo = [w for w in W if w["ep"] not in have]
    CH = int(os.environ.get("CHUNK", "6")); NW = min(16, int(os.environ.get("NW", "8")))
    base = dict(weights=ck, team=team, tau=tau, dev=os.environ.get("LA_DEVICE", "cpu"), alts=os.environ.get("ALTS", "012"),
                mode=os.environ.get("MODE", "lookahead"), check_a0=int(os.environ.get("CHECK_A0", "0")), days=_days())
    jobs = [dict(base, worlds=todo[k:k + CH]) for k in range(0, len(todo), CH)]
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: {len(todo)} worlds to do ({len(have)} done), {len(jobs)} jobs of <= {CH}, "
          f"{NW} workers, {base['dev']}, alts {base['alts']}, mode {base['mode']}, days {base['days'][0]}-{base['days'][-1]}", flush=True)
    t0 = time.time()
    rows = list(done)
    if NW <= 1:
        it = map(run_job, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(NW, maxtasksperchild=int(os.environ.get("MAXTASKS", "20")))
        it = pool.imap_unordered(run_job, jobs)
    for res in it:
        rows += res
        _atomic_dump(rows, fn)
        _atomic_dump([dict(ep=r["ep"], opp=r["opp"], opp_rating=r["opp_rating"], margin=r["ctrl_margin"], bank=r["ctrl_bank"],
                           opp_bank=r["ctrl_opp_bank"], rec_margin=r["rec_margin"], src=r["src"]) for r in rows],
                     fn.replace(".json", "_control.json"))
        d = np.array([r["margin"] - r["ctrl_margin"] for r in rows])
        r0 = res[0]
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)}/{len(W)} worlds, +{time.time() - t0:.0f}s | job {r0['job_n']} games "
              f"{r0['job_secs']:.0f}s ({r0['job_secs'] / r0['job_n']:.0f}s/game, control {r0['ctrl_secs']:.0f}s/game, "
              f"{r0['job_rows']} rows, fwd {r0['job_fwd']:.0f}s, policy {r0['job_pol']:.0f}s, {r0['job_branches']} branches) | "
              f"lookahead - control {d.mean():+,.0f}/game (se {d.std() / max(1, np.sqrt(len(d))):,.0f})", flush=True)
    if pool is not None:
        pool.close(); pool.join()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {tag} in {time.time() - t0:.0f}s", flush=True)


def report(tag, stored=None):
    import glob
    rows = [r for fn in sorted(glob.glob(f"{R}/il/logs/lookahead_{tag}.json")) for r in json.load(open(fn))]
    ln = []
    m = np.array([r["margin"] for r in rows], float); c = np.array([r["ctrl_margin"] for r in rows], float); d = m - c
    ln.append(f"lookahead_{tag}: n {len(rows)}")
    if stored:
        S = {r["ep"]: r for r in json.load(open(stored))}
        sm = np.array([S[r["ep"]]["margin"] for r in rows], float)
        ok = (sm == c)
        ln.append(f"  control vs stored benchmark ({os.path.basename(stored)}): exact {ok.sum()}/{len(ok)}; "
                  f"mean control {c.mean():+,.0f} vs stored {sm.mean():+,.0f}" +
                  ("" if ok.all() else f"; mismatches (ep, stored, control): {[(r['ep'], S[r['ep']]['margin'], r['ctrl_margin']) for r, o in zip(rows, ok) if not o][:8]}"))
    mism = [x for r in rows for x in r.get("a0_mismatch", [])]
    if any("a0_mismatch" in r for r in rows):
        ln.append(f"  A0 re-roll mismatches (CHECK_A0): {len(mism)} {mism[:5]}")

    def block(sel, name):
        s = np.array(sel)
        if not s.any():
            return
        dd = d[s]; n = s.sum()
        ln.append(f"  {name:10s} n {n:3d}  control {c[s].mean():+9,.0f}  lookahead {m[s].mean():+9,.0f}  diff {dd.mean():+8,.0f} "
                  f"(se {dd.std(ddof=1) / np.sqrt(n) if n > 1 else 0:,.0f}, median {np.median(dd):+,.0f})  wins {(c[s] > 0).sum()} -> {(m[s] > 0).sum()} "
                  f"(flips +{((c[s] <= 0) & (m[s] > 0)).sum()} -{((c[s] > 0) & (m[s] <= 0)).sum()})  improved {(dd > 0).sum()}  worse {(dd < 0).sum()}")
    block([True] * len(rows), "all")
    block([r["opp_rating"] >= 2400 for r in rows], "2400+")
    block([r["opp_rating"] < 2400 for r in rows], "<2400")
    ch = collections.Counter(x["k"] for r in rows for x in r["dec"])
    nd = sum(len(r["dec"]) for r in rows)
    ln.append(f"  decisions with a distinct alternative: {nd} ({nd / len(rows):.1f}/game); chosen: " +
              ", ".join(f"A{k} {v} ({v / max(1, nd):.0%})" for k, v in sorted(ch.items())))
    byday = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        for x in r["dec"]:
            e = byday[x["d"]]; e[0] += 1; e[1] += x["k"] != 0; e[2] += max(x["v"].values()) - x["v"]["0"]
    ln.append("  per day (decisions / switched / $ gained at that decision): " +
              " ".join(f"d{k}:{v[0]}/{v[1]}/{v[2] / max(1, len(rows)):+,.0f}" for k, v in sorted(byday.items())))
    alt_gain = collections.defaultdict(list)
    for r in rows:
        for x in r["dec"]:
            for k, v in x["v"].items():
                if k != "0":
                    alt_gain[k].append(v - x["v"]["0"])
    for k, v in sorted(alt_gain.items()):
        v = np.array(v)
        ln.append(f"  A{k} vs A0 at the decision (one-step deviation value): n {len(v)}  mean {v.mean():+,.0f}  better {(v > 0).mean():.0%}  "
                  f"worse {(v < 0).mean():.0%}  equal {(v == 0).mean():.0%}")
    js = {}
    for r in rows:
        js[(r["job_secs"], r["job_n"])] = r
    secs = np.array([r["secs"] for r in rows]); cs = np.array([r["ctrl_secs"] for r in rows])
    ln.append(f"  runtime: {secs.mean():.0f} s/game wall per worker (control rollout alone {cs.mean():.0f} s/game); "
              f"per job: {np.mean([r['job_rows'] / r['job_n'] for r in js.values()]):,.0f} policy rows/game, "
              f"forward {np.sum([r['job_fwd'] for r in js.values()]) / np.sum([r['job_secs'] for r in js.values()]):.0%} of the time")
    print("\n".join(ln))
    return ln


def combine(*tags):
    """Approximate full-horizon ceiling from runs over disjoint day ranges from the SAME control line (e.g. days 1-7 and
    days 8-28): per world, control + the sum of the parts' gains.  Exact only if the earlier range's switches do not change
    the later range's gains (they are rare: ~1 switch per world-week); reported as an approximation."""
    import glob
    parts = []
    for tag in tags:
        parts.append({r["ep"]: r for fn in sorted(glob.glob(f"{R}/il/logs/lookahead_{tag}.json")) for r in json.load(open(fn))})
    common = sorted(set.intersection(*[set(p) for p in parts]))
    if not common:
        print("no common worlds"); return
    ctrl = np.array([parts[0][e]["ctrl_margin"] for e in common], float)
    for p in parts[1:]:
        bad = [e for e in common if p[e]["ctrl_margin"] != parts[0][e]["ctrl_margin"]]
        if bad:
            print(f"  warning: control differs between parts for {len(bad)} worlds {bad[:5]}")
    gains = np.array([[p[e]["margin"] - p[e]["ctrl_margin"] for e in common] for p in parts], float)
    tot = gains.sum(0); full = ctrl + tot; n = len(common)
    print(f"combine {' + '.join(tags)}: n {n} common worlds")
    for t, g in zip(tags, gains):
        print(f"  {t:28s} gain {g.mean():+8,.0f} (se {g.std(ddof=1) / np.sqrt(n) if n > 1 else 0:,.0f})  improved {(g > 0).sum()}")
    print(f"  {'sum (approx. full ceiling)':28s} gain {tot.mean():+8,.0f} (se {tot.std(ddof=1) / np.sqrt(n) if n > 1 else 0:,.0f}, median {np.median(tot):+,.0f})  "
          f"wins {(ctrl > 0).sum()} -> {(full > 0).sum()} of {n}  (control {ctrl.mean():+,.0f} -> {full.mean():+,.0f})")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        main_run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "report":
        report(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "combine":
        combine(*sys.argv[2:])
