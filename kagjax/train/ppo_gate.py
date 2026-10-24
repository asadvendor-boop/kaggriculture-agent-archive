"""FROZEN copy of the trainer exactly as it passed the pre-training gate (Sep 26, md5 570e33e3 of ppo.py at the time), used
by the overnight run (overnight.sh).  Do not edit: ppo.py is where development continues.

Self-play PPO for the Kaggriculture policy, pure JAX (rollouts and updates on the accelerator).

Replicates il/ppo_live.py + il/ppo.py (near-greedy PPO: T=0.2 unit / qty sampling, sharpened market fires KSHARP=10
around TAU, per-decision clipped surrogate over unit-op / qty / market-fire log-probs, KL penalty BETA to a frozen
reference policy, separate value network with warm-up, reward = change of the money margin / 2e4 (+ terminal
margin, + optional win bonus), GAE(gamma=1, LAM), minibatch Adam with grad-norm clip 1.0, PFSP opponent weights) on
top of the JAX engine (backend.py) with B games stepped in lockstep under jit + vmap, trajectories kept on device,
K opponent blocks (frozen snapshots / IL style clones / the current learner), updates on whole episodes (SEG=720) or
truncated segments with value bootstrap (design.md).

Usage:  ppo.py            (all settings from env vars, see Cfg below; RUN_DIR is created)
Outputs in RUN_DIR: log.txt, state_<it>.npz (resume: RESUME=<file>), pt/it<it>.pt (torch, il/export.py-ready),
best.pt (best greedy eval score), snaps/snap_<it>.npz (opponent pool), eval.jsonl.
"""
import os, sys, time, json, math, dataclasses
from functools import partial
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax
import jax.numpy as jnp
from jax import lax

import actions as A
import sample as S
import ckpt_io
from backend import load_backend, Obs

PS = jax.sharding.PartitionSpec
Q255 = jnp.asarray(np.arange(256, dtype=np.float32) / np.float32(255.0))       # correctly rounded, as numpy / torch
_CACHE = os.environ.get("JAX_CACHE_DIR", os.path.join(HERE, ".jax_cache"))      # persistent compilation cache
if _CACHE != "off":
    jax.config.update("jax_compilation_cache_dir", _CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)


def _env(name, default, typ=str):
    v = os.environ.get(name)
    return typ(v) if v is not None else default


@dataclasses.dataclass
class Cfg:
    backend: str = _env("KAGJAX_BACKEND", "")          # mock | real ("" = real if kagjax/engine.py exists)
    episode_slots: int = _env("EPISODE_SLOTS", 48, int)  # mock only (the real engine: 720)
    init: str = _env("INIT", "/Users/asad.ali/kaggriculture/il/ckpt/g20_u_dsm_s0.pt")
    ref: str = _env("REF", "")                          # KL anchor (default INIT)
    il: str = _env("IL", "")                            # frozen imitation model for the opponent pool (default REF)
    value_init: str = _env("VALUE_INIT", "policy")      # policy (INIT's vstate if present, else INIT) | <ckpt> | small:WxB
    resume: str = _env("RESUME", "")
    run_dir: str = _env("RUN_DIR", "runs/jax_selfplay")
    team: int = _env("TEAM", 10, int)
    T: float = _env("T", 0.2, float)
    ks: float = _env("KSHARP", 10.0, float)
    tau: str = _env("TAU", "0.5")                       # float or a per-head JSON file (export.py format)
    lr: float = _env("LR", 5e-6, float)
    vlr: float = _env("VLR", 1e-4, float)
    beta: float = _env("BETA", 0.05, float)
    clip: float = _env("CLIP", 0.2, float)
    lam: float = _env("LAM", 0.95, float)
    epochs: int = _env("EPOCHS", 2, int)
    mb: int = _env("MB", 2048, int)                     # minibatch (per device)
    vwarm: int = _env("VWARM", 0, int)                  # iterations with the policy frozen (value net only)
    winb: float = _env("WINB", 0.0, float)
    rscale: float = _env("RSCALE", 2e4, float)
    games: int = _env("B", 1024, int)                   # games per device (all in lockstep: the engine shares t)
    blocks: int = _env("K", 16, int)                    # opponent blocks (games per block = B / K)
    seg: int = _env("SEG", 720, int)                    # slots per update (divides 720; < 720 = truncated + bootstrap)
    iters: int = _env("ITERS", 1000, int)               # training segments
    opps: str = _env("OPPS", "LATEST=2,SELF=2,IL=4")    # opponent kinds and weights (+ TOP = recorded leader tapes)
    tape_dirs: str = _env("TAPE_DIRS", "/Users/asad.ali/kaggriculture/data/tapes_top12")   # TOP worlds (tapes.py)
    tape_max: int = _env("TAPE_MAX", 512, int)          # leader seats kept in the device tape bank
    il_teams: str = _env("IL_TEAMS", "10:3,5:2,3:2,4:2,0:2")   # style ids (weights) the frozen IL model plays as
    self_max: int = _env("SELF_MAX", 8, int)            # snapshot slots in the bank
    snap_every: int = _env("SNAP_EVERY", 40, int)
    pfsp: int = _env("PFSP", 1, int)
    opp_tau: float = _env("OPP_TAU", 0.5, float)
    opp_sample: int = _env("OPP_SAMPLE", 0, int)        # 0 = greedy opponents (as BCAgent), 1 = sampled like the learner
    opp_map: str = _env("OPP_MAP", "map")                # opponent blocks: map (sequential forwards, faster: 151 vs 275 ms at B=1024 K=4) | vmap
    midnight: int = _env("MIDNIGHT", 1, int)
    finalfix: int = _env("FINALFIX", 1, int)
    homing: int = _env("HOMING", 1, int)
    se_guard: int = _env("SE_GUARD", 0, int)
    sheep_cap: int = _env("SHEEP_CAP", 0, int)
    seed_budget: int = _env("SEED_BUDGET", 0, int)
    rollout_precision: str = _env("ROLLOUT_PRECISION", "default")   # default (TF32 on Ampere) | highest | bf16 (not advised)
    eval_precision: str = _env("EVAL_PRECISION", "highest")         # greedy evaluation / checkpoint selection: full fp32
    update_precision: str = _env("UPDATE_PRECISION", "default")     # default | highest
    uin_dtype: str = _env("UIN_DTYPE", "float32")        # storage of unit features (float32 = the exact rollout inputs)
    old_logp: str = _env("OLD_LOGP", "behavior")         # behavior: log-probs stored at rollout | recompute (ppo_live)
    diag_old: int = _env("DIAG_OLD", 0, int)             # also recompute them every iteration (max |stored - recomputed|)
    diag_every: int = _env("DIAG_EVERY", 10, int)        # ... and in any case every N iterations (0 = never)
    ref_online: int = _env("REF_ONLINE", 0, int)         # 1 = reference log-probs per minibatch (saves 2.8 KB/sample)
    kl_max: float = _env("KL_MAX", 1.0, float)           # stop the run when the KL to the reference exceeds this
    value_target: str = _env("VALUE_TARGET", "togo")     # togo: value net predicts the return-to-go (ppo_live);
                                                         # final: it predicts the final margin (the IL value head's target),
                                                         # V(s) = V'(s) - current margin (same TD errors, calibrated start)
    eval_every: int = _env("EVAL_EVERY", 40, int)
    eval_n: int = _env("EVAL_N", 256, int)              # games per eval opponent (per device)
    eval_opps: str = _env("EVAL_OPPS", "IL@10,IL@0,IL@5")
    ckpt_every: int = _env("CKPT_EVERY", 40, int)
    pt_every: int = _env("PT_EVERY", 40, int)            # torch .pt export every N iterations ...
    pt_every_min: float = _env("PT_EVERY_MIN", 0.0, float)  # ... and/or every M minutes of wall time (0 = off)
    seed: int = _env("SEED", 0, int)
    devices: int = _env("DEVICES", 0, int)              # 0 = all local devices (data parallel)
    log_every: int = _env("LOG_EVERY", 1, int)


# ================================================================================================ optimizer
def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return dict(m=z, v=jax.tree_util.tree_map(jnp.zeros_like, params), t=jnp.int32(0))


def clip_by_global_norm(g, max_norm=1.0):
    """torch.nn.utils.clip_grad_norm_ semantics."""
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(g)))
    coef = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda x: x * coef, g), norm


def adam_step(params, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    """torch.optim.Adam (no weight decay)."""
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, st["m"], g)
    v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, st["v"], g)
    bc1 = 1 - b1 ** t.astype(jnp.float32); bc2 = 1 - b2 ** t.astype(jnp.float32)
    new = jax.tree_util.tree_map(lambda p, m_, v_: p - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps), params, m, v)
    return new, dict(m=m, v=v, t=t)


def freeze_grads(g):
    """Buffers that are not torch parameters (coords) never change."""
    if "coords" in g:
        g = dict(g); g["coords"] = jnp.zeros_like(g["coords"])
    return g


# ================================================================================================ losses
def gae(rew, v, v_boot, done, valid, lam):
    """Time-major GAE with gamma = 1 (il/ppo_live.py): rew, v, done, valid [L, B]; v_boot [B] = V(s_L).
    delta_t = r_t + (1 - done_t) V_{t+1} - V_t, A_t = delta_t + lam (1 - done_t) A_{t+1}; invalid slots get A = 0
    (they only occur after a done step or before a game starts, so they never feed a valid step)."""
    def body(c, x):
        gnext, vnext = c
        r, v_, d, ok = x
        nt = 1.0 - d
        g = r + nt * vnext - v_ + lam * nt * gnext
        g = jnp.where(ok, g, 0.0)
        return (g, v_), g

    _, adv = lax.scan(body, (jnp.zeros_like(v_boot), v_boot), (rew, v, done.astype(jnp.float32), valid), reverse=True)
    return adv


def policy_loss(cur, d, a, old, refp, clip, beta):
    """il/ppo_live.py policy loss: per-decision clipped surrogates (unit ops, qty x 0.5, market fires) with the sample
    advantage, + beta * (KL(pi || ref) on unit ops of valid units + Bernoulli KL of the market fires).
    cur = sample.logp_terms of the current params; old = dict(lpu, lpq, lpm); refp = dict(rlpu_all, rpf)."""
    sv = d["valid"]
    a = a[:, None]
    ru = jnp.exp(cur["lpu"] - old["lpu"]); vm = cur["valid"] & sv[:, None]
    su = jnp.minimum(ru * a, jnp.clip(ru, 1 - clip, 1 + clip) * a)
    nvm = jnp.maximum(vm.sum(), 1)
    l_u = -jnp.where(vm, su, 0.0).sum() / nvm
    rq = jnp.exp(cur["lpq"] - old["lpq"]); qm = cur["qvalid"] & sv[:, None]
    sq = jnp.minimum(rq * a, jnp.clip(rq, 1 - clip, 1 + clip) * a)
    l_q = -jnp.where(qm, sq, 0.0).sum() / jnp.maximum(qm.sum(), 1)
    rm = jnp.exp(cur["lpm"] - old["lpm"]); mm = jnp.broadcast_to(sv[:, None], rm.shape)
    sm = jnp.minimum(rm * a, jnp.clip(rm, 1 - clip, 1 + clip) * a)
    l_m = -jnp.where(mm, sm, 0.0).sum() / jnp.maximum(mm.sum(), 1)
    pu = jnp.exp(cur["lpu_all"])
    klu = jnp.where(d["umask"], pu * (cur["lpu_all"] - refp["rlpu_all"]), 0.0).sum(-1)
    klu = jnp.where(vm, klu, 0.0).sum() / nvm
    pf, pr = cur["pf"], refp["rpf"]
    klm = jnp.where(mm, pf * jnp.log(pf / pr) + (1 - pf) * jnp.log((1 - pf) / (1 - pr)), 0.0).sum() / jnp.maximum(mm.sum(), 1)
    loss = l_u + 0.5 * l_q + l_m + beta * (klu + klm)
    ent = jnp.where(vm, -(jnp.where(d["umask"], pu * cur["lpu_all"], 0.0)).sum(-1), 0.0).sum() / nvm
    lr_ = cur["lpu"] - old["lpu"]
    akl = jnp.where(vm, (jnp.exp(lr_) - 1) - lr_, 0.0).sum() / nvm
    cf = jnp.where(vm, jnp.abs(ru - 1) > clip, False).sum() / nvm
    dev = lambda r, m: jnp.where(m, jnp.abs(r - 1), 0.0).max()
    frac = lambda r, m, e: jnp.where(m, jnp.abs(r - 1) > e, False).sum() / jnp.maximum(m.sum(), 1)
    return loss, dict(l_u=l_u, l_q=l_q, l_m=l_m, klu=klu, klm=klm, ent=ent, akl=akl, clipfrac=cf,
                      ru_mean=jnp.where(vm, ru, 0.0).sum() / nvm, ru_maxdev=dev(ru, vm), rq_maxdev=dev(rq, qm), rm_maxdev=dev(rm, mm),
                      ru_frac1e3=frac(ru, vm, 1e-3), ru_frac1e2=frac(ru, vm, 1e-2), rm_frac1e3=frac(rm, mm, 1e-3), rm_frac1e2=frac(rm, mm, 1e-2))


def value_loss(v, target, valid):
    sv = valid.astype(jnp.float32)
    return (jnp.square(v - target) * sv).sum() / jnp.maximum(sv.sum(), 1.0)


# ================================================================================================ opponent pool
START_SLOT = 2                                 # the warm-start policy (INIT), never overwritten (SNAP0 evaluation)


class Pool:
    """Host-side opponent pool.  Bank slots: 0 = LATEST (current learner, refreshed every iteration), 1 = IL (frozen
    imitation model), 2 = START (the warm-start policy, fixed), 3.. = SELF snapshots (ring of SELF_MAX).  SELF draws
    pick uniformly among START and the ring (past selves incl. the start).  Entries: (kind, bank slot, team)."""

    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.kinds = []
        for part in cfg.opps.split(","):
            k, w = part.split("=") if "=" in part else (part, "1")
            self.kinds.append((k, float(w)))
        self.il_teams = [(int(t.split(":")[0]), float(t.split(":")[1]) if ":" in t else 1.0) for t in cfg.il_teams.split(",")]
        self.snaps = []                       # [(bank slot, iteration)]
        self.hist = {}                        # kind name -> list of losses (1 = learner lost)

    def entry_names(self):
        return [k for k, _ in self.kinds]

    def add_snapshot(self, it):
        if len(self.snaps) < self.cfg.self_max:
            slot = START_SLOT + 1 + len(self.snaps); self.snaps.append((slot, it))
        else:                                 # ring: replace the oldest snapshot
            old = min(range(len(self.snaps)), key=lambda i: self.snaps[i][1])
            slot = self.snaps[old][0]; self.snaps[old] = (slot, it)
        return slot

    def draw(self):
        """-> (kind label, bank slot, team)."""
        names = [k for k, _ in self.kinds]
        w = np.array([w for _, w in self.kinds], np.float64)
        if self.cfg.pfsp:
            w = w * np.array([0.25 + (np.mean(self.hist[k][-64:]) if self.hist.get(k) else 0.5) for k in names])
        k = names[self.rng.choice(len(names), p=w / w.sum())]
        if k == "LATEST":
            return "LATEST", 0, self.cfg.team
        if k == "SELF":
            cand = [START_SLOT] + [s for s, _ in self.snaps]
            return "SELF", cand[self.rng.randint(len(cand))], self.cfg.team
        if k == "TOP":                        # recorded leader tapes (the network output of slot 1 is ignored)
            return "TOP", 1, self.cfg.team
        tw = np.array([w for _, w in self.il_teams]); t = self.il_teams[self.rng.choice(len(tw), p=tw / tw.sum())][0]
        return f"IL", 1, t

    def record(self, kind, margins):
        self.hist.setdefault(kind, []).extend([float(m <= 0) for m in margins])
        self.hist[kind] = self.hist[kind][-256:]


# ================================================================================================ trainer
class Trainer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        kind = cfg.backend or os.environ.get("KAGJAX_BACKEND") or None
        kw = {"episode_slots": cfg.episode_slots} if kind == "mock" else {}
        self.be = load_backend(kind, **kw)
        self.E = self.be.episode_slots
        assert self.E % cfg.seg == 0, f"SEG {cfg.seg} must divide the episode length {self.E}"
        assert cfg.games % cfg.blocks == 0, "B must be a multiple of K"
        self.bk = cfg.games // cfg.blocks
        n = jax.local_device_count() if cfg.devices <= 0 else cfg.devices
        self.ndev = n
        self.mesh = jax.make_mesh((n,), ("d",)) if n > 1 else None
        self.N = cfg.seg * cfg.games                               # samples per device per iteration
        assert self.N % cfg.mb == 0, f"MB {cfg.mb} must divide SEG*B = {self.N}"
        self.rules = A.RuleCfg(midnight=cfg.midnight, final_fix=cfg.finalfix, homing=cfg.homing, se_guard=cfg.se_guard,
                               sheep_cap=cfg.sheep_cap, seed_budget=cfg.seed_budget)
        if cfg.tau.endswith(".json"):
            tj = json.load(open(cfg.tau))
            self.tau = np.array([float(tj.get(h, tj.get("_default", 0.5))) for h, _ in A.MARKET_HEADS], np.float32)
        else:
            self.tau = np.full(A.N_HEADS, float(cfg.tau), np.float32)
        self.uin_dtype = {"float16": jnp.float16, "bfloat16": jnp.bfloat16}.get(cfg.uin_dtype, jnp.float32)
        kinds = [x.split("=")[0] for x in cfg.opps.split(",")]
        self.use_tapes = "TOP" in kinds or "TOP" in cfg.eval_opps.split(",")
        self.tape_np = {}
        if self.use_tapes:
            assert hasattr(self.be, "step_tapes"), "TOP (recorded tapes) needs the real engine backend"
            import tapes as TP
            dirs = [d for d in cfg.tape_dirs.split(",") if d]
            if "TOP" in kinds:
                self.tape_np["train"] = TP.build_bank(dirs, split="train", max_n=cfg.tape_max)
            if "TOP" in cfg.eval_opps.split(","):
                self.tape_np["eval"] = TP.build_bank(dirs, split="heldout", max_n=cfg.eval_n)
        self._build()

    @staticmethod
    def _sel(flag, a, b):
        """flag [B] bool: b where flag else a (pytrees with leading B)."""
        return jax.tree_util.tree_map(lambda x, y: jnp.where(flag.reshape((-1,) + (1,) * (x.ndim - 1)), y, x), a, b)

    def _fwd(self, params, obs, team, precision):
        pl = Q255[obs.planes.astype(jnp.int32)]            # exact uint8 / 255 (XLA may rewrite x / c as x * (1 / c))
        return self.be.net_apply(params, pl, obs.glob, obs.pos, obs.uin.astype(jnp.float32), team, precision)

    # -------------------------------------------------------------------------------------------- rollout
    def rollout_local(self, params, bank, cs, tapes, opp_idx, opp_team, carry, key, *, L, learner_greedy, record, axis, precision=None):
        """L lockstep steps of the local games.  carry = (engine state, prev ops [B,2,16], learner seat [B], tape id [B]
        (-1 = a normal game; >= 0: a recorded world whose other seat replays the tape; tapes = device bank or None)).
        Returns carry, traj (time-major, learner seat), (finished [B], final margin [B]), bootstrap obs."""
        cfg, be = self.cfg, self.be
        if axis is not None:
            key = jax.random.fold_in(key, lax.axis_index(axis))
        B = carry[2].shape[0]
        K = opp_idx.shape[0]; Bk = B // K
        tau_l = jnp.asarray(self.tau)
        tau_o = jnp.full((A.N_HEADS,), cfg.opp_tau, jnp.float32)
        team_l = jnp.full((B,), cfg.team, jnp.int32)
        rules = self.rules
        prec_r = precision or cfg.rollout_precision

        def margin(mon, lseat):
            return (jnp.take_along_axis(mon, lseat[:, None], 1) - jnp.take_along_axis(mon, 1 - lseat[:, None], 1))[:, 0]

        def step(c, k):
            state, prev, lseat, tid = c
            kL, kO, kW = jax.random.split(k, 3)
            t = be.t(state)
            one = lseat == 1
            raw0, raw1 = be.observe(state, 0, cs), be.observe(state, 1, cs)
            rawL, rawO = self._sel(one, raw0, raw1), self._sel(one, raw1, raw0)
            prevL = jnp.where(one[:, None], prev[:, 1], prev[:, 0]); prevO = jnp.where(one[:, None], prev[:, 0], prev[:, 1])
            oL, oO = be.encode(rawL, prevL, cs), be.encode(rawO, prevO, cs)
            mL, mO = be.legal(rawL), be.legal(rawO)
            vL, vO = be.view(rawL), be.view(rawO)
            outL = self._fwd(params, oL, team_l, prec_r)

            def opp_block(xs):
                idx, tm, ob = xs
                prm = jax.tree_util.tree_map(lambda b: b[idx], bank)
                return self._fwd(prm, ob, jnp.full((Bk,), tm, jnp.int32), prec_r)

            oOb = jax.tree_util.tree_map(lambda x: x.reshape((K, Bk) + x.shape[1:]), oO)
            if cfg.opp_map == "vmap":                             # one batched (grouped-conv) forward over all blocks
                outO = jax.vmap(opp_block)((opp_idx, opp_team, oOb))
            else:                                                 # K sequential forwards (less memory)
                outO = lax.map(opp_block, (opp_idx, opp_team, oOb))
            outO = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), outO)
            dL = S.decide(outL, mL, oL.nu, kL, cfg.T, cfg.ks, tau_l, learner_greedy)
            dO = S.decide(outO, mO, oO.nu, kO, cfg.T, cfg.ks, tau_o, not cfg.opp_sample)
            build = jax.vmap(lambda d, nu, v: S.build_act(d, nu, v, rules))
            aL, aO = build(dL, oL.nu, vL), build(dO, oO.nu, vO)
            a0, a1 = self._sel(one, aL, aO), self._sel(one, aO, aL)
            mb = margin(be.money(state), lseat)
            ns = be.step(state, a0, a1, cs) if tapes is None else be.step_tapes(state, a0, a1, tid, 1 - lseat, tapes, kW, cs)
            valid = t <= be.last_step
            done = t == be.last_step
            ma = margin(be.money(ns), lseat)                          # money == the final reward once done
            rew = jnp.where(valid, (ma - mb) / cfg.rscale, 0.0) + jnp.where(done, cfg.winb * jnp.sign(ma), 0.0)
            kk = jnp.arange(A.NU)[None, :]
            pL = jnp.where(kk < oL.nu[:, None], dL.op, -1); pO = jnp.where(kk < oO.nu[:, None], dO.op, -1)
            p0 = jnp.where(one[:, None], pO, pL); p1 = jnp.where(one[:, None], pL, pO)
            nprev = jnp.where(valid, jnp.stack([p0, p1], 1), prev)
            out = dict(fin=jnp.where(done, ma, 0.0), done=jnp.broadcast_to(done, (B,)))
            if record:
                bl = S.logp_terms(outL, mL, dL.uact, dL.qact, dL.fire, cfg.T, cfg.ks, tau_l)     # behavior log-probs
                out.update(planes=oL.planes, glob=oL.glob, pos=oL.pos.astype(jnp.int8), uin=oL.uin.astype(self.uin_dtype),
                           umask=mL, uact=dL.uact, qact=dL.qact, mfire=dL.fire, rew=rew, valid=jnp.broadcast_to(valid, (B,)),
                           blpu=bl["lpu"], blpq=bl["lpq"], blpm=bl["lpm"], marg=mb / cfg.rscale)
            return (ns, nprev, lseat, tid), out

        carry, tr = lax.scan(step, tuple(carry), jax.random.split(key, L))
        finished = tr["done"].any(0)
        fin_margin = tr["fin"].sum(0)
        if record:
            state, prev, lseat, _ = carry
            one = lseat == 1
            rawL = self._sel(one, be.observe(state, 0, cs), be.observe(state, 1, cs))
            boot = (be.encode(rawL, jnp.where(one[:, None], prev[:, 1], prev[:, 0]), cs), margin(be.money(state), lseat) / cfg.rscale)
        else:
            boot = (jnp.zeros((B,), jnp.int8), jnp.zeros((B,), jnp.float32))      # placeholder (fixed output structure)
        return carry, tr, (finished, fin_margin), boot

    # -------------------------------------------------------------------------------------------- update
    def update_local(self, params, vparams, opt_p, opt_v, ref, tr, boot, key, *, train_policy, axis, diag=False):
        cfg = self.cfg
        if axis is not None:
            key = jax.random.fold_in(key, lax.axis_index(axis))
        L, B = tr["rew"].shape
        N = L * B
        C = cfg.mb
        team = jnp.full((C,), cfg.team, jnp.int32)
        tau = jnp.asarray(self.tau)
        prec = cfg.update_precision
        flat = {k: tr[k].reshape((N,) + tr[k].shape[2:]) for k in ("planes", "glob", "pos", "uin", "umask", "uact", "qact", "mfire", "valid")}

        def obs_of(d):
            return Obs(d["planes"], d["glob"], d["pos"].astype(jnp.int32), d["uin"].astype(jnp.float32), None)

        def terms(p, d):
            out = self._fwd(p, obs_of(d), team, prec)
            return out, S.logp_terms(out, d["umask"], d["uact"], d["qact"], d["mfire"], cfg.T, cfg.ks, tau)

        beh = {k: tr["b" + k].reshape((N,) + tr["b" + k].shape[2:]) for k in ("lpu", "lpq", "lpm")}
        need_old = cfg.old_logp == "recompute" or cfg.diag_old or diag

        def pre(idx):                                            # recomputed old / reference terms and values, chunks of MB
            d = {k: v[idx] for k, v in flat.items()}
            r_ = {}
            if need_old:
                _, o = terms(params, d)
                r_.update(lpu=o["lpu"], lpq=o["lpq"], lpm=o["lpm"])
            if not cfg.ref_online:
                _, r = terms(ref, d)
                r_.update(rlpu_all=r["lpu_all"], rpf=r["pf"])
            r_["v"] = self._fwd(vparams, obs_of(d), team, prec)["v"]
            return r_

        pc = lax.map(pre, jnp.arange(N).reshape(N // C, C))
        pc = {k: v.reshape((N,) + v.shape[2:]) for k, v in pc.items()}
        diag = {}
        if need_old:                                             # stored (rollout) vs recomputed (update) log-probs
            fv = flat["valid"]
            for k, m in (("lpu", (flat["uact"] >= 0) & fv[:, None]), ("lpq", (flat["qact"] >= 0) & fv[:, None]),
                         ("lpm", jnp.broadcast_to(fv[:, None], beh["lpm"].shape))):
                diag["old_diff_" + k] = jnp.where(m, jnp.abs(beh[k] - pc[k]), 0.0).max()
        oldsrc = {k: pc[k] for k in ("lpu", "lpq", "lpm")} if cfg.old_logp == "recompute" else beh
        bobs, bmarg = boot
        vb = self._fwd(vparams, Obs(bobs.planes, bobs.glob, bobs.pos, bobs.uin, None), jnp.full((B,), cfg.team, jnp.int32), prec)["v"]
        fin_param = cfg.value_target == "final"
        marg = tr["marg"].reshape(N)
        if fin_param:                                            # V = V' - margin (V' = predicted final margin)
            vb = vb - bmarg
            pc["v"] = pc["v"] - marg
        vt = pc["v"].reshape(L, B)
        rew, done, valid = tr["rew"], tr["done"], tr["valid"]
        adv = gae(rew, vt, vb, done, valid, cfg.lam)
        ret = (adv + vt).reshape(N)
        adv = adv.reshape(N)
        vf = flat["valid"].astype(jnp.float32)
        s1, s0 = (adv * vf).sum(), vf.sum()
        if axis is not None:
            s1, s0 = lax.psum(s1, axis), lax.psum(s0, axis)
        mean = s1 / jnp.maximum(s0, 1.0)
        s2 = (jnp.square(adv - mean) * vf).sum()
        if axis is not None:
            s2 = lax.psum(s2, axis)
        std = jnp.sqrt(s2 / jnp.maximum(s0 - 1.0, 1.0))           # torch .std() is unbiased
        advn = (adv - mean) / (std + 1e-8)

        def ploss(p, d, a, old, refp):
            _, cur = terms(p, d)
            if cfg.ref_online:
                _, r = terms(ref, d)
                refp = dict(rlpu_all=lax.stop_gradient(r["lpu_all"]), rpf=lax.stop_gradient(r["pf"]))
            return policy_loss(cur, d, a, old, refp, cfg.clip, cfg.beta)

        def vloss(vp, d, target):                                # target: return-to-go (togo) or final margin
            return value_loss(self._fwd(vp, obs_of(d), team, prec)["v"], target, d["valid"])

        vtarget = ret + marg if fin_param else ret

        def mb_body(c, idx):
            p, vp, op, ov = c
            d = {k: v[idx] for k, v in flat.items()}
            old = {k: oldsrc[k][idx] for k in ("lpu", "lpq", "lpm")}
            refp = {} if cfg.ref_online else {k: pc[k][idx] for k in ("rlpu_all", "rpf")}
            (l, st), g = jax.value_and_grad(ploss, has_aux=True)(p, d, advn[idx], old, refp)
            lv, gv = jax.value_and_grad(vloss)(vp, d, vtarget[idx])
            if axis is not None:
                g, gv = lax.pmean(g, axis), lax.pmean(gv, axis)
            gv, vn = clip_by_global_norm(freeze_grads(gv), 1.0)
            vp, ov = adam_step(vp, gv, ov, cfg.vlr)
            if train_policy:
                g, gn = clip_by_global_norm(freeze_grads(g), 1.0)
                p, op = adam_step(p, g, op, cfg.lr)
            else:
                gn = jnp.float32(0)
            return (p, vp, op, ov), dict(st, loss=l, l_v=lv, gnorm=gn, vgnorm=vn)

        def ep_body(c, k):
            perm = jax.random.permutation(k, N).reshape(N // C, C)
            return lax.scan(mb_body, c, perm)

        (params, vparams, opt_p, opt_v), st = lax.scan(ep_body, (params, vparams, opt_p, opt_v), jax.random.split(key, cfg.epochs))
        ep_loss, ep_lv = st["loss"].mean(1), st["l_v"].mean(1)                # [epochs]: the fit on this batch per epoch
        first = {"r0_" + k: st[k][0, 0] for k in ("ru_maxdev", "rq_maxdev", "rm_maxdev", "clipfrac", "klu", "klm")}   # 1st minibatch
        st = {k: (v.max() if k.endswith("maxdev") else v.mean()) for k, v in st.items()}
        st.update(loss_ep0=ep_loss[0], loss_epN=ep_loss[-1], lv_ep0=ep_lv[0], lv_epN=ep_lv[-1], **first, **diag)
        leaves = jax.tree_util.tree_leaves((params, vparams))
        st["finite"] = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]) &
                               jnp.all(jnp.stack([jnp.isfinite(v) for v in jax.tree_util.tree_leaves(st)])))
        rm = (ret * vf).sum() / jnp.maximum(vf.sum(), 1)
        st.update(ret_mean=rm, v_res2=(jnp.where(flat["valid"], ret - pc["v"], 0.0) ** 2).sum() / jnp.maximum(vf.sum(), 1),
                  ret_var=((ret - rm) ** 2 * vf).sum() / jnp.maximum(vf.sum(), 1), rew_sum=(rew * valid).sum() / B)
        if axis is not None:
            red = lambda k, v: (lax.pmin(v.astype(jnp.float32), axis) if k == "finite" else
                                lax.pmax(v, axis) if (k.endswith("maxdev") or k.startswith("old_diff")) else lax.pmean(v, axis))
            st = {k: red(k, v) for k, v in st.items()}
        return params, vparams, opt_p, opt_v, st

    # -------------------------------------------------------------------------------------------- jit / shard_map
    def _build(self):
        be = self.be
        if self.mesh is None:
            self._rollout = jax.jit(partial(self.rollout_local, axis=None), static_argnames=("L", "learner_greedy", "record", "precision"))
            self._update = jax.jit(partial(self.update_local, axis=None), static_argnames=("train_policy", "diag"), donate_argnums=(0, 1, 2, 3))
        else:
            m = self.mesh
            rep, bat = PS(), PS("d")
            cspec = (be.state_spec("d"), bat, bat, bat)

            def ro(params, bank, cs, tapes, opp_idx, opp_team, carry, key, L, learner_greedy, record, precision=None):
                f = partial(self.rollout_local, L=L, learner_greedy=learner_greedy, record=record, axis="d", precision=precision)
                return jax.shard_map(f, mesh=m, in_specs=(rep, rep, rep, rep, rep, rep, cspec, rep),
                                     out_specs=(cspec, PS(None, "d"), (bat, bat), bat), check_vma=False)(
                    params, bank, cs, tapes, opp_idx, opp_team, tuple(carry), key)

            def up(params, vparams, opt_p, opt_v, ref, tr, boot, key, train_policy, diag=False):
                f = partial(self.update_local, train_policy=train_policy, axis="d", diag=diag)
                return jax.shard_map(f, mesh=m, in_specs=(rep, rep, rep, rep, rep, PS(None, "d"), bat, rep),
                                     out_specs=(rep, rep, rep, rep, rep), check_vma=False)(params, vparams, opt_p, opt_v, ref, tr, boot, key)

            self._rollout = jax.jit(ro, static_argnames=("L", "learner_greedy", "record", "precision"))
            self._update = jax.jit(up, static_argnames=("train_policy", "diag"), donate_argnums=(0, 1, 2, 3))
        self._set_bank = jax.jit(lambda bank, p, s: jax.tree_util.tree_map(lambda b, x: b.at[s].set(x), bank, p), donate_argnums=(0,))
        self._reset = jax.jit(be.reset, static_argnums=(1,))

    def put(self, x, spec=None):
        """Device placement: replicated (spec None) or sharded over the mesh."""
        if self.mesh is None:
            return x
        sh = jax.sharding.NamedSharding(self.mesh, spec if spec is not None else PS())
        return jax.device_put(x, sh)

    def fresh_games(self, key, n, tid=None, lseat=None):
        """n new games; the learner seat alternates game by game, except in tape worlds (tid >= 0: fixed by the tape)."""
        st = self._reset(key, n)
        tid = -np.ones(n, np.int32) if tid is None else np.asarray(tid, np.int32)
        ls = (np.arange(n) % 2).astype(np.int32) if lseat is None else np.where(tid >= 0, np.asarray(lseat), np.arange(n) % 2).astype(np.int32)
        carry = (st, -jnp.ones((n, 2, A.NU), jnp.int32), jnp.asarray(ls), jnp.asarray(tid))
        if self.mesh is None:
            return carry
        specs = (self.be.state_spec("d"), PS("d"), PS("d"), PS("d"))
        return jax.tree_util.tree_map(lambda x, sp: self.put(x, sp), carry, specs, is_leaf=lambda z: isinstance(z, PS))


def game_block_index(cfg, ndev):
    """Global game index -> block id (game axis = [device][block][game in block])."""
    g = np.arange(ndev * cfg.games)
    return (g % cfg.games) // (cfg.games // cfg.blocks)


# ================================================================================================ main loop
def log_line(path, s):
    print(s, flush=True)
    with open(path, "a") as f:
        f.write(s + "\n")


def mem_note():
    try:
        st = jax.devices()[0].memory_stats() or {}
        return f" | mem peak {st.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB"
    except Exception:
        return ""


def main(cfg: Cfg = None):
    cfg = cfg or Cfg()
    for d in ("", "pt", "snaps"):
        os.makedirs(os.path.join(cfg.run_dir, d), exist_ok=True)
    LOG = os.path.join(cfg.run_dir, "log.txt")
    tr = Trainer(cfg)
    be, D = tr.be, tr.ndev
    log_line(LOG, f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} devices {jax.devices()} x{D} backend {type(be).__name__} | "
                  f"cfg {json.dumps(dataclasses.asdict(cfg))}")
    rng = np.random.RandomState(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # ---- params: policy, KL reference, frozen IL opponent, value net, optimizers
    tiny = cfg.init.startswith("tiny:")                       # tiny:WxB = random small net (mock / smoke tests)
    if tiny:
        w, b = [int(x) for x in cfg.init.split(":")[1].split("x")]
        params = be.init_params(jax.random.PRNGKey(cfg.seed + 1), width=w, blocks=b)
    else:
        params, _ = be.load_params(cfg.init, "state")
    copy = lambda t: jax.tree_util.tree_map(jnp.array, t)     # the update donates params: never alias them
    start = copy(params)                                      # the warm start, fixed (bank slot START, SNAP0 eval)
    ref = copy(params) if not cfg.ref else be.load_params(cfg.ref, "state")[0]
    il = ref if not cfg.il else be.load_params(cfg.il, "state")[0]
    if cfg.value_init.startswith("small:"):
        w, b = [int(x) for x in cfg.value_init.split(":")[1].split("x")]
        vparams = be.init_params(jax.random.PRNGKey(cfg.seed + 2), width=w, blocks=b, n_teams=ckpt_io.arch_of(params)["n_teams"])
    elif cfg.value_init == "policy":
        vparams = None
        if not tiny:
            try:
                vparams = be.load_params(cfg.init, "vstate")[0]            # an RL checkpoint carries its value net
            except Exception:
                vparams = None
        vparams = vparams if vparams is not None else copy(params)
    else:
        try:
            vparams = be.load_params(cfg.value_init, "vstate")[0]
        except Exception:
            vparams = be.load_params(cfg.value_init, "state")[0]
    opt_p, opt_v = adam_init(params), adam_init(vparams)
    it0 = 0
    pool = Pool(cfg, rng)
    if cfg.resume:
        params, vparams, opt_p, opt_v, meta = ckpt_io.load_train_state(cfg.resume)
        it0 = int(meta["it"])
        pool.snaps = [tuple(x) for x in meta.get("snaps", []) if x[0] > START_SLOT]; pool.hist = meta.get("hist", {})
        rng = np.random.RandomState([cfg.seed, it0]); pool.rng = rng          # new random streams after a resume
        key = jax.random.fold_in(key, it0)
        log_line(LOG, f"[resume] {cfg.resume} at iteration {it0} (random streams advanced; START = {cfg.init})")
    log_line(LOG, f"[model] policy {ckpt_io.arch_of(params)} value {ckpt_io.arch_of(vparams)} ref {cfg.ref or 'INIT'} il {cfg.il or 'REF'}")

    # ---- opponent bank: 0 LATEST, 1 IL, 2 START, 3.. SELF snapshots (one stacked pytree: IL must share the architecture)
    assert ckpt_io.arch_of(il) == ckpt_io.arch_of(params), f"IL {ckpt_io.arch_of(il)} != policy {ckpt_io.arch_of(params)}"
    bank = jax.tree_util.tree_map(lambda p, q, s0: jnp.stack([p, q, s0] + [p] * cfg.self_max), params, il, start)
    del start
    if pool.snaps:                            # restore snapshot params into the bank (skip any that no longer exist)
        keep = []
        for s, it_ in pool.snaps:
            fn = os.path.join(cfg.run_dir, "snaps", f"snap_{it_}.npz")
            if os.path.exists(fn):
                bank = tr._set_bank(bank, ckpt_io.load_params_npz(fn)[0], s); keep.append((s, it_))
            else:
                log_line(LOG, f"[resume] snapshot {fn} is gone (left the ring after this state was saved); not in the pool")
        pool.snaps = keep
    params, vparams, ref, opt_p, opt_v, bank = [tr.put(x) for x in (params, vparams, ref, opt_p, opt_v, bank)]
    cs = tr.put(be.consts())                                                  # engine / encoder tables (jit arguments)
    dev_bank = lambda b: tr.put({k: jnp.asarray(v) for k, v in b.items() if k != "meta"})
    tapes_train = dev_bank(tr.tape_np["train"]) if "train" in tr.tape_np else None
    tapes_eval = dev_bank(tr.tape_np["eval"]) if "eval" in tr.tape_np else None

    K = cfg.blocks
    blk_of_game = game_block_index(cfg, D)
    blk = [pool.draw() for _ in range(K)]

    def new_games(k_):
        """Fresh games for the current block assignment: TOP blocks get random training tapes (learner seat = the
        tape's non-leader seat)."""
        tid = -np.ones(D * cfg.games, np.int32); ls = None
        if tapes_train is not None:
            lsb = tr.tape_np["train"]["lseat"]
            for b_ in range(K):
                if blk[b_][0] == "TOP":
                    g_ = np.flatnonzero(blk_of_game == b_)
                    tid[g_] = rng.randint(len(lsb), size=len(g_))
            ls = np.where(tid >= 0, lsb[np.maximum(tid, 0)], 0)
        return tr.fresh_games(k_, D * cfg.games, tid, ls)

    key, k0 = jax.random.split(key)
    carry = new_games(k0)

    def opp_arrays(entries):
        return tr.put(jnp.asarray([b[1] for b in entries], jnp.int32)), tr.put(jnp.asarray([b[2] for b in entries], jnp.int32))

    # ---- evaluation: fixed worlds, greedy learner, paired vs the first evaluation
    eval_specs = [s for s in cfg.eval_opps.split(",") if s]
    base_eval = {}

    def evaluate(it):
        res, t_ev = {}, time.time()
        for sp in eval_specs:
            tp, tid_e, ls_e = None, None, None
            if sp.startswith("IL@"):
                ent = [("IL", 1, int(sp[3:]))]
            elif sp == "SNAP0":                                               # the warm start (fixed bank slot)
                ent = [("SELF", START_SLOT, cfg.team)]
            elif sp == "TOP" and tapes_eval is not None:                     # held-out recorded leader worlds
                ent = [("TOP", 1, cfg.team)]
                ne = len(tr.tape_np["eval"]["lseat"])
                tid_e = np.arange(D * cfg.eval_n) % ne; ls_e = tr.tape_np["eval"]["lseat"][tid_e]; tp = tapes_eval
            else:
                continue
            ev = tr.fresh_games(jax.random.PRNGKey(10_000 + cfg.seed), D * cfg.eval_n, tid_e, ls_e)   # same worlds every time
            oi, ot = opp_arrays(ent)
            _, _, (fin, mar), _ = tr._rollout(params, bank, cs, tp, oi, ot, ev, jax.random.PRNGKey(20_000), L=tr.E, learner_greedy=True, record=False,
                                              precision=cfg.eval_precision)
            res[sp] = np.asarray(mar)
        line = f"[eval it {it}] {time.time() - t_ev:.0f}s"
        score = 0.0
        for sp, m in res.items():
            line += f" | vs {sp}: W {(m > 0).mean():.3f} margin {m.mean():+,.0f}"
            if sp in base_eval:
                d_ = m - base_eval[sp]
                line += f" (paired {d_.mean():+,.0f} se {d_.std() / np.sqrt(len(d_)):,.0f})"
            score += (m > 0).mean()
        if not base_eval:
            base_eval.update(res)
        score /= max(1, len(res))
        log_line(LOG, line + f" | score {score:.3f}")
        with open(os.path.join(cfg.run_dir, "eval.jsonl"), "a") as f:
            f.write(json.dumps({"it": it, "score": score, **{k: [float((v > 0).mean()), float(v.mean())] for k, v in res.items()}}) + "\n")
        return score

    t0 = time.time()
    best = evaluate(it0) if cfg.eval_every > 0 else -1.0
    samples = 0
    snap_due = False
    last_pt = time.time()
    for it in range(it0 + 1, cfg.iters + 1):
        ti = time.time()
        bank = tr._set_bank(bank, params, 0)                                 # LATEST = current learner
        oi, ot = opp_arrays(blk)
        key, kr, ku, kn = jax.random.split(key, 4)
        carry, traj, (fin, mar), boot = tr._rollout(params, bank, cs, tapes_train, oi, ot, carry, kr, L=cfg.seg, learner_greedy=False, record=True)
        jax.block_until_ready(traj["rew"]); t_roll = time.time() - ti
        diag = bool(cfg.diag_every) and it % cfg.diag_every == 1
        params, vparams, opt_p, opt_v, st = tr._update(params, vparams, opt_p, opt_v, ref, traj, boot, ku, train_policy=it > cfg.vwarm, diag=diag)
        st = {k: float(v) for k, v in st.items()}
        t_upd = time.time() - ti - t_roll
        if st["finite"] < 1:
            log_line(LOG, f"[STOP] non-finite params or loss after the update of iteration {it}: {json.dumps(st)}")
            raise SystemExit(3)
        if max(st["klu"], st["klm"]) > cfg.kl_max:
            log_line(LOG, f"[STOP] KL to the reference {st['klu']:.3f}/{st['klm']:.3f} > KL_MAX {cfg.kl_max} at iteration {it}")
            raise SystemExit(4)
        with open(os.path.join(cfg.run_dir, "stats.jsonl"), "a") as f:
            f.write(json.dumps({"it": it, "t_roll": t_roll, "t_upd": t_upd, **st}) + "\n")
        del traj, boot
        per = {}
        if int(be.t(carry[0])) > be.last_step:                               # episodes over: score, new games / opponents
            mar_h = np.asarray(mar)
            for k in range(K):
                ms = mar_h[blk_of_game == k]
                kind = blk[k][0] if blk[k][0] != "IL" else f"IL@{blk[k][2]}"
                per.setdefault(kind, []).extend(ms.tolist())
                pool.record(blk[k][0], ms)
            blk = [pool.draw() for _ in range(K)]
            carry = new_games(kn)
        samples += tr.N * D
        dt = time.time() - ti
        if it % cfg.log_every == 0 or per:
            log_line(LOG, f"it {it} {time.time() - t0:.0f}s ({dt:.1f}s: rollout {t_roll:.1f} update {t_upd:.1f}; {tr.N * D / dt:,.0f} samples/s) | "
                          + " ".join(f"{k} {np.mean(v):+,.0f} W{np.mean(np.array(v) > 0):.0%}({len(v)})" for k, v in sorted(per.items()))
                          + f" | loss {st['loss']:+.4f} u/q/m {st['l_u']:+.4f}/{st['l_q']:+.4f}/{st['l_m']:+.4f} kl u/m {st['klu']:.4f}/{st['klm']:.4f}"
                          + f" ent {st['ent']:.3f} akl {st['akl']:.5f} clip {st['clipfrac']:.3f} | epochs loss {st['loss_ep0']:+.4f}->{st['loss_epN']:+.4f}"
                          + f" v {st['lv_ep0']:.4f}->{st['lv_epN']:.4f} | v {st['l_v']:.4f} (res {st['v_res2']:.4f} / var {st['ret_var']:.4f})"
                          + f" rew/game {st['rew_sum']:+.4f} | gn {st['gnorm']:.3f}/{st['vgnorm']:.3f}" + mem_note())
        if cfg.snap_every and it % cfg.snap_every == 0:
            snap_due = True
        if snap_due and per:                                                  # only at an episode boundary (no opponent
            snap_due = False                                                  # changes inside a running game)
            before = {i for _, i in pool.snaps}
            s_ = pool.add_snapshot(it)
            bank = tr._set_bank(bank, params, s_)
            ckpt_io.save_params_npz(os.path.join(cfg.run_dir, "snaps", f"snap_{it}.npz"), params, {"it": it})
            for gone in before - {i for _, i in pool.snaps}:                  # left the ring: free the disk
                try:                                                           # (a resume from an older state skips it)
                    os.remove(os.path.join(cfg.run_dir, "snaps", f"snap_{gone}.npz"))
                except OSError:
                    pass
        if cfg.eval_every and it % cfg.eval_every == 0:
            sc = evaluate(it)
            if sc > best:
                best = sc
                ckpt_io.save_pt(os.path.join(cfg.run_dir, "best.pt"), params, vparams, team=cfg.team, it=it, score=sc)
                log_line(LOG, f"  new best eval score {sc:.3f} -> best.pt")
        due_t = cfg.pt_every_min > 0 and time.time() - last_pt >= 60 * cfg.pt_every_min
        if (cfg.pt_every and it % cfg.pt_every == 0) or due_t or it == cfg.iters:     # policy only (export.py needs "state")
            ckpt_io.save_pt(os.path.join(cfg.run_dir, "pt", f"it{it}.pt"), params, vparams if it == cfg.iters else None, team=cfg.team, it=it)
            last_pt = time.time()
            log_line(LOG, f"  exported pt/it{it}.pt (il/export.py-ready)")
        if (cfg.ckpt_every and it % cfg.ckpt_every == 0) or it == cfg.iters:
            meta = {"it": it, "team": cfg.team, "cfg": dataclasses.asdict(cfg), "snaps": pool.snaps, "hist": pool.hist, "best": best}
            ckpt_io.save_train_state(os.path.join(cfg.run_dir, f"state_{it}.npz"), params, vparams, opt_p, opt_v, meta)
            olds = sorted(int(f[6:-4]) for f in os.listdir(cfg.run_dir) if f.startswith("state_") and f.endswith(".npz"))
            for o in olds[:-2]:
                os.remove(os.path.join(cfg.run_dir, f"state_{o}.npz"))
    log_line(LOG, f"[done] {cfg.iters - it0} iterations, {samples:,} samples, {time.time() - t0:.0f}s")
    return dict(params=params, vparams=vparams, trainer=tr, best=best, base_eval=base_eval)


if __name__ == "__main__":
    main()
