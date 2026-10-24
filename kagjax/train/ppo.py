"""Self-play PPO for the Kaggriculture policy, pure JAX (rollouts and updates on the accelerator).

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
    init: str = _env("INIT", "/Users/asad.ali/kaggriculture/il/ckpt/g20_u_dsm_s0.pt")   # a checkpoint | tiny:WxB (mock) |
                                                        # random[:<template ckpt>]: fresh torch-default init with the
                                                        # template's architecture (default template: IL, else REF)
    ref: str = _env("REF", "")                          # KL anchor (default INIT)
    il: str = _env("IL", "")                            # frozen imitation model for the opponent pool (default REF)
    value_init: str = _env("VALUE_INIT", "policy")      # policy (INIT's vstate if present, else INIT) | <ckpt> | small:WxB
                                                        # | head: the policy's own value head on its stop-gradient market
                                                        #   features (no value network: V is free with the policy forward)
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
    mb: int = _env("MB", 2048, int)                     # optimizer minibatch (per device)
    micro: int = _env("MICRO", 0, int)                  # memory microbatch (gradient accumulation; 0 = MB); same maths
    ref_frac: float = _env("REF_FRAC", 1.0, float)      # share of samples that get reference (KL anchor) terms; the KL is
                                                        # reweighted by 1 / REF_FRAC (unbiased); 1 = every sample
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
    opp_map: str = _env("OPP_MAP", "map")                # opponent forwards: map (one per param group, sequential) | vmap
    opp_groups: int = _env("OPP_GROUPS", 0, int)        # param groups per episode: 0 = one per block (K); G = the K blocks
                                                        # are split into G groups that share one bank slot (per-row style
                                                        # ids), i.e. G opponent forwards of B/G rows per step
    midnight: int = _env("MIDNIGHT", 1, int)
    finalfix: int = _env("FINALFIX", 1, int)
    homing: int = _env("HOMING", 1, int)
    se_guard: int = _env("SE_GUARD", 0, int)
    sheep_cap: int = _env("SHEEP_CAP", 0, int)
    seed_budget: int = _env("SEED_BUDGET", 0, int)
    rollout_precision: str = _env("ROLLOUT_PRECISION", "default")   # default (TF32 on Ampere) | highest | bf16 (not advised)
    eval_precision: str = _env("EVAL_PRECISION", "highest")         # greedy evaluation / checkpoint selection: full fp32
    opp_precision: str = _env("OPP_PRECISION", "")                  # opponents' rollout forwards ("" = ROLLOUT_PRECISION);
    #                                                                 bf16: ~2.5x faster, opponent actions are not trained on
    update_precision: str = _env("UPDATE_PRECISION", "default")     # default (TF32) | highest | bf16 (bf16 compute, fp32
                                                                    # master weights / Adam / log-softmax / ratios / losses)
                                                                    # | bf16grad (values of the TF32 forward - ratios, clip,
                                                                    # KL exactly as default - gradients of a bf16 forward)
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
    eval_base: str = _env("EVAL_BASE", "")               # npz of a previous run's iteration-0 evaluation (same INIT, IL,
                                                        # worlds): reused as the paired baseline instead of re-evaluating
    ckpt_every: int = _env("CKPT_EVERY", 40, int)
    pt_every: int = _env("PT_EVERY", 40, int)            # torch .pt export every N iterations ...
    pt_every_min: float = _env("PT_EVERY_MIN", 0.0, float)  # ... and/or every M minutes of wall time (0 = off)
    seed: int = _env("SEED", 0, int)
    devices: int = _env("DEVICES", 0, int)              # 0 = all local devices (data parallel)
    log_every: int = _env("LOG_EVERY", 1, int)
    # ---- Sep 27 (10M A / B plan): schedules, entropy bonus, specialists, passive opponents (all off by default)
    beta_end: float = _env("BETA_END", -1.0, float)      # < 0: constant BETA; else BETA -> BETA_END over the horizon below
    beta_decay_iters: int = _env("BETA_DECAY_ITERS", 0, int)          # horizon in iterations ...
    beta_decay_samples: float = _env("BETA_DECAY_SAMPLES", 0.0, float)  # ... or in learner samples (takes precedence)
    beta_schedule: str = _env("BETA_SCHEDULE", "exp")    # exp (geometric, BETA and BETA_END > 0) | linear; BETA = 0 at an
                                                        # iteration = no reference forward and no KL term at all
    ent_coef: float = _env("ENT_COEF", 0.0, float)       # entropy bonus ENT_COEF x (H(unit ops) + 0.5 H(qty) + H(fires)) of
                                                        # the sampling distributions (temperature T, sharpened fires)
    value_grad: float = _env("VALUE_GRAD", 0.0, float)   # VALUE_INIT=head: weight of the value-loss gradient that flows into
                                                        # the policy trunk (0 = stop-gradient, the validated default)
    winb_ramp_iters: int = _env("WINB_RAMP_ITERS", 0, int)  # WINB grows linearly 0 -> WINB over the first N iterations ...
    winb_ramp_samples: float = _env("WINB_RAMP_SAMPLES", 0.0, float)  # ... or learner samples (takes precedence)
    pt_every_samples: float = _env("PT_EVERY_SAMPLES", 0.0, float)  # torch .pt export every S learner samples (0 = off)
    top_teams: str = _env("TOP_TEAMS", "")               # "|"-separated leader team names of TOP tapes ("" = tapes.
                                                        # TOP_TEAMS_DEFAULT: leaders + named specialists; "*" = any)
    spec_dirs: str = _env("SPEC_DIRS", "")               # tape dirs of OUR games vs specialists (index teams "us:..."):
                                                        # every non-"us:" seat is a tape opponent (kind SPEC; held-out
                                                        # ones: evaluation SPEC)
    spec_max: int = _env("SPEC_MAX", 1024, int)          # specialist seats kept in the device tape bank
    # ---- Sep 27 sprint (8x H100, 1-hour runs, resume winner): horizon-aware schedules, wall-clock stop, win-only phase
    horizon_samples: float = _env("HORIZON_SAMPLES", 0.0, float)  # planned learner samples of the whole run (0 = off):
                                                        # BETA and ENT_COEF reach their end values at *_DECAY_FRAC of it
    beta_decay_frac: float = _env("BETA_DECAY_FRAC", 0.9, float)
    ent_end: float = _env("ENT_END", -1.0, float)        # < 0: constant ENT_COEF; else ENT_COEF -> ENT_END (linear)
    ent_decay_frac: float = _env("ENT_DECAY_FRAC", 0.9, float)
    schedule_reset: int = _env("SCHEDULE_RESET", 0, int)  # RESUME continues BETA / ENT from their saved values toward the
                                                        # end values over the remaining horizon (1 = restart the curves)
    margin_w: float = _env("MARGIN_W", 1.0, float)       # weight of the margin shaping reward (0 + WINB 1 = win-only phase)
    vwarm_resume: int = _env("VWARM_RESUME", 0, int)     # value-only iterations right after a RESUME (reward changes)
    max_minutes: float = _env("MAX_MINUTES", 0.0, float) # wall-clock budget from process start: the run ends in time with
                                                        # a final evaluation, state_<it>.npz, pt/it<it>.pt and FINAL.json
    eval_every_min: float = _env("EVAL_EVERY_MIN", 0.0, float)  # evaluation every M minutes (in addition to EVAL_EVERY)
    lr_scale: str = _env("LR_SCALE", "none")             # sqrt: LR and VLR x sqrt(devices x MB / LR_REF_MB) (global batch)
    lr_ref_mb: int = _env("LR_REF_MB", 4096, int)        # the global minibatch LR / VLR were set for
    pt_keep: int = _env("PT_KEEP", 0, int)               # keep only the last N pt/it*.pt (0 = all) ...
    pt_milestone_samples: float = _env("PT_MILESTONE_SAMPLES", 5e8, float)  # ... plus one per S samples (and the final)


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


def minibatch_counts(uact, qact, valid):
    """Loss normalizers of one optimizer minibatch (they do not depend on the params): valid unit decisions, valid
    quantity decisions, market-head decisions (21 per valid sample), valid samples."""
    sv = valid[:, None]
    return dict(nvm=jnp.maximum(((uact >= 0) & sv).sum(), 1), nq=jnp.maximum(((qact >= 0) & sv).sum(), 1),
                nm=jnp.maximum(valid.sum() * A.N_HEADS, 1), nv=jnp.maximum(valid.sum(), 1))


def policy_loss(cur, d, a, old, refp, clip, beta, norm=None, rw=None, ent_coef=None):
    """il/ppo_live.py policy loss: per-decision clipped surrogates (unit ops, qty x 0.5, market fires) with the sample
    advantage, + beta * (KL(pi || ref) on unit ops of valid units + Bernoulli KL of the market fires).
    cur = sample.logp_terms of the current params; old = dict(lpu, lpq, lpm); refp = dict(rlpu_all, rpf).
    norm: the minibatch's counts (minibatch_counts) when this is one microbatch of it: every term is a sum divided by the
    minibatch count, so the microbatch losses (and gradients) add up exactly to the minibatch loss; None = this batch.
    rw [N]: per-sample weights of the KL terms (REF_FRAC: 1 / inclusion probability for samples with reference terms,
    0 for the others: an unbiased estimate of the full KL and of its gradient); None = 1.
    refp None: no KL anchor (BETA = 0).  ent_coef: entropy bonus on the sampling distributions of unit ops, quantities
    (x 0.5, as their surrogate) and market fires (Bernoulli), each averaged like its surrogate term."""
    sv = d["valid"]
    a = a[:, None]
    ru = jnp.exp(cur["lpu"] - old["lpu"]); vm = cur["valid"] & sv[:, None]
    rq = jnp.exp(cur["lpq"] - old["lpq"]); qm = cur["qvalid"] & sv[:, None]
    rm = jnp.exp(cur["lpm"] - old["lpm"]); mm = jnp.broadcast_to(sv[:, None], rm.shape)
    if norm is None:
        norm = dict(nvm=jnp.maximum(vm.sum(), 1), nq=jnp.maximum(qm.sum(), 1), nm=jnp.maximum(mm.sum(), 1))
    nvm, nq, nm = norm["nvm"], norm["nq"], norm["nm"]
    su = jnp.minimum(ru * a, jnp.clip(ru, 1 - clip, 1 + clip) * a)
    l_u = -jnp.where(vm, su, 0.0).sum() / nvm
    sq = jnp.minimum(rq * a, jnp.clip(rq, 1 - clip, 1 + clip) * a)
    l_q = -jnp.where(qm, sq, 0.0).sum() / nq
    sm = jnp.minimum(rm * a, jnp.clip(rm, 1 - clip, 1 + clip) * a)
    l_m = -jnp.where(mm, sm, 0.0).sum() / nm
    w = 1.0 if rw is None else rw[:, None]
    pu = jnp.exp(cur["lpu_all"])
    pf = cur["pf"]
    if refp is not None:
        klu = jnp.where(d["umask"], pu * (cur["lpu_all"] - refp["rlpu_all"]), 0.0).sum(-1)
        klu = jnp.where(vm, klu * w, 0.0).sum() / nvm
        pr = refp["rpf"]
        klm = jnp.where(mm, (pf * jnp.log(pf / pr) + (1 - pf) * jnp.log((1 - pf) / (1 - pr))) * w, 0.0).sum() / nm
        loss = l_u + 0.5 * l_q + l_m + beta * (klu + klm)
    else:
        klu = klm = jnp.float32(0)
        loss = l_u + 0.5 * l_q + l_m
    ent = jnp.where(vm, -(jnp.where(d["umask"], pu * cur["lpu_all"], 0.0)).sum(-1), 0.0).sum() / nvm
    ent_q = jnp.where(qm, -(jnp.exp(cur["lpq_all"]) * cur["lpq_all"]).sum(-1), 0.0).sum() / nq
    ent_m = jnp.where(mm, -(pf * cur["lpf1"] + (1 - pf) * cur["lpf0"]), 0.0).sum() / nm
    if ent_coef is not None:                          # (a python float or a traced schedule value)
        loss = loss - ent_coef * (ent + 0.5 * ent_q + ent_m)
    lr_ = cur["lpu"] - old["lpu"]
    akl = jnp.where(vm, (jnp.exp(lr_) - 1) - lr_, 0.0).sum() / nvm
    cf = jnp.where(vm, jnp.abs(ru - 1) > clip, False).sum() / nvm
    dev = lambda r, m: jnp.where(m, jnp.abs(r - 1), 0.0).max()
    frac = lambda r, m, e, n: jnp.where(m, jnp.abs(r - 1) > e, False).sum() / n
    return loss, dict(l_u=l_u, l_q=l_q, l_m=l_m, klu=klu, klm=klm, ent=ent, ent_q=ent_q, ent_m=ent_m, akl=akl, clipfrac=cf,
                      ru_mean=jnp.where(vm, ru, 0.0).sum() / nvm, ru_maxdev=dev(ru, vm), rq_maxdev=dev(rq, qm), rm_maxdev=dev(rm, mm),
                      ru_frac1e3=frac(ru, vm, 1e-3, nvm), ru_frac1e2=frac(ru, vm, 1e-2, nvm), rm_frac1e3=frac(rm, mm, 1e-3, nm),
                      rm_frac1e2=frac(rm, mm, 1e-2, nm), ru_frac5e2=frac(ru, vm, 5e-2, nvm), rm_frac5e2=frac(rm, mm, 5e-2, nm))


def value_loss(v, target, valid, nv=None):
    sv = valid.astype(jnp.float32)
    return (jnp.square(v - target) * sv).sum() / (jnp.maximum(sv.sum(), 1.0) if nv is None else nv)


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
        if k in ("TOP", "SPEC", "PASS"):      # recorded leader / specialist tapes, or a passive opponent (PASS every
            return k, 1, self.cfg.team        # step): the network output of slot 1 is ignored
        return "IL", 1, self.draw_team()

    def draw_team(self):
        tw = np.array([w for _, w in self.il_teams])
        return self.il_teams[self.rng.choice(len(tw), p=tw / tw.sum())][0]

    def draw_blocks(self, K, G=0):
        """Opponents of the K blocks for one episode.  G = 0: every block draws its own entry.  G > 0: the blocks form G
        groups of K / G; a group draws one entry (kind, bank slot) and IL groups give each block its own style id, so the
        rollout runs one opponent forward per group with per-row team ids."""
        if not G:
            return [self.draw() for _ in range(K)]
        out = []
        for _ in range(G):
            kind, slot, team = self.draw()
            out += [(kind, slot, self.draw_team() if kind == "IL" else team) for _ in range(K // G)]
        return out

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
        self.micro = cfg.micro or cfg.mb
        assert cfg.mb % self.micro == 0, f"MICRO {self.micro} must divide MB {cfg.mb}"
        assert self.N % self.micro == 0, f"MICRO {self.micro} must divide SEG*B = {self.N}"
        assert cfg.mb <= self.N, f"MB {cfg.mb} > SEG*B = {self.N}"   # N % MB != 0: each epoch drops the random remainder
        assert 0.0 < cfg.ref_frac <= 1.0
        self.value_head = cfg.value_init == "head"
        self.use_ent = cfg.ent_coef > 0 or cfg.ent_end > 0
        if cfg.opp_groups:
            assert cfg.blocks % cfg.opp_groups == 0, "K must be a multiple of OPP_GROUPS"
        self.rules = A.RuleCfg(midnight=cfg.midnight, final_fix=cfg.finalfix, homing=cfg.homing, se_guard=cfg.se_guard,
                               sheep_cap=cfg.sheep_cap, seed_budget=cfg.seed_budget)
        if cfg.tau.endswith(".json"):
            tj = json.load(open(cfg.tau))
            self.tau = np.array([float(tj.get(h, tj.get("_default", 0.5))) for h, _ in A.MARKET_HEADS], np.float32)
        else:
            self.tau = np.full(A.N_HEADS, float(cfg.tau), np.float32)
        self.uin_dtype = {"float16": jnp.float16, "bfloat16": jnp.bfloat16}.get(cfg.uin_dtype, jnp.float32)
        kinds = [x.split("=")[0] for x in cfg.opps.split(",")]
        evk = cfg.eval_opps.split(",")
        self.use_tapes = any(k in kinds or k in evk for k in ("TOP", "SPEC"))
        self.tape_np = {}
        if self.use_tapes:
            assert hasattr(self.be, "step_tapes"), "TOP / SPEC (recorded tapes) need the real engine backend"
            import tapes as TP
            spec = [d.rstrip("/") for d in cfg.spec_dirs.split(",") if d]
            dirs = [d.rstrip("/") for d in cfg.tape_dirs.split(",") if d and d.rstrip("/") not in spec]
            teams = cfg.top_teams or TP.TOP_TEAMS_DEFAULT
            if "TOP" in kinds or "SPEC" in kinds:        # one bank: leader seats, then specialist seats (bank["spec"])
                self.tape_np["train"] = TP.build_bank(dirs, teams, "train", cfg.tape_max, spec_dirs=spec, spec_max=cfg.spec_max)
            if "TOP" in evk:
                self.tape_np["eval_TOP"] = TP.build_bank(dirs, teams, "heldout", cfg.eval_n)
            if "SPEC" in evk and spec:
                self.tape_np["eval_SPEC"] = TP.build_bank([], teams, "heldout", 0, spec_dirs=spec, spec_max=cfg.eval_n)
        self._build()

    @staticmethod
    def _sel(flag, a, b):
        """flag [B] bool: b where flag else a (pytrees with leading B)."""
        return jax.tree_util.tree_map(lambda x, y: jnp.where(flag.reshape((-1,) + (1,) * (x.ndim - 1)), y, x), a, b)

    def _fwd(self, params, obs, team, precision, with_hm=False):
        if precision == "bf16grad":                         # value: TF32 forward; derivative: bf16 forward (its VJP)
            o32 = self._fwd(lax.stop_gradient(params), obs, team, "default", with_hm=with_hm)
            o16 = self._fwd(params, obs, team, "bf16", with_hm=with_hm)
            return jax.tree_util.tree_map(lambda a, b: lax.stop_gradient(a - b) + b, o32, o16)
        pl = Q255[obs.planes.astype(jnp.int32)]            # exact uint8 / 255 (XLA may rewrite x / c as x * (1 / c))
        return self.be.net_apply(params, pl, obs.glob, obs.pos, obs.uin.astype(jnp.float32), team, precision, with_hm=with_hm)

    # -------------------------------------------------------------------------------------------- rollout
    def rollout_local(self, params, bank, cs, tapes, opp_idx, opp_team, carry, key, opp_mode=None, winb=None, *, L, learner_greedy,
                      record, axis, precision=None):
        """L lockstep steps of the local games.  carry = (engine state, prev ops [B,2,16], learner seat [B], tape id [B]
        (-1 = a normal game; >= 0: a recorded world whose other seat replays the tape; tapes = device bank or None)).
        opp_mode [K] int (per block): 1 = the opponent PASSes every step (no unit command, no order), 0 = its network;
        winb: the terminal win bonus (WINB schedule; None = cfg.winb).
        Returns carry, traj (time-major, learner seat), (finished [B], final margin [B]), bootstrap obs."""
        cfg, be = self.cfg, self.be
        winb_ = cfg.winb if winb is None else winb
        if axis is not None:
            key = jax.random.fold_in(key, lax.axis_index(axis))
        B = carry[2].shape[0]
        G = opp_idx.shape[0]; Bg = B // G                       # opponent param groups (one bank slot each)
        team_o = jnp.repeat(opp_team, B // opp_team.shape[0])   # per-row style ids (one per block)
        tau_l = jnp.asarray(self.tau)
        tau_o = jnp.full((A.N_HEADS,), cfg.opp_tau, jnp.float32)
        team_l = jnp.full((B,), cfg.team, jnp.int32)
        rules = self.rules
        prec_r = precision or cfg.rollout_precision
        prec_o = precision or cfg.opp_precision or cfg.rollout_precision   # (an explicit precision, e.g. evaluation: both seats)

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
            outL = self._fwd(params, oL, team_l, prec_r, with_hm=record and self.value_head)

            def opp_block(xs):
                idx, tm, ob = xs
                prm = jax.tree_util.tree_map(lambda b: b[idx], bank)
                return self._fwd(prm, ob, tm, prec_o)

            oOb = jax.tree_util.tree_map(lambda x: x.reshape((G, Bg) + x.shape[1:]), oO)
            tOb = team_o.reshape(G, Bg)
            if cfg.opp_map == "vmap":                             # one batched (grouped-conv) forward over all groups
                outO = jax.vmap(opp_block)((opp_idx, tOb, oOb))
            else:                                                 # G sequential forwards
                outO = lax.map(opp_block, (opp_idx, tOb, oOb))
            outO = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), outO)
            dL = S.decide(outL, mL, oL.nu, kL, cfg.T, cfg.ks, tau_l, learner_greedy)
            dO = S.decide(outO, mO, oO.nu, kO, cfg.T, cfg.ks, tau_o, not cfg.opp_sample)
            build = jax.vmap(lambda d, nu, v: S.build_act(d, nu, v, rules))
            aL, aO = build(dL, oL.nu, vL), build(dO, oO.nu, vO)
            if opp_mode is not None:                              # passive opponents: PASS (no command, no order)
                pas = jnp.repeat(opp_mode, B // opp_mode.shape[0]) == 1
                pact = jax.tree_util.tree_map(jnp.zeros_like, aO)._replace(u_n=jnp.ones_like(aO.u_n))
                aO = self._sel(pas, aO, pact)
            a0, a1 = self._sel(one, aL, aO), self._sel(one, aO, aL)
            mb = margin(be.money(state), lseat)
            ns = be.step(state, a0, a1, cs) if tapes is None else be.step_tapes(state, a0, a1, tid, 1 - lseat, tapes, kW, cs)
            valid = t <= be.last_step
            done = t == be.last_step
            ma = margin(be.money(ns), lseat)                          # money == the final reward once done
            shaping = (ma - mb) / cfg.rscale if cfg.margin_w == 1.0 else cfg.margin_w * (ma - mb) / cfg.rscale
            rew = jnp.where(valid, shaping, 0.0) + jnp.where(done, winb_ * jnp.sign(ma), 0.0)
            kk = jnp.arange(A.NU)[None, :]
            pL = jnp.where(kk < oL.nu[:, None], dL.op, -1); pO = jnp.where(kk < oO.nu[:, None], dO.op, -1)
            if opp_mode is not None:
                pO = jnp.where(pas[:, None], -1, pO)
            p0 = jnp.where(one[:, None], pO, pL); p1 = jnp.where(one[:, None], pL, pO)
            nprev = jnp.where(valid, jnp.stack([p0, p1], 1), prev)
            out = dict(fin=jnp.where(done, ma, 0.0), done=jnp.broadcast_to(done, (B,)))
            if record:
                bl = S.logp_terms(outL, mL, dL.uact, dL.qact, dL.fire, cfg.T, cfg.ks, tau_l)     # behavior log-probs
                out.update(planes=oL.planes, glob=oL.glob, pos=oL.pos.astype(jnp.int8), uin=oL.uin.astype(self.uin_dtype),
                           umask=mL, uact=dL.uact, qact=dL.qact, mfire=dL.fire, rew=rew, valid=jnp.broadcast_to(valid, (B,)),
                           blpu=bl["lpu"], blpq=bl["lpq"], blpm=bl["lpm"], marg=mb / cfg.rscale)
                if self.value_head:                               # value-head input (V_old without a value forward)
                    out["hm"] = outL["hm"]
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
    def update_local(self, params, vparams, opt_p, opt_v, ref, tr, boot, key, beta=None, ent=None, lrs=None, *, train_policy, axis,
                     diag=False, anchor=True):
        """PPO update on one iteration's trajectories: EPOCHS passes of optimizer minibatches of MB samples, each
        accumulated over MB / MICRO microbatches (identical maths, less memory).  Value: a separate network (vparams =
        net params) or, with VALUE_INIT=head, the policy's own value head on stop-gradient features (vparams = head).
        beta: KL weight of this iteration (BETA schedule; None = cfg.beta); anchor (static): False = no reference terms
        and no KL at all (BETA = 0: from scratch, or the end of a schedule to 0)."""
        cfg = self.cfg
        beta = jnp.float32(cfg.beta) if beta is None else beta
        ent_c = (jnp.float32(cfg.ent_coef) if ent is None else ent) if self.use_ent else None
        lr_p, lr_v = (cfg.lr, cfg.vlr) if lrs is None else (lrs[0], lrs[1])     # (LR_SCALE: the global-batch LRs)
        anchor = bool(anchor) and train_policy
        if axis is not None:
            key = jax.random.fold_in(key, lax.axis_index(axis))
        k_ep, k_sub = key, jax.random.fold_in(key, 7)            # k_ep = key: the gate trainer's minibatch permutations
        L, B = tr["rew"].shape
        N = L * B
        MB, MI = cfg.mb, self.micro
        head = self.value_head
        tau = jnp.asarray(self.tau)
        prec = cfg.update_precision
        prec_ng = "default" if prec == "bf16grad" else prec      # forwards without a gradient (pre-passes, bootstrap)
        flat = {k: tr[k].reshape((N,) + tr[k].shape[2:]) for k in ("planes", "glob", "pos", "uin", "umask", "uact", "qact", "mfire", "valid")}
        gather = lambda idx: {k: v[idx] for k, v in flat.items()}
        team_of = lambda n: jnp.full((n,), cfg.team, jnp.int32)

        def obs_of(d):
            return Obs(d["planes"], d["glob"], d["pos"].astype(jnp.int32), d["uin"].astype(jnp.float32), None)

        def terms(p, d, with_hm=False, grad=False):
            out = self._fwd(p, obs_of(d), team_of(d["glob"].shape[0]), prec if grad else prec_ng, with_hm=with_hm)
            return out, S.logp_terms(out, d["umask"], d["uact"], d["qact"], d["mfire"], cfg.T, cfg.ks, tau)

        def head_v(vp, hm):
            return (hm @ vp["w"] + vp["b"])[:, 0]

        beh = {k: tr["b" + k].reshape((N,) + tr["b" + k].shape[2:]) for k in ("lpu", "lpq", "lpm")}
        need_old = cfg.old_logp == "recompute" or cfg.diag_old or diag
        chunks = jnp.arange(N).reshape(N // MI, MI)

        # (1) recomputed old log-probs (diagnostic, or OLD_LOGP=recompute) and old values of a separate value network
        pc = {}
        if need_old or not head:
            def pre(idx):
                d = gather(idx)
                r_ = {}
                if need_old:
                    _, o = terms(params, d)
                    r_.update(lpu=o["lpu"], lpq=o["lpq"], lpm=o["lpm"])
                if not head:
                    r_["v"] = self._fwd(vparams, obs_of(d), team_of(MI), prec_ng)["v"]
                return r_

            pc = lax.map(pre, chunks)
            pc = {k: v.reshape((N,) + v.shape[2:]) for k, v in pc.items()}
        if head:                                                  # V_old from the features recorded at rollout
            pc["v"] = head_v(vparams, tr["hm"].reshape(N, -1))

        # (2) reference (KL anchor) terms: every sample, or a uniform subsample of M samples without replacement
        #     (inclusion probability exactly M / N; the KL terms are weighted by N / M: unbiased)
        spos = rw_all = None
        M = N
        if not cfg.ref_online and anchor:                        # (value-only iterations / no anchor: no reference terms)
            if cfg.ref_frac < 1.0:
                M = max(MI, int(round(cfg.ref_frac * N / MI)) * MI)
                sub = jax.random.permutation(k_sub, N)[:M]
                spos = jnp.full((N,), -1, jnp.int32).at[sub].set(jnp.arange(M, dtype=jnp.int32))
                rw_all = jnp.where(spos >= 0, jnp.float32(N / M), jnp.float32(0))
                ridx = sub.reshape(M // MI, MI)
            else:
                M, ridx = N, chunks

            def pre_ref(idx):
                _, r = terms(ref, gather(idx))
                return dict(rlpu_all=r["lpu_all"], rpf=r["pf"])

            rc = lax.map(pre_ref, ridx)
            rc = {k: v.reshape((M,) + v.shape[2:]) for k, v in rc.items()}
        diag_ = {}
        if need_old:                                             # stored (rollout) vs recomputed (update) log-probs
            fv = flat["valid"]
            for k, m in (("lpu", (flat["uact"] >= 0) & fv[:, None]), ("lpq", (flat["qact"] >= 0) & fv[:, None]),
                         ("lpm", jnp.broadcast_to(fv[:, None], beh["lpm"].shape))):
                diag_["old_diff_" + k] = jnp.where(m, jnp.abs(beh[k] - pc[k]), 0.0).max()
        oldsrc = {k: pc[k] for k in ("lpu", "lpq", "lpm")} if cfg.old_logp == "recompute" else beh

        # (3) advantages: GAE(gamma 1, LAM) with the value bootstrap at a truncated segment's end
        bobs, bmarg = boot
        if cfg.seg == self.E:                                     # whole episodes: every game is done, no bootstrap
            vb = jnp.zeros((B,), jnp.float32)
        else:
            bo = Obs(bobs.planes, bobs.glob, bobs.pos, bobs.uin, None)
            vb = (head_v(vparams, self._fwd(params, bo, team_of(B), prec_ng, with_hm=True)["hm"]) if head
                  else self._fwd(vparams, bo, team_of(B), prec_ng)["v"])
        fin_param = cfg.value_target == "final"
        marg = tr["marg"].reshape(N)
        if cfg.margin_w != 1.0:                                  # MARGIN_W: the shaping part of the return is w x margin
            marg, bmarg = cfg.margin_w * marg, cfg.margin_w * bmarg
        v_old = pc["v"]
        if fin_param:                                            # V = V' - margin (V' = predicted final margin)
            vb = vb - bmarg
            v_old = v_old - marg
        vt = v_old.reshape(L, B)
        rew, done, valid = tr["rew"], tr["done"], tr["valid"]
        adv = gae(rew, vt, vb, done, valid, cfg.lam)
        ret = (adv + vt).reshape(N)
        g1 = (gae(rew, vt, vb, done, valid, 1.0) + vt).reshape(N)  # Monte-Carlo return-to-go (bootstrapped if truncated)
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
        vtarget = ret + marg if fin_param else ret                # the value regression target in the net's own terms

        def loss_p(p, d, a, old, refp, rw, norm):
            out, cur = terms(p, d, with_hm=head, grad=True)
            if cfg.ref_online and anchor:
                _, r = terms(ref, d)
                refp = dict(rlpu_all=lax.stop_gradient(r["lpu_all"]), rpf=lax.stop_gradient(r["pf"]))
            lp, st = policy_loss(cur, d, a, old, refp if anchor else None, cfg.clip, beta, norm, rw, ent_c)
            return lp, (st, out["hm"] if head else jnp.zeros(()))

        def loss_joint(p, vp, d, a, old, refp, rw, target, norm):  # head: one forward serves both losses
            lp, (st, hm) = loss_p(p, d, a, old, refp, rw, norm)
            hv = lax.stop_gradient(hm)
            if cfg.value_grad:                                   # VALUE_GRAD: that share of the value gradient reaches
                hv = hv + cfg.value_grad * (hm - lax.stop_gradient(hm))   # the trunk (forward value unchanged)
            lv = value_loss(head_v(vp, hv), target, d["valid"], norm["nv"])
            return lp + lv, (lp, lv, st)

        def loss_v(vp, d, target, norm):
            return value_loss(self._fwd(vp, obs_of(d), team_of(MI), prec)["v"], target, d["valid"], norm["nv"])

        MAXK = ("ru_maxdev", "rq_maxdev", "rm_maxdev")
        STATK = ("l_u", "l_q", "l_m", "klu", "klm", "ent", "ent_q", "ent_m", "akl", "clipfrac", "ru_mean", "ru_frac1e3", "ru_frac1e2",
                 "rm_frac1e3", "rm_frac1e2", "ru_frac5e2", "rm_frac5e2") + MAXK

        def mb_body(c, idx):
            p, vp, op, ov = c
            norm = minibatch_counts(flat["uact"][idx], flat["qact"][idx], flat["valid"][idx])

            def micro(acc, midx):
                gp_a, gv_a, st_a = acc
                d = gather(midx)
                old = {k: oldsrc[k][midx] for k in ("lpu", "lpq", "lpm")}
                a, target = advn[midx], vtarget[midx]
                if cfg.ref_online or not anchor:
                    refp, rw = {}, None
                elif spos is None:
                    refp, rw = {k: rc[k][midx] for k in rc}, None
                else:
                    refp, rw = {k: rc[k][jnp.maximum(spos[midx], 0)] for k in rc}, rw_all[midx]
                if not train_policy:                             # value only (VWARM, value experiments)
                    gp = jax.tree_util.tree_map(jnp.zeros_like, p)
                    st = {k: jnp.float32(0) for k in STATK}; lp = jnp.float32(0)
                    if head:
                        hm = lax.stop_gradient(self._fwd(p, obs_of(d), team_of(MI), prec_ng, with_hm=True)["hm"])
                        lv, gv = jax.value_and_grad(lambda vp_: value_loss(head_v(vp_, hm), target, d["valid"], norm["nv"]))(vp)
                    else:
                        lv, gv = jax.value_and_grad(loss_v)(vp, d, target, norm)
                elif head:
                    (_, (lp, lv, st)), (gp, gv) = jax.value_and_grad(loss_joint, argnums=(0, 1), has_aux=True)(
                        p, vp, d, a, old, refp, rw, target, norm)
                else:
                    (lp, (st, _)), gp = jax.value_and_grad(loss_p, has_aux=True)(p, d, a, old, refp, rw, norm)
                    lv, gv = jax.value_and_grad(loss_v)(vp, d, target, norm)
                st = dict(st, loss=lp, l_v=lv)
                add = lambda x, y: jax.tree_util.tree_map(jnp.add, x, y)
                st_a = {k: (jnp.maximum(st_a[k], st[k]) if k in MAXK else st_a[k] + st[k]) for k in st_a}
                return (add(gp_a, gp), add(gv_a, gv), st_a), None

            zeros = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
            st0 = {k: jnp.float32(0) for k in STATK + ("loss", "l_v")}
            (g, gv, st), _ = lax.scan(micro, (zeros(p), zeros(vp), st0), idx.reshape(MB // MI, MI))
            if axis is not None:
                g, gv = lax.pmean(g, axis), lax.pmean(gv, axis)
            gv, vn = clip_by_global_norm(freeze_grads(gv), 1.0)
            vp, ov = adam_step(vp, gv, ov, lr_v)
            if train_policy:
                g, gn = clip_by_global_norm(freeze_grads(g), 1.0)
                p, op = adam_step(p, g, op, lr_p)
            else:
                gn = jnp.float32(0)
            return (p, vp, op, ov), dict(st, gnorm=gn, vgnorm=vn)

        def ep_body(c, k):
            perm = jax.random.permutation(k, N)[:(N // MB) * MB].reshape(N // MB, MB)
            return lax.scan(mb_body, c, perm)

        (params, vparams, opt_p, opt_v), st = lax.scan(ep_body, (params, vparams, opt_p, opt_v), jax.random.split(k_ep, cfg.epochs))
        ep_loss, ep_lv = st["loss"].mean(1), st["l_v"].mean(1)                # [epochs]: the fit on this batch per epoch
        first = {"r0_" + k: st[k][0, 0] for k in ("ru_maxdev", "rq_maxdev", "rm_maxdev", "clipfrac", "klu", "klm", "ru_frac1e2",
                                                   "ru_frac5e2", "rm_frac1e2", "rm_frac5e2")}   # 1st minibatch = LR-0 noise
        st = {k: (v.max() if k.endswith("maxdev") else v.mean()) for k, v in st.items()}
        st.update(loss_ep0=ep_loss[0], loss_epN=ep_loss[-1], lv_ep0=ep_lv[0], lv_epN=ep_lv[-1], **first, **diag_)
        leaves = jax.tree_util.tree_leaves((params, vparams))
        st["finite"] = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in leaves]) &
                               jnp.all(jnp.stack([jnp.isfinite(v) for v in jax.tree_util.tree_leaves(st)])))
        nvf = jnp.maximum(vf.sum(), 1)
        wmean = lambda x: (x * vf).sum() / nvf
        wvar = lambda x: wmean(jnp.square(x - wmean(x)))
        rm = wmean(ret)
        st.update(ret_mean=rm, v_res2=wmean(jnp.square(jnp.where(flat["valid"], ret - v_old, 0.0))), ret_var=wvar(ret),
                  rew_sum=(rew * valid).sum() / B,
                  ev_gae=1.0 - wvar(ret - v_old) / jnp.maximum(wvar(ret), 1e-12),          # value fit before this update
                  ev_mc=1.0 - wvar(g1 - v_old) / jnp.maximum(wvar(g1), 1e-12),              # ... vs the MC return
                  adv_std=jnp.sqrt(wvar(adv)), g_std=jnp.sqrt(wvar(g1)),
                  ref_samples=jnp.float32(M if (anchor and not cfg.ref_online) else 0), beta=beta,
                  ent_coef=ent_c if ent_c is not None else jnp.float32(0))
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
            self._update = jax.jit(partial(self.update_local, axis=None), static_argnames=("train_policy", "diag", "anchor"),
                                   donate_argnums=(0, 1, 2, 3))
        else:
            m = self.mesh
            rep, bat = PS(), PS("d")
            cspec = (be.state_spec("d"), bat, bat, bat)
            cfg = self.cfg

            def ro(params, bank, cs, tapes, opp_idx, opp_team, carry, key, opp_mode=None, winb=None, *, L, learner_greedy, record,
                   precision=None):
                opp_mode = jnp.zeros_like(opp_team) if opp_mode is None else opp_mode
                winb = jnp.float32(cfg.winb) if winb is None else winb
                f = partial(self.rollout_local, L=L, learner_greedy=learner_greedy, record=record, axis="d", precision=precision)
                return jax.shard_map(f, mesh=m, in_specs=(rep, rep, rep, rep, rep, rep, cspec, rep, rep, rep),
                                     out_specs=(cspec, PS(None, "d"), (bat, bat), bat), check_vma=False)(
                    params, bank, cs, tapes, opp_idx, opp_team, tuple(carry), key, opp_mode, winb)

            def up(params, vparams, opt_p, opt_v, ref, tr, boot, key, beta=None, ent=None, lrs=None, *, train_policy, diag=False, anchor=True):
                beta = jnp.float32(cfg.beta) if beta is None else beta
                ent = jnp.float32(cfg.ent_coef) if ent is None else ent
                lrs = jnp.asarray([cfg.lr, cfg.vlr], jnp.float32) if lrs is None else lrs
                f = partial(self.update_local, train_policy=train_policy, axis="d", diag=diag, anchor=anchor)
                return jax.shard_map(f, mesh=m, in_specs=(rep, rep, rep, rep, rep, PS(None, "d"), bat, rep, rep, rep, rep),
                                     out_specs=(rep, rep, rep, rep, rep), check_vma=False)(params, vparams, opt_p, opt_v, ref, tr, boot, key,
                                                                                           beta, ent, lrs)

            self._rollout = jax.jit(ro, static_argnames=("L", "learner_greedy", "record", "precision"))
            self._update = jax.jit(up, static_argnames=("train_policy", "diag", "anchor"), donate_argnums=(0, 1, 2, 3))
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
def _interp(v0, v1, s0, s1, s, kind):
    """v0 at s0 -> v1 at s1, then held; exp = geometric (v0, v1 > 0), else linear."""
    if s1 <= s0:
        return v1
    x = min(1.0, max(0.0, (s - s0) / (s1 - s0)))
    if kind == "exp" and v0 > 0 and v1 > 0:
        return v0 * (v1 / v0) ** x
    return v0 + (v1 - v0) * x


def beta_schedule(cfg, it, smp, anchor=None):
    """KL weight of iteration it (smp = learner samples before it): constant BETA (BETA_END < 0), or BETA -> BETA_END
    geometric (exp) or linear, then held, over BETA_DECAY_SAMPLES samples, else BETA_DECAY_FRAC x HORIZON_SAMPLES, else
    BETA_DECAY_ITERS iterations.  anchor = (samples, value) at a RESUME: the curve continues from there to the end value
    (horizon-aware: a longer horizon after a resume never tightens the anchor again)."""
    if cfg.beta_end < 0:
        return cfg.beta
    h = getattr(cfg, "horizon_samples", 0.0)
    s1 = cfg.beta_decay_samples if cfg.beta_decay_samples > 0 else getattr(cfg, "beta_decay_frac", 0.9) * h if h > 0 else 0.0
    if s1 > 0:
        s0, v0 = anchor if anchor is not None else (0.0, cfg.beta)
        return _interp(v0, cfg.beta_end, s0, s1, smp, cfg.beta_schedule)
    x = (it - 1) / cfg.beta_decay_iters if cfg.beta_decay_iters > 0 else 1.0
    return _interp(cfg.beta, cfg.beta_end, 0.0, 1.0, x, cfg.beta_schedule)


def ent_schedule(cfg, smp, anchor=None):
    """Entropy bonus weight: constant ENT_COEF (ENT_END < 0 or no horizon), else ENT_COEF -> ENT_END linearly over
    ENT_DECAY_FRAC x HORIZON_SAMPLES (anchor: as beta_schedule)."""
    h = getattr(cfg, "horizon_samples", 0.0)
    if cfg.ent_end < 0 or h <= 0:
        return cfg.ent_coef
    s0, v0 = anchor if anchor is not None else (0.0, cfg.ent_coef)
    return _interp(v0, cfg.ent_end, s0, cfg.ent_decay_frac * h, smp, "linear")


def winb_schedule(cfg, it, smp=0):
    """Win bonus of iteration it (smp = learner samples before it): WINB, or a linear ramp 0 -> WINB over the first
    WINB_RAMP_SAMPLES samples (else WINB_RAMP_ITERS iterations)."""
    if getattr(cfg, "winb_ramp_samples", 0) > 0:
        return cfg.winb * min(1.0, smp / cfg.winb_ramp_samples)
    return cfg.winb * min(1.0, it / cfg.winb_ramp_iters) if cfg.winb_ramp_iters > 0 else cfg.winb


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
    T_START = time.time()                                     # MAX_MINUTES counts from here (start-up and compiles included)
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
    scratch = cfg.init == "random" or cfg.init.startswith("random:")
    if tiny:
        w, b = [int(x) for x in cfg.init.split(":")[1].split("x")]
        params = be.init_params(jax.random.PRNGKey(cfg.seed + 1), width=w, blocks=b)
    elif scratch:                                             # from scratch: torch-default init, the template's architecture
        tmpl = cfg.init.split(":", 1)[1] if ":" in cfg.init else (cfg.il or cfg.ref)
        assert tmpl, "INIT=random needs an architecture template: INIT=random:<ckpt>, IL or REF"
        arch = ckpt_io.arch_of(be.load_params(tmpl, "state")[0])
        params = be.init_params(jax.random.PRNGKey(cfg.seed + 1), width=arch["width"], blocks=arch["blocks"], n_teams=arch["n_teams"],
                                g_in=arch["g_in"], n_aux=arch["n_aux"])
        log_line(LOG, f"[init] random weights (seed {cfg.seed + 1}), architecture of {tmpl}: {arch}")
    else:
        params, _ = be.load_params(cfg.init, "state")
    copy = lambda t: jax.tree_util.tree_map(jnp.array, t)     # the update donates params: never alias them
    start = copy(params)                                      # the warm start, fixed (bank slot START, SNAP0 eval)
    ref = copy(params) if not cfg.ref else be.load_params(cfg.ref, "state")[0]
    il = ref if not cfg.il else be.load_params(cfg.il, "state")[0]
    if cfg.value_init == "head":                              # the policy's own value head (INIT's, IL-trained)
        vparams = {"w": jnp.array(params["v"]["w"]), "b": jnp.array(params["v"]["b"])}
    elif cfg.value_init.startswith("small:"):
        w, b = [int(x) for x in cfg.value_init.split(":")[1].split("x")]
        vparams = be.init_params(jax.random.PRNGKey(cfg.seed + 2), width=w, blocks=b, n_teams=ckpt_io.arch_of(params)["n_teams"])
    elif cfg.value_init == "policy":
        vparams = None
        if not tiny and not scratch:
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
    it0 = samples0 = 0
    meta_r = {}
    pool = Pool(cfg, rng)
    if cfg.resume:
        params, vparams, opt_p, opt_v, meta = ckpt_io.load_train_state(cfg.resume)
        meta_r = meta
        it0 = int(meta["it"])
        samples0 = int(meta.get("samples", it0 * tr.N * D))            # (schedules and PT_EVERY_SAMPLES count samples)
        pool.snaps = [tuple(x) for x in meta.get("snaps", []) if x[0] > START_SLOT]; pool.hist = meta.get("hist", {})
        rng = np.random.RandomState([cfg.seed, it0]); pool.rng = rng          # new random streams after a resume
        key = jax.random.fold_in(key, it0)
        log_line(LOG, f"[resume] {cfg.resume} at iteration {it0} (random streams advanced; START = {cfg.init})")
    varch = "head (policy features)" if tr.value_head else ckpt_io.arch_of(vparams)
    log_line(LOG, f"[model] policy {ckpt_io.arch_of(params)} value {varch} target {cfg.value_target} ref {cfg.ref or 'INIT'} il {cfg.il or 'REF'}")
    with_head = lambda p, vp: dict(p, v=vp) if tr.value_head else p       # exported .pt: the trained value head in "v"
    pt_value = lambda vp: None if tr.value_head else vp

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
    import tapes as TP_
    dev_bank = lambda b: tr.put({k: jnp.asarray(v) for k, v in TP_.device_arrays(b).items()})
    tapes_train = dev_bank(tr.tape_np["train"]) if "train" in tr.tape_np else None
    tapes_eval = {k[5:]: dev_bank(v) for k, v in tr.tape_np.items() if k.startswith("eval_") and len(v["lseat"])}
    tape_ids = {}                                             # kind -> training bank rows it draws from
    if tapes_train is not None:
        spf = tr.tape_np["train"]["spec"]
        kinds_ = pool.entry_names()
        tape_ids["TOP"] = np.flatnonzero(~spf) if "SPEC" in kinds_ else np.arange(len(spf))
        tape_ids["SPEC"] = np.flatnonzero(spf)
        for k_ in ("TOP", "SPEC"):
            if k_ in kinds_ and not len(tape_ids[k_]):
                log_line(LOG, f"[tapes] no {k_} seats in the training bank: {k_} removed from the opponent pool")
                pool.kinds = [(k, w) for k, w in pool.kinds if k != k_]
        log_line(LOG, f"[tapes] training bank: {len(spf)} seats (leaders {int((~spf).sum())}, specialists {int(spf.sum())}); "
                      f"TOP draws {len(tape_ids['TOP'])}, SPEC draws {len(tape_ids['SPEC'])}")
    else:
        for k_ in ("TOP", "SPEC"):
            assert k_ not in pool.entry_names(), f"{k_} in OPPS without a tape bank"

    K = cfg.blocks
    G = cfg.opp_groups
    blk_of_game = game_block_index(cfg, D)
    blk = pool.draw_blocks(K, G)

    def new_games(k_):
        """Fresh games for the current block assignment: TOP blocks get random training tapes (learner seat = the
        tape's non-leader seat)."""
        tid = -np.ones(D * cfg.games, np.int32); ls = None
        if tapes_train is not None:
            lsb = tr.tape_np["train"]["lseat"]
            for b_ in range(K):
                if blk[b_][0] in ("TOP", "SPEC"):
                    g_ = np.flatnonzero(blk_of_game == b_)
                    ids = tape_ids[blk[b_][0]]
                    tid[g_] = ids[rng.randint(len(ids), size=len(g_))]
            ls = np.where(tid >= 0, lsb[np.maximum(tid, 0)], 0)
        return tr.fresh_games(k_, D * cfg.games, tid, ls)

    key, k0 = jax.random.split(key)
    carry = new_games(k0)

    def opp_arrays(entries):
        """-> (bank slot per param group [G or K], style id per block [K], mode per block [K]: 1 = PASS)."""
        slots = [entries[g * (len(entries) // G)][1] for g in range(G)] if (G and len(entries) == K) else [b[1] for b in entries]
        return (tr.put(jnp.asarray(slots, jnp.int32)), tr.put(jnp.asarray([b[2] for b in entries], jnp.int32)),
                tr.put(jnp.asarray([int(b[0] == "PASS") for b in entries], jnp.int32)))

    # ---- evaluation: fixed worlds, greedy learner, paired vs the first evaluation
    eval_specs = [s for s in cfg.eval_opps.split(",") if s]
    base_eval = {}

    def evaluate(it):
        if it == it0 and cfg.eval_base and os.path.exists(cfg.eval_base):     # reuse a previous run's iteration-0 eval
            z = np.load(cfg.eval_base)
            base_eval.update({k: z[k] for k in z.files if k in eval_specs})
            sc = float(np.mean([(v > 0).mean() for v in base_eval.values()]))
            log_line(LOG, f"[eval it {it}] loaded from {cfg.eval_base}: " + " | ".join(
                f"vs {k}: W {(v > 0).mean():.3f} margin {v.mean():+,.0f}" for k, v in base_eval.items()) + f" | score {sc:.3f}")
            return sc
        res, t_ev = {}, time.time()
        for sp in eval_specs:
            tp, tid_e, ls_e, n_e, n_keep = None, None, None, D * cfg.eval_n, None
            if sp.startswith("IL@"):
                ent = [("IL", 1, int(sp[3:]))]
            elif sp == "SNAP0":                                               # the warm start (fixed bank slot)
                ent = [("SELF", START_SLOT, cfg.team)]
            elif sp == "PASS":                                                # a passive opponent (PASS every step)
                ent = [("PASS", 1, cfg.team)]
            elif sp in ("TOP", "SPEC") and sp in tapes_eval:                 # held-out recorded leader / specialist
                ent = [(sp, 1, cfg.team)]                                     # worlds, each played once (distinct worlds)
                lse = tr.tape_np["eval_" + sp]["lseat"]
                n_keep = len(lse); n_e = D * -(-n_keep // D)
                tid_e = np.arange(n_e) % n_keep; ls_e = lse[tid_e]; tp = tapes_eval[sp]
            else:
                continue
            ev = tr.fresh_games(jax.random.PRNGKey(10_000 + cfg.seed), n_e, tid_e, ls_e)   # same worlds every time
            oi, ot, om = opp_arrays(ent)
            _, _, (fin, mar), _ = tr._rollout(params, bank, cs, tp, oi, ot, ev, jax.random.PRNGKey(20_000), om, jnp.float32(0),
                                              L=tr.E, learner_greedy=True, record=False, precision=cfg.eval_precision)
            res[sp] = np.asarray(mar)[:n_keep]
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
        np.savez(os.path.join(cfg.run_dir, f"eval_{it}.npz"), **res)
        score /= max(1, len(res))
        log_line(LOG, line + f" | score {score:.3f}")
        with open(os.path.join(cfg.run_dir, "eval.jsonl"), "a") as f:
            f.write(json.dumps({"it": it, "score": score, **{k: [float((v > 0).mean()), float(v.mean())] for k, v in res.items()}}) + "\n")
        return score

    anc_b = anc_e = None                                      # RESUME: the schedules continue from their saved values
    if cfg.resume and not cfg.schedule_reset:
        if meta_r.get("beta_now") is not None:
            anc_b = (float(samples0), float(meta_r["beta_now"]))
        if meta_r.get("ent_now") is not None:
            anc_e = (float(samples0), float(meta_r["ent_now"]))
    beta_at = lambda it_, smp: beta_schedule(cfg, it_, smp, anc_b)
    ent_at = lambda smp: ent_schedule(cfg, smp, anc_e)
    winb_at = partial(winb_schedule, cfg)
    lrs_dev, lr_fac = None, 1.0
    if cfg.lr_scale == "sqrt":                                # the LRs follow the GLOBAL minibatch (devices x MB)
        lr_fac = math.sqrt(D * cfg.mb / cfg.lr_ref_mb)
        lrs_dev = tr.put(jnp.asarray([cfg.lr * lr_fac, cfg.vlr * lr_fac], jnp.float32))
    log_line(LOG, f"[lr] global minibatch {D} x {cfg.mb} = {D * cfg.mb:,}: LR {cfg.lr * lr_fac:.3g}, VLR {cfg.vlr * lr_fac:.3g}"
                  f" (LR_SCALE={cfg.lr_scale}, x{lr_fac:.3f}) | margin weight {cfg.margin_w}, WINB {cfg.winb}"
                  + (f" | horizon {cfg.horizon_samples:.3g} samples" if cfg.horizon_samples else "")
                  + (f" | schedules resume from beta {anc_b} ent {anc_e}" if (anc_b or anc_e) else ""))
    pt_index_fn = os.path.join(cfg.run_dir, "pt", "index.json")
    pt_index = json.load(open(pt_index_fn)) if os.path.exists(pt_index_fn) else []

    def export_pt(it_, smp, final=False):
        """pt/it<it>.pt (il/export.py-ready; the final one also carries the value net) in a ring of PT_KEEP files, plus
        one per PT_MILESTONE_SAMPLES and the final export, which are always kept."""
        fn = os.path.join(cfg.run_dir, "pt", f"it{it_}.pt")
        ckpt_io.save_pt(fn, with_head(params, vparams), pt_value(vparams) if final else None, team=cfg.team, it=it_,
                        extra={"samples": int(smp)})
        prev = max([e["samples"] for e in pt_index], default=-1)
        ms = cfg.pt_milestone_samples > 0 and (prev < 0 or int(smp // cfg.pt_milestone_samples) > int(prev // cfg.pt_milestone_samples))
        pt_index.append(dict(it=it_, samples=int(smp), file=os.path.basename(fn), keep=bool(final or ms)))
        if cfg.pt_keep > 0:
            ring = [e for e in pt_index if not e["keep"]]
            for e in ring[:-cfg.pt_keep]:
                try:
                    os.remove(os.path.join(cfg.run_dir, "pt", e["file"]))
                except OSError:
                    pass
                pt_index.remove(e)
        json.dump(pt_index, open(pt_index_fn, "w"), indent=0)
        log_line(LOG, f"  exported pt/it{it_}.pt at {smp / 1e6:,.0f}M learner samples (il/export.py-ready)" + (" [kept]" if ms or final else ""))

    def save_state(it_, smp, beta_now, ent_now):
        meta = {"it": it_, "team": cfg.team, "cfg": dataclasses.asdict(cfg), "snaps": pool.snaps, "hist": pool.hist, "best": best,
                "samples": int(smp), "beta_now": beta_now, "ent_now": ent_now}
        fn = os.path.join(cfg.run_dir, f"state_{it_}.npz")
        ckpt_io.save_train_state(fn, params, vparams, opt_p, opt_v, meta)
        olds = sorted(int(f[6:-4]) for f in os.listdir(cfg.run_dir) if f.startswith("state_") and f.endswith(".npz"))
        for o in olds[:-2]:
            os.remove(os.path.join(cfg.run_dir, f"state_{o}.npz"))
        return fn

    def replica_diff(tree):
        """Data parallel: max |difference| of the replicated params between devices (must stay ~0)."""
        m_ = 0.0
        for x in jax.tree_util.tree_leaves(tree):
            sh = [np.asarray(z.data) for z in getattr(x, "addressable_shards", [])]
            for z in sh[1:]:
                m_ = max(m_, float(np.abs(z.astype(np.float64) - sh[0]).max()))
        return m_

    t0 = time.time()
    evals_on = cfg.eval_every > 0 or cfg.eval_every_min > 0
    t_ev0 = time.time()
    best = evaluate(it0) if evals_on else -1.0
    t_eval, last_eval = time.time() - t_ev0, time.time()
    samples = samples0
    next_pt_s = (samples // cfg.pt_every_samples + 1) * cfg.pt_every_samples if cfg.pt_every_samples > 0 else float("inf")
    snap_due = False
    last_pt = time.time()
    beta_it = ent_it = None
    stop_reason = None
    it = it0
    for it in range(it0 + 1, cfg.iters + 1):
        ti = time.time()
        bank = tr._set_bank(bank, params, 0)                                 # LATEST = current learner
        oi, ot, om = opp_arrays(blk)
        beta_it, winb_it, ent_it = beta_at(it, samples), winb_at(it, samples), ent_at(samples)
        anchor = beta_it > 0
        train_pol = it > cfg.vwarm and not (cfg.resume and it - it0 <= cfg.vwarm_resume)
        key, kr, ku, kn = jax.random.split(key, 4)
        carry, traj, (fin, mar), boot = tr._rollout(params, bank, cs, tapes_train, oi, ot, carry, kr, om, tr.put(jnp.float32(winb_it)),
                                                    L=cfg.seg, learner_greedy=False, record=True)
        jax.block_until_ready(traj["rew"]); t_roll = time.time() - ti
        diag = bool(cfg.diag_every) and it % cfg.diag_every == 1
        params, vparams, opt_p, opt_v, st = tr._update(params, vparams, opt_p, opt_v, ref, traj, boot, ku, tr.put(jnp.float32(beta_it)),
                                                       tr.put(jnp.float32(ent_it)) if tr.use_ent else None, lrs_dev,
                                                       train_policy=train_pol, diag=diag, anchor=anchor)
        st = {k: float(v) for k, v in st.items()}
        st["winb"] = winb_it
        t_upd = time.time() - ti - t_roll
        if D > 1 and (it - it0 <= 3 or it % 20 == 0):
            st["dp_replica_diff"] = replica_diff((params, vparams))
            log_line(LOG, f"[dp] iteration {it}: max |param difference| between the {D} replicas {st['dp_replica_diff']:.3g}")
        if st["finite"] < 1:
            log_line(LOG, f"[STOP] non-finite params or loss after the update of iteration {it}: {json.dumps(st)}")
            raise SystemExit(3)
        if anchor and max(st["klu"], st["klm"]) > cfg.kl_max:
            log_line(LOG, f"[STOP] KL to the reference {st['klu']:.3f}/{st['klm']:.3f} > KL_MAX {cfg.kl_max} at iteration {it}")
            raise SystemExit(4)
        with open(os.path.join(cfg.run_dir, "stats.jsonl"), "a") as f:
            f.write(json.dumps({"it": it, "t_roll": t_roll, "t_upd": t_upd, "samples": samples + tr.N * D, **st}) + "\n")
        del traj, boot
        per = {}
        if int(be.t(carry[0])) > be.last_step:                               # episodes over: score, new games / opponents
            mar_h = np.asarray(mar)
            for k in range(K):
                ms = mar_h[blk_of_game == k]
                kind = blk[k][0] if blk[k][0] != "IL" else f"IL@{blk[k][2]}"
                per.setdefault(kind, []).extend(ms.tolist())
                pool.record(blk[k][0], ms)
            blk = pool.draw_blocks(K, G)
            carry = new_games(kn)
            with open(os.path.join(cfg.run_dir, "games.jsonl"), "a") as f:      # per-iteration W / margin by opponent
                f.write(json.dumps({"it": it, "samples": samples + tr.N * D, **{k: dict(n=len(v), w=float(np.mean(np.array(v) > 0)),
                                    margin=float(np.mean(v))) for k, v in per.items()}}) + "\n")
        samples += tr.N * D
        dt = time.time() - ti
        if it % cfg.log_every == 0 or per:
            log_line(LOG, f"it {it} {time.time() - t0:.0f}s ({dt:.1f}s: rollout {t_roll:.1f} update {t_upd:.1f}; {tr.N * D / dt:,.0f} samples/s) | "
                          + " ".join(f"{k} {np.mean(v):+,.0f} W{np.mean(np.array(v) > 0):.0%}({len(v)})" for k, v in sorted(per.items()))
                          + f" | loss {st['loss']:+.4f} u/q/m {st['l_u']:+.4f}/{st['l_q']:+.4f}/{st['l_m']:+.4f} kl u/m {st['klu']:.4f}/{st['klm']:.4f}"
                          + (f" beta {beta_it:.4f}" if cfg.beta_end >= 0 else "")
                          + (f" winb {winb_it:.3f}" if (cfg.winb_ramp_iters or cfg.winb_ramp_samples) else "")
                          + f" ent {st['ent']:.3f}" + (f"/{st['ent_q']:.3f}/{st['ent_m']:.3f} x{ent_it:.4f}" if tr.use_ent else "")
                          + f" akl {st['akl']:.5f} clip {st['clipfrac']:.3f} | epochs loss {st['loss_ep0']:+.4f}->{st['loss_epN']:+.4f}"
                          + f" v {st['lv_ep0']:.4f}->{st['lv_epN']:.4f} | v {st['l_v']:.4f} (res {st['v_res2']:.4f} / var {st['ret_var']:.4f})"
                          + f" EV gae/mc {st['ev_gae']:.3f}/{st['ev_mc']:.3f} adv sd {st['adv_std']:.4f}"
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
        eval_due = bool((cfg.eval_every and it % cfg.eval_every == 0) or
                        (cfg.eval_every_min > 0 and time.time() - last_eval >= 60 * cfg.eval_every_min))
        # wall-clock budget: stop now unless (an evaluation due now +) the next iteration + a final evaluation + saving fit
        last_it = it == cfg.iters
        if cfg.max_minutes > 0 and not last_it:
            left = T_START + 60 * cfg.max_minutes - time.time()
            t_ev_est = t_eval if t_eval > 5 else 60.0                         # (no evaluation measured yet: guess 1 min)
            if left < 1.1 * dt + (t_ev_est * (1 + eval_due) if evals_on else 0) + 90:
                stop_reason, last_it = f"MAX_MINUTES {cfg.max_minutes:g} ({left / 60:.1f} min left, iteration {dt / 60:.1f} min)", True
        if evals_on and (eval_due or last_it):
            t_ev0 = time.time()
            sc = evaluate(it)
            t_eval, last_eval = time.time() - t_ev0, time.time()
            if sc > best:
                best = sc
                ckpt_io.save_pt(os.path.join(cfg.run_dir, "best.pt"), with_head(params, vparams), pt_value(vparams), team=cfg.team, it=it, score=sc)
                log_line(LOG, f"  new best eval score {sc:.3f} -> best.pt")
        due_t = cfg.pt_every_min > 0 and time.time() - last_pt >= 60 * cfg.pt_every_min
        due_s = samples >= next_pt_s
        if due_s:
            next_pt_s = (samples // cfg.pt_every_samples + 1) * cfg.pt_every_samples
        if (cfg.pt_every and it % cfg.pt_every == 0) or due_t or due_s or last_it:     # policy only (export.py needs "state")
            export_pt(it, samples, final=last_it)
            last_pt = time.time()
        if (cfg.ckpt_every and it % cfg.ckpt_every == 0) or last_it:       # (+ the schedule values of the NEXT iteration:
            fn_state = save_state(it, samples, beta_at(it + 1, samples), ent_at(samples))   # a resume continues the curves)
        if last_it:
            json.dump(dict(it=it, samples=int(samples), state=fn_state, pt=os.path.join(cfg.run_dir, "pt", f"it{it}.pt"),
                           reason=stop_reason or "ITERS", minutes=round((time.time() - T_START) / 60, 1), beta=beta_it, ent=ent_it,
                           margin_w=cfg.margin_w, winb=winb_it),
                      open(os.path.join(cfg.run_dir, "FINAL.json"), "w"), indent=1)
            if stop_reason:
                log_line(LOG, f"[stop] {stop_reason}: final state {fn_state}, pt/it{it}.pt, FINAL.json")
                break
    log_line(LOG, f"[done] {it - it0 if cfg.iters > it0 else 0} iterations, {samples:,} samples, {time.time() - t0:.0f}s")
    return dict(params=params, vparams=vparams, trainer=tr, best=best, base_eval=base_eval)


if __name__ == "__main__":
    main()
