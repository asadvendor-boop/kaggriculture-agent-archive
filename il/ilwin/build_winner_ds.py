#!/usr/bin/env python3
"""Winner-imitation fine-tune data (Sep 29 track): for each of OUR live games, the WINNER's seat becomes one training seat
file in exactly the format of il/build_dataset3.py + il/live/srv_build_x6.py (the format v13_p3 was trained on):
  <out>/seats/<ep>_<seat>.npz  planes u8, glob f16 (98), pos i8, ufeat f16, prev_op i8, op i8, qty i8, market i8 (21 heads),
                               money f32, meta i64
  <out>/x6/<ep>_<seat>.npy     features6.future_features (81, f16)
We won  -> our own moves are the labels (after LABEL HYGIENE, below).  We lost -> the OPPONENT's moves are the labels.

LABEL HYGIENE (our seat, i.e. games we won)
 * The rule layer's edits are reconstructed EXACTLY with the rule code that ran live (the features.py embedded in each live
   package, extracted to pkg_src/): the tape's action A_t = straddle_fix(real obs, M_t), M_t = midnight_fix(policy obs, P_t).
 * STRADDLE legs (ST2 only, STRADDLE=2: the legs go LAST in the order list, or FIRST once the rival is seen straddling) are
   decomposed with the package's own state machine: at each step the candidates "no leg / last order is the leg / first order
   is the leg" are replayed through straddle_fix and must reproduce the tape's list exactly (a tiny beam keeps ambiguous
   branches until the next step decides). Labels use M_t (legs removed). The policy's input features use the straddle_obs
   view (shed without the straddle wheat, cash + sell-back value), exactly what the live policy saw.
 * SE guard BUY_LAND (default DROPPED): at every step where the guard could have fired (3 quadrants, day 10 hour >= 12 ..
   day 13, cash >= 4,300) and M_t holds a BUY_LAND, the family's policy (v13_p3 for ST2/SOH/N20MW8) is run on that one state;
   if the policy did not order the land itself, the label is midnight_fix(...) with SE_GUARD=0 (the exact list without the
   rule's land). Reason: the SE decision stays with the rule layer (SE_GUARD / SE_SHOPS), the net must not learn to bypass it.
 * Kept as labels (consistent with rules we keep): SELL_ORDER=held (slot order is invisible to the 21 quantity heads),
   MIDNIGHT overflow sells (the rule only sells what is still over after the net's own sells), HOMING / FINAL_FIX (idempotent
   end-of-game rules; the final turn's market is overwritten at play time anyway).
OPPONENT seat (games we lost): their own moves; wheat "drain straddle" legs of straddling opponents (BUY_PRODUCT WHEAT >= 10
at a drain step, SELL WHEAT >= half of it the next step) -> those two market heads are IGNORED (-100): our STRADDLE rule
already does it and the net must not learn it twice.
FAILED ORDERS (both seats; engine receipts): unit ops the engine did not execute -> op/qty label -1 (ignored by the loss);
market heads whose orders were ALL rejected (INSUFFICIENT_CASH / SHED_FULL / LAND_EXHAUSTED / UNIT_CAP / MALFORMED /
DEAD_SLOT) -> -100 (ignored). Phantom SELLs (SHED_EMPTY) keep their requested quantity: standing sell orders are part of the
leaders' style the net was trained on. Only the first 10 orders count (the engine ignores the rest). Steps with no recorded
action (timeouts) are ignored entirely.
meta = [ep, seat, bank, opp_bank, exact, seed, team_id, win=1, true_team_id, src_idx, ours, |margin|, weight_milli,
        straddle_legs_removed, se_lands_removed, verify_permille]   (train6.py reads meta[2], [3], [6], [7] only)

Usage (one worker):  build_winner_ds.py --out DIR --tapes data/tapes_st2:WL data/tapes_npse:L ... [--worker i --workers n]
          merge:     build_winner_ds.py --out DIR --merge [--margin-weight none|loss|all]
ROLE per tape dir: W = only games we won (our moves), L = only games we lost (their moves), WL = both.
"""
import sys, os, json, gzip, glob, copy, time, argparse, types, collections, math
import numpy as np

K = os.environ.get("KROOT", "/Users/asad.ali/kaggriculture")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{K}/tools"); sys.path.insert(0, f"{K}/il")
from harness import kagsim_trace, PASS          # noqa: E402
import features as F                            # noqa: E402  repo encoder (encode_obs identical in every package; checked)
import features6 as F6                          # noqa: E402

OUR_NAME = "Syed Asad Ali"
OUR_TEAM = 10                                   # the style id we play with (DSM's embedding; TEAM=10 in every package)
UB = F.MAX_UNITS + 2 + len(F.ITEMS) + 2
IGN = -100                                      # market-head "ignore" label (torch cross_entropy ignore_index)
FAIL_REASONS = {"INSUFFICIENT_CASH", "SHED_FULL", "LAND_EXHAUSTED", "UNIT_CAP", "MALFORMED_ITEM", "DEAD_SLOT"}
FAMILIES = {   # tape dir -> the live package (its embedded rule code) and the policy checkpoint that played it
    "tapes_st2": dict(pkg="NEURAL_20MW8_SOH_ST2", ckpt="ilsp_v13_p3"),
    "tapes_soh": dict(pkg="NEURAL_20MW8_SE_SOH", ckpt="ilsp_v13_p3"),
    "tapes_n20mw8": dict(pkg="NEURAL_20MW8_SE", ckpt="ilsp_v13_p3"),
    "tapes_npse": dict(pkg="NEURAL_20MW8NP_SE_SOH", ckpt="ilsp_v13_newplan"),
    "tapes_nps": dict(pkg="NEURAL_20MW8NPS_SOH", ckpt="ilsp_v13_newplan_soup3"),
    "tapes_n20mw5": dict(pkg="NEURAL_20MW5_SE", ckpt="g20_w5_dsm_s0"),
    # Sep 29: the ALL-rules builds (v13_p3 + ST2 + wool/eggs/milk/demand rules; ALLG + STRADDLE_PREBUY). rule_ign: in OUR wins,
    # every head / unit op where the tape differs from v13_p3's own pre-rule output is IGNORED (the rules still fire at play time;
    # learning their moves too could double them - the sheep tip-over lesson). Labels there = the network's own moves only.
    "tapes_all": dict(pkg="NEURAL_20MW8_ST2_ALL", ckpt="ilsp_v13_p3", rule_ign=True),
    "tapes_allg": dict(pkg="NEURAL_20MW8_ST2_ALLG", ckpt="ilsp_v13_p3", rule_ign=True),
}
SRC_ORDER = ["tapes_n20mw5", "tapes_n20mw8", "tapes_nps", "tapes_npse", "tapes_soh", "tapes_st2",   # = sorted(old FAMILIES)
             "tapes_today", "tapes_top12", "tapes_neural", "tapes_all", "tapes_allg", "tapes_lead29"]  # stable meta src ids
HEAD = {h: i for i, (h, _) in enumerate(F.MARKET_HEADS)}


def head_of(o):
    if not (isinstance(o, (list, tuple)) and o):
        return None
    if o[0] in ("HIRE", "BUY_LAND"):
        return o[0]
    if len(o) >= 2:
        k = f"{o[0]}:{o[1]}"
        return k if k in HEAD else None
    return None


# ---------------------------------------------------------------------------------------------------------------- setup
_PKG = {}


def pkg_module(name):
    """The live package's own features.py (RULES baked in), as a fresh module (own straddle state)."""
    if name not in _PKG:
        src = open(f"{HERE}/pkg_src/{name}_features.py").read()
        m = types.ModuleType(f"pkgF_{name}"); m.__file__ = f"pkg_src/{name}_features.py"
        exec(compile(src, m.__file__, "exec"), m.__dict__)
        _PKG[name] = m
    return _PKG[name]


_NET = {}


def policy_net(ckpt, device):
    import torch
    if ckpt not in _NET:
        from model import Net
        c = torch.load(f"{K}/il/ckpt/{ckpt}.pt", map_location="cpu", weights_only=False)
        n = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"),
                n_aux=c.get("n_aux", 0), ar=c.get("ar", 0), menu=c.get("menu", 0))
        n.load_state_dict(c["state"]); n.eval()
        if device != "cpu":
            torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        _NET[ckpt] = n.to(device)
    return _NET[ckpt]


def policy_actions(net, device, FM, obs_list, prev_list, team=OUR_TEAM, tau=0.5):
    """NumpyAgent's decode (il/np_runtime.py) on a torch forward: greedy legal op, clamped quantity, market heads fire when
    P(order) > tau; decode_market with the package's SELL_ORDER. Returns the pre-rule actions P_t and their op lists."""
    import torch
    from np_runtime import legal_mask_np
    enc = [F6.encode_obs(o, p) for o, p in zip(obs_list, prev_list)]
    n = len(enc); nu = [len(e[2]) for e in enc]
    pos = np.zeros((n, F.MAX_UNITS, 2), np.int64); uin = np.zeros((n, F.MAX_UNITS, F.U), np.float32)
    for j, e in enumerate(enc):
        pos[j, :nu[j]] = e[2]; uin[j, :nu[j]] = e[3]
    off = np.cumsum([0] + [h for _, h in F.MARKET_HEADS])
    outs = []
    with torch.no_grad():
        for b in range(0, n, 256):
            sl = slice(b, min(n, b + 256))
            pl = torch.from_numpy(np.stack([e[0] for e in enc[sl]]).astype(np.float32) / 255.0).to(device)
            gl = torch.from_numpy(np.stack([e[1] for e in enc[sl]])).to(device)
            tt = torch.full((pl.shape[0],), team, dtype=torch.long, device=device) if net.n_teams else None
            lop, lq, lm, _ = net(pl, gl, torch.from_numpy(pos[sl]).to(device), torch.from_numpy(uin[sl]).to(device), tt)
            outs.append((lop.cpu().numpy(), lq.cpu().numpy(), lm.cpu().numpy()))
    lop = np.concatenate([o[0] for o in outs]); lq = np.concatenate([o[1] for o in outs]); lm = np.concatenate([o[2] for o in outs])
    acts, opss = [], []
    for j, obs in enumerate(obs_list):
        priv = obs["private"]; shed = priv.get("shed", {}) or {}; invs = priv.get("inventories") or [{}]
        cmds, ops = [], []
        for k in range(nu[j]):
            allowed = legal_mask_np(obs, k, pos[j, k], F)
            oc = int(np.argmax(np.where(allowed, lop[j, k], -1e9))); name = F.OPS[oc]; qc = 0
            if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                qc = int(np.argmax(lq[j, k])); it = name.split("_", 1)[1]
                have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                qc = max(0, min(qc, int(have) - 1))
            cmds.append(F.unit_command(oc, qc)); ops.append(oc)
        classes = []
        for h in range(len(F.MARKET_HEADS)):
            z = lm[j, off[h]:off[h + 1]].astype(np.float64); p = np.exp(z - z.max()); p /= p.sum()
            classes.append(int(1 + np.argmax(p[1:])) if 1.0 - p[0] > tau else 0)
        acts.append({"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": FM.decode_market(classes, obs)})
        opss.append(ops)
    return acts, opss


def unit_ops(a):
    if not isinstance(a, dict):
        return None
    return [F.unit_op_label(c)[0] for c in [a.get("farmer")] + list(a.get("hands") or [])]


# ------------------------------------------------------------------------------------------------- straddle decomposition
def _is_wheat_leg(o):
    return isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] == "WHEAT" and o[0] in ("BUY_PRODUCT", "SELL")


def _policy_grammar_ok(M, t):
    """Could the policy (+ the non-straddle rules) have emitted M's wheat orders? decode_market emits at most one order per
    head, BUY_PRODUCT WHEAT in {1..16, 35} and SELL in {1..16, 1000}; only midnight_fix (hour 23) and FINAL_FIX (step 718)
    produce other SELL quantities. A straddle leg (10..100 wheat) left inside M is therefore usually impossible."""
    if t >= 718 or not isinstance(M, dict):
        return True
    cnt = collections.Counter()
    for o in M.get("market") or []:
        if not _is_wheat_leg(o):
            continue
        cnt[o[0]] += 1
        try:
            q = int(o[2])
        except Exception:
            return False
        if o[0] == "BUY_PRODUCT" and not (1 <= q <= 16 or q == 35):
            return False
        if o[0] == "SELL" and t % 24 != 23 and not (1 <= q <= 16 or q == 1000):
            return False
    return cnt["BUY_PRODUCT"] <= 1 and cnt["SELL"] <= 1


def decompose_straddle(FM, obs_real, A, seat, n_steps, stats=None):
    """Exact inversion of FM.straddle_fix over a whole game: returns (M list of pre-straddle actions, held list, legs, n_ambiguous)
    or None when no consistent explanation exists. held[t] = straddle wheat open at the START of step t (policy view).
    Candidates per step: no leg / the last order is the leg / the first order is the leg; a candidate must reproduce the tape's
    list through the package's straddle_fix AND leave a pre-straddle list the policy could have emitted (_policy_grammar_ok).
    Remaining ties (a leg of 10..16 or 35 wheat) keep up to 8 branches, the ones with more legs first; the next steps decide
    (a wrong 'leg' branch expects a sell-back the tape does not have)."""
    key = (0, seat)                                   # the package's straddle state key (Kaggle obs have no _seed_tag)
    FM._STRAD.clear()
    beams = [(None, [], [], [])]                      # (state, M list, held list, legs list)
    amb = 0
    for t in range(n_steps):
        a = A[t]
        new, loose = [], []
        for st, Ms, Hs, Ls in beams:
            held_now = int(st["held"]) if (st is not None and st.get("step", -1) < t) else 0
            if not isinstance(a, dict):               # no action recorded (timeout): the agent was not called
                new.append((st, Ms + [None], Hs + [held_now], Ls)); continue
            mk = list(a.get("market") or [])
            cands = [(a, None)]
            if mk and _is_wheat_leg(mk[-1]):
                cands.append((dict(a, market=mk[:-1]), ("last", mk[-1])))
            if len(mk) > 1 and _is_wheat_leg(mk[0]):
                cands.append((dict(a, market=mk[1:]), ("first", mk[0])))
            for M, leg in cands:
                if st is None:
                    FM._STRAD.pop(key, None)
                else:
                    FM._STRAD[key] = copy.deepcopy(st)
                out = FM.straddle_fix(obs_real[t], copy.deepcopy(M))
                if out == a:
                    s2 = copy.deepcopy(FM._STRAD.get(key))
                    b = (s2, Ms + [M], Hs + [held_now], Ls + ([(t, leg[0], leg[1])] if leg else []))
                    (new if _policy_grammar_ok(M, t) else loose).append(b)
        if not new:
            new = loose                               # nothing grammatical: keep the consistent ones (counted)
            if stats is not None and loose:
                stats["straddle_grammar_fallback"] += 1
        if not new:
            return None
        if len(new) > 1:
            new.sort(key=lambda b: -len(b[3]))
            uniq, seen = [], set()
            for b in new:
                sig = (json.dumps(b[0], sort_keys=True, default=str), len(b[3]))
                if sig not in seen:
                    seen.add(sig); uniq.append(b)
            if len(uniq) > 1:
                amb += 1
            new = uniq[:8]
        beams = new
    st, Ms, Hs, Ls = max(beams, key=lambda b: len(b[3]))
    return Ms, Hs, Ls, amb


def straddle_view(FM, obs, held, seat):
    """FM.straddle_obs with the straddle state set to `held` (what the live policy saw)."""
    if held <= 0:
        return obs
    FM._STRAD[(0, seat)] = {"step": int(obs["step"]) - 1, "held": int(held)}
    o = FM.straddle_obs(obs)
    FM._STRAD.pop((0, seat), None)
    return o


# ------------------------------------------------------------------------------------------------------------ game build
def se_window(FM, obs, t):
    day, hour = divmod(t, 24)
    farm = obs["farms"][int(obs["player"])]
    nq = len(farm.get("unlocked_quadrants") or [])
    if not (getattr(FM, "SE_GUARD", 0) and nq == 3 and t < 718):
        return False
    if not ((day == 10 and hour >= FM.SE_HOUR) or 10 < day <= FM.SE_LAST_DAY):
        return False
    if float(farm["money"]) < 4000 + FM.SE_RESERVE:
        return False
    if getattr(FM, "SE_SHOPS", 0):
        shops = ((obs.get("town") or {}).get("unlocked_shops")) or []
        if sum(x in FM.CROP_SHOPS for x in shops[:3]) < FM.SE_SHOPS:
            return False
    return True


def midnight_with(FM, obs, act, se):
    old = FM.SE_GUARD; FM.SE_GUARD = se
    try:
        return FM.midnight_fix(obs, copy.deepcopy(act))
    finally:
        FM.SE_GUARD = old


def build_game(path, src_name, args, voc, stats):
    d = json.load(gzip.open(path, "rt"))
    rw = d.get("rewards") or [None, None]
    if rw[0] is None or rw[1] is None:
        stats["skip_no_reward"] += 1; return None
    us = d.get("seat")
    names = d.get("names") or d.get("teams") or [None, None]
    W = 0 if rw[0] > rw[1] else 1
    if rw[0] == rw[1]:
        stats["skip_tie"] += 1; return None
    margin = abs(rw[0] - rw[1])
    ours = us is not None and W == us
    fam = FAMILIES.get(src_name) if ours else None
    Araw = [list(d["actions"][s]) + [None] * (720 - len(d["actions"][s])) for s in (0, 1)]
    Aplay = [[x if isinstance(x, dict) else dict(PASS) for x in Araw[s]] for s in (0, 1)]
    g = kagsim_trace.Game(int(d["seed"]), 720, True)
    obs_real, rec, urec = [], [], []
    n_steps = 0
    for t in range(720):
        obs_real.append(g.observe(W))
        g.step(Aplay[0][t], Aplay[1][t])
        tr = g.trace_step(g.trace_len() - 1)
        rec.append([r for r in tr.get("receipts", []) if r["seat"] == W])
        urec.append({r["unit"]: r for r in tr.get("unit_receipts", []) if r["seat"] == W})
        n_steps = t + 1
        if g.done:
            break
    if not (abs(g.reward(0) - rw[0]) < 0.5 and abs(g.reward(1) - rw[1]) < 0.5):
        stats["skip_not_exact"] += 1; return None
    AW = Araw[W][:n_steps]
    info = dict(ep=int(d["ep"]), src=src_name, winner=W, ours=bool(ours), margin=float(margin), steps=n_steps,
                opp=d.get("opp"), opp_rating=d.get("opp_rating"), names=names)
    # ---- label actions + policy-view observations
    L = list(AW); obs_feat = obs_real; se_removed = 0; legs = []; verify = -1; P_pol = None
    if ours and fam is not None:
        FM = pkg_module(fam["pkg"])
        if getattr(FM, "STRADDLE", 0):
            dec = decompose_straddle(FM, obs_real, AW, W, n_steps, stats)
            if dec is None:
                stats["skip_straddle_undecomposable"] += 1; return None
            Ms, Hs, legs, amb = dec
            stats["straddle_ambiguous_steps"] += amb
            obs_feat = [straddle_view(FM, o, h, W) for o, h in zip(obs_real, Hs)]
            L = Ms
        # SE guard land: policy check at every step the guard could have fired with a land in the pre-straddle list
        cand = [t for t in range(n_steps) if isinstance(L[t], dict) and any(head_of(o) == "BUY_LAND" for o in (L[t].get("market") or []))
                and se_window(FM, obs_feat[t], t)]
        if cand and args.se_label == "drop":
            ck = fam["ckpt"]
            if os.path.exists(f"{K}/il/ckpt/{ck}.pt") and not args.no_policy:
                net = policy_net(ck, args.device)
                prevs = [unit_ops(AW[t - 1]) if t > 0 else None for t in cand]
                P, _ = policy_actions(net, args.device, FM, [obs_feat[t] for t in cand], prevs)
                for t, p in zip(cand, P):
                    if any(head_of(o) == "BUY_LAND" for o in p["market"]):
                        stats["se_land_by_policy"] += 1; continue
                    m1 = midnight_with(FM, obs_feat[t], p, FM.SE_GUARD)
                    if m1 == L[t]:
                        L[t] = midnight_with(FM, obs_feat[t], p, 0); stats["se_land_removed_exact"] += 1
                    else:
                        L[t] = dict(L[t], market=[o for o in L[t]["market"] if head_of(o) != "BUY_LAND"])
                        stats["se_land_removed_fallback"] += 1
                    se_removed += 1
            else:
                for t in cand[:1]:                     # no policy: the first eligible land is the guard's (flagged)
                    L[t] = dict(L[t], market=[o for o in L[t]["market"] if head_of(o) != "BUY_LAND"])
                    se_removed += 1; stats["se_land_removed_heuristic"] += 1
        if args.verify or fam.get("rule_ign"):        # full-game check: policy + the package's rules == the tape
            verify, P_pol = verify_game(FM, fam, args, obs_feat, AW, Ms if getattr(FM, "STRADDLE", 0) else AW, n_steps, stats)
            if fam.get("rule_ign") and P_pol is None:
                stats["skip_rule_ign_no_policy"] += 1; return None
    elif ours:
        stats["ours_unknown_family"] += 1
    # ---- arrays
    T = n_steps
    b = dict(planes=np.zeros((T, F.C, F.B, F.B), np.uint8), glob=np.zeros((T, F.G), np.float16),
             pos=np.zeros((T, F.MAX_UNITS, 2), np.int8), ufeat=np.zeros((T, F.MAX_UNITS, UB), np.float16),
             prev_op=np.full((T, F.MAX_UNITS), -1, np.int8), op=np.full((T, F.MAX_UNITS), -1, np.int8),
             qty=np.full((T, F.MAX_UNITS), -1, np.int8), market=np.zeros((T, F.N_HEADS), np.int8),
             money=np.zeros((T, 2), np.float32))
    x6 = np.zeros((T, F6.E), np.float16)
    prev = None; n_mask_u = n_ign_h = n_churn = n_noact = 0
    n_rule_u = n_rule_h = 0
    rule_ign = bool(ours and fam is not None and fam.get("rule_ign") and P_pol is not None)
    churn_sell_next = False
    for t in range(T):
        o = obs_feat[t]
        p, gl, ps, u = F.encode_obs(o, None)
        n = len(ps)
        b["planes"][t] = p; b["glob"][t] = gl; b["pos"][t, :n] = ps; b["ufeat"][t, :n] = u[:, :UB]
        x6[t] = F6.future_features(o)
        if prev is not None:
            b["prev_op"][t, :n] = [prev[k] if k < len(prev) else -1 for k in range(n)]
        ro = obs_real[t]
        b["money"][t] = (ro["farms"][W]["money"], ro["farms"][1 - W]["money"])
        a = L[t]
        if not isinstance(a, dict):                   # no recorded action: ignore the step (all labels -1 / -100)
            n_noact += 1; prev = None
            b["market"][t] = IGN; churn_sell_next = False
            continue
        cmds = [a.get("farmer")] + list(a.get("hands") or [])
        pcmds = ([P_pol[t].get("farmer")] + list(P_pol[t].get("hands") or [])) if rule_ign else None
        cur = []
        for k in range(n):
            oc, qc = F.unit_op_label(cmds[k] if k < len(cmds) else ["PASS"])
            cur.append(oc)
            if rule_ign and (oc, qc) != F.unit_op_label(pcmds[k] if k < len(pcmds) else ["PASS"]):
                b["op"][t, k] = -1; b["qty"][t, k] = -1; n_rule_u += 1   # a rule's unit edit (or HOMING): not a label
                continue
            ur = urec[t].get(k)
            if args.fail_mask and ur is not None and not ur.get("executed") and F.OPS[oc] != "PASS":
                oc = qc = -1; n_mask_u += 1
            b["op"][t, k] = oc; b["qty"][t, k] = qc
        prev = cur
        mk = [o_ for o_ in (a.get("market") or []) if isinstance(o_, (list, tuple))][:10]
        lab = F.market_labels(mk).astype(np.int16)
        if args.fail_mask:                            # map label orders onto the recorded slots (greedy, in order)
            recs = {r["slot"]: r for r in rec[t]}; rec_list = [x for x in (AW[t].get("market") or []) if isinstance(x, (list, tuple))][:10] if isinstance(AW[t], dict) else []
            used = set(); per_head = collections.defaultdict(list)
            for o_ in mk:
                h = head_of(o_)
                if h is None:
                    continue
                j = next((i for i, r_ in enumerate(rec_list) if i not in used and list(r_) == list(o_)), None)
                if j is None:
                    per_head[h].append(None); continue
                used.add(j); per_head[h].append(recs.get(j))
            for h, rs in per_head.items():
                if rs and all(r_ is not None and int(r_.get("executed", 0)) == 0 and r_.get("reason") in FAIL_REASONS for r_ in rs):
                    lab[HEAD[h]] = IGN; n_ign_h += 1
        if rule_ign:                                  # heads a rule changed (goose/sheep/cow/straw buys, midnight tops) -> ignored
            pm = [o_ for o_ in (P_pol[t].get("market") or []) if isinstance(o_, (list, tuple))][:10]
            lp = F.market_labels(pm).astype(np.int16)
            for h in range(len(lab)):
                if lab[h] != IGN and lab[h] != lp[h]:
                    lab[h] = IGN; n_rule_h += 1
        if churn_sell_next:
            lab[HEAD["SELL:WHEAT"]] = IGN; churn_sell_next = False
        if not ours and args.strip_churn and t % 4 == 0 and t + 1 < T:
            bought = sum(int(r_.get("executed", 0)) for r_ in rec[t] if r_.get("op") == "BUY_PRODUCT" and r_.get("item") == "WHEAT")
            nxt = AW[t + 1] if isinstance(AW[t + 1], dict) else {}
            sold_req = 0
            for o_ in (nxt.get("market") or [])[:10]:
                if isinstance(o_, (list, tuple)) and len(o_) >= 3 and o_[0] == "SELL" and o_[1] == "WHEAT":
                    try:
                        sold_req += max(0, int(o_[2]))
                    except Exception:
                        pass
            if bought >= 10 and sold_req >= 0.5 * bought:
                lab[HEAD["BUY_PRODUCT:WHEAT"]] = IGN; churn_sell_next = True; n_churn += 1
        b["market"][t] = lab.astype(np.int8)
    # ---- team ids
    true_team = OUR_TEAM
    wname = names[W] if names else None
    if not ours:
        if wname == OUR_NAME:                         # our other submission beat us: its family is unknown here
            stats["skip_self_match"] += 1; return None
        true_team = voc.get(wname, args.generic_team)
    team = OUR_TEAM if (ours or args.team_mode == "ours") else true_team
    meta = np.array([int(d["ep"]), W, int(rw[W]), int(rw[1 - W]), 1, int(d["seed"]) % (2 ** 31), team, 1, true_team,
                     args.src_index.get(src_name, -1), int(ours), int(round(margin)), 1000, len(legs), se_removed, verify], np.int64)
    info.update(team=int(team), true_team=int(true_team), unit_masked=n_mask_u, heads_ignored=n_ign_h, churn_legs=n_churn,
                straddle_legs=len(legs), se_removed=se_removed, no_action_steps=n_noact, verify_permille=verify,
                rule_units_ignored=n_rule_u, rule_heads_ignored=n_rule_h)
    stats["rule_units_ignored"] += n_rule_u; stats["rule_heads_ignored"] += n_rule_h
    return b, x6, meta, info


def verify_game(FM, fam, args, obs_feat, AW, M, n_steps, stats):
    """Replays the family's policy on every step (prev = the policy's own previous ops) and applies the package's rules:
    returns the per-mille of steps whose pre-straddle action equals the tape's exactly."""
    ck = fam["ckpt"]
    if not os.path.exists(f"{K}/il/ckpt/{ck}.pt"):
        return -1, None
    net = policy_net(ck, args.device)
    cut = min(n_steps, 691)                             # before HOMING can edit units, prev = the tape's own unit ops
    prevs = [None] + [unit_ops(AW[t - 1]) for t in range(1, cut)]
    P, ops = policy_actions(net, args.device, FM, obs_feat[:cut], prevs)
    for t in range(cut, n_steps):                     # last turns: prev = the policy's own previous ops
        p1, o1 = policy_actions(net, args.device, FM, [obs_feat[t]], [ops[-1]])
        P.append(p1[0]); ops.append(o1[0])
    ok = 0; first_bad = None
    for t in range(n_steps):
        if not isinstance(AW[t], dict):
            continue
        m = FM.midnight_fix(obs_feat[t], copy.deepcopy(P[t]))
        if m == M[t]:
            ok += 1
        elif first_bad is None:
            first_bad = t
    stats["verify_steps"] += n_steps; stats["verify_ok"] += ok
    if first_bad is not None:
        stats["verify_first_bad_examples"] = stats.get("verify_first_bad_examples", 0) + 1
    return int(round(1000 * ok / max(1, n_steps))), P


# ----------------------------------------------------------------------------------------------------------- selection
def load_heldout(args):
    held = set()
    try:
        held |= {int(w["ep"]) for w in json.load(open(f"{K}/il/live/ladder_worlds.json"))}     # the 464 (both splits)
    except Exception as e:
        print("WARN ladder_worlds:", e)
    for f in glob.glob(f"{K}/il/live/top_worlds_*_300.json"):
        if "llall" in f or "livelosses" in f or "speclosses" in f:
            continue
        try:
            held |= {int(w["ep"]) for w in json.load(open(f))}
        except Exception:
            pass
    try:
        held |= set(int(x) for x in json.load(open(f"{K}/il/live/fresh_heldout_eps.json")))
    except Exception:
        pass
    ll = set()
    for f in glob.glob(f"{K}/il/live/top_worlds_llall_300.json") + glob.glob(f"{K}/il/live/top_worlds_livelosses_300.json"):
        try:
            ll |= {int(w["ep"]) for w in json.load(open(f))}
        except Exception:
            pass
    return held, ll


def load_dates():
    end = {}
    for f in glob.glob(f"{K}/data/rows_*.json"):
        try:
            for r in json.load(open(f)):
                end[int(r["ep"])] = r.get("end") or ""
        except Exception:
            pass
    for f in glob.glob(f"{K}/data/tapes_*/index.json"):
        try:
            for ep, v in json.load(open(f)).items():
                if isinstance(v, dict) and (v.get("end") or v.get("day")):
                    end.setdefault(int(ep), v.get("end") or v.get("day"))
        except Exception:
            pass
    return end


def select_games(args):
    held, ll = load_heldout(args)
    ends = load_dates() if args.since else {}
    wn = {n.strip() for n in json.load(open(args.winner_names))} if args.winner_names else None
    excl = set()
    for mp in args.exclude_manifest:
        try:
            excl |= {(int(g["ep"]), int(g["winner"])) for g in json.load(open(mp))["games"]}
        except Exception as ex:
            print("WARN exclude-manifest", mp, ex)
    jobs, why = [], collections.Counter()
    for spec in args.tapes:
        dpath, role = (spec.rsplit(":", 1) + ["WL"])[:2] if ":" in spec else (spec, "WL")
        dname = os.path.basename(dpath.rstrip("/"))
        dfull = dpath if os.path.isabs(dpath) else f"{K}/{dpath}"
        for f in sorted(glob.glob(f"{dfull}/*.json.gz")):
            ep = int(os.path.basename(f).split(".")[0])
            if args.eps and ep not in args.eps:
                continue
            if ep in held:
                why[f"{dname} held-out ruler world"] += 1; continue
            if args.exclude_llall and ep in ll:
                why[f"{dname} llall world"] += 1; continue
            if args.since and (ends.get(ep) or "") < args.since:
                why[f"{dname} before {args.since}"] += 1; continue
            try:
                d = json.load(gzip.open(f, "rt"))
            except Exception:
                why[f"{dname} unreadable"] += 1; continue
            rw = d.get("rewards") or [None, None]; us = d.get("seat")
            if rw[0] is None or rw[1] is None:
                why[f"{dname} no rewards"] += 1; continue
            if us is None:                             # a leader tape (both seats top-band): the winner, as "L" role
                won = False
            else:
                won = rw[us] > rw[1 - us]
            if (won and "W" not in role) or ((not won) and "L" not in role):
                why[f"{dname} role filter"] += 1; continue
            if abs(rw[0] - rw[1]) < args.min_margin:
                why[f"{dname} near tie"] += 1; continue
            r = d.get("opp_rating")
            if args.min_opp_rating and r is not None and r < args.min_opp_rating:
                why[f"{dname} opp rating < {args.min_opp_rating}"] += 1; continue
            W = 0 if rw[0] > rw[1] else 1
            if wn is not None:                            # Sep 29: only games whose WINNER is one of the listed teams
                nm = d.get("names") or d.get("teams") or [None, None]
                if (nm[W] or "").strip() not in wn:
                    why[f"{dname} winner not in list"] += 1; continue
            if (ep, W) in excl:
                why[f"{dname} already in exclude-manifest"] += 1; continue
            jobs.append((f, dname, ep, W, bool(won)))
    # dedupe by (ep, winner seat): games we won first (their rule edits are known), then losses
    jobs.sort(key=lambda j: (not j[4], args.order.index(j[1]) if j[1] in args.order else 99, j[2]))
    seen, out = set(), []
    for j in jobs:
        if (j[2], j[3]) in seen:
            why[f"{j[1]} duplicate game"] += 1; continue
        seen.add((j[2], j[3])); out.append(j)
    return out, why


def merge(args):
    info = []
    for f in sorted(glob.glob(f"{args.out}/manifest_part*.jsonl")):
        info += [json.loads(l) for l in open(f) if l.strip()]
    stats = collections.Counter()
    for f in sorted(glob.glob(f"{args.out}/stats_part*.json")):
        stats.update(json.load(open(f)))
    # sample weights per seat file: role weight x optional margin factor, normalised to mean 1 over the set
    w = {}
    for x in info:
        role_w = args.w_win if x["ours"] else args.w_loss
        mf = 1.0
        if args.margin_weight == "all" or (args.margin_weight == "loss" and not x["ours"]):
            mf = min(args.mw_max, max(args.mw_min, x["margin"] / args.mw_scale))
        w[f"{x['ep']}_{x['winner']}"] = role_w * mf
    if w:
        mean = sum(w.values()) / len(w)
        w = {k: v / mean for k, v in w.items()}
    json.dump(w, open(f"{args.out}/win_weights.json", "w"))
    by = collections.defaultdict(lambda: collections.Counter())
    for x in info:
        c = by[x["src"]]; c["games"] += 1; c["W(our moves)" if x["ours"] else "L(their moves)"] += 1; c["steps"] += x["steps"]
    summ = dict(games=len(info), steps=sum(x["steps"] for x in info), by_dir={k: dict(v) for k, v in by.items()},
                stats=dict(stats), weights=dict(n=len(w), min=min(w.values()) if w else None, max=max(w.values()) if w else None))
    json.dump(dict(summary=summ, games=info), open(f"{args.out}/manifest.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tapes", nargs="*", default=[])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--since", default="", help="UTC ISO prefix, e.g. 2026-09-27T12:00 (dates from data/rows_*.json)")
    ap.add_argument("--min-opp-rating", type=float, default=0.0)
    ap.add_argument("--min-margin", type=float, default=500.0, help="drop near ties (|margin| below this)")
    ap.add_argument("--team-mode", choices=["ours", "true"], default="ours",
                    help="ours: every winner sample is team 10 (what we play with); true: opponents keep their vocab id")
    ap.add_argument("--generic-team", type=int, default=48, help="team id for winners outside team_vocab.json (needs N_TEAMS=49)")
    ap.add_argument("--se-label", choices=["drop", "keep"], default="drop")
    ap.add_argument("--fail-mask", type=int, default=1)
    ap.add_argument("--strip-churn", type=int, default=1)
    ap.add_argument("--exclude-llall", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0, help="full policy+rules replay check of our winning games")
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--device", default=os.environ.get("BUILD_DEVICE", "cpu"))
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--margin-weight", choices=["none", "loss", "all"], default="none")
    ap.add_argument("--mw-scale", type=float, default=5000.0); ap.add_argument("--mw-min", type=float, default=0.5)
    ap.add_argument("--mw-max", type=float, default=2.0)
    ap.add_argument("--w-win", type=float, default=1.0); ap.add_argument("--w-loss", type=float, default=1.0)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--eps", default="", help="comma list: only these episodes (dry runs)")
    ap.add_argument("--winner-names", default="", help="json list of team names: keep only games whose winner is one of them")
    ap.add_argument("--exclude-manifest", nargs="*", default=[], help="manifest.json files: skip games (ep, winner) already built there")
    args = ap.parse_args()
    args.eps = {int(x) for x in args.eps.split(",") if x}
    args.order = [os.path.basename(s.split(":")[0].rstrip("/")) for s in args.tapes]
    args.src_index = {n: i for i, n in enumerate(SRC_ORDER)}
    if args.merge:
        merge(args); return
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    os.makedirs(f"{args.out}/seats", exist_ok=True); os.makedirs(f"{args.out}/x6", exist_ok=True)
    voc = {t: i for i, t in enumerate(json.load(open(os.environ.get("VOCAB", f"{K}/il/data/team_vocab.json"))))}   # Sep 30: VOCAB = extended vocab (own slot per top-25 team)
    jobs, why = select_games(args)
    if args.max_games:
        jobs = jobs[:args.max_games]
    if args.worker == 0:
        print(f"selected {len(jobs)} games; excluded: {dict(why)}", flush=True)
        c = collections.Counter((j[1], "W" if j[4] else "L") for j in jobs)
        print("per dir:", {f"{k[0]}:{k[1]}": v for k, v in sorted(c.items())}, flush=True)
    if args.list_only:
        return
    mine = jobs[args.worker::args.workers]
    stats = collections.Counter(); t0 = time.time()
    man = open(f"{args.out}/manifest_part{args.worker}.jsonl", "a")
    for i, (f, dname, ep, W, won) in enumerate(mine):
        fn = f"{args.out}/seats/{ep}_{W}.npz"
        if os.path.exists(fn) and os.path.exists(f"{args.out}/x6/{ep}_{W}.npy"):
            stats["already_built"] += 1; continue
        try:
            r = build_game(f, dname, args, voc, stats)
        except Exception as ex:
            import traceback; traceback.print_exc()
            stats["failed"] += 1; print("FAIL", f, str(ex)[:200], flush=True); continue
        if r is None:
            continue
        b, x6, meta, info = r
        np.savez_compressed(fn, **b, meta=meta)
        np.save(f"{args.out}/x6/{ep}_{W}.npy", x6)
        man.write(json.dumps(info, default=str) + "\n"); man.flush()
        stats["written"] += 1; stats["steps"] += info["steps"]
        if (i + 1) % 10 == 0 or i + 1 == len(mine):
            print(f"worker {args.worker}: {i+1}/{len(mine)} games, {dict(stats)} {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(stats), open(f"{args.out}/stats_part{args.worker}.json", "w"))
    print(f"worker {args.worker} done: {dict(stats)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
