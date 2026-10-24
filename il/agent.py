import os
"""Runtime imitation agent: observation -> network -> per-unit commands (legal-action masked) + market orders.

BCAgent(ckpt) is a callable agent(obs, cfg=None) usable by the harness; it keeps the previous op of each unit
(an input feature) per player and resets at step 0.  Torch inference (cpu or mps); the Kaggle build exports the
same network to numpy.
"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/Users/asad.ali/kaggriculture/il")
import features as F
from model import Net, split_heads

MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def legal_mask(obs, k, pos, shed=None, seeds=None):
    """Boolean [N_OPS] of commands that would change state for unit k at pos (engine rules).  shed / seeds: live
    copies already reduced by the units that acted before this one in the same step."""
    me = int(obs["player"]); farm = obs["farms"][me]; priv = obs["private"]
    x, y = int(pos[0]), int(pos[1])
    t = farm["tiles"][y][x]
    iv = (priv.get("inventories") or [{}])
    iv = iv[k] if k < len(iv) else {}
    shed = shed if shed is not None else (priv.get("shed", {}) or {})
    seeds = seeds if seeds is not None else (priv.get("seeds", {}) or {})
    day = int(obs["step"]) // 24
    at_shed = (x, y) in F.SHED_ACCESS
    m = np.zeros(F.N_OPS, bool)
    ok = lambda name: m.__setitem__(F.OP_INDEX[name], True)
    ok("PASS")
    for mv, (dx, dy) in MOVES.items():
        if 0 <= x + dx < F.B and 0 <= y + dy < F.B:
            ok(mv)
    is_d = isinstance(t, dict)
    kind = t.get("kind") if is_d else t
    animal = is_d and "animal" in t
    if kind == "PLANT":
        if not t.get("watered_today"):
            ok("WATER")
        ci = F.CROP_INFO.get(t.get("crop"))
        if ci and int(t.get("yield_units", 0)) > 0 and day - int(t.get("planted_day", day)) >= ci[0]:
            ok("HARVEST")
        if iv.get("FERTILIZER", 0) > 0:
            ok("FERTILIZE")
    if animal:
        if int(t.get("yield_units", 0)) > 0:
            ok("HARVEST")
        if not t.get("fed_today") and iv.get("WHEAT", 0) > 0:
            ok("FEED")
        if t.get("fertilizer_available"):
            ok("COLLECT_FERTILIZER")
        if not t.get("cared_today"):
            ok("CARE")
    if t is not None and t != "LOCKED" and not animal:
        ok("DIG")
    if t is None:
        ok("BUILD_COOP"); ok("BUILD_PASTURE")
        for c in F.CROPS:
            if seeds.get(c, 0) > 0:
                ok("PLANT_" + c)
    if at_shed:
        if any(v > 0 for v in iv.values()):
            ok("DROP")
        for it in F.ITEMS:
            if shed.get(it, 0) > 0:
                ok("PICKUP_" + it)
    for it in F.ITEMS:
        if iv.get(it, 0) > 0:
            if it in F.ANIMALS:
                if is_d and not animal and kind == {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}[it]:
                    ok("PLACE_" + it)
                elif at_shed:
                    ok("PLACE_" + it)
            elif at_shed:
                ok("PLACE_" + it)
    return m


class BCAgent:
    def __init__(self, ckpt, device="cpu", mask=True, greedy=True, temp=1.0, seed=0, mk_tau=None, intent=False, team=0):
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"), n_aux=c.get("n_aux", 0), ar=c.get("ar", 0), menu=c.get("menu", 0))
        self.team = torch.tensor([team], dtype=torch.long) if c.get("n_teams", 0) else None
        self.net.load_state_dict(c["state"]); self.net.eval().to(device)
        self.dev = torch.device(device); self.mask = mask; self.greedy = greedy; self.temp = temp
        self.rng = np.random.RandomState(seed)
        self.mk_tau = mk_tau          # market heads: order when P(class > 0) > tau (None = argmax)
        self.intent = intent          # buy what a unit's preferred (unmasked) PLANT / PICKUP WHEAT needs
        self.state = {}
        self.midnight = int(os.environ.get("MIDNIGHT", "1"))   # rule layer: sell the shed overflow before midnight
        if getattr(self.net, "g_in", F.G) != F.G:
            import features6
            self.FE = features6
        else:
            self.FE = F

    def _pick(self, logits, allowed=None):
        z = logits.astype(np.float64)
        if allowed is not None:
            z = np.where(allowed, z, -1e9)
        if self.greedy:
            return int(np.argmax(z))
        p = np.exp((z - z.max()) / self.temp); p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    @torch.no_grad()
    def __call__(self, obs, cfg=None):
        me, step = int(obs["player"]), int(obs["step"])
        st = self.state.get(me)
        if st is None or step == 0 or step <= st["step"]:
            st = self.state[me] = {"step": -1, "prev": None}
        st["step"] = step
        planes, g, pos, uf = self.FE.encode_obs(obs, st["prev"])
        n = len(pos)
        tp = torch.from_numpy(planes[None].astype(np.float32) / 255.0).to(self.dev)
        tg = torch.from_numpy(g[None]).to(self.dev)
        tpos = torch.from_numpy(pos[None].astype(np.int64)).to(self.dev)
        tu = torch.from_numpy(uf[None]).to(self.dev)
        outs = self.net(tp, tg, tpos, tu, None if self.team is None else self.team.to(self.dev), ret_hm=True)
        lop, lq, lm, self._hm = outs[0], outs[1], outs[2], outs[5]
        lop = lop[0].cpu().numpy(); lq = lq[0].cpu().numpy()
        cmds, ops = [], []
        me_priv = obs["private"]; invs = me_priv.get("inventories") or [{}]
        shed = dict(me_priv.get("shed", {}) or {}); seeds = dict(me_priv.get("seeds", {}) or {})
        want_seed, want_wheat = {}, 0
        for k in range(n):
            allowed = None
            if self.mask == "shared":
                allowed = legal_mask(obs, k, pos[k], shed, seeds)
            elif self.mask:
                allowed = legal_mask(obs, k, pos[k])
            oc = self._pick(lop[k], allowed)
            if self.intent and allowed is not None:
                top = F.OPS[int(np.argmax(lop[k]))]
                x, y = int(pos[k][0]), int(pos[k][1])
                tile = obs["farms"][me]["tiles"][y][x]
                if top.startswith("PLANT_") and tile is None and seeds.get(top[6:], 0) <= 0:
                    want_seed[top[6:]] = want_seed.get(top[6:], 0) + 1      # plant next step
                    oc = F.OP_INDEX["PASS"]
                elif top == "PICKUP_WHEAT" and (x, y) in F.SHED_ACCESS and shed.get("WHEAT", 0) <= 0:
                    want_wheat += int(np.argmax(lq[k])) + 1
                    oc = F.OP_INDEX["PASS"]
            name = F.OPS[oc]; qc = 0
            if name.startswith("PICKUP_") or name.startswith("PLACE_"):
                qc = int(np.argmax(lq[k]))
                it = name.split("_", 1)[1]
                have = shed.get(it, 0) if name.startswith("PICKUP_") else (invs[k] if k < len(invs) else {}).get(it, 0)
                qc = max(0, min(qc, int(have) - 1))
                if name.startswith("PICKUP_"):
                    shed[it] = shed.get(it, 0) - (qc + 1)
            elif name.startswith("PLANT_"):
                seeds[name[6:]] = seeds.get(name[6:], 0) - 1
            cmds.append(F.unit_command(oc, qc)); ops.append(oc)
        st["prev"] = ops
        heads = split_heads(lm[0], self.net.head_sizes)
        MK = F
        if getattr(self.net, "menu", 0) == 8:        # extended fixed menu
            import features8 as MK
        if self.mk_tau is None:
            classes = [self._pick(h.cpu().numpy()) for h in heads]
        else:
            classes = []
            for j, h in enumerate(heads):
                tau = self.mk_tau.get(MK.MARKET_HEADS[j][0], self.mk_tau.get('_default', 0.6)) if isinstance(self.mk_tau, dict) else self.mk_tau
                z = h.cpu().numpy().astype(np.float64); p = np.exp(z - z.max()); p /= p.sum()
                classes.append(int(1 + np.argmax(p[1:])) if 1.0 - p[0] > tau else 0)
        if self.intent and MK is F:
            for c, q in want_seed.items():
                h = F.HEAD_INDEX["BUY_SEED:" + c]
                classes[h] = max(int(classes[h]), min(q, F.MARKET_HEADS[h][1] - 1))
            if want_wheat:
                h = F.HEAD_INDEX["BUY_PRODUCT:WHEAT"]
                classes[h] = max(int(classes[h]), min(want_wheat, 16))
        if getattr(self.net, "ar", 0):              # order-by-order head
            import features_ar as FA
            tau_ = self.mk_tau.get("_default", 0.5) if isinstance(self.mk_tau, dict) else (self.mk_tau if self.mk_tau is not None else 0.5)
            sq = self.net.ar_greedy(self._hm, float(tau_))[0].cpu().numpy()
            market = FA.decode_seq(sq[:, 0], sq[:, 1], sq[:, 2])
        else:
            market = MK.decode_market(classes, obs)
        act = {"farmer": cmds[0] if cmds else ["PASS"], "hands": cmds[1:], "market": market}
        return F.midnight_fix(obs, act) if self.midnight else act


def factory(ckpt, **kw):
    def make():
        return BCAgent(ckpt, **kw)
    return make
