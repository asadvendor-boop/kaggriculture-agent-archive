"""Policy network for the imitation agent (PyTorch; exported to numpy for the Kaggle runtime).

grid (C planes, 10x10) + 2 coordinate planes -> stem conv -> R residual 3x3 blocks, with the global vector
broadcast into every cell (FiLM-style add) -> per-unit head gathers the cell under each unit and concatenates the
unit features (index, position, inventory, previous op) -> op logits (44) + quantity logits (12);
pooled grid (mean, max) + global -> market head (21 quantity heads) and a value head (final margin, for RL).
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import features as F

UB = F.MAX_UNITS + 2 + len(F.ITEMS) + 2          # 32 stored unit features (+ previous-op one-hot at run time)
UIN = UB + F.N_OPS                                # 76
HEAD_SIZES = [n for _, n in F.MARKET_HEADS]


def menu_sizes(menu=0):
    """Market head sizes of a menu: 0 = features.py (21 heads, 249 classes), 8 = features8 (23 heads, extended)."""
    if int(menu or 0) == 8:
        import features8 as F8
        return list(F8.HEAD_SIZES)
    return list(HEAD_SIZES)


class Res(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return Fn.relu(x + self.c2(Fn.relu(self.c1(x))))


class Net(nn.Module):
    def __init__(self, width=96, blocks=5, gdim=160, hid=256, n_teams=0, team_dim=16, g_in=None, n_aux=0, ar=0, menu=0):
        """g_in: width of the global feature vector (None = features.G, v5; 179 = features6.G, v6).
        n_aux: extra training-only heads that predict future quantities (read with forward(..., aux=True))."""
        super().__init__()
        self.width = width
        self.g_in = g_in or F.G
        self.n_teams = n_teams
        self.team = nn.Embedding(n_teams, team_dim) if n_teams else None
        self.stem = nn.Conv2d(F.C + 2, width, 3, padding=1)
        self.blocks = nn.ModuleList([Res(width) for _ in range(blocks)])
        self.g1 = nn.Linear(self.g_in + (team_dim if n_teams else 0), gdim)
        self.g2 = nn.Linear(gdim, gdim)
        self.film = nn.Linear(gdim, width)
        self.u1 = nn.Linear(width + gdim + UIN, hid)
        self.u2 = nn.Linear(hid, hid)
        self.uop = nn.Linear(hid, F.N_OPS)
        self.uqty = nn.Linear(hid, F.N_QTY)
        self.m1 = nn.Linear(2 * width + gdim, hid)
        self.m2 = nn.Linear(hid, hid)
        self.menu = int(menu or 0)        # 8 = extended fixed menu (features8)
        self.head_sizes = menu_sizes(self.menu)
        self.mk = nn.Linear(hid, sum(self.head_sizes))
        self.v = nn.Linear(hid, 1)
        self.aux = nn.Linear(hid, n_aux) if n_aux else None
        self.ar = int(ar)                 # 1 = order-by-order market head (features_ar): up to 10 (op, item, qty) slots
        if self.ar:
            import features_ar as FA
            E = 32; self.ar_slots = FA.SLOTS
            self.ar_op_emb = nn.Embedding(len(FA.AR_OPS), E); self.ar_item_emb = nn.Embedding(len(FA.AR_ITEMS), E)
            self.ar_qty_emb = nn.Embedding(len(FA.AR_QTY), E); self.ar_slot_emb = nn.Embedding(FA.SLOTS, E)
            self.ar_h0 = nn.Linear(hid, hid); self.ar_cell = nn.GRUCell(3 * E, hid)
            self.ar_op = nn.Linear(hid + E, len(FA.AR_OPS)); self.ar_item = nn.Linear(hid + E, len(FA.AR_ITEMS))
            self.ar_qty = nn.Linear(hid + 2 * E, len(FA.AR_QTY))
            mask = torch.full((len(FA.AR_OPS), len(FA.AR_ITEMS)), -1e9)
            mask[torch.from_numpy(FA.ITEM_MASK)] = 0.0
            mask[~torch.from_numpy(FA.ITEM_MASK).any(1)] = 0.0        # ops without an item: unmasked (loss ignores them)
            self.register_buffer("ar_item_bias", mask)
            has = torch.tensor([o in FA.HAS_ITEM for o in FA.AR_OPS])          # ops that carry an item and a quantity
            self.register_buffer("ar_has_item", has, persistent=False)
        yy, xx = torch.meshgrid(torch.arange(F.B), torch.arange(F.B), indexing="ij")
        self.register_buffer("coords", torch.stack([xx / 9.0, yy / 9.0]).float())

    def trunk(self, planes, glob, team=None):
        """planes float [N,C,10,10] in 0..1, glob [N,G], team long [N] (style id) -> grid features [N,W,10,10],
        global embedding [N,gdim]."""
        n = planes.shape[0]
        if self.team is not None:
            glob = torch.cat([glob, self.team(team)], 1)
        x = torch.cat([planes, self.coords.expand(n, -1, -1, -1)], 1)
        g = Fn.relu(self.g2(Fn.relu(self.g1(glob))))
        x = Fn.relu(self.stem(x) + self.film(g)[:, :, None, None])
        for b in self.blocks:
            x = b(x)
        return x, g

    def ar_tf(self, hm, lab):
        """Teacher forcing: hm [N,hid], lab long [N,SLOTS,3] -> op [N,S,ops], item [N,S,items], qty [N,S,qtys] logits."""
        h = torch.tanh(self.ar_h0(hm)); ops, its, qts = [], [], []
        for k in range(self.ar_slots):
            se = self.ar_slot_emb.weight[k].expand(h.shape[0], -1)
            oe = self.ar_op_emb(lab[:, k, 0]); ie = self.ar_item_emb(lab[:, k, 1]); qe = self.ar_qty_emb(lab[:, k, 2])
            ops.append(self.ar_op(torch.cat([h, se], 1)))
            its.append(self.ar_item(torch.cat([h, oe], 1)) + self.ar_item_bias[lab[:, k, 0]])
            qts.append(self.ar_qty(torch.cat([h, oe, ie], 1)))
            h = self.ar_cell(torch.cat([oe, ie, qe], 1), h)
        return torch.stack(ops, 1), torch.stack(its, 1), torch.stack(qts, 1)

    @torch.no_grad()
    def ar_greedy(self, hm, tau=0.5):
        """Greedy order-by-order decoding: at each slot fire the most likely non-STOP op when P(not STOP) > tau."""
        n = hm.shape[0]; h = torch.tanh(self.ar_h0(hm)); out = torch.zeros(n, self.ar_slots, 3, dtype=torch.long, device=hm.device)
        alive = torch.ones(n, dtype=torch.bool, device=hm.device)
        for k in range(self.ar_slots):
            se = self.ar_slot_emb.weight[k].expand(n, -1)
            p = torch.softmax(self.ar_op(torch.cat([h, se], 1)), 1)
            fire = alive & (1.0 - p[:, 0] > tau)
            op = torch.where(fire, 1 + p[:, 1:].argmax(1), torch.zeros_like(fire, dtype=torch.long))
            oe = self.ar_op_emb(op)
            has = self.ar_has_item[op]                   # STOP / EMPTY / HIRE / BUY_LAND: item 0 and qty 0, exactly as in the
            it = torch.where(has, (self.ar_item(torch.cat([h, oe], 1)) + self.ar_item_bias[op]).argmax(1), 0)   # training labels
            ie = self.ar_item_emb(it)
            q = torch.where(has, self.ar_qty(torch.cat([h, oe, ie], 1)).argmax(1), 0)
            out[:, k, 0] = op; out[:, k, 1] = it; out[:, k, 2] = q
            alive = fire
            h = self.ar_cell(torch.cat([oe, ie, self.ar_qty_emb(q)], 1), h)
        return out

    def forward(self, planes, glob, pos, uin, team=None, aux=False, ret_hm=False):
        """pos long [N,U,2] (x,y), uin [N,U,UIN], team long [N] -> op logits [N,U,44], qty logits [N,U,12],
        market logits [N,sum(heads)], value [N]."""
        x, g = self.trunk(planes, glob, team)
        n, w = x.shape[0], x.shape[1]
        flat = x.flatten(2)                                              # [N,W,100]
        idx = (pos[..., 1] * F.B + pos[..., 0]).clamp(0, F.B * F.B - 1)  # [N,U]
        cell = torch.gather(flat, 2, idx[:, None, :].expand(-1, w, -1)).transpose(1, 2)   # [N,U,W]
        u = torch.cat([cell, g[:, None, :].expand(-1, idx.shape[1], -1), uin], 2)
        h = Fn.relu(self.u2(Fn.relu(self.u1(u))))
        pooled = torch.cat([flat.mean(2), flat.amax(2), g], 1)
        hm = Fn.relu(self.m2(Fn.relu(self.m1(pooled))))
        if ret_hm:
            return self.uop(h), self.uqty(h), self.mk(hm), self.v(hm).squeeze(1), (self.aux(hm) if self.aux is not None else None), hm
        if aux:
            return self.uop(h), self.uqty(h), self.mk(hm), self.v(hm).squeeze(1), self.aux(hm)
        return self.uop(h), self.uqty(h), self.mk(hm), self.v(hm).squeeze(1)


def split_heads(mlogits, sizes=None):
    out, i = [], 0
    for n in (sizes or HEAD_SIZES):
        out.append(mlogits[..., i:i + n]); i += n
    return out
