"""Logit-averaging ensemble of trained policy networks (same interface as model.Net for rollouts / PPO)."""
import torch
import torch.nn as nn
from model import Net


class EnsembleNet(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.members = nn.ModuleList(nets)
        self.n_teams = nets[0].n_teams

    def forward(self, planes, glob, pos, uin, team=None):
        outs = [m(planes, glob, pos, uin, team) for m in self.members]
        return tuple(sum(o[i] for o in outs) / len(outs) for i in range(4))


def load_ensemble(ckpts, dev):
    nets = []
    for ck in ckpts:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        n = Net(width=c["width"], blocks=c.get("blocks", 5), n_teams=c.get("n_teams", 0), g_in=c.get("g_in"), n_aux=c.get("n_aux", 0), ar=c.get("ar", 0)); n.load_state_dict(c["state"]); nets.append(n)
    return EnsembleNet(nets).eval().to(dev)
