"""Export a trained checkpoint to a self-contained Kaggle main.py (numpy + stdlib only).

Usage: export.py <ckpt> <out_main.py> [tau] [team]
The file embeds features.py, np_runtime.py and the weights (float32, zlib, base85) and ends with a module-level
`agent` callable (Kaggle loads the last callable)."""
import sys, zlib, base64, io
import numpy as np
import torch

ckpt, out = sys.argv[1], sys.argv[2]
import json as _json
tau = (_json.load(open(sys.argv[3])) if sys.argv[3].endswith('.json') else float(sys.argv[3])) if len(sys.argv) > 3 else 0.35
team = int(sys.argv[4]) if len(sys.argv) > 4 else 0
c = torch.load(ckpt, map_location="cpu", weights_only=False)
state = {k: v.numpy().astype(np.float32) for k, v in c["state"].items() if k != "coords" and not k.startswith("aux.")}
v6 = int(c.get("feat", 5)) == 6
buf = io.BytesIO(); np.savez(buf, **state)
import os as _os_b
B64 = _os_b.environ.get("B64", "0") == "1"   # Sep 26: base64 (C decoder, 0.5 s) instead of base85 (pure Python, 20 s for 20M)
blob = (base64.b64encode if B64 else base64.b85encode)(zlib.compress(buf.getvalue(), 9)).decode()
blocks = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
feat_src = open("/Users/asad.ali/kaggriculture/il/features.py").read()
import os as _os                                      # Sep 26: RULES="SE_GUARD=1,SHEEP_CAP=15" switches the rule layer on
import re as _re                                      # Sep 28: any integer default (e.g. TOMATO_FROM_DAY = 2 -> 15)
for _kv in [x for x in _os.environ.get("RULES", "").split(",") if x]:
    _k, _v = _kv.split("=")
    feat_src, _n = _re.subn(rf"\n{_re.escape(_k)} = -?\d+(?=[ \n])", f"\n{_k} = {int(_v)}", feat_src)
    assert _n == 1, (_k, _n)
    print(f"rule layer: {_k} = {int(_v)}")
for _kv in [x for x in _os.environ.get("RULES_STR", "").split(",") if x]:   # Sep 28: string rules, e.g. SELL_ORDER=base
    _k, _v = _kv.split("=")
    _old = {"SELL_ORDER": f'\n{_k} = "canonical"', "FINAL_ORDER": f'\n{_k} = "qty"'}.get(_k)
    assert _old and feat_src.count(_old) == 1, _k
    feat_src = feat_src.replace(_old, f'\n{_k} = "{_v}"', 1)
    print(f"rule layer: {_k} = {_v!r}")
rt_src = open("/Users/asad.ali/kaggriculture/il/np_runtime.py").read()
f6_src = open("/Users/asad.ali/kaggriculture/il/features6.py").read() if v6 else ""
ar = int(c.get("ar", 0)) == 1
menu8 = int(c.get("menu", 0)) == 8
fa_src = open("/Users/asad.ali/kaggriculture/il/features_ar.py").read() if (ar or menu8) else ""
f8_src = open("/Users/asad.ali/kaggriculture/il/features8.py").read() if menu8 else ""
fa_code = (f"import sys as _sys\n_sys.modules['features'] = _F\n_FA = _types.ModuleType('features_ar'); "
           f"exec(compile({fa_src!r}, 'features_ar.py', 'exec'), _FA.__dict__); _F6.FA = _FA\n") if ar else ""
f8_code = (f"import sys as _sys\n_sys.modules['features'] = _F\n_FA8 = _types.ModuleType('features_ar'); "
           f"exec(compile({fa_src!r}, 'features_ar.py', 'exec'), _FA8.__dict__); _sys.modules['features_ar'] = _FA8\n"
           f"_F8 = _types.ModuleType('features8'); exec(compile({f8_src!r}, 'features8.py', 'exec'), _F8.__dict__); _F6.F8 = _F8\n") if menu8 else ""
f6_code = (f"import sys as _sys\n_sys.modules['features'] = _F\n_F6 = _types.ModuleType('features6'); "
           f"exec(compile({f6_src!r}, 'features6.py', 'exec'), _F6.__dict__)\n") if v6 else "_F6 = _F\n"
src = f'''# Kaggriculture neural agent: behaviour cloning on top-10 leader games (+ RL fine-tuning), numpy inference.
import types as _types, zlib as _zlib, base64 as _b64, io as _io
import numpy as _np
_F = _types.ModuleType("features"); exec(compile({feat_src!r}, "features.py", "exec"), _F.__dict__)
{f6_code}{fa_code}{f8_code}_R = _types.ModuleType("np_runtime"); exec(compile({rt_src!r}, "np_runtime.py", "exec"), _R.__dict__)
_BLOB = {blob!r}
_W = dict(_np.load(_io.BytesIO(_zlib.decompress(_b64.{"b64decode" if B64 else "b85decode"}(_BLOB)))))
_NET = _R.NumpyNet(_W, width={int(c["width"])}, blocks={blocks}, n_teams={int(c.get("n_teams", 0))})
_AGENT = _R.NumpyAgent(_NET, _F6, tau={tau}, team={team})


def _fallback():
    return {{"farmer": ["PASS"], "hands": [], "market": []}}


def agent(observation, configuration=None):
    try:
        if getattr(_F, "STRADDLE", 0):    # wheat drain straddle: the policy sees the farm without it, its legs go last
            return _F.straddle_fix(observation, _AGENT(_F.straddle_obs(observation), configuration))
        return _AGENT(observation, configuration)
    except Exception:
        return _fallback()
'''
open(out, "w").write(src)
print(f"wrote {out}: {len(src)/1e6:.1f} MB, blocks {blocks}, width {c['width']}, teams {c.get('n_teams', 0)}, tau {tau}, team {team}")
