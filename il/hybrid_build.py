"""Hybrid agent: a neural policy's move as the BASE of a STACK1 script's layer stack (operator, Sep 25).

The STACK1 scripts build their base move with `_IMPL = make_agent(...)` (a recorded route schedule adapted by the
Chassis) and then stack ~30 reactive layers on top (`_SHOP_PARENT = agent`, ...).  This builder keeps the script
byte-identical except at its single hook: the base agent still calls `_IMPL` every turn (layers read its chassis
state and helpers), but returns the neural agent's move instead.  The neural package's main.py is executed in its own
module namespace, so the two files cannot clash.

Usage: hybrid_build.py <script main.py> <neural main.py> <out main.py>
"""
import sys

HOOK = "        action=_IMPL(observation,configuration)\n"
NEW = ("        _base=_IMPL(observation,configuration)   # schedule + chassis still run: layers read their state\n"
       "        action=_NEURAL_AGENT(observation,configuration)\n")


def build(script_src, neural_src):
    assert script_src.count(HOOK) == 1, "hook not found exactly once"
    head = ("# HYBRID: neural policy as the base move of the STACK1 layer stack (see il/hybrid_build.py).\n"
            "import types as _hy_types\n"
            "_HY_NEURAL = _hy_types.ModuleType('neural_policy')\n"
            f"exec(compile({neural_src!r}, 'neural_main.py', 'exec'), _HY_NEURAL.__dict__)\n"
            "_NEURAL_AGENT = _HY_NEURAL.agent\n\n")
    anchor = "_IMPL=make_agent("                  # after the script's imports (incl. its `from __future__`), before the base
    assert script_src.count("\n" + anchor) == 1, "base definition not found exactly once"
    i = script_src.index("\n" + anchor) + 1
    return (script_src[:i] + head + script_src[i:]).replace(HOOK, NEW)


if __name__ == "__main__":
    s, n, out = sys.argv[1:4]
    src = build(open(s).read(), open(n).read())
    open(out, "w").write(src)
    print(f"wrote {out}: {len(src)/1e6:.1f} MB (script {s} + neural {n})")
