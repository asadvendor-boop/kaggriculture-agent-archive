def _fallback():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def agent(observation, configuration=None):
    try:
        if getattr(_F, "STRADDLE", 0):    # wheat drain straddle: the policy sees the farm without it, its legs go last
            return _F.straddle_fix(observation, _AGENT(_F.straddle_obs(observation), configuration))
        return _AGENT(observation, configuration)
    except Exception:
        return _fallback()
