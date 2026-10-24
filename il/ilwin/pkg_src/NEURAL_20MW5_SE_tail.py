def _fallback():
    return {"farmer": ["PASS"], "hands": [], "market": []}


def agent(observation, configuration=None):
    try:
        return _AGENT(observation, configuration)
    except Exception:
        return _fallback()
