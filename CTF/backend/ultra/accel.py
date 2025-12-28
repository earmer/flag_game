try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional
    np = None


def argmin(scores):
    if not scores:
        return None
    if np is not None and len(scores) >= 32:
        return int(np.argmin(np.asarray(scores, dtype=float)))
    return min(range(len(scores)), key=scores.__getitem__)
