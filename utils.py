import numpy as np

def norm(v):
    return float(np.linalg.norm(v))

def compute_error(stop, x_prev, x, f_prev, f, g):
    if stop == "grad_norm":
        return norm(g)
    elif stop == "step_norm":
        if x_prev is None: return np.inf
        return norm(x - x_prev)
    elif stop == "f_change":
        if f_prev is None: return np.inf
        return abs(f - f_prev)
    else:
        raise ValueError("stop debe ser 'grad_norm' | 'step_norm' | 'f_change'")


