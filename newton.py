import numpy as np
from utils import *

def newton_method(f, df, ddf, x0, alpha, tol=1e-6, maxIter=200, prt=0, stop="grad_norm"):
    x = np.array(x0, dtype=float).copy()
    fx = f(x)
    g  = df(x)

    z = [x.copy()]
    iters = 0
    conv = 0

    err = compute_error(stop, None, x, None, fx, g)
    if err <= tol:
        return x, fx, z, iters, 1

    for k in range(maxIter):
        H = ddf(x)
        try:
            p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = -g

        if np.dot(p, g) >= 0:
            p = -g

        x_prev, f_prev = x, fx
        x = x + alpha * p
        fx = f(x)
        g  = df(x)

        z.append(x.copy())
        iters += 1

        if prt:
            print(f"[newton] iter={iters}  f={fx:.6e}  ||g||={norm(g):.3e}")

        err = compute_error(stop, x_prev, x, f_prev, fx, g)
        if err <= tol:
            conv = 1
            break

    return x, fx, z, iters, conv