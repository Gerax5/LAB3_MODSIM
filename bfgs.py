import numpy as np
from utils import *

def bfgs(f, df, x0, alpha, tol=1e-6, maxIter=1000, prt=0, stop="grad_norm"):
    x = np.array(x0, dtype=float).copy()
    n = x.size
    H = np.eye(n)  # aprox. del inverso de Hessiano

    fx = f(x)
    g  = df(x)

    z = [x.copy()]
    iters = 0
    conv = 0

    err = compute_error(stop, None, x, None, fx, g)
    if err <= tol:
        return x, fx, z, iters, 1

    for k in range(maxIter):
        p = -H @ g
        if np.dot(p, g) >= 0:
            p = -g

        x_prev, f_prev, g_prev = x, fx, g
        x = x + alpha * p
        fx = f(x)
        g  = df(x)

        s = x - x_prev
        y = g - g_prev
        ys = float(np.dot(y, s))

        if ys > 1e-12:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        z.append(x.copy())
        iters += 1

        if prt:
            print(f"[bfgs] iter={iters}  f={fx:.6e}  ||g||={norm(g):.3e}")

        err = compute_error(stop, x_prev, x, f_prev, fx, g)
        if err <= tol:
            conv = 1
            break

    return x, fx, z, iters, conv