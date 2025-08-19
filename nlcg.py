from utils import *
import numpy as np

# def nonlinear_cg(f, df, x0, alpha, tol=1e-6, maxIter=1000, prt=0, stop="grad_norm", formula="PR"):
#     x = np.array(x0, dtype=float).copy()
#     fx = f(x)
#     g  = df(x)
#     p  = -g.copy()

#     z = [x.copy()]
#     iters = 0
#     conv = 0

#     err = compute_error(stop, None, x, None, fx, g)
#     if err <= tol:
#         return x, fx, z, iters, 1

#     for k in range(maxIter):
#         x_prev, f_prev, g_prev, p_prev = x, fx, g, p
#         x = x + alpha * p_prev
#         fx = f(x)
#         g  = df(x)

#         if formula.upper() == "FR":
#             denom = np.dot(g_prev, g_prev)
#             beta = 0.0 if denom <= 1e-32 else np.dot(g, g) / denom
#         elif formula.upper() in ("PR", "PR+"):
#             denom = np.dot(g_prev, g_prev)
#             beta_raw = 0.0 if denom <= 1e-32 else np.dot(g, g - g_prev) / denom
#             beta = max(0.0, beta_raw) 
#         elif formula.upper() in ("HS", "HS+"):
#             y = g - g_prev
#             denom = np.dot(p_prev, y)
#             beta_raw = 0.0 if abs(denom) <= 1e-32 else np.dot(g, y) / denom
#             beta = max(0.0, beta_raw)  
#         else:
#             raise ValueError("formula debe ser 'FR', 'PR', o 'HS'")

#         p = -g + beta * p_prev
#         if np.dot(p, g) >= 0:
#             p = -g  

#         z.append(x.copy())
#         iters += 1

#         if prt:
#             print(f"[cg-{formula}] iter={iters}  f={fx:.6e}  ||g||={norm(g):.3e}")

#         err = compute_error(stop, x_prev, x, f_prev, fx, g)
#         if err <= tol:
#             conv = 1
#             break

#     return x, fx, z, iters, conv

def nonlinear_cg(f, df, x0, alpha, tol=1e-6, maxIter=1000, prt=0, stop="grad_norm", formula="PR"):
    x  = np.array(x0, dtype=float).copy()
    fx = f(x); g = df(x); p = -g.copy()
    xs, fxs, errs = [x.copy()], [fx], [compute_error(stop, None, x, None, fx, g)]
    iters, conv = 0, 0
    if errs[-1] <= tol:
        return x, xs, fxs, errs, iters, 1
    for _ in range(maxIter):
        x_prev, f_prev, g_prev, p_prev = x, fx, g, p
        x = x + alpha * p_prev
        fx = f(x); g = df(x)
        if formula.upper() == "FR":
            denom = np.dot(g_prev, g_prev); beta = 0.0 if denom <= 1e-32 else np.dot(g, g)/denom
        elif formula.upper() in ("PR", "PR+"):
            denom = np.dot(g_prev, g_prev)
            beta_raw = 0.0 if denom <= 1e-32 else np.dot(g, g - g_prev)/denom
            beta = max(0.0, beta_raw)
        elif formula.upper() in ("HS", "HS+"):
            y = g - g_prev; denom = np.dot(p_prev, y)
            beta_raw = 0.0 if abs(denom) <= 1e-32 else np.dot(g, y)/denom
            beta = max(0.0, beta_raw)
        else:
            raise ValueError("formula debe ser 'FR', 'PR' o 'HS'")
        p = -g + beta * p_prev
        if np.dot(p, g) >= 0: p = -g
        xs.append(x.copy()); fxs.append(fx); iters += 1
        err = compute_error(stop, x_prev, x, f_prev, fx, g); errs.append(err)
        if prt: print(f"[cg-{formula}] iter={iters}  f={fx:.6e}  ||g||={norm(g):.3e}")
        if err <= tol: conv = 1; break
    return x, xs, fxs, errs, iters, conv