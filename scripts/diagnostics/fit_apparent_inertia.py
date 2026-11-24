"""Fit the fingertip's apparent translational inertia from a usb_probe.py --mode response log.

Model (free space, DR off, task frame): M_eff * acc = Kp * 0.02 * a - Kd * v + f0, where a is the (EMA'd)
action and f0 a constant force (e.g. the held asset's weight). Least squares over the physics-rate log.

    python scripts/diagnostics/fit_apparent_inertia.py logs/usb/response_*.json
"""

import json
import sys

import numpy as np

np.set_printoptions(precision=2, suppress=True)
for path in sys.argv[1:]:
    d = json.load(open(path))
    dt = d.get("physics_dt", 1 / 120)
    p = np.array(d["log"]["pos"])
    a = np.array(d["log"]["action"])[:, 0:3]
    kp = np.array(d["gains"][0:3])
    kd = np.array(d["deriv_gains"][0:3])
    v = np.gradient(p, dt, axis=0)
    acc = np.gradient(v, dt, axis=0)
    F = kp * 0.02 * a - kd * v
    X = np.hstack((acc, -np.ones((len(acc), 1))))  # F = M acc - f0
    sol, *_ = np.linalg.lstsq(X[5:-5], F[5:-5], rcond=None)
    M, f0 = sol[0:3].T, sol[3]
    resid = F[5:-5] - X[5:-5] @ sol
    r2 = 1 - (resid**2).sum(0) / ((F[5:-5] - F[5:-5].mean(0)) ** 2).sum(0)
    print(path, "\n  M_eff [kg]:\n", (M + M.T) / 2, "\n  eig", np.linalg.eigvalsh((M + M.T) / 2),
          "\n  f0 [N]", f0, " R2", r2)
    if "lambda_mean" in d:
        L = np.array(d["lambda_mean"]).reshape(6, 6)
        print("  PhysX Lambda_tt:\n", L[0:3, 0:3], "\n  inv((Lambda^-1)_tt):\n", np.linalg.inv(np.linalg.inv(L)[0:3, 0:3]))
