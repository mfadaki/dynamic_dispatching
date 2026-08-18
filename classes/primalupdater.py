"""
classes/primalupdater.py
========================
Universal primal update step for any ALP / PSMD problem.
No problem-specific code.

Implements paper Algorithm 1, Step 3 (unconstrained case):
    theta_{t+1} = theta_t + eta_t * grad

Optionally supports box-constrained projection when lb/ub are provided:
    theta_{t+1} = clip(theta_t + eta_t * grad, lb, ub)
"""

import numpy as np


class Primal:
    """
    Gradient ascent step on theta. Universal across all ALP problems.

    Usage
    -----
    primal = Primal()                          # unconstrained
    primal = Primal(lb=lb_vec, ub=ub_vec)      # box-constrained
    theta  = primal.update(theta, grad, eta)
    """

    def __init__(self, lb=None, ub=None):
        """
        Parameters
        ----------
        lb : (B,) array, list (may contain None for unconstrained), or None
        ub : (B,) array, list (may contain None for unconstrained), or None
        If lb/ub are lists with None entries, those dimensions are unconstrained.
        """
        import numpy as np
        if lb is not None:
            # Convert list-with-Nones to float array using -inf for None
            lb_arr = np.array([-np.inf if v is None else float(v) for v in lb])
            self.lb = lb_arr if np.any(np.isfinite(lb_arr)) else None
        else:
            self.lb = None
        if ub is not None:
            ub_arr = np.array([+np.inf if v is None else float(v) for v in ub])
            self.ub = ub_arr if np.any(np.isfinite(ub_arr)) else None
        else:
            self.ub = None

    def update(self, theta, grad, eta):
        """
        One gradient ascent step.

        Parameters
        ----------
        theta : (B,) current weight vector
        grad  : (B,) gradient vector (pre-averaged over samples)
        eta   : scalar step size eta_t

        Returns
        -------
        (B,) updated theta
        """
        theta_new = theta + eta * grad

        if self.lb is not None:
            theta_new = np.where(np.isfinite(self.lb), np.maximum(theta_new, self.lb), theta_new)
        if self.ub is not None:
            theta_new = np.where(np.isfinite(self.ub), np.minimum(theta_new, self.ub), theta_new)

        return theta_new