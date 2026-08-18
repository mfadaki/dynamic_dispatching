"""
classes/dualupdater.py
======================
Universal dual updater for any ALP / PSMD problem.
No problem-specific code — all problem logic goes through alp.f_hat().

Implements paper Algorithm 1, Step 4 (eq. 12 / 15).

Weight arrays store the FULL accumulated history:
    weightcost[t]    : scalar — cost weight at iteration t (includes the
                       c_w = 1/(1-gamma) factor from f_hat's c_w*r(s,u) term)
    weighttheta[t,:] : (B,)  — theta weight at iteration t

Past entries are scaled by alpha = 1/(1+lam*eta) each iteration.

Key universality fix:
    log_y_t_batch uses alp.phi, alp.E_phi_next, mdp.cost — no hardcoded
    inventory formula. Works for any MDP/ALP.

    Formula:
        log y_t(s,a) = wc_sum * cost(s,a;G_t) + grad_f(s,a;G_t) . wt_sum
    where
        grad_f = [gamma*E[phi(s')] - phi(s)] / (1-gamma) + E_q[phi]
        G_t    = DSampleHist[t]  (current iteration exog)
        wc_sum, wt_sum = accumulated weight sums (encode full history);
                         wc_sum already carries the c_w = 1/(1-gamma)
                         factor (see update()), so no further division by
                         (1-gamma) is applied here.
"""

import numpy as np


class Dual:
    """
    Full-history dual weight tracker. Universal across all ALP problems.

    Usage
    -----
    dual = Dual(mdp, alp, T=1000, N_mh=50)
    dual.set_exog_history(DSampleHist)
    # inside loop:
    dual.update(eta, lam, theta)
    log_vals = dual.log_y_t_batch(s_batch, a_batch)   # (H,)
    wc, wt   = dual.weight_sums()
    """

    def __init__(self, mdp, alp, T, N_mh):
        self.mdp      = mdp
        self.alp      = alp
        self.gamma    = mdp.gamma
        self.B        = alp.B
        self.T        = T
        self.N_mh     = N_mh
        self._E_phi_q = alp.E_phi(mdp)   # (B,) cached

        self.weightcost  = np.zeros(T)           # (T,)
        self.weighttheta = np.zeros((T, self.B)) # (T, B)
        self.DSampleHist = None
        self.t           = -1

    def set_exog_history(self, DSampleHist):
        """
        Set pre-generated exog history.
        DSampleHist[k] = exog samples used at iteration k.
        Shape: (T, N_mh) for 1D exog; list of T arrays for structured exog.
        """
        self.DSampleHist = DSampleHist

    # ── dual weight update ────────────────────────────────────────────────
    def update(self, eta, lam, theta):
        """
        Scale all past weights, write new entry at position t.

            c_w    = 1 / (1 - gamma)          [matches f_hat's c_w * r(s,u) term]
            scale  = 1 / (1 + lam*eta)
            beta   = eta * scale
            new_wc = -beta * c_w              [cost-weight recursion]
            new_wt = -beta * theta             [theta-weight recursion, no c_w]
            weightcost[:t]    *= scale
            weighttheta[:t]   *= scale
            weightcost[t]     = new_wc
            weighttheta[t,:]  = new_wt
        """
        self.t += 1
        t      = self.t
        c_w    = 1.0 / (1.0 - self.gamma)
        scale  = 1.0 / (1.0 + lam * eta)
        beta   = eta * scale
        new_wc = -beta * c_w

        self.weightcost[:t]   *= scale
        self.weighttheta[:t]  *= scale
        self.weightcost[t]     = new_wc
        self.weighttheta[t, :] = -beta * theta

    # ── log y_t — fully universal ─────────────────────────────────────────
    def log_y_t_batch(self, s, a, t=None):
        """
        log y_t(s, a) for a batch of H (s,a) pairs.

        Fully universal — calls mdp.cost, alp.phi, alp.E_phi_next.
        No hardcoded cost formula.

        Parameters
        ----------
        s : (H,) or (H, s_dim)
        a : (H,) or (H, a_dim)
        t : int or None

        Returns (H,) log-density values, -inf for infeasible points.
        """
        if t is None:
            t = self.t

        s = np.asarray(s, dtype=float)
        a = np.asarray(a, dtype=float)
        H = s.shape[0]

        valid = self._check_bounds(s, a)
        if not np.any(valid):
            return np.full(H, -np.inf)

        exog = self.DSampleHist[t]   # (N_mh,) or structured

        # grad_f: (H, B)
        phi_s      = self.alp.phi(s)                           # (H, B)
        E_phi_next = self.alp.E_phi_next(self.mdp, s, a, exog) # (H, B)
        grad_f     = ((self.gamma * E_phi_next - phi_s)
                      / (1.0 - self.gamma) + self._E_phi_q)    # (H, B)

        cost = self.mdp.cost(s, a, exog)                       # (H,)

        wc_sum = self.weightcost[:t+1].sum()
        wt_sum = self.weighttheta[:t+1].sum(axis=0)            # (B,)

        log_y             = wc_sum * cost + grad_f @ wt_sum    # (H,)
        log_y[~valid]     = -np.inf
        return log_y

    def log_y_t(self, s, a, t=None):
        """Scalar convenience wrapper."""
        return float(self.log_y_t_batch(
            np.atleast_1d(np.asarray(s, dtype=float)),
            np.atleast_1d(np.asarray(a, dtype=float)), t)[0])

    # ── bounds check — any dimensionality ────────────────────────────────
    def _check_bounds(self, s, a):
        H     = s.shape[0]
        valid = np.ones(H, dtype=bool)
        if s.ndim == 1:
            lo, hi = self.mdp.state_bounds[0]
            valid &= (s >= lo) & (s <= hi)
        else:
            for dim, (lo, hi) in enumerate(self.mdp.state_bounds):
                valid &= (s[:, dim] >= lo) & (s[:, dim] <= hi)
        if a.ndim == 1:
            lo, hi = self.mdp.action_bounds[0]
            valid &= (a >= lo) & (a <= hi)
        else:
            for dim, (lo, hi) in enumerate(self.mdp.action_bounds):
                valid &= (a[:, dim] >= lo) & (a[:, dim] <= hi)
        return valid

    # ── accessors ─────────────────────────────────────────────────────────
    def weight_sums(self):
        """Return (wc_sum, wt_sum) for iterations 0..t."""
        t = self.t
        return (self.weightcost[:t+1].sum(),
                self.weighttheta[:t+1].sum(axis=0))