"""
classes/gradient.py
===================
Universal gradient computer for any ALP / PSMD problem.
Supports arbitrary state dimensionality.

sa_samples layout: columns 0..s_dim-1 = state, column s_dim = action.
s_dim is read from len(mdp.state_bounds) — never hardcoded to 1.
"""

import numpy as np


class Gradient:
    """
    Stochastic gradient computer. Universal across all ALP problems.

    Usage
    -----
    grad = Gradient(mdp, alp)
    g    = grad.compute_batch(sa_samples, exog_samples)   # -> (B,)
    """

    def __init__(self, mdp, alp):
        self.mdp      = mdp
        self.alp      = alp
        self.gamma    = mdp.gamma
        self.s_dim    = len(mdp.state_bounds)   # number of state dimensions
        self._E_phi_q = alp.E_phi(mdp)          # (B,) cached

    def compute_batch(self, sa_samples, exog_samples):
        """
        Average gradient over H (s,a) samples.

        grad_f(s,a) = [gamma * E[phi(s')] - phi(s)] / (1-gamma) + E_q[phi]

        Parameters
        ----------
        sa_samples   : (H, s_dim + 1) array
                       Columns 0..s_dim-1 = state
                       Column  s_dim      = action (scalar per sample)
        exog_samples : (N,) or (N, d_exog) array

        Returns
        -------
        (B,) gradient vector
        """
        s_dim = self.s_dim

        if s_dim == 1:
            s = sa_samples[:, 0]        # (H,)
            a = sa_samples[:, 1]        # (H,)
        else:
            s = sa_samples[:, :s_dim]   # (H, s_dim)
            a = sa_samples[:, s_dim]    # (H,)  action is always scalar

        phi_s      = self.alp.phi(s)                                    # (H, B)
        E_phi_next = self.alp.E_phi_next(self.mdp, s, a, exog_samples) # (H, B)

        grads = (self.gamma * E_phi_next - phi_s) / (1.0 - self.gamma) \
                + self._E_phi_q                                          # (H, B)

        return grads.mean(axis=0)   # (B,)

    def compute_single(self, s, a, exog_samples):
        """Gradient at a single (s, a) — for debugging."""
        s  = np.atleast_1d(np.asarray(s, dtype=float))
        sa = np.concatenate([s, np.atleast_1d(a)])[None, :]
        return self.compute_batch(sa, exog_samples)