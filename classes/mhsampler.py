"""
classes/mhsampler.py
====================
Universal Metropolis-Hastings sampler for any ALP / PSMD problem.
Supports both continuous and discrete action spaces, and arbitrary
state dimensionality.

sa_chain layout: columns 0..s_dim-1 = state, column s_dim = action scalar.
Action is always stored as a scalar (not a vector), matching the convention
in gradient.py and psmd.py.

Action space type is determined by mdp.action_type:
    'continuous' : Gaussian proposal clipped to action_bounds  (default)
    'discrete'   : uniformly samples from mdp.action_set
"""

import numpy as np


class MHSampler:
    """
    MH sampler over log y_t(s, a). Universal across all ALP problems.

    Usage
    -----
    sampler  = MHSampler(mdp, dual, proposal_std=0.2, n_total=400, n_keep=200)
    chain, last_s, last_a = sampler.sample(start_s, start_a)
    # chain: (n_keep, s_dim + 1)  columns: [s..., a_scalar]
    """

    def __init__(self, mdp, dual, proposal_std=0.2, n_total=400, n_keep=200):
        self.mdp          = mdp
        self.dual         = dual
        self.proposal_std = proposal_std
        self.n_total      = n_total
        self.n_keep       = n_keep
        self.n_burnin     = n_total - n_keep

        self.s_bounds    = mdp.state_bounds
        self.a_bounds    = mdp.action_bounds
        self.s_dim       = len(self.s_bounds)
        self.action_type = getattr(mdp, 'action_type', 'continuous')

        if self.action_type == 'discrete':
            self.action_set = np.asarray(mdp.action_set)

    # ── state proposal ────────────────────────────────────────────────────
    def _propose_state(self, s):
        """Gaussian proposal clipped to state_bounds. s: (s_dim,)"""
        s_new = s + np.random.randn(self.s_dim) * self.proposal_std
        for i, (lo, hi) in enumerate(self.s_bounds):
            s_new[i] = np.clip(s_new[i], lo, hi)
        return s_new

    # ── action proposal ───────────────────────────────────────────────────
    def _propose_action(self, a):
        """
        Returns proposed action as a scalar float/int.
        Discrete: uniform random from action_set.
        Continuous: Gaussian clipped.
        """
        if self.action_type == 'discrete':
            return float(np.random.choice(self.action_set))
        else:
            lo, hi = self.a_bounds[0]
            return float(np.clip(a + np.random.randn() * self.proposal_std,
                                 lo, hi))

    # ── main sampler ──────────────────────────────────────────────────────
    def sample(self, start_s, start_a):
        """
        Run MH chain from (start_s, start_a).

        Parameters
        ----------
        start_s : scalar or (s_dim,) array
        start_a : scalar action value

        Returns
        -------
        chain  : (n_keep, s_dim + 1)
                 Columns 0..s_dim-1 = state, column s_dim = action scalar
        last_s : (s_dim,) final state
        last_a : scalar final action
        """
        s = np.atleast_1d(np.asarray(start_s, dtype=float)).copy()  # (s_dim,)
        a = float(start_a)                                            # scalar

        chain = []

        for i in range(self.n_total):
            s_new = self._propose_state(s)
            a_new = self._propose_action(a)

            # Build (2,) or (2, s_dim) batches — action always (2,)
            if self.s_dim == 1:
                s_batch = np.array([s[0], s_new[0]])   # (2,)
            else:
                s_batch = np.stack([s, s_new])         # (2, s_dim)

            a_batch = np.array([a, a_new])             # (2,) always scalar

            logp = self.dual.log_y_t_batch(s_batch, a_batch)   # (2,)

            if np.log(np.random.rand()) <= logp[1] - logp[0]:
                s, a = s_new, a_new

            if i >= self.n_burnin:
                # Store as flat row: [s_0, ..., s_{s_dim-1}, a]
                chain.append(np.append(s, a))

        chain = np.array(chain)   # (n_keep, s_dim + 1)
        return chain, s, a