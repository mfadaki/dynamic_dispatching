"""
classes/bounds.py
=================
Universal lower and upper bound computation.
Updated to handle:
    - Discrete action spaces (mdp.action_type == 'discrete')
    - Multi-dimensional state vectors
    - Arbitrary exog types
"""

import numpy as np


# =============================================================================
# _GibbsSAATarget — helper for SAALowerBound
# =============================================================================

class _GibbsSAATarget:
    """
    Lightweight stand-in for the Dual object, exposing log_y_t_batch(s,a), so
    the existing MHSampler machinery can sample from the SAA lower-bound
    density (Section 5.3, eq. lb-sampling):

        y_hat_{kappa_bar,theta_bar}(s,u) ~ exp(-f_hat^{N'}(theta_bar,s,u)/kappa_bar)

    This is independent of the training-time omega_c/omega_theta compact
    dual representation — it only needs the FINAL theta_bar and kappa_bar.
    """

    def __init__(self, mdp, alp, theta_bar, kappa_bar, exog_Nprime):
        self.mdp         = mdp
        self.alp         = alp
        self.theta_bar   = theta_bar
        self.kappa_bar   = kappa_bar
        self.exog_Nprime = exog_Nprime

    def log_y_t_batch(self, s, a, t=None):
        s = np.atleast_2d(np.asarray(s, dtype=float))
        a = np.atleast_1d(np.asarray(a, dtype=float))
        valid = self._check_bounds(s, a)
        if not np.any(valid):
            return np.full(s.shape[0], -np.inf)
        fN = np.atleast_1d(self.alp.f_hat(self.mdp, self.theta_bar, s, a,
                                          self.exog_Nprime))
        log_y = -fN / self.kappa_bar
        log_y[~valid] = -np.inf
        return log_y

    def _check_bounds(self, s, a):
        H     = s.shape[0]
        valid = np.ones(H, dtype=bool)
        for dim, (lo, hi) in enumerate(self.mdp.state_bounds):
            valid &= (s[:, dim] >= lo) & (s[:, dim] <= hi)
        lo, hi = self.mdp.action_bounds[0]
        valid &= (a >= lo) & (a <= hi)
        return valid


# =============================================================================
# SAALowerBound — the actual Section 5.3 bound (replaces the Cbar=0 stub)
# =============================================================================

class SAALowerBound:
    """
    Sample-average approximation of ell(theta_bar) (Section 5.3, eq. lb-saa):

        ell_hat(theta_bar) := (1/H') sum_h f_hat^{N'}(theta_bar, s_h, a_h)
                               + kappa_bar*Cbar + n*kappa_bar*log(kappa_bar)

    where (s_h,a_h) ~ y_hat_{kappa_bar,theta_bar} (see _GibbsSAATarget above),
    sampled via the SAME Metropolis-Hastings machinery used during PSMD
    training (classes.mhsampler.MHSampler), targeting this SAA Gibbs density
    directly rather than the training-time compact dual representation.

    Requires alp.compute_geometry_and_lipschitz(mdp) to have been called
    once beforehand (populates alp.n_dim, alp.R, alp.diam, alp.p_bar, alp.L,
    alp.Cbar). This is a VALID but genuinely WEAKER bound than the ad hoc
    LP-based LowerBound above — see the Cbar derivation's own docstring for
    why (the discontinuous EOD/expiry indicator forces a large, grid-scale
    Lipschitz constant, which materially loosens Cbar and hence this bound).
    Use LowerBound for a tighter (but not theoretically certified) day-to-day
    diagnostic, and this class when you need an actually-valid certificate.
    """

    def __init__(self, mdp, alp):
        self.mdp = mdp
        self.alp = alp
        if not hasattr(alp, 'n_dim'):
            alp.compute_geometry_and_lipschitz(mdp)

    def compute(self, theta_bar, kappa_bar, H_prime=100, N_prime=500,
                n_total=400, n_keep=200, proposal_std=0.5, seed=None):
        from classes.mhsampler import MHSampler

        if kappa_bar <= 0:
            return float('-inf'), None   # bound undefined / not yet meaningful

        if seed is not None:
            np.random.seed(seed)

        exog_Nprime = self.mdp.sample_exog(N_prime)
        target  = _GibbsSAATarget(self.mdp, self.alp, theta_bar, kappa_bar,
                                  exog_Nprime)
        sampler = MHSampler(self.mdp, target, proposal_std=proposal_std,
                            n_total=n_total, n_keep=n_keep)

        discrete = (getattr(self.mdp, 'action_type', 'continuous') == 'discrete')
        s0 = self.mdp.sample_initial_state()
        a0 = (float(np.random.choice(self.mdp.action_set)) if discrete
              else float(np.random.uniform(*self.mdp.action_bounds[0])))

        collected = []
        n_have = 0
        while n_have < H_prime:
            chain, s0, a0 = sampler.sample(s0, a0)
            collected.append(chain)
            n_have += chain.shape[0]
        sa = np.vstack(collected)[:H_prime]

        s_dim = len(self.mdp.state_bounds)
        S_h   = sa[:, :s_dim]
        A_h   = sa[:, s_dim]

        f_vals = np.atleast_1d(self.alp.f_hat(self.mdp, theta_bar, S_h, A_h,
                                              exog_Nprime))

        correction = kappa_bar * self.alp.Cbar + self.alp.n_dim * kappa_bar * np.log(kappa_bar)
        ell_hat = float(np.mean(f_vals)) + correction
        se      = float(np.std(f_vals) / np.sqrt(len(f_vals)))
        return ell_hat, se


# =============================================================================
# LowerBound
# =============================================================================

class LowerBound:
    """
    LP-based (constraint-sampling) lower bound on V*(s_0).

    The ALP lower bound is: max_{theta} E_q[phi].theta
                            s.t. Alp(S,A).theta <= blp(S,A)
    This is the LP objective value; under IDEALIZED (full or sufficiently
    representative) constraint coverage this is a valid lower bound
    (theta.E_q[phi] = E_q[theta.phi(s)] <= E_q[V*(s)]), but with a FINITE,
    resampled-every-checkpoint constraint set it is only approximately so —
    see classes.mdp.sample_reachable_states() for how the constraint sample
    is constructed, and the caveats there.

    For an actually CERTIFIED lower bound (at the cost of being materially
    looser), use SAALowerBound above, which implements Section 5.3's
    ell_hat(theta_bar) directly, with a properly-derived (no longer
    hardcoded-to-zero) Cbar via alp.compute_geometry_and_lipschitz(). This
    class remains useful as a cheaper, tighter day-to-day training
    diagnostic, just not a theoretically certified one.

    At each evaluation, we re-solve the LP warm-started from thetabar
    using fresh random sample points. The LP objective is our LB.
    """

    def __init__(self, mdp, alp):
        self.mdp      = mdp
        self.alp      = alp
        self.discrete = (getattr(mdp, 'action_type', 'continuous') == 'discrete')
        if self.discrete:
            self.action_set = np.asarray(mdp.action_set)
        print(f"LowerBound: LP-based (valid for high-dim state spaces)")

    def _solve_once(self, n_init, N):
        """One independent LP re-solve: fresh random constraint sample,
        fresh exog, fresh action draws. Returns the LP objective (or the
        fallback dot-product if the solve itself fails)."""
        from scipy.optimize import linprog

        if self.discrete:
            action_weights = getattr(
                __import__('inputs.inputs', fromlist=['ACTION_WEIGHTS']),
                'ACTION_WEIGHTS', None)
            w = np.asarray(action_weights, dtype=float) / sum(action_weights) \
                if action_weights else None
            A_samp = np.random.choice(self.action_set, n_init, p=w)
        else:
            lo, hi = self.mdp.action_bounds[0]
            A_samp = np.random.uniform(lo, hi, n_init)

        S_samp = self.mdp.sample_reachable_states(n_init)
        exog   = self.mdp.sample_exog(N)

        gamma     = self.mdp.gamma
        phi_s     = self.alp.phi(S_samp)
        E_pn      = self.alp.E_phi_next(self.mdp, S_samp, A_samp, exog)
        blp       = self.mdp.cost(S_samp, A_samp, exog)
        Alp       = phi_s - gamma * E_pn
        Alp[:, 0] = 1.0 - gamma

        E_phi = self.alp.E_phi(self.mdp)

        theta_bounds = [(None, None)] + [(0, None)] * (self.alp.B - 1)
        res = linprog(-E_phi, A_ub=Alp, b_ub=blp,
                      bounds=theta_bounds, method='highs')
        return float(-res.fun) if res.success else None

    def compute(self, thetabar, lambdabar, H=300, N=1000, n_resolves=5, **kwargs):
        """
        Compute LP lower bound as the MEDIAN of n_resolves independent LP
        re-solves (each with its own fresh random constraint sample and
        exog draw), rather than a single draw.

        This does not fix the underlying weakness of constraint sampling
        with a modest H — it reduces the chance that any ONE unlucky draw
        (which can occasionally land as a visible outlier against a
        thetabar-independent Monte Carlo UB — see this class's docstring)
        gets reported as "the" checkpoint value. The median, rather than
        the mean, is used specifically because it is robust to a single
        anomalous high draw without being dragged toward it the way a mean
        would be.

        Returns (lb scalar, None). lambdabar accepted for interface
        compatibility but not used. thetabar is used only as the fallback
        value if every one of the n_resolves LP solves fails outright.
        """
        vals = [self._solve_once(H, N) for _ in range(max(1, n_resolves))]
        vals = [v for v in vals if v is not None]

        if not vals:
            lb = float(np.dot(self.alp.E_phi(self.mdp), thetabar))  # fallback
        else:
            lb = float(np.median(vals))

        return lb, None


# =============================================================================
# UpperBound
# =============================================================================

class UpperBound:
    """
    E[V^pi(s)] under the greedy ALP policy via Monte Carlo simulation.

    Greedy policy:
        Continuous: argmin_{a in grid} c(s,a) + gamma*theta.E[phi(s')]
        Discrete:   argmin_{a in action_set} c(s,a) + gamma*theta.E[phi(s')]
    """

    def __init__(self, mdp, alp):
        self.mdp      = mdp
        self.alp      = alp
        self.gamma    = mdp.gamma
        self.discrete = (getattr(mdp, 'action_type', 'continuous') == 'discrete')
        if self.discrete:
            self.action_set = np.asarray(mdp.action_set)

    def greedy_action(self, s, thetabar, exog):
        """
        Choose best action at state s.
        s    : (s_dim,) array (single state)
        exog : (N,) exog samples for expectation
        Returns best action (int for discrete, float for continuous).
        """
        s_dim = len(self.mdp.state_bounds)

        if self.discrete:
            actions = self.action_set
        else:
            lo, hi  = self.mdp.action_bounds[0]
            actions = np.linspace(lo, hi, 101)

        M     = len(actions)
        S_rep = np.tile(s[None, :], (M, 1))           # (M, s_dim)
        A_rep = np.asarray(actions)                    # (M,)

        cost  = self.mdp.cost(S_rep, A_rep, exog)      # (M,)
        Ephi  = self.alp.E_phi_next(self.mdp, S_rep,
                                    A_rep, exog)        # (M, B)
        value = cost + self.gamma * (Ephi * thetabar).sum(axis=-1)  # (M,)

        best_idx = int(np.argmin(value))
        best_a   = actions[best_idx]
        return int(best_a) if self.discrete else float(best_a)

    def _simulate_trajectory(self, thetabar, num_stage=200, n_exog_policy=100):
        """Simulate one trajectory. Returns cumulative discounted cost."""
        s       = self.mdp.sample_initial_state()   # (s_dim,)
        cumcost = 0.0

        for t in range(num_stage):
            exog_pol = self.mdp.sample_exog(n_exog_policy)
            a        = self.greedy_action(s, thetabar, exog_pol)

            g        = self.mdp.sample_single_exog()
            cost     = self.mdp.cost_realised(s, a, g)
            s_next   = self.mdp.transition(s, a, g)
            # transition with scalar exog may return (1, N_STATE) — always ravel
            s        = np.asarray(s_next, dtype=float).ravel()
            cumcost += self.gamma ** t * cost

            if t > 0 and (self.gamma**t * cost) / max(abs(cumcost), 1e-8) < 1e-5:
                break

        return cumcost

    def compute(self, thetabar, num_stage=200, num_traj=50,
                n_action_pts=None, n_exog_policy=100):
        """
        Compute upper bound.
        Returns (ub mean, ub standard error).
        """
        costs = []
        for _ in range(num_traj):
            c = self._simulate_trajectory(thetabar,
                                          num_stage=num_stage,
                                          n_exog_policy=n_exog_policy)
            costs.append(c)
            arr = np.array(costs)
            if len(arr) > 1:
                se = arr.std() / np.sqrt(len(arr))
                if se < abs(arr.mean()) * 0.001:
                    break

        arr = np.array(costs)
        return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))