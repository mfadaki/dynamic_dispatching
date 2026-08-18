"""
classes/alp_evaluate_approximation.py
=======================================
ISOLATED COPY of classes/alp.py, used ONLY by evaluate_approximation.py.

Identical to the production classes/alp.py except for one fix:
E_phi_next() now computes its H x N (states x exog) broadcast explicitly,
via the deterministic single-state transition, rather than delegating to
mdp.transition()'s public broadcasting heuristic (which infers "broadcast"
vs "1-to-1 aligned" from len(exog) != H -- silently wrong whenever H
happens to equal N, e.g. two sample-size hyperparameters coinciding, which
this script's small-instance config makes more likely to occur).

Deliberately kept as a SEPARATE file rather than edited in place, so the
production classes/alp.py -- and therefore every existing/future training
run through main.py -- is completely unaffected. Do not import this
module from anything other than evaluate_approximation.py.

------------------------------------------------------------------------
Original classes/alp.py docstring follows, unmodified:
------------------------------------------------------------------------

classes/alp.py — B=7 VFA

FEATURE SET (B=7):
  0  const         1
  1  depot_age1    h_{0,1}·n_{0,1}                          θ ≥ 0  (break-even floor)
  2  depot_age2    h_{0,2}·n_{0,2}                          θ ≥ 0  (break-even floor)
  3  depot_age3    h_{0,3}·n_{0,3}                          θ ≥ 0  (break-even floor)
  4  lab_shortfall Σ_{p≥1} max(0, μ_p·τ − N_lab_p)          θ ≥ 0  (break-even floor)
  5  imbalance     |N_1/μ_1 − N_2/μ_2|                      θ ≥ 0  (break-even floor)
  6  exp_risk      C_dep·n_{0,1} + C_lab·Σ_{p≥1} n_{p,1}        θ ≥ 0  (break-even floor)

DESIGN PRINCIPLE
================
The Bellman recursion V_t(s) = min_u { g(s) + c(s,u) + γ E[V_{t+1}(s')] }
already embeds every future g and c inside V_{t+1}. Features are therefore NOT
re-encodings of g/c; they span what TODAY'S STATE forecasts about the FUTURE
cost stream, on top of the current-epoch g+c on the same Bellman line.

SIGN REQUIREMENT
Every non-const feature should DECREASE when a loaded depot is dispatched into an
under-utilised lab, so a non-negative weight REWARDS the useful dispatch. A
feature that RISES on every dispatch is a dispatch BRAKE (the removed hold_max
did this and caused the "stall"). imbalance is the one feature whose sign is
ACTION-DEPENDENT — see its note — and it earns its place precisely because it is
the only term that rewards EVENING OUT the two labs.

  depot_age{1,2,3} — hold_depot split by age class. The aggregate hold_depot
                  Σ_a h_{0,a} n_{0,a} forced the three ages to share the fixed
                  cost ratio h_{0,·}=[3,2,1]. Splitting lets the VFA weight
                  URGENCY independently of holding cost — an age-1 depot kit
                  (about to expire, dispatch-saveable) can be valued differently
                  from an age-3 kit of equal holding cost. The three age counts
                  are ~uncorrelated (|corr|<0.02), so each carries independent
                  signal. Dispatch empties the depot ⇒ each Δφ < 0 ⇒ break-even
                  floor. (depot_age1 vs exp_risk corr ≈ 0.69: related but not
                  redundant — exp_risk also weights lab age-1 kits.)

  lab_shortfall — forecasts IDLE processing capacity, AGGREGATED over labs:
                  Σ_p max(0, μ_p·τ − N_p). Drives dispatch VOLUME (how badly the
                  labs need feeding overall). Kept aggregate, NOT split per lab:
                  a per-lab split still biases toward the faster lab (μ_2>μ_1)
                  and does not by itself fix starvation. Dispatch raises N_p ⇒
                  Δφ < 0 ⇒ break-even floor.

  imbalance     — |N_1/μ_1 − N_2/μ_2|, the throughput-normalised backlog gap
                  (backlog measured in TIME units, so the target is equal CLEAR
                  TIME, not equal counts). This is the ONLY feature that fixes
                  lab-1 starvation: lab_shortfall alone always prefers the faster
                  lab, so the depot piles onto lab 2 while lab 1 sits idle and
                  parallel throughput is wasted. imbalance rewards balancing.
                  SIGN IS ACTION-DEPENDENT: dispatching the whole depot into the
                  EMPTIER lab lowers it (rewarded), into the FULLER lab raises it
                  (penalised). With θ≥0 it therefore steers dispatch to the
                  under-loaded lab. Because dispatching into already-balanced labs
                  can raise it, it carries mild brake risk; the break-even floor
                  keeps θ from collapsing and a sign-behaviour check (run before
                  wiring) confirms it does not suppress useful dispatch.

  exp_risk      — cost-weighted count of OLDEST-class (age-1) kits across depot
                  and labs: C_dep·n_{0,1} + C_lab·Σ_p n_{p,1}. The only kits one
                  epoch from expiry. Restricted to age-1 (older depot kits are
                  already in depot_age2/3; including them duplicated depot volume
                  and zeroed the weight). No survival discounting (τ is large
                  enough that survival ≈ 1 almost everywhere, which made the
                  discounted form vanish). Rises as kits age into class 1 under
                  no-dispatch; falls as oldest-first processing clears lab age-1
                  kits. Depot part is dispatch-actionable; lab part adds the
                  independent lab-side expiry signal (≈uncorrelated with the
                  depot features).

REMOVED / NOT ADDED (with reasons)
  depot_total   : redundant — captured by the depot_age features.
  hold_min      : redundant — labs identical in holding; collapses to ~0.
  hold_max      : WRONG SIGN — rose on dispatch into empty labs; caused the stall.
  unmet_demand  : redundant with lab_shortfall (corr ≈ 0.999).
  per-lab split of lab_shortfall : still speed-biased; does not fix starvation,
                  so the aggregate is kept and imbalance does the balancing.
  depot_urgency Σ_a (L−a+1) n_{0,a} : EXACTLY hold_depot (weights [3,2,1] = h_0);
                  even the 1/a form is 0.993-correlated. Subsumed by depot_age*.
  τ-scaling / EOD-proximity : deferred — measure the gap with this set first.
"""
import numpy as np
from inputs.inputs import (
    NO_BASIS_FN, N_LABS, L_AGE, N_INV, N_STATE,
    H_HOLD, C_EXP_DEPOT, C_EXP_LAB, TAU_MAX, MU,
    STATE_BOUNDS, ACTION_BOUNDS, LAMBDA_TOTAL, DELTA_T, THETA_UB,
)
_MU_TOTAL = float(MU.sum())
_MU       = np.asarray(MU, float)        # per-lab throughput rates μ_p


class ALP:
    B        = NO_BASIS_FN   # 7
    n_sa_dim = N_STATE + 1
    Cbar     = 0.0   # overwritten by compute_geometry_and_lipschitz() below

    def compute_geometry_and_lipschitz(self, mdp, theta_ub=None,
                                        n_state_samples=3000, n_boundary_samples=1000,
                                        exog_N=300, safety_factor=1.25, seed=123):
        """
        Populate the geometric constants (n, R, diam(S x U), p_bar) and an
        empirically-estimated Lipschitz constant L for f(theta,s,u) over
        Theta x S x U, then compute Cbar (Section 5.3, eq. Cbar):

            Cbar = log(p_bar) - L*(R + diam(SxU)) + n*log(R)
                   - log[Gamma(n/2+1)*pi^{-n/2}]

        GEOMETRY: n, R, diam, p_bar come directly from mdp.state_bounds /
        mdp.action_bounds (the discrete action set is embedded in its convex
        hull [lo,hi] from ACTION_BOUNDS for this purpose — the paper's own
        note that "low-dimensional discrete states can easily be
        incorporated" extends the same way to a low-cardinality discrete
        action).

        LIPSCHITZ CONSTANT: estimated by finite differences of f_hat(theta_ub,
        s, u) using COMMON RANDOM NUMBERS (one fixed exog sample reused for
        every perturbed/unperturbed evaluation, isolating the slope from
        sampling noise), evaluated at theta_ub — the upper corner of Theta —
        since f is affine in theta with non-negative coefficients for b>=1,
        so the worst-case (s,u)-sensitivity over Theta is attained there.

        The EOD/expiry indicator 1{tau-Delta_t<=0} in r(s,u) is genuinely
        discontinuous in a continuously-embedded tau, so f is NOT globally
        Lipschitz in the classical sense (this is a real tension with
        Assumption 1 of Lin et al. 2020, not a bug). But this model is
        fundamentally discrete-time: tau only ever takes values on the
        Delta_t grid (tau_max - k*Delta_t), never in between. We therefore
        estimate the tau-direction slope using Delta_t itself as the step
        (the true smallest resolvable distance in this chain), NOT an
        arbitrarily small epsilon — using a vanishing epsilon here would
        inflate L (and hence make Cbar arbitrarily, uselessly negative)
        without reflecting anything the chain can actually distinguish.
        Even so, this DOES make Cbar substantially more negative than the
        Cbar=0 stub it replaces: expect a materially looser (but now
        actually valid) lower bound.
        """
        from scipy.special import gammaln

        if theta_ub is None:
            theta_ub = np.array([50.0 if v is None else float(v) for v in THETA_UB])

        # ---- geometry: n, R, diam(S x U), p_bar ------------------------------
        sides = np.asarray(
            [hi - lo for (lo, hi) in STATE_BOUNDS] +
            [hi - lo for (lo, hi) in ACTION_BOUNDS],
            dtype=float,
        )
        n     = len(sides)
        R     = float(sides.min() / 2.0)
        diam  = float(np.sqrt((sides**2).sum()))
        p_bar = 1.0 / float(np.prod(sides))

        # ---- empirical, grid-aware Lipschitz constant ------------------------
        prev = np.random.get_state()
        np.random.seed(seed)
        exog = mdp.sample_exog(exog_N)
        lo = np.array([b[0] for b in STATE_BOUNDS])
        hi = np.array([b[1] for b in STATE_BOUNDS])
        discrete = (getattr(mdp, 'action_type', 'continuous') == 'discrete')
        action_set = np.asarray(mdp.action_set, dtype=float) if discrete else None

        def _sample_sa(H):
            S = np.random.uniform(lo, hi, size=(H, len(lo)))
            A = (np.random.choice(action_set, size=H) if discrete
                 else np.random.uniform(*ACTION_BOUNDS[0], size=H))
            return S, A.astype(float)

        L_candidates = []

        # (a) general state-coordinate sweep, unit/grid-scale steps
        H = n_state_samples
        S, A = _sample_sa(H)
        f0 = np.atleast_1d(self.f_hat(mdp, theta_ub, S, A, exog))
        dims = np.random.randint(0, len(lo), size=H)
        side = hi - lo
        tau_dim = len(lo) - 1
        eps = np.where(dims == tau_dim, DELTA_T, np.maximum(1.0, 1e-2 * side[dims]))
        S2 = S.copy()
        idx = np.arange(H)
        S2[idx, dims] = np.clip(S[idx, dims] + eps, lo[dims], hi[dims])
        f1 = np.atleast_1d(self.f_hat(mdp, theta_ub, S2, A, exog))
        dS = np.abs(S2[idx, dims] - S[idx, dims])
        mask = dS > 1e-9
        if mask.any():
            L_candidates.append(np.max(np.abs(f1[mask] - f0[mask]) / dS[mask]))

        # (b) action-coordinate sweep
        if discrete:
            A2 = np.array([np.random.choice(action_set[action_set != a])
                            if (action_set != a).any() else a for a in A])
        else:
            eps_a = max(1e-6, 1e-2 * (ACTION_BOUNDS[0][1] - ACTION_BOUNDS[0][0]))
            A2 = np.clip(A + eps_a, *ACTION_BOUNDS[0])
        f2 = np.atleast_1d(self.f_hat(mdp, theta_ub, S, A2, exog))
        dA = np.abs(A2 - A)
        mask2 = dA > 1e-9
        if mask2.any():
            L_candidates.append(np.max(np.abs(f2[mask2] - f0[mask2]) / dA[mask2]))

        # (c) dedicated EOD/expiry boundary straddle, at Delta_t grid resolution
        Hb = n_boundary_samples
        Sb, Ab = _sample_sa(Hb)
        Sb_lo = Sb.copy(); Sb_lo[:, tau_dim] = DELTA_T
        Sb_hi = Sb.copy(); Sb_hi[:, tau_dim] = 2 * DELTA_T
        f_lo = np.atleast_1d(self.f_hat(mdp, theta_ub, Sb_lo, Ab, exog))
        f_hi = np.atleast_1d(self.f_hat(mdp, theta_ub, Sb_hi, Ab, exog))
        L_candidates.append(np.max(np.abs(f_hi - f_lo)) / DELTA_T)

        np.random.set_state(prev)
        L = float(max(L_candidates) * safety_factor)

        # ---- Cbar -------------------------------------------------------------
        log_gamma_term = gammaln(n / 2.0 + 1.0) - (n / 2.0) * np.log(np.pi)
        Cbar = float(np.log(p_bar) - L * (R + diam) + n * np.log(R) - log_gamma_term)

        self.n_dim   = n
        self.R       = R
        self.diam    = diam
        self.p_bar   = p_bar
        self.L       = L
        self.Cbar    = Cbar
        self._theta_ub_for_L = theta_ub.copy()

        print(f"Geometry/Lipschitz: n={n}, R={R:.4f}, diam={diam:.4f}, "
              f"p_bar={p_bar:.3e}, L={L:.3e}, Cbar={Cbar:.3e}")
        return dict(n=n, R=R, diam=diam, p_bar=p_bar, L=L, Cbar=Cbar)

    # ── Feature normalization (scale-only) ─────────────────────────────────────
    # φ̃_b = φ_b / σ_b with σ_0 ≡ 1 (constant). Defaults to ones (no-op) so the
    # class is usable before calibration; calibrate_normalization() fills it from
    # the initial-state distribution ν. Stored as a (B,) vector for broadcasting.
    _phi_scale = np.ones(NO_BASIS_FN)

    def calibrate_normalization(self, mdp, n_samples=20000, seed=12345):
        """Estimate per-feature scales σ_b = std_ν[φ_b] once, under ν.

        σ_0 (constant) is forced to 1. Any σ_b that is ~0 (a feature with no
        spread under ν) is also set to 1 to avoid division blow-up. Deterministic
        given the seed ⇒ reproducible. Returns the scale vector and also the raw
        means (for reporting / de-normalizing θ back to raw coordinates).
        """
        prev = np.random.get_state()
        np.random.seed(seed)
        S = np.array([mdp.sample_initial_state() for _ in range(n_samples)])
        np.random.set_state(prev)
        raw   = self._phi_raw(S)                    # (n_samples, B)
        sigma = raw.std(axis=0)
        mu    = raw.mean(axis=0)
        sigma[0] = 1.0                              # const: leave as-is
        sigma[sigma < 1e-8] = 1.0                   # guard zero-spread features
        self._phi_scale = sigma
        return sigma, mu

    def feature_scale(self):
        """Return the current (B,) normalization scale vector σ."""
        return self._phi_scale.copy()

    # ── state parsing ─────────────────────────────────────────────────────────
    @staticmethod
    def _parse(s):
        if s.ndim == 1:
            n   = s[:N_INV].reshape(N_LABS + 1, L_AGE)
            tau = s[N_INV]
        else:
            n   = s[:, :N_INV].reshape(-1, N_LABS + 1, L_AGE)
            tau = s[:, N_INV]
        return n, tau

    # ── basis functions ───────────────────────────────────────────────────────
    @staticmethod
    def phi_const(s):
        """φ₀ = 1."""
        return 1.0 if s.ndim == 1 else np.ones(s.shape[0])

    @staticmethod
    def phi_depot_age(s, a_idx):
        """φ = h_{0,a}·n_{0,a} for a single depot age class a_idx (0-based).

        hold_depot split by age so the VFA can weight urgency independently of
        the fixed holding-cost ratio. Dispatch empties the depot ⇒ Δφ < 0.
        """
        n, _ = ALP._parse(s)
        if s.ndim == 1:
            return float(H_HOLD[0, a_idx] * n[0, a_idx])
        return H_HOLD[0, a_idx] * n[:, 0, a_idx]

    @staticmethod
    def phi_lab_shortfall(s):
        """φ = Σ_{p≥1} max(0, μ_p·τ − N_lab_p)  (aggregate idle capacity).

        Drives dispatch VOLUME. Kept aggregate (not per-lab): a per-lab split
        still biases toward the faster lab and does not fix starvation; imbalance
        handles balancing. Dispatch raises N_lab_p ⇒ shortfall drops ⇒ Δφ < 0.
        """
        n, tau = ALP._parse(s)
        if s.ndim == 1:
            N_lab = n[1:].sum(axis=1)                       # (N_LABS,)
            return float(np.maximum(0., _MU * tau - N_lab).sum())
        N_lab = n[:, 1:, :].sum(axis=2)                     # (H, N_LABS)
        return np.maximum(0., _MU[None, :] * tau[:, None] - N_lab).sum(1)

    @staticmethod
    def phi_imbalance(s):
        """φ₅ = |N_1/μ_1 − N_2/μ_2|  (throughput-normalised lab backlog gap).

        Backlog measured in TIME units, so the target is equal CLEAR TIME across
        labs. The ONLY feature that rewards balancing the labs and so fixes
        lab-1 starvation. SIGN IS ACTION-DEPENDENT: dispatching the whole depot
        into the emptier lab lowers it (rewarded), into the fuller lab raises it
        (penalised). With θ≥0 this steers dispatch toward the under-loaded lab.
        """
        n, _ = ALP._parse(s)
        if s.ndim == 1:
            N = n[1:].sum(axis=1)                           # (N_LABS,)
            return float(abs(N[0] / _MU[0] - N[1] / _MU[1]))
        N = n[:, 1:, :].sum(axis=2)                         # (H, N_LABS)
        return np.abs(N[:, 0] / _MU[0] - N[:, 1] / _MU[1])

    @staticmethod
    def phi_exp_risk(s):
        """φ₆ = C_dep·n_{0,1} + C_lab·Σ_{p≥1} n_{p,1}   (age-1 expiry risk).

        The cost-weighted count of OLDEST-class (age-1) kits — the only kits one
        epoch from expiry — across the depot and both labs. Depot age-1 kits are
        weighted by C_dep (=20), lab age-1 kits by C_lab (=15), matching the two
        expiry costs they would incur at the next end-of-day sweep.

        WHY THIS FORM (replaces the survival-weighted all-age version):
          • Restricted to age-1 only. Older depot kits (age-2, age-3) are not at
            imminent expiry risk and are already represented by depot_age2 /
            depot_age3; including them here only duplicated depot inventory
            (the old all-age depot term was perfectly collinear with depot_total,
            corr 1.000, which drove its weight to ~0).
          • No survival discounting. With τ ~ Uniform[1, τ_max] the epochs-left
            k = round(τ/Δt) ≥ 8 in essentially every state, so the survival
            probability π ≈ 1 and the discounted lab term vanished (~0 in 100% of
            sampled states). The bare age-1 lab count is the operative signal.
          • Captures BOTH locations' age-1 exposure with the correct relative
            weight C_dep > C_lab, so the feature rises as kits age into class 1
            (no-dispatch lets age-2 → age-1) and falls as oldest-first processing
            clears lab age-1 kits. The depot part is dispatch-actionable (moving a
            depot age-1 kit to a lab gives it a processing chance); the lab part
            adds the independent lab-side expiry signal (corr ≈ 0.005 with
            depot_age1, so it is genuinely new information).
        """
        n, _ = ALP._parse(s)
        if s.ndim == 1:
            return float(C_EXP_DEPOT * n[0, 0] + C_EXP_LAB * n[1:, 0].sum())
        return C_EXP_DEPOT * n[:, 0, 0] + C_EXP_LAB * n[:, 1:, 0].sum(1)

    # ── feature matrix ────────────────────────────────────────────────────────
    def _phi_raw(self, s):
        """Raw (unnormalized) basis matrix -- exactly the canonical formula:
            theta^(0)
          + sum_a theta^(h_a) * h_{0,a} * n_{0,a,t}                    [depot_age, x L_AGE]
          + theta^(sigma) * sum_p max(0, mu_p*tau - sum_a n_{p,a,t})   [lab_shortfall, aggregate]
          + theta^(Delta) * |N_1/mu_1 - N_2/mu_2|                      [imbalance, N_LABS>=2 only]
          + theta^(r) * (C_dep*n_{0,1,t} + C_lab*sum_p n_{p,1,t})       [exp_risk]
        B = L_AGE + 4 for N_LABS>=2 (imbalance term defined), or L_AGE + 3
        for N_LABS==1 (imbalance is undefined with only one lab -- there's
        nothing to compare it against -- so it's dropped rather than
        erroring). Depot-age features are built dynamically from L_AGE so
        this also works for L_AGE != 3.
        """
        s  = np.asarray(s, float)
        sc = (s.ndim == 1)
        if sc:
            s = s[None, :]
        depot_age_cols = [self.phi_depot_age(s, a_idx)[:, None]
                          for a_idx in range(L_AGE)]
        # np.column_stack(...) here, not np.c_[...]: np.c_ uses NumPy's
        # special subscript (__getitem__) syntax, and star-unpacking a
        # variable-length list inside a subscript (np.c_[a, *b, c]) only
        # became valid Python syntax in 3.11 (PEP 646) -- a SyntaxError on
        # any earlier Python. np.column_stack is a normal function call
        # taking a plain list, built here with ordinary list concatenation
        # (+), which works on every Python 3 version.
        cols = [self.phi_const(s)[:, None]] + depot_age_cols + [
            self.phi_lab_shortfall(s)[:, None],
        ]
        if N_LABS >= 2:
            cols += [
                self.phi_imbalance(s)[:, None],
                self.phi_exp_risk(s)[:, None],
            ]
        else:
            cols += [self.phi_exp_risk(s)[:, None]]
        r = np.column_stack(cols)
        return r[0] if sc else r

    def phi(self, s):
        """Normalized basis matrix used everywhere in the ALP/PSMD pipeline.

        Scale-only normalization (see set_feature_scale / inputs):
            φ̃_b(s) = φ_b(s) / σ_b          for b ≥ 1   (σ_b = std_ν[φ_b])
            φ̃_0(s) = 1                                  (constant left as-is)

        This is a pure positive rescaling of each coordinate. It spans the SAME
        function space as the raw basis, so the ALP optimum V̂* and the greedy
        policy are EXACTLY invariant; only the θ coordinates change
        (θ̃_b = θ_b·σ_b). Its sole effect is to precondition the PSMD subgradient
        geometry so that no large-magnitude feature dominates the step or is
        suppressed in the objective E_ν[θ·φ]. σ_b are fixed constants estimated
        once under ν (initial-state distribution) by Monte Carlo with a fixed
        seed — reproducible preprocessing, reported in the paper.
        """
        r = self._phi_raw(s)
        return r / self._phi_scale          # broadcasts over (B,) or (H, B)

    # ── expectations ──────────────────────────────────────────────────────────
    def E_phi(self, mdp=None):
        # Save/restore the RNG state, matching calibrate_normalization's
        # own pattern above -- NOT np.random.seed(None), which reseeds
        # from OS entropy and silently defeats any run_seed passed to
        # run_psmd, since E_phi is called (via Dual/Gradient's
        # constructors and lp_warm_start) before the main training loop
        # even starts. This was the actual cause of "same seed, different
        # result" when testing run_seed reproducibility for this script.
        prev = np.random.get_state()
        np.random.seed(42)
        s = np.array([mdp.sample_initial_state() for _ in range(5000)])
        r = self.phi(s).mean(0)
        np.random.set_state(prev)
        return r

    def E_phi_next(self, mdp, s, a, exog):
        s    = np.atleast_2d(np.asarray(s, float))
        a    = np.atleast_1d(np.asarray(a))
        H, N = s.shape[0], len(exog)
        # Explicit H x N broadcast via the deterministic single-state
        # transition, rather than mdp.transition()'s public broadcasting
        # heuristic (which infers "broadcast" vs "1-to-1 aligned" from
        # len(exog) != H -- ambiguous, and wrong, whenever H happens to
        # equal N, e.g. two sample-size hyperparameters coinciding).
        sn = np.zeros((H, N, mdp.n_state))
        for h in range(H):
            for k in range(N):
                sn[h, k, :] = mdp._transition_single(s[h], a[h], exog[k])
        return self.phi(sn.reshape(H * N, mdp.n_state)).reshape(H, N, self.B).mean(1)

    def f_hat(self, mdp, theta, s, a, exog):
        g  = mdp.gamma
        s  = np.asarray(s, float)
        sc = (s.ndim == 1)
        s  = np.atleast_2d(s)
        a  = np.atleast_1d(np.asarray(a))
        c  = mdp.cost(s, a, exog)
        ps = self.phi(s)
        ep = self.E_phi_next(mdp, s, a, exog)
        eq = self.E_phi(mdp)
        v  = ((c + g * (ep * theta).sum(-1) - (ps * theta).sum(-1))
              / (1 - g) + (theta * eq).sum())
        return float(v[0]) if sc else v
