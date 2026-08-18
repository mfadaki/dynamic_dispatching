"""
classes/mdp.py  — Dynamic Dispatching Problem
==============================================
Time-driven MDP using uniformisation. Infinite-horizon, stationary (Section
3.3): every day looks statistically identical to the last, so the Markov
state carries only what the transition kernel and the value function
actually need going forward.

State vector (flattened, shape (N_STATE,)):
    s[0 : N_INV] = n_{p,a}  inventory, row-major (location, age)
                   index = p * L_AGE + (a-1)
                   p=0 depot, p=1..N_LABS labs
                   a=1 (1 period left) .. L_AGE (freshest)
    s[N_INV]     = tau      remaining working time in the current day

No expiry counter and no day index are carried in the state:
  - Expiry: billed as an exact closed-form expectation inside cost(), taken
    over the (small, discrete) set of possible next-epoch events, on the one
    epoch per day whose transition triggers end-of-day. This needs no memory
    of a past, already-realised expiry count.
  - Day index: under a stationary, infinite-horizon MDP the optimal policy
    cannot depend on which day it is (no terminal condition, no day-varying
    parameters) — day is never read by cost() or by any basis function, so
    it is external simulation/reporting bookkeeping, not part of s.

Action (scalar integer):
    0      = no dispatch
    p=1..N_LABS = dispatch entire depot to lab p

Exogenous information per decision epoch (uniformisation):
    One event drawn from {arrival_a, processing_p, dummy} with probabilities
    proportional to their rates divided by Lambda_total.
    Represented as a single integer:
        0..L_AGE-1  = arrival of kit with age index a
        L_AGE..L_AGE+N_LABS-1 = processing completion at lab p-1
        L_AGE+N_LABS = dummy (no event, only clock ticks)

Engine interface
----------------
    gamma, state_bounds, action_bounds
    action_type = 'discrete', action_set = [0,1,...,N_LABS]
    sample_exog(n), sample_single_exog()
    cost(s, a, exog), cost_realised(s, a, g)
    transition(s, a, exog)
    sample_initial_state()
    clip_state(s)
"""

import numpy as np
from inputs.inputs import (
    GAMMA, N_LABS, LAB_IDS, L_AGE,
    TAU_MAX, DELTA_T, LAMBDA_AGE, LAMBDA, MU, LAMBDA_TOTAL,
    C_DISPATCH, H_HOLD, C_EXP_DEPOT, C_EXP_LAB, K_CAPACITY,
    N_MAX, N_MIN,
    N_INV, N_STATE, STATE_BOUNDS, ACTION_SET, ACTION_BOUNDS
)


class MDP:

    # ── engine interface attributes ───────────────────────────────────────
    gamma         = GAMMA
    state_bounds  = STATE_BOUNDS
    action_bounds = ACTION_BOUNDS
    action_type   = 'discrete'
    action_set    = ACTION_SET

    # convenience
    n_labs  = N_LABS
    l_age   = L_AGE
    n_inv   = N_INV
    n_state = N_STATE

    # ── Cost normalization (scale-only) ───────────────────────────────────
    # c̃(s,a) = c_raw(s,a) / cost_scale,  cost_scale = E_{ν,a}[c_raw].
    # A single positive scalar; scaling the stage cost scales the whole value
    # function by the same factor (Bellman operator is positively homogeneous),
    # so the greedy policy is EXACTLY invariant. Defaults to 1 (no-op) until
    # calibrate_cost_scale() sets it. Reported costs multiply back by cost_scale.
    cost_scale = 1.0

    def calibrate_cost_scale(self, n_samples=20000, seed=12345):
        """Estimate cost_scale = E_{ν,a}[c_raw(s,a)] once, under ν and a uniform
        action draw. Deterministic given the seed ⇒ reproducible preprocessing.
        Returns the scalar scale."""
        prev = np.random.get_state()
        np.random.seed(seed)
        acc = 0.0
        for _ in range(n_samples):
            s = self.sample_initial_state()
            a = np.random.randint(0, N_LABS + 1)
            acc += self._cost_single_raw(s, a)
        np.random.set_state(prev)
        scale = acc / n_samples
        self.cost_scale = float(scale) if scale > 1e-8 else 1.0
        return self.cost_scale

    # ── state indexing helpers ────────────────────────────────────────────
    @staticmethod
    def inv_idx(p, a):
        """
        Index into the inventory part of the state vector.
        p : location (0=depot, 1..N_LABS=lab)
        a : age index (0-based, 0=age 1 = expires soonest)
        """
        return p * L_AGE + a

    TAU_IDX = N_INV

    # ── state parsing ─────────────────────────────────────────────────────
    def parse_state(self, s):
        """
        Parse flat state vector into structured components.

        Returns
        -------
        n   : (N_LABS+1, L_AGE) inventory matrix
        tau : float remaining working time
        """
        n   = s[:N_INV].reshape(N_LABS + 1, L_AGE)
        tau = float(s[self.TAU_IDX])
        return n, tau

    def build_state(self, n, tau):
        """Assemble flat state vector from components."""
        s = np.zeros(N_STATE)
        s[:N_INV]       = n.ravel()
        s[self.TAU_IDX] = tau
        return s

    # ── exogenous information ─────────────────────────────────────────────
    # Event types (integer codes):
    #   0..L_AGE-1          : arrival of kit with that age index
    #   L_AGE..L_AGE+N_LABS-1: processing completion at lab (index - L_AGE)
    #   L_AGE + N_LABS      : dummy event (no action, clock ticks)

    N_EVENT_TYPES = L_AGE + N_LABS + 1

    # Event probabilities under uniformisation
    _probs = np.concatenate([
        LAMBDA_AGE / LAMBDA_TOTAL,
        MU         / LAMBDA_TOTAL,
        [1.0 - (LAMBDA + MU.sum()) / LAMBDA_TOTAL]
    ])

    def sample_exog(self, n):
        """
        Draw n i.i.d. event codes.
        Returns (n,) integer array.
        """
        return np.random.choice(self.N_EVENT_TYPES, size=n, p=self._probs)

    def sample_single_exog(self):
        """Draw one event code (scalar int)."""
        return int(np.random.choice(self.N_EVENT_TYPES, p=self._probs))

    # ── transition ────────────────────────────────────────────────────────
    def transition(self, s, a, exog):
        """
        Next state after action a and event exog.

        s    : (N_STATE,) or (H, N_STATE)
        a    : scalar int or (H,) int array
        exog : scalar int or (N,) int array

        Returns
        -------
        If s is (N_STATE,) and exog is (N,): returns (N, N_STATE)
        If s is (H, N_STATE) and exog is (N,): returns (H, N, N_STATE)
        If s is (N_STATE,) and exog is scalar: returns (N_STATE,)
        """
        s    = np.asarray(s, dtype=float)
        a    = np.asarray(a)
        exog = np.asarray(exog)

        scalar_s    = (s.ndim == 1)
        scalar_exog = (exog.ndim == 0)

        if scalar_s:
            s    = s[None, :]    # (1, N_STATE)
            a    = np.atleast_1d(a)
        if scalar_exog:
            exog = exog[None]    # (1,)

        H = s.shape[0]
        N = exog.shape[0]

        # Output: (H, N, N_STATE)
        s_next = np.zeros((H, N, N_STATE))

        for h in range(H):
            for j in range(N):
                s_next[h, j] = self._transition_single(
                    s[h], int(a[h]), int(exog[j]))

        # Clip to valid bounds
        s_next = self._clip_state_batch(s_next)

        if scalar_s and scalar_exog:
            return s_next[0, 0]      # (N_STATE,)
        if scalar_s and N == 1:
            return s_next[0, 0]      # (N_STATE,) when exog was 1-element
        if scalar_s:
            return s_next[0]         # (N, N_STATE)
        return s_next                # (H, N, N_STATE)

    @staticmethod
    def apply_action_and_event(n, a, event):
        """
        Deterministically apply dispatch action a and one exogenous event to
        inventory n. Returns a NEW (N_LABS+1, L_AGE) array — n itself is not
        mutated. Shared by _transition_single (which then ages/resets at EOD)
        and _expected_expiry_cost (which enumerates every possible event
        without ever touching the state), so the two stay mechanically
        consistent by construction.
        """
        n = n.copy()

        # ── 1. Apply action: dispatch entire depot inventory ──────────────
        if a > 0:
            p_lab = int(a) - 1          # 0-based lab index
            n[p_lab + 1] += n[0]        # transfer all depot kits to lab
            n[0, :]       = 0.0         # empty depot

        # ── 2. Apply exogenous event ──────────────────────────────────────
        if event < L_AGE:
            # Arrival: kit with age_idx = event arrives at depot
            age_idx = int(event)
            n[0, age_idx] = min(n[0, age_idx] + 1, N_MAX)

        elif event < L_AGE + N_LABS:
            # Processing completion at lab p_lab
            p_lab = event - L_AGE       # 0-based
            # Remove one kit, OLDEST first. Age index 0 = age-1 = closest to
            # expiry, so process it first (FIFO by remaining shelf life). This
            # minimises lab expiry: a kit one epoch from expiring is cleared
            # ahead of fresher kits that can still wait.
            for age_idx in range(L_AGE):
                if n[p_lab + 1, age_idx] > 0:
                    n[p_lab + 1, age_idx] -= 1
                    break
        # else: dummy event — nothing changes

        return n

    def _transition_single(self, s, a, event):
        """
        Transition for one (s, a, event) triple.
        s     : (N_STATE,)
        a     : int action
        event : int event code
        Returns (N_STATE,) next state.
        """
        n, tau = self.parse_state(s)
        n = self.apply_action_and_event(n.astype(float), a, event)

        # ── Clock update ───────────────────────────────────────────────────
        tau_new = tau - DELTA_T

        if tau_new <= 0:
            # End of day: age all inventories. (Expiry itself is not recorded
            # here — it is billed, in expectation, by cost() on this same
            # epoch; see _expected_expiry_cost.)
            n = np.roll(n, -1, axis=1)   # n[p,a] <- n[p,a+1]; n[p,L-1] <- 0
            n[:, -1] = 0.0
            tau_new = TAU_MAX

        return self.build_state(n, tau_new)

    # ── immediate cost ────────────────────────────────────────────────────
    def cost(self, s, a, exog):
        """
        Expected immediate cost E[c(s, a, event)] averaged over exog samples.

        Cost = dispatch_cost(a)
             + (1/Lambda) * sum_{p,a} h_{p,a} * n_{p,a}
             + 1{tau - Delta_t <= 0} * E_event[ c_exp_0 * n^+_0,1 + c_exp * sum_p n^+_p,1 ]
        where n^+ is the inventory after applying action a and one exogenous
        event to the CURRENT n (see apply_action_and_event), and the
        expectation is an exact closed-form sum over the small, discrete set
        of possible events — no state memory of a past expiry count is
        needed, and no Monte Carlo sampling is needed either (the sum is over
        at most L_AGE+N_LABS+1 outcomes with known probabilities). The
        indicator restricts the charge to the one epoch per day whose
        transition triggers end-of-day, so a day's expirations are billed
        exactly once, not once per epoch.

        s    : (N_STATE,) or (H, N_STATE)
        a    : scalar or (H,)
        exog : (N,) — not used (the expiry expectation is computed exactly
               inside cost() itself; no sampling is needed given (s,a))

        Returns scalar or (H,).
        """
        s = np.asarray(s, dtype=float)
        a = np.asarray(a)
        scalar = (s.ndim == 1)

        if scalar:
            return self._cost_single(s, int(a))
        else:
            return np.array([self._cost_single(s[h], int(a[h]))
                             for h in range(s.shape[0])])

    def _cost_single(self, s, a):
        """Normalized cost for one (s, a) pair:  c̃ = c_raw / cost_scale.

        Pure positive scaling by a single constant cost_scale = E_{ν,a}[c_raw].
        Scaling the stage cost by 1/cost_scale scales the whole value function by
        the same factor (the Bellman operator is positively homogeneous), so the
        greedy argmin — and hence the optimal policy — is EXACTLY invariant. The
        scale only conditions the magnitude PSMD works with, putting cost on the
        same O(1) footing as the normalized features. cost_scale defaults to 1
        (no-op) until calibrate_cost_scale() sets it.
        """
        return self._cost_single_raw(s, a) / self.cost_scale

    def _cost_single_raw(self, s, a):
        """Raw (unnormalized) cost for one (s, a) pair.

        Expiry cost is charged exactly once per day, on the epoch whose
        transition triggers end-of-day (tau - DELTA_T <= 0), as an exact
        closed-form expectation over the epoch's possible events — see
        _expected_expiry_cost. No eps state variable is needed: whether this
        epoch is EOD-triggering is deterministic given tau alone, so the cost
        remains a pure, closed-form function of (s, a), as required by the
        ALP/PSMD saddle-point machinery.
        """
        n, tau = self.parse_state(s)

        # Pure dispatch cost — lab selection is driven by lab_shortfall in V(s)
        c = C_DISPATCH[a - 1] if a > 0 else 0.0

        # Holding cost (scaled by 1/Lambda as in value function)
        holding = 0.0
        for p in range(N_LABS + 1):
            for age_idx in range(L_AGE):
                holding += H_HOLD[p, age_idx] * n[p, age_idx]
        c += holding / LAMBDA_TOTAL

        # Expected expiry cost — charged once, only on an EOD-triggering epoch.
        if tau - DELTA_T <= 0:
            c += self._expected_expiry_cost(n, a)

        return c

    def _expected_expiry_cost(self, n, a):
        """
        E[expiry cost | s, a] on an EOD-triggering epoch, computed as an
        EXACT finite sum over the N_EVENT_TYPES possible events (arrivals,
        processing completions, dummy), weighted by their known
        uniformisation probabilities — no Monte Carlo, no state memory.

        For each candidate event, apply_action_and_event gives the resulting
        age-1 counts at each location (the kits that would expire if that
        event is the one realised this epoch); these are combined with the
        event probabilities to give the exact expectation.
        """
        expected = 0.0
        for event in range(self.N_EVENT_TYPES):
            n_e = self.apply_action_and_event(n, a, event)
            exp_cost = C_EXP_DEPOT * n_e[0, 0] + C_EXP_LAB * n_e[1:, 0].sum()
            expected += self._probs[event] * exp_cost
        return expected

    def cost_realised(self, s, a, g):
        """Realised cost at single event g. Used in trajectory simulation."""
        return self._cost_single(np.asarray(s, dtype=float), int(a))

    # ── helpers ───────────────────────────────────────────────────────────
    def sample_initial_state(self):
        """Random feasible initial state (near-empty trajectory start).

        Intentionally narrow — represents a fresh depot/lab at the start of
        a rollout, not a general-purpose state sampler. Use
        sample_representative_state() for anywhere that needs to cover the
        FULL reachable state space (ALP objective/constraints, PSMD
        sampling, normalization calibration).
        """
        n   = np.random.randint(0, 4, size=(N_LABS + 1, L_AGE)).astype(float)
        tau = float(np.random.uniform(1.0, TAU_MAX))
        return self.build_state(n, tau)

    def sample_representative_state(self):
        """Broad state covering the FULL feasible inventory range
        [0, N_MAX] independently per cell, not just near-empty conditions.

        NOTE: do not use this for the ALP's constraint sample either — see
        sample_reachable_states() below. Sampling each of the N_INV
        inventory cells independently and uniformly is a poor proxy for
        "states the system can actually be in": with N_INV=9 dimensions,
        the near-empty region that matters most for nu occupies a tiny
        fraction of the full box (4**9/11**9 ~ 0.01% here), so a modest
        constraint sample essentially never lands near it, while wasting
        coverage on combinations (e.g. every cell simultaneously near
        N_MAX) that the real arrival/processing/aging dynamics would
        almost never jointly produce. Kept only as a utility for ad hoc
        broad testing; not used anywhere in the ALP/PSMD pipeline.
        """
        n   = np.random.randint(0, N_MAX + 1, size=(N_LABS + 1, L_AGE)).astype(float)
        tau = float(np.random.uniform(1.0, TAU_MAX))
        return self.build_state(n, tau)

    def sample_reachable_states(self, n_states, rollout_len=150):
        """Sample states via short random-action rollouts from near-empty
        starts — the ALP's CONSTRAINT sample (bounds.py's
        LowerBound.compute(), psmd.py's warm-start sample).

        This captures dynamically-consistent combinations of inventory
        levels — exactly the joint combinations arrivals, processing, and
        aging can actually produce, spanning from near-empty up through
        higher-inventory states reachable after arrival bursts — without
        the curse-of-dimensionality failure of sampling each of the N_INV
        cells independently and uniformly over [0, N_MAX] (see
        sample_representative_state()): that approach needs astronomically
        more than a few hundred samples before any of them land near the
        practically-important near-empty region, in N_INV=9+ dimensions.
        """
        states = []
        while len(states) < n_states:
            s = self.sample_initial_state()
            for _ in range(rollout_len):
                states.append(s.copy())
                if len(states) >= n_states:
                    break
                a = np.random.choice(self.action_set)
                g = self.sample_single_exog()
                s = np.asarray(self.transition(s, a, g), dtype=float).ravel()
        idx = np.random.choice(len(states), n_states, replace=False)
        return np.array(states)[idx]

    def _clip_state_batch(self, s):
        """Clip state array (any shape [..., N_STATE]) to valid bounds."""
        for dim, (lo, hi) in enumerate(STATE_BOUNDS):
            s[..., dim] = np.clip(s[..., dim], lo, hi)
        return s

    def clip_state(self, s):
        return self._clip_state_batch(np.asarray(s, dtype=float).copy())

    def clip_action(self, a):
        return np.clip(np.asarray(a), 0, N_LABS)