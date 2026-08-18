"""
inputs/inputs_evaluate_approximation_L2_canonical.py
=======================================================
2-lab instance running the CANONICAL VFA, exactly matching the formula:

    V_hat(S_t) = theta^(0)
      + sum_{a=1}^{L} theta^(h_a) * h_{0,a} * n_{0,a,t}
      + theta^(sigma) * sum_{p=1}^{P} max(0, mu_p*tau_t - sum_a n_{p,a,t})
      + theta^(Delta) * |N_1/mu_1 - N_2/mu_2|
      + theta^(r) * (C_dep*n_{0,1,t} + C_lab*sum_p n_{p,1,t})

B = L_AGE + 4. This replaces an earlier config (previously named
..._richfeat_fastcmp) that had drifted from this formula by adding two
per-lab shortfall features and an imbalance*exp_risk interaction term
(B = L_AGE + N_LABS + 5) while chasing agreement-rate experiments --
those features are NOT part of the specified model and have been removed
from classes/alp_evaluate_approximation.py entirely, not just excluded
here. Results from the earlier config should be treated as an
experimental variant, not a measurement of this model.

Same L_AGE=2 sizing rationale as before -- a genuinely smaller problem,
not just a smaller K_CAPACITY/epochs_per_day. Requires two things
elsewhere to generalize correctly (both already in place):
  - classes/alp_evaluate_approximation.py's _phi_raw() builds its
    depot-age features by looping over range(L_AGE).
  - functions/psmd_evaluate_approximation.py's break-even floor
    representative state is built as [1.0]*(L_AGE-1) + [2.0].

Sizing: K_CAPACITY=1, epochs_per_day=3 (matching the default L_AGE=3
instance's epochs_per_day) =>
    |S| = (K_CAPACITY+1)^{(N_LABS+1)*L_AGE} * epochs_per_day
        = 2^6 * 3 = 192 states
    |S| x |U| = 192 x 3 = 576 exact-LP constraints -- solves in well under
a minute, since the state count is 8x smaller than the L_AGE=3 default's
1,536 states purely from dropping one age class.
"""
import numpy as np

# ── Problem structure ──────────────────────────────────────────────────────
N_LABS = 2
LAB_IDS = list(range(1, N_LABS + 1))
L_AGE = 2                # <-- the actual change: 2 age classes instead of 3

GAMMA = 0.95

# ── Small-instance sizing ─────────────────────────────────────────────────
K_CAPACITY = 2
epochs_per_day = 3

LAMBDA_AGE = np.array([0.3, 0.3])        # one arrival rate per age class (L_AGE=2)
LAMBDA = float(LAMBDA_AGE.sum())
MU = np.array([0.6, 0.6])
LAMBDA_TOTAL = float(LAMBDA_AGE.sum() + MU.sum())
DELTA_T = 1.0 / LAMBDA_TOTAL
TAU_MAX = epochs_per_day * DELTA_T

C_DISPATCH = np.array([2.0, 2.0])
# H_HOLD collapsed to 2 age columns: oldest (was age-1) and freshest (was
# age-3), dropping the middle age class's holding rate -- keeps the same
# depot > lab, oldest-costs-more pattern with one fewer column.
H_HOLD = np.array([
    [3.0, 1.0],
    [1.5, 0.5],
    [1.5, 0.5],
])
C_EXP_DEPOT = 20.0
C_EXP_LAB = 15.0

N_MAX = K_CAPACITY
N_MIN = 0

# ── State / action space ──────────────────────────────────────────────────
N_INV = (N_LABS + 1) * L_AGE
N_STATE = N_INV + 1

STATE_BOUNDS = [(N_MIN, N_MAX)] * N_INV + [(0.0, TAU_MAX)]
ACTION_SET = list(range(0, N_LABS + 1))
ACTION_BOUNDS = [(0, N_LABS)]
ACTION_WEIGHTS = [0.5] + [0.5 / N_LABS] * N_LABS

SIM_DAYS_DEFAULT = 5

# ── VFA / Theta -- derived from L_AGE by formula, not hardcoded ──────────
# Canonical model, exactly the formula: const + L_AGE depot-age +
# aggregate lab_shortfall + imbalance + exp_risk = L_AGE + 4. Every
# non-const coefficient is theta>=0 with an LP-derived (break-even) floor,
# per the formula's own annotations -- no exceptions needed here, unlike
# the earlier (non-canonical) experimental variant that added a
# sign-unconstrained interaction term.
NO_BASIS_FN = L_AGE + 4
_B = NO_BASIS_FN
THETA_LB = [None] + [0.0] * (_B - 1)
THETA_UB = [None] + [50.0] * (_B - 1)
THETA_BREAK_EVEN_IDX = list(range(1, NO_BASIS_FN))   # every non-const feature

# ── PSMD hyperparameters ───────────────────────────────────────────────────
T = 15000
ETA0 = 0.1
LAM0 = 0.0001
H_GRAD = 30
N_SAMPLES = 100
N_MH = 30
N_MH_TOTAL = 200
N_MH_KEEP = 100
N_INIT = 100
EVAL_EVERY = 1500
LP_N_INIT = 150
LP_N_EXOG = 140
MH_PROPOSAL_STD = 0.5
NORMALIZE = True
NORM_N_SAMPLES = 5000
NORM_SEED = 12345
H_BOUND = 100
N_BOUND = 200
UB_NUM_STAGES = 100
UB_NUM_TRAJ = 20
UB_N_EXOG_POL = 100
SAA_H_PRIME = 50
SAA_N_PRIME = 150
COMPUTE_SAA_LB = False
