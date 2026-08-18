"""
inputs/inputs_evaluate_approximation.py
========================================
A SMALL instance of the dispatching MDP, sized so its state space can be
enumerated exhaustively and solved with the EXACT linear program (one
variable per state, Eq. (26) restricted to a finite S) rather than the
ALP/VFA restriction. Used only by evaluate_approximation.py, to check the
PSMD-trained VFA-greedy policy against the true optimal policy on a problem
small enough to solve exactly.

Mirrors inputs.py's variable names exactly, so the isolated
classes/mdp_evaluate_approximation.py and classes/alp_evaluate_approximation.py
(see those files' own docstrings) work against this module unchanged --
swapping
which config they see is handled by evaluate_approximation.py's import
machinery, not by any change to MDP/ALP themselves.

Sizing: N_LABS and L_AGE are kept at their production values (2, 3) because
classes/alp.py's phi_depot_age / phi_imbalance are written for exactly this
shape (3 explicit age-resolved depot features, a 2-lab imbalance gap).
Shrinking K_CAPACITY and epochs-per-day is what keeps the state space small:
    |S| = (K_CAPACITY+1)^{(N_LABS+1)*L_AGE} * epochs_per_day
    K_CAPACITY=1, epochs_per_day=3  =>  |S| = 2^9 * 3 = 1,536 states
    |S| x |U| = 1,536 x 3 = 4,608 exact-LP constraints -- solves in seconds.
Increase K_CAPACITY / epochs_per_day if you want a larger (still exact,
just slower) instance; the state count above should guide how far you can
push this before enumeration/LP-build time becomes unwieldy.
"""
import numpy as np

# ── Problem structure (kept at production values -- see note above) ──────
N_LABS = 2
LAB_IDS = list(range(1, N_LABS + 1))
L_AGE = 3

GAMMA = 0.95

# ── Small-instance sizing ─────────────────────────────────────────────────
K_CAPACITY = 1          # per-cell inventory cap: cells take values {0,...,K_CAPACITY}
epochs_per_day = 3      # tau grid points per day (small -> few states, few dynamics)

LAMBDA_AGE = np.array([0.3, 0.3, 0.3])   # arrival rate per age class
LAMBDA = float(LAMBDA_AGE.sum())
MU = np.array([0.6, 0.6])                # per-lab processing rate
LAMBDA_TOTAL = float(LAMBDA_AGE.sum() + MU.sum())
DELTA_T = 1.0 / LAMBDA_TOTAL
# TAU_MAX set as an EXACT multiple of DELTA_T so the tau grid is clean
# (tau_max, tau_max-Delta_t, ..., Delta_t) with no floating-point drift.
TAU_MAX = epochs_per_day * DELTA_T

C_DISPATCH = np.array([2.0, 2.0])
H_HOLD = np.array([
    [3.0, 2.0, 1.0],
    [1.5, 1.0, 0.5],
    [1.5, 1.0, 0.5],
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

# ── VFA / Theta (unchanged structure; B is still L_AGE+4=7) ──────────────
NO_BASIS_FN = 7
_B = NO_BASIS_FN
THETA_LB = [None] + [0.0] * (_B - 1)
THETA_UB = [None] + [50.0] * (_B - 1)
THETA_BREAK_EVEN_IDX = [1, 2, 3, 4, 5, 6]

# ── PSMD hyperparameters — much smaller budget; the instance itself is tiny ─
T = 800
ETA0 = 0.1
LAM0 = 0.0001
H_GRAD = 30
N_SAMPLES = 100
N_MH = 30
N_MH_TOTAL = 200
N_MH_KEEP = 100
N_INIT = 100
EVAL_EVERY = 100
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
