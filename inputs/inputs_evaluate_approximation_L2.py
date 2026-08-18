"""
inputs/inputs_evaluate_approximation_L2.py
=============================================
Same idea as inputs_evaluate_approximation.py, but with L_AGE=2 instead of
the production value of 3 -- a genuinely smaller problem, not just a
smaller K_CAPACITY/epochs_per_day. This requires two things elsewhere to
generalize correctly (both already done, not something you need to redo):
  - classes/alp_evaluate_approximation.py's _phi_raw() builds its
    depot-age features by looping over range(L_AGE), not by hardcoding 3
    explicit calls the way production classes/alp.py does.
  - functions/psmd_evaluate_approximation.py's break-even floor
    representative state is built as [1.0]*(L_AGE-1) + [2.0], not the
    hardcoded 3-element [1.0, 1.0, 2.0] the production functions/psmd.py
    uses (which would silently write into the wrong inventory slot for
    any L_AGE != 3).

NO_BASIS_FN and THETA_BREAK_EVEN_IDX are derived from L_AGE by formula
here (L_AGE+4, and every non-const index) rather than hardcoded, so this
file is the actual source of truth for "how many age classes" -- change
L_AGE and everything else in this file follows automatically.

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
K_CAPACITY = 1
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
NO_BASIS_FN = L_AGE + 4     # const + L_AGE depot-age features + 3 aggregate features
_B = NO_BASIS_FN
THETA_LB = [None] + [0.0] * (_B - 1)
THETA_UB = [None] + [50.0] * (_B - 1)
THETA_BREAK_EVEN_IDX = list(range(1, NO_BASIS_FN))   # every non-const feature

# ── PSMD hyperparameters ───────────────────────────────────────────────────
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
