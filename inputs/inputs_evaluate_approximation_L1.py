"""
inputs/inputs_evaluate_approximation_L1.py
=============================================
A single-lab (N_LABS=1) instance, designed specifically to test whether
removing "which lab" as a decision produces meaningfully higher exact-vs-
VFA policy agreement than the 2-lab instances tried so far.

Motivation: check_exact_tie_structure.py found that 75% of ALL states in
the 2-lab instance have their best-vs-second-best action within 5% of
each other -- and with N_LABS=2 and IDENTICAL processing rates/costs for
both labs, "which lab do I send this to" is very often a genuine
near-tie by symmetry alone, independent of anything the VFA does right or
wrong. With only one lab, that entire decision axis -- and its associated
symmetric ties -- doesn't exist: the only choice is hold vs dispatch, a
single threshold decision driven by urgency (exp_risk) vs cost
(C_DISPATCH), which should have a much more decisively separated
optimum.

Requires one change elsewhere, already made:
  - classes/alp_evaluate_approximation.py's _phi_raw() now conditionally
    skips phi_imbalance, phi_imbalance_urgency_interaction, and the
    per-lab-shortfall features when N_LABS<2 (imbalance-between-labs is
    undefined with only one lab; per-lab shortfall would exactly
    duplicate the aggregate). Feature set here is just
    [const, depot_age1, depot_age2, lab_shortfall, exp_risk] -- B=5.

Sizing: K_CAPACITY=1, epochs_per_day=3 =>
    |S| = (K_CAPACITY+1)^{(N_LABS+1)*L_AGE} * epochs_per_day
        = 2^4 * 3 = 48 states
    |S| x |U| = 48 x 2 = 96 exact-LP constraints -- trivially fast, since
there are only 2 actions (hold, dispatch) instead of 3.
"""
import numpy as np

# ── Problem structure ──────────────────────────────────────────────────────
N_LABS = 1                # <-- the actual change: single lab, no lab-choice axis
LAB_IDS = list(range(1, N_LABS + 1))
L_AGE = 2

GAMMA = 0.95

# ── Small-instance sizing ─────────────────────────────────────────────────
K_CAPACITY = 2
epochs_per_day = 3

LAMBDA_AGE = np.array([0.3, 0.3])
LAMBDA = float(LAMBDA_AGE.sum())
MU = np.array([0.6])                     # single lab's processing rate
LAMBDA_TOTAL = float(LAMBDA_AGE.sum() + MU.sum())
DELTA_T = 1.0 / LAMBDA_TOTAL
TAU_MAX = epochs_per_day * DELTA_T

C_DISPATCH = np.array([2.0])             # single dispatch cost
H_HOLD = np.array([
    [3.0, 1.0],    # depot
    [1.5, 0.5],    # the one lab
])
C_EXP_DEPOT = 20.0
C_EXP_LAB = 15.0

N_MAX = K_CAPACITY
N_MIN = 0

# ── State / action space ──────────────────────────────────────────────────
N_INV = (N_LABS + 1) * L_AGE
N_STATE = N_INV + 1

STATE_BOUNDS = [(N_MIN, N_MAX)] * N_INV + [(0.0, TAU_MAX)]
ACTION_SET = list(range(0, N_LABS + 1))       # [0, 1]: hold or dispatch
ACTION_BOUNDS = [(0, N_LABS)]
ACTION_WEIGHTS = [0.5] + [0.5 / N_LABS] * N_LABS   # [0.5, 0.5]

SIM_DAYS_DEFAULT = 5

# ── VFA / Theta -- derived from L_AGE by formula, not hardcoded ──────────
# With N_LABS==1: const + L_AGE depot-age + lab_shortfall + exp_risk. No
# per-lab-shortfall, imbalance, or imbalance*exp_risk terms -- see
# classes/alp_evaluate_approximation.py's _phi_raw for why.
NO_BASIS_FN = L_AGE + 3
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
COMPUTE_SAA_LB = False   # skip -- not needed for this experiment, saves time
