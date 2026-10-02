"""
inputs/inputs_exact.py
========================
Instance for testing structural conjectures about the EXACT (not
approximated) optimal value function against ground truth -- queue-
balancing dominance, submodularity across age pairs, etc. -- via
solve_exact_cplex.py.

Deliberately symmetric labs (identical MU, identical H_HOLD rows for both
labs): "identical processing rates and holding costs" is the condition
requested, and several of the structural conjectures under discussion are
only well-posed / only expected to hold under this symmetry (see the
queue-balancing and submodularity write-ups earlier in this project --
without mu_1=mu_2, a faster lab can rationally receive more inventory,
which isn't what those conjectures are about).

Sizing rationale -- WHY L_AGE=3 and K_CAPACITY=2:
  - L_AGE=3 gives THREE distinct age-class pairs to test submodularity on
    ((1,2), (1,3), (2,3)) instead of just one -- a broader empirical check
    before committing to a proof attempt.
  - K_CAPACITY=2 allows imbalance magnitudes beyond a bare 0-vs-1 split
    (e.g. 2-vs-0, 2-vs-1) for the queue-balancing conjecture -- K=1 would
    only ever let one lab have "more" of a given age by exactly one kit,
    a narrow test of a conjecture stated as a general inequality.

Sizing: K_CAPACITY=2, L_AGE=3, N_LABS=2, tau_max targeted at the real 8-hour
day, epochs_per_day DERIVED as round(8.0/Delta_t) rather than hardcoded =>
    Delta_t = 1/Lambda_total = 1/2.1 = 0.4762 hours
    epochs_per_day = round(8.0/0.4762) = round(16.8) = 17
    tau_max = 17*0.4762 = 8.095 hours (nearest exact multiple of Delta_t
              to the targeted 8.0 -- see the epochs_per_day derivation
              below for why this can't land on exactly 8.0)
    |S| = (2+1)^{(2+1)*3} * 17 = 3^9 * 17 = 334,611 states
    |S| x |U| = 334,611 x 3 = 1,003,833 exact-LP constraints
Verified directly: enumerates in well under a second despite the size.

CPLEX LICENSING -- READ BEFORE RUNNING:
This instance's constraint count (177,147) is far beyond the free CPLEX
Community Edition's 1000-constraint limit (confirmed directly against
this project's own CPLEX installation: fails at 1001, works at 1000). You
need a full academic/commercial CPLEX license properly configured to
solve this instance via CPLEX. solve_exact_cplex.py will detect the
Community Edition error specifically and fall back to scipy's HiGHS
solver with a loud, explicit warning if that happens -- correct, but not
CPLEX, and the saved output records which solver actually ran.
"""
import numpy as np

# 2 Ages
L_AGE = 2

H_HOLD = np.array([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],])
C_DISPATCH = np.array([10.0, 10.0])
C_EXP_DEPOT = 100.0
C_EXP_LAB = 100.0
LAMBDA_AGE = np.array([0.1, 0.1]) #Arrival Rates
MU = np.array([0.2, 0.2]) #Processing rates
K_CAPACITY = 5


# ── Problem structure ──────────────────────────────────────────────────────
N_LABS = 2
LAB_IDS = list(range(1, N_LABS + 1))
#L_AGE = 3

GAMMA = 0.95

# ── Instance sizing ────────────────────────────────────────────────────────
#K_CAPACITY = 2

# tau_max is the REAL day length (8 hours, matching the case-study
# convention) -- FIXED. Delta_t = 1/Lambda is fixed by the arrival/
# processing rates below, not chosen. epochs_per_day is DERIVED from
# these two by division, not hardcoded to an arbitrary small number for
# state-space tractability (that was the pattern used in this project's
# earlier small VALIDATION instances -- appropriate there, not here,
# since it silently shrinks the real day down to whatever kept the state
# space small, rather than reflecting the actual rates).
#
# 8.0 / Delta_t is generally NOT an integer -- uniformization requires a
# CONSTANT step size, so epochs_per_day must be a whole number. Rounding
# to the nearest integer and then setting tau_max = epochs_per_day *
# Delta_t (rather than forcing tau_max to exactly 8.0 and leaving a
# leftover partial step) keeps every epoch's step size identical, at the
# cost of tau_max landing near, not exactly at, 8.0. Verified directly:
# 8.0/Delta_t = 16.8 here, rounds to 17, giving tau_max=8.095 hours.
_TARGET_TAU_MAX = 8.0

#LAMBDA_AGE = np.array([0.3, 0.3, 0.3])
LAMBDA = float(LAMBDA_AGE.sum())
#MU = np.array([0.6, 0.6])                # IDENTICAL processing rates (symmetric labs)
LAMBDA_TOTAL = float(LAMBDA_AGE.sum() + MU.sum())
DELTA_T = 1.0 / LAMBDA_TOTAL
epochs_per_day = round(_TARGET_TAU_MAX / DELTA_T)   # DERIVED, not hardcoded
TAU_MAX = epochs_per_day * DELTA_T                  # exact multiple of DELTA_T by construction

#C_DISPATCH = np.array([2.0, 2.0])


# Lab holding cost now UNIFORM across ages (1.0 regardless of age) at both
# labs, per explicit instruction -- previously age-decreasing ([1.5,1.0,0.5],
# older cheaper to hold). Depot's row is UNCHANGED (still 3.0/2.0/1.0,
# strictly exceeding the labs at every age, still oldest-costs-more) --
# only the LAB rows were asked to become age-uniform, not the depot.
# Labs remain identical to each other (both [1.0,1.0,1.0]), so the
# symmetric-labs premise this whole config is built around still holds.

#H_HOLD = np.array([
#    [3.0, 2.0, 1.0],
#    [1.0, 1.0, 1.0],
#    [1.0, 1.0, 1.0],
#])

#H_HOLD = np.array([
#    [1.0, 1.0, 1.0],
#    [1.0, 1.0, 1.0],
#    [1.0, 1.0, 1.0],
#])

#C_EXP_DEPOT = 20.0
#C_EXP_LAB = 15.0

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

# NO_BASIS_FN etc. are UNUSED by solve_exact_cplex.py -- it never touches
# ALP/theta, only classes/mdp_evaluate_approximation.py. Kept only in case
# this same config module is later reused for actual PSMD training (in
# which case NO_BASIS_FN etc. would need to be checked against whatever
# alp_evaluate_approximation.py currently implements at that time -- not
# relevant to the exact-solve task this file was written for).
NO_BASIS_FN = L_AGE + 4
_B = NO_BASIS_FN
THETA_LB = [None] + [0.0] * (_B - 1)
THETA_UB = [None] + [50.0] * (_B - 1)
THETA_BREAK_EVEN_IDX = list(range(1, NO_BASIS_FN))
