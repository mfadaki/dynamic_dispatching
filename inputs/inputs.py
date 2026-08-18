"""
inputs/inputs.py — Dynamic Dispatching Problem
"""
import numpy as np

# ── Problem parameters ────────────────────────────────────────────────────────
GAMMA       = 0.95      # UNCHANGED -- not provided in the calibrated data; confirm before trusting this
N_LABS      = 2
LAB_IDS     = list(range(1, N_LABS + 1))
L_AGE       = 3
TAU_MAX     = 8.0       # UNCHANGED -- not provided in the calibrated data; confirm before trusting this
# Arrival pattern is now heavily concentrated at age-3 (fresh): lambda_1/lambda_2
# are together under 1% of arrivals, lambda_3 is ~99%. Different in shape from
# both the original placeholder (monotonically increasing with freshness) and
# the previous calibration (hump at age-2) -- essentially all volume now
# enters freshly, then ages down through the epoch-based aging process.
LAMBDA_AGE  = np.array([0.0012, 0.0041, 0.8293])
LAMBDA      = float(LAMBDA_AGE.sum())
# Lab speed ordering FLIPS relative to the previous calibration: mu_1 > mu_2
# now, so Lab 1 is the faster lab (previously Lab 2 was). Any manuscript text
# built on "mu_1 < mu_2, lab 2 is faster" needs both the inequality direction
# and the lab identity swapped, not just the numbers updated.
MU          = np.array([0.4535, 0.3577])
LAMBDA_TOTAL= float(LAMBDA + MU.sum())
DELTA_T     = 1.0 / LAMBDA_TOTAL
# Traffic intensity rho = Sum(lambda)/Sum(mu) = 0.8346/0.8112 = 1.029 here --
# still (barely) above the rho=1 stability threshold, i.e. still structurally
# overloaded in queueing terms, but far closer to critically loaded than the
# previous calibration's rho=1.892. Verified independently before this edit,
# not just asserted.

# ── Cost structure ────────────────────────────────────────────────────────────
# Dispatch (order) cost: identical across labs. UNCHANGED from the previous
# calibration -- no new dispatch-cost data was provided with this update.
#   Lab selection is now driven by mu=[0.4535, 0.3577] (Lab 1 now faster),
#   not by a cost spread.
C_DISPATCH  = np.array([1.0, 1.0])

# Holding cost h_{p,a}: depot STRICTLY exceeds both labs for EVERY age class,
# and the two labs are IDENTICAL. UNCHANGED -- no new holding-cost data was
# provided with this update.
#   depot age-1 3.0 > lab 1.5   depot age-2 2.0 > lab 1.0   depot age-3 1.0 > lab 0.5
# Economic consequence: a kit is cheaper to hold once at a lab, so dispatch
# reduces the holding stream for every age class (reinforced by C_EXP below).
H_HOLD      = np.array([[3.0, 2.0, 1.0],   # depot:  strictly highest, all ages
                         [1.5, 1.0, 0.5],   # lab 1
                         [1.5, 1.0, 0.5]])  # lab 2  (identical to lab 1)
# Strict inequality C_EXP_DEPOT > C_EXP_LAB now RESTORED with this update
# (2.0 > 1.5) -- the previous calibration had these exactly equal, which
# had broken the manuscript's "a kit expiring at the depot was never
# utilised at all" claim. That claim is valid again with these values.
C_EXP_DEPOT = 2.0
C_EXP_LAB   = 1.5
K_CAPACITY  = 7

N_MAX = K_CAPACITY; N_MIN = 0

N_INV   = (N_LABS + 1) * L_AGE
N_STATE = N_INV + 1   # inventory + tau.
# Infinite-horizon, stationary MDP (Section 3.3): the expiry counter and the
# day index carry no information the value function or the transition kernel
# needs (neither ever appears in cost() or in any basis function beyond tau
# itself), so both are left out of the formal state. Expiry is billed as an
# exact closed-form expectation inside cost() instead of being memorised
# across epochs; the day count, where still useful (e.g. the Excel report),
# is tracked as external simulation bookkeeping, not as part of s.

STATE_BOUNDS   = [(N_MIN, N_MAX)] * N_INV + [(0.0, TAU_MAX)]
ACTION_SET     = list(range(0, N_LABS + 1))
ACTION_BOUNDS  = [(0, N_LABS)]
ACTION_WEIGHTS = [0.5] + [0.5 / N_LABS] * N_LABS

# Reporting-only default: number of days to simulate for the Excel policy
# report (functions/export_policy_excel.py). NOT a planning horizon and NOT a
# state bound — the MDP itself is infinite-horizon and stationary.
SIM_DAYS_DEFAULT = 5

# ── Basis functions (B = 7) ───────────────────────────────────────────────────
# Index map (must match phi() in alp.py AND labels in export_policy_excel.py):
#   0: const
#   1: depot_age1   2: depot_age2   3: depot_age3   (age-split depot holding, B1)
#   4: lab_shortfall                                (aggregate idle capacity)
#   5: imbalance                                    (throughput-normalised, A2)
#   6: exp_risk                                     (age-1 expiry risk)
#
# ── Design rationale ──────────────────────────────────────────────────────────
# The Bellman recursion embeds every g and c inside V_{t+1}; features only span
# what TODAY'S STATE forecasts about the FUTURE cost stream beyond current g+c.
#
#   1–3  depot_age{1,2,3}   h_{0,a}·n_{0,a}   for a = 1,2,3
#        hold_depot split by age so the VFA can weight URGENCY (age-1, about to
#        expire, dispatch-saveable) independently of the fixed holding-cost ratio
#        h_{0,·}=[3,2,1]. The three age counts are ~uncorrelated (|corr|<0.02), so
#        each carries independent signal. Dispatch empties the depot ⇒ each
#        Δφ < 0. (A single Σ_a (L−a+1) n_{0,a} "urgency" feature was REJECTED: its
#        weights (3,2,1) equal h_{0,·} ⇒ corr 1.000 with hold_depot; even the 1/a
#        form is 0.993 correlated.)
#
#   4    lab_shortfall   Σ_{p≥1} max(0, μ_p·τ − N_p)        [N_p = Σ_a n_{p,a}]
#        Aggregate idle processing capacity — drives dispatch VOLUME. Kept
#        AGGREGATE, not per-lab: a per-lab split steers correctly only when one
#        lab is already saturated (clipping), but TIES when both labs have room,
#        and the tie breaks toward the faster lab (μ₂>μ₁) — so the split does NOT
#        fix lab-1 starvation in the common case. Balancing is delegated to the
#        imbalance feature instead. Dispatch raises N_p ⇒ Δφ < 0.
#
#   5    imbalance   |N_1/μ_1 − N_2/μ_2|
#        Throughput-normalised backlog gap (backlog in TIME units ⇒ target is
#        equal CLEAR TIME, not equal counts). The ONLY feature that rewards
#        BALANCING the labs, and so the one that actually fixes lab-1 starvation:
#        lab_shortfall alone always prefers the faster lab, piling the depot onto
#        lab 2 while lab 1 sits idle and parallel throughput is wasted.
#        SIGN IS ACTION-DEPENDENT: dispatch into the emptier lab lowers it
#        (rewarded), into the fuller lab raises it (penalised) ⇒ with θ≥0 it
#        steers to the under-loaded lab. It RISES for both dispatch actions when
#        both labs are empty (mild brake), BUT this was tested: the depot_age
#        features dominate, so dispatch still fires at empty labs — the brake is
#        harmless. Break-even floor keeps θ from collapsing.
#
#   6    exp_risk   C_dep·n_{0,1} + C_lab·Σ_{p≥1} n_{p,1}
#        Age-1 expiry jump cost across depot and labs. LIVE per epoch: under
#        no-dispatch, age-2 kits age into class 1 and n_{·,1} climbs ⇒ rises
#        toward EOD. C_dep > C_lab ⇒ dispatch reduces it (Δφ < 0). (corr 0.69 with
#        depot_age1 — related but not redundant: exp_risk also weights lab age-1.)
#
# ── Removed / not added across the project ─────────────────────────────────────
#   depot_total  : redundant with depot holding (volume w/o age/cost weight)
#   hold_min     : redundant — identical labs ⇒ collapses to ~0
#   hold_max     : WRONG SIGN — rose on dispatch into empty labs; caused stall
#   unmet_demand : redundant with lab_shortfall (corr ≈ 0.999)
#   depot_urgency: redundant — corr 1.000 with hold_depot in this parametrization
#   per-lab lab_shortfall split : ties when both labs have room ⇒ doesn't fix
#                  starvation; aggregate kept and imbalance does the balancing
#   hold_depot   : replaced by its age-split components (B1)
#
# IN g(s) (known, already incurred — NOT re-encoded as features):
#   C_EXP_DEPOT·ε₀ + C_EXP_LAB·Σε_p   (realised past expiry)
#   (1/Λ)·Σ h_{p,a} n_{p,a}            (current-epoch holding)

NO_BASIS_FN = 7
# LIPSCHITZ_L is no longer a hand-set guess: L is now estimated empirically
# (grid-scale, accounting for the discontinuous EOD/expiry indicator) by
# ALP.compute_geometry_and_lipschitz(), which populates alp.L and alp.Cbar.
# Kept only as a documented fallback if that estimation cannot be run.
LIPSCHITZ_L_FALLBACK = 200.0

# ── Theta constraints — sign constraints only ─────────────────────────────────
# Every non-const feature has negative PSMD gradient at violation states ⇒ θ→0
# without a floor. θ_b ≥ 0 (b ≥ 1); θ₀ free. A break-even floor (from C_DISPATCH,
# h, μ, Λ, γ — NOT LP values) is applied to every non-const index. imbalance can
# rise on dispatch into balanced/empty labs, but the depot_age features dominate
# so this does not block dispatch (verified by sign test).
_B       = NO_BASIS_FN
THETA_LB = [None] + [0.0] * (_B - 1)    # θ₀ free; θ₁..θ₆ ≥ 0
# Theta must be COMPACT (bounded) for the ALP/PSMD theory (Section 5.1's Theta,
# and the Lipschitz-constant/Cbar derivation in Section 5.3) to apply at all —
# an unbounded Theta has no well-defined Lipschitz constant for f(theta,s,u)
# over Theta x S x U. Reuses the same LP_CAP=50.0 scale already used to bound
# the LP warm-start solve in functions/psmd.py's lp_warm_start().
THETA_UB = [None] + [50.0] * (_B - 1)    # θ₀ free; θ₁..θ₆ ≤ 50

# Features receiving a break-even floor (all non-const indices):
THETA_BREAK_EVEN_IDX = [1, 2, 3, 4, 5, 6]

LP_CAP = 50.0

# =============================================================================
# PSMD ALGORITHM PARAMETERS
# =============================================================================
ETA0      = 0.001
LAM0      = 0.0001
H_GRAD    = 10
N_SAMPLES = 50

N_MH            = 50
N_MH_TOTAL      = 400
N_MH_KEEP       = 200
MH_PROPOSAL_STD = 0.5

N_INIT     = 200
T          = 5000
EVAL_EVERY = 200

H_BOUND    = 200
N_BOUND    = 500
UB_NUM_STAGES = 200
UB_NUM_TRAJ   = 50
UB_N_EXOG_POL = 300

LP_N_INIT = 300
LP_N_EXOG = 2000

# ── Normalization (scale-only, policy-invariant preprocessing) ────────────────
# φ̃_b = φ_b / σ_b (σ_b = std_ν[φ_b], σ_0≡1);  c̃ = c / cost_scale (E_{ν,a}[c]).
# Pure positive rescalings ⇒ ALP optimum and greedy policy EXACTLY invariant;
# only the PSMD subgradient geometry is conditioned. Constants estimated once
# under ν by Monte Carlo with a fixed seed (reproducible). Set NORMALIZE=False
# to recover the un-normalized pipeline (e.g. for an ablation in the paper).
NORMALIZE      = True
NORM_N_SAMPLES = 20000
NORM_SEED      = 12345