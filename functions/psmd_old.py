"""
functions/psmd.py
=================
PSMD engine — Primal-Dual Subgradient Method (Lin et al. 2020).

KEY DESIGN CHOICES
==================

1. POLYAK-RUPPERT WEIGHTED AVERAGE
   θ̄_T = Σ η_t · θ_{t+1} / Σ η_t
   Early iterates (large η) carry more weight — required by the Lin et al.
   convergence proof. Implemented via theta_eta_sum / eta_sum_acc.

2. LP WARM-START — INITIALISATION ONLY
   The LP is solved once to find a good starting θ. It does NOT constrain the
   PSMD search afterwards.

3. BREAK-EVEN FLOORS FOR NEGATIVE-GRADIENT FEATURES
   All non-const features (depot_age1..3, lab_shortfall, imbalance, exp_risk)
   have negative PSMD gradient at violation states: dispatch reduces each (in
   expectation), so E[φ(s')] < φ(s) ⇒ grad < 0 ⇒ θ → 0 without a floor (the
   no-dispatch collapse).

   The floor is derived from the break-even dispatch condition at a
   representative state s* (small mixed-age depot, labs empty, τ = τ_max/2):

       dispatch worthwhile iff  Σ_b θ_b · Δφ_b(s*) > c_min_dispatch
       where  Δφ_b(s*) = γ · (E[φ_b|no-dispatch] − E[φ_b|dispatch_cheapest])

   Treating all other θ at their current (LP/PSMD) values:
       θ_b^floor = max(0, (c_min − σ_others) / Δφ_b(s*))

   Derived purely from problem parameters (C_DISPATCH, h_{p,a}, μ, Λ, γ) plus
   the warm-start θ — NO magic numbers. Indices in cfg.THETA_BREAK_EVEN_IDX.

4. ON THE imbalance FEATURE (action-dependent sign)
   imbalance |N₁/μ₁ − N₂/μ₂| drops on dispatch into the emptier lab but rises on
   dispatch into the fuller lab, and rises for either action when both labs are
   empty. It is therefore a mild, situational brake rather than a pure dispatch
   driver. It is retained because it is the ONLY feature that rewards balancing
   the labs (fixing lab-1 starvation), and a sign test confirms the depot_age
   features dominate so dispatch still fires at empty labs. Its break-even floor
   is computed at s* like the others; at s* (labs empty) its Δφ may be ≤ 0, in
   which case _compute_break_even_floor returns 0 and only the sign constraint
   LB=0 binds — which is the intended, safe behaviour.

FEATURE SET (B=7, indices match alp.py and inputs.py)
  0: const  1: depot_age1  2: depot_age2  3: depot_age3
  4: lab_shortfall  5: imbalance  6: exp_risk

ALP FORMULATION (upper-bound, Lin et al.)
  min_θ  E_ν[θ·φ]   s.t.   θ·φ(s) ≥ g(s,u)+c(u)+γE[θ·φ(s')|s,u]  ∀s,u
  PSMD saddle-point: min_θ max_{λ≥0} L(θ,λ)
"""

import numpy as np
from scipy.optimize import linprog


def build_alp_constraints(mdp, alp, S, A, exog):
    gamma      = mdp.gamma
    phi_s      = alp.phi(S)
    E_phi_next = alp.E_phi_next(mdp, S, A, exog)
    blp        = mdp.cost(S, A, exog)
    Alp        = phi_s - gamma * E_phi_next
    Alp[:, 0]  = 1.0 - gamma
    return Alp, blp


def violation_score(Alp, blp, wc_sum, wt_sum, gamma):
    return (blp * wc_sum
            - (Alp[:, 1:] * wt_sum[1:]).sum(axis=1)) / (1.0 - gamma)


def sample_initial_points(mdp, n_init):
    n_random = n_init // 2
    n_mdp    = n_init - n_random
    cols = [np.random.uniform(lo, hi, n_random) for lo, hi in mdp.state_bounds]
    S_rand = np.stack(cols, axis=1)
    S_mdp  = mdp.sample_reachable_states(n_mdp)
    S      = np.vstack([S_rand, S_mdp])

    action_set = np.asarray(mdp.action_set)
    try:
        from inputs.inputs import ACTION_WEIGHTS
        weights = np.asarray(ACTION_WEIGHTS, float)
        weights = weights / weights.sum()
    except Exception:
        weights = None
    A = np.random.choice(action_set, size=n_init, p=weights)
    return S, A


def lp_warm_start(mdp, alp, n_init, n_exog, theta_lb=None, theta_ub=None):
    """Solve a sampled ALP for an initial θ. Initialisation only — the LP
    solution does not floor or ceiling the PSMD search."""
    np.random.seed(0)
    S, A  = sample_initial_points(mdp, n_init)
    exog  = mdp.sample_exog(n_exog)
    Alp, blp = build_alp_constraints(mdp, alp, S, A, exog)
    E_phi = alp.E_phi(mdp)

    LP_CAP = 50.0
    lo = [max(-LP_CAP, theta_lb[b] if (theta_lb and theta_lb[b] is not None) else -LP_CAP)
          for b in range(alp.B)]
    hi = [min(+LP_CAP, theta_ub[b] if (theta_ub and theta_ub[b] is not None) else +LP_CAP)
          for b in range(alp.B)]

    res = linprog(-E_phi, A_ub=Alp, b_ub=blp,
                  bounds=list(zip(lo, hi)), method='highs')

    if res.success and res.fun is not None:
        print(f"LP warm start: theta={np.round(res.x, 4)}, obj={-res.fun:.4f}")
        return res.x

    print("LP warm start failed — using zeros")
    return np.zeros(alp.B)


def _compute_break_even_floor(mdp, alp, theta_ref, exog, feature_idx):
    """
    Minimum θ_b floor from the break-even dispatch condition.

    At representative state s* (small mixed-age depot, labs empty, τ = τ_max/2)
    dispatch is worthwhile iff:
        Σ_b θ_b · Δφ_b(s*) > c_min_dispatch
        Δφ_b = γ · (E[φ_b|no-dispatch] − E[φ_b|dispatch_cheapest])

    Treating all b' ≠ b at their reference (warm-start) θ values:
        θ_b^floor = max(0, (c_min − σ_others) / Δφ_b)

    Positive Δφ_b means the feature DECREASES on dispatch (a dispatch signal);
    only such features need a floor. Returns 0.0 otherwise.

    All quantities come from problem parameters + reference θ — no magic numbers.
    """
    from inputs.inputs import N_STATE, TAU_MAX, C_DISPATCH

    gamma    = mdp.gamma
    # c_min is a RAW dispatch cost; the feature Δφ below come from the NORMALIZED
    # basis (alp.phi). To keep the break-even inequality dimensionally consistent
    # after cost/feature normalization, express c_min in the SAME normalized cost
    # units as the ALP right-hand side: divide by mdp.cost_scale (≡1 if costs are
    # not calibrated). Feature normalization (÷σ_b) is already baked into Δφ via
    # alp.phi, so no extra factor is needed there.
    c_min    = float(C_DISPATCH.min()) / mdp.cost_scale
    cheapest = int(np.argmin(C_DISPATCH)) + 1        # 1-indexed lab action
    tau_rep  = TAU_MAX / 2.0
    exog_sm  = exog[:min(500, len(exog))]

    # Representative state: 1 age-1 + 1 age-2 + 2 age-3 kits at depot, labs empty
    s_rep        = np.zeros(N_STATE)
    s_rep[:3]    = [1.0, 1.0, 2.0]
    s_rep[mdp.TAU_IDX] = tau_rep

    S_rep = s_rep[None, :]
    E_no  = alp.E_phi_next(mdp, S_rep, np.array([0]),        exog_sm)[0]
    E_dis = alp.E_phi_next(mdp, S_rep, np.array([cheapest]), exog_sm)[0]

    delta_phi = gamma * (E_no - E_dis)               # shape (B,)

    if delta_phi[feature_idx] <= 1e-6:
        return 0.0

    B = alp.B
    sigma_others = sum(
        theta_ref[b] * delta_phi[b] for b in range(B) if b != feature_idx
    )
    floor = max(0.0, (c_min - sigma_others) / delta_phi[feature_idx])
    print(f"  break-even floor θ_{feature_idx}: {floor:.4f}  "
          f"(Δφ={delta_phi[feature_idx]:.4f}, σ_others={sigma_others:.4f}, "
          f"c_min={c_min:.1f})")
    return floor


def run_psmd(mdp, alp, lb_calc, ub_calc, cfg):
    """
    Main PSMD loop.
    Returns: thetabar, Thetabar, LB_history, LB_best, UB_history, UB_best, best_thetabar
    """
    from classes.dualupdater   import Dual
    from classes.gradient      import Gradient
    from classes.primalupdater import Primal
    from classes.mhsampler     import MHSampler

    T          = cfg.T
    eta0       = cfg.ETA0
    lam0       = cfg.LAM0
    H          = cfg.H_GRAD
    N          = cfg.N_SAMPLES
    N_mh       = cfg.N_MH
    n_total    = cfg.N_MH_TOTAL
    n_keep     = cfg.N_MH_KEEP
    n_init     = cfg.N_INIT
    eval_every = cfg.EVAL_EVERY
    B          = alp.B
    gamma      = mdp.gamma

    _lb_static = list(getattr(cfg, 'THETA_LB', [None] * B))
    _ub_static = list(getattr(cfg, 'THETA_UB', [None] * B))
    break_even_idx = list(getattr(cfg, 'THETA_BREAK_EVEN_IDX', []))

    # ── Normalization preprocessing (scale-only; policy-invariant) ────────────
    # Calibrate ONCE under ν before any optimization, deterministic given seeds.
    #   features: φ̃_b = φ_b / σ_b,  σ_b = std_ν[φ_b]   (σ_0 ≡ 1)   → alp._phi_scale
    #   cost/g  : c̃ = c / cost_scale, cost_scale = E_{ν,a}[c]       → mdp.cost_scale
    # Both are pure positive rescalings ⇒ the ALP optimum V̂* and the greedy
    # policy are exactly invariant; only the PSMD step geometry (conditioning)
    # changes. All downstream calls to alp.phi(...) and mdp.cost(...) are now in
    # normalized units automatically, so the break-even floor, gradients, LP
    # warm-start and bounds are all mutually consistent.
    if getattr(cfg, 'NORMALIZE', True):
        sigma, mu_raw = alp.calibrate_normalization(
            mdp, n_samples=getattr(cfg, 'NORM_N_SAMPLES', 20000),
            seed=getattr(cfg, 'NORM_SEED', 12345))
        cscale = mdp.calibrate_cost_scale(
            n_samples=getattr(cfg, 'NORM_N_SAMPLES', 20000),
            seed=getattr(cfg, 'NORM_SEED', 12345))
        print("Normalization (scale-only, under ν):")
        print(f"  feature σ = {np.round(sigma, 4)}")
        print(f"  cost_scale = {cscale:.4f}")
        # store for de-normalizing θ back to raw coordinates when reporting
        alp._phi_sigma_report = sigma
        alp._cost_scale_report = cscale

    dual    = Dual(mdp, alp, T=T, N_mh=N_mh)
    grad_   = Gradient(mdp, alp)
    sampler = MHSampler(mdp, dual,
                        proposal_std=cfg.MH_PROPOSAL_STD,
                        n_total=n_total, n_keep=n_keep)

    np.random.seed(1)
    DSampleHist = np.vstack([mdp.sample_exog(N_mh) for _ in range(T)])
    dual.set_exog_history(DSampleHist)

    np.random.seed(2)
    S_init, A_init = sample_initial_points(mdp, n_init)
    exog_large     = mdp.sample_exog(2000)
    Alp_init, blp_init = build_alp_constraints(mdp, alp, S_init, A_init, exog_large)

    # ── LP warm start (initialisation only) ──────────────────────────────────
    theta = lp_warm_start(mdp, alp, cfg.LP_N_INIT, cfg.LP_N_EXOG,
                          theta_lb=_lb_static, theta_ub=_ub_static)

    # Enforce sign constraints on the LP starting point
    for b in range(B):
        if _lb_static[b] is not None:
            theta[b] = max(theta[b], float(_lb_static[b]))
        if _ub_static[b] is not None:
            theta[b] = min(theta[b], float(_ub_static[b]))
    print(f"\nLP warm start theta (sign-clipped): {np.round(theta, 4)}")

    # ── Break-even floors for negative-gradient features ─────────────────────
    # Derived from problem parameters + warm-start θ; not from LP values directly.
    lb_live = list(_lb_static)
    ub_live = list(_ub_static)

    print("\nComputing break-even floors:")
    for idx in break_even_idx:
        floor   = _compute_break_even_floor(mdp, alp, theta, exog_large, idx)
        lb_base = float(_lb_static[idx]) if _lb_static[idx] is not None else 0.0
        lb_live[idx] = max(lb_base, floor)
        theta[idx]   = max(theta[idx], lb_live[idx])

    print(f"\nInitial theta (floors applied): {np.round(theta, 4)}")
    print(f"Active lower bounds: "
          f"{[f'θ_{b}≥{lb_live[b]:.4f}' for b in range(B) if lb_live[b] is not None]}")

    primal = Primal(lb=lb_live, ub=ub_live)

    thetabar      = theta.copy()
    Thetabar      = np.zeros((T, B))
    lam_sum       = 0.0
    eta_sum_acc   = 0.0
    theta_eta_sum = np.zeros(B)
    LB_history    = []; LB_best = []
    UB_history    = []; UB_best = []
    best_thetabar = theta.copy()
    best_ub_val   = float('inf')

    for t in range(T):
        eta = eta0 / np.sqrt(t + 1)
        lam = lam0 / np.sqrt(t + 1)
        lam_sum     += lam * eta
        eta_sum_acc += eta

        dual.update(eta, lam, theta)

        wc_sum, wt_sum = dual.weight_sums()
        scores  = violation_score(Alp_init, blp_init, wc_sum, wt_sum, gamma)
        tempid  = int(np.argmax(scores))
        start_s = S_init[tempid]
        start_a = A_init[tempid]

        sa_chain, _, _ = sampler.sample(start_s, start_a)

        exog_grad = mdp.sample_exog(N)
        grad_vec  = grad_.compute_batch(sa_chain[:H, :], exog_grad)

        theta = primal.update(theta, grad_vec, eta)

        # Polyak-Ruppert weighted average: θ̄_T = Σ η_t·θ_{t+1} / Σ η_t
        theta_eta_sum += eta * theta
        thetabar = theta_eta_sum / eta_sum_acc
        Thetabar[t, :] = thetabar

        if (t + 1) % eval_every == 0 or t == T - 1:
            lambdabar = lam_sum / eta_sum_acc

            print(f"\n--- t={t+1}  lambdabar={lambdabar:.6f} ---")
            print(f"    thetabar = {np.round(thetabar, 3)}")

            lb, _ = lb_calc.compute(thetabar, lambdabar,
                                    H=cfg.H_BOUND, N=cfg.N_BOUND)
            LB_history.append((t + 1, lb))
            best_lb = max(v for _, v in LB_history)
            LB_best.append((t + 1, best_lb))

            ub, ub_se = ub_calc.compute(
                thetabar,
                num_stage    = cfg.UB_NUM_STAGES,
                num_traj     = cfg.UB_NUM_TRAJ,
                n_exog_policy= cfg.UB_N_EXOG_POL
            )
            UB_history.append((t + 1, ub, ub_se))
            best_ub = min(v for _, v, _ in UB_history)
            UB_best.append((t + 1, best_ub))
            if ub < best_ub_val:
                best_ub_val   = ub
                best_thetabar = thetabar.copy()

            gap      = (ub - lb) / abs(ub) * 100 if ub != 0 else float('nan')
            best_gap = (best_ub - best_lb) / abs(best_ub) * 100 \
                       if best_ub != 0 else float('nan')
            print(f"    LB={lb:.2f}  UB={ub:.2f}±{ub_se:.2f}  "
                  f"gap={gap:.2f}%  best_gap={best_gap:.2f}%")

        if t % 50 == 0:
            print(f"t={t:4d} | thetabar={np.round(thetabar, 3)}")

    print(f"\nFinal thetabar: {np.round(thetabar, 4)}")
    print(f"Best thetabar (at lowest UB={best_ub_val:.4f}): "
          f"{np.round(best_thetabar, 4)}")
    return (thetabar, Thetabar, LB_history, LB_best,
            UB_history, UB_best, best_thetabar)
