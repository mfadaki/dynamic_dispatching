"""
functions/psmd_evaluate_approximation.py
==========================================
ISOLATED COPY of functions/psmd.py, used ONLY by evaluate_approximation.py.

Identical to the production functions/psmd.py except for two changes:

  1. _compute_break_even_floor()'s representative state s* hardcoded
     exactly 3 depot age classes (s_rep[:3] = [1.0, 1.0, 2.0]). For any
     L_AGE != 3 (e.g. the L_AGE=2 small instance) this silently writes
     into the WRONG inventory slot -- with L_AGE=2 the 3rd element of the
     flattened state is lab 1's age-1 cell, not a (nonexistent) depot
     age-3 cell. Generalized to build s* from L_AGE dynamically:
     s_rep[:L_AGE] = [1.0]*(L_AGE-1) + [2.0], which reduces to the
     original [1.0, 1.0, 2.0] exactly when L_AGE=3.

  2. run_psmd() now accepts a `run_seed` parameter (default 0, matching
     the production file's hardcoded behaviour exactly when left at its
     default). All of the internal seed resets that the production file
     hardcodes as 0/1/2 are offset by run_seed*1000 instead, so different
     run_seed values produce genuinely different -- but each individually
     reproducible -- training trajectories. This is what run_multiseed.py
     uses to get an actual distribution across seeds rather than relying
     on incidental, unlabelled run-to-run variance.

Deliberately kept as a SEPARATE file rather than edited in place, so the
production functions/psmd.py -- and therefore every existing/future
training run through main.py -- is completely unaffected. Do not import
this module from anything other than evaluate_approximation.py /
run_multiseed.py.
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


def lp_warm_start(mdp, alp, n_init, n_exog, theta_lb=None, theta_ub=None, seed=0):
    """Solve a sampled ALP for an initial θ. Initialisation only — the LP
    solution does not floor or ceiling the PSMD search."""
    np.random.seed(seed)
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
    from inputs.inputs import N_STATE, TAU_MAX, C_DISPATCH, L_AGE

    gamma    = mdp.gamma
    c_min    = float(C_DISPATCH.min()) / mdp.cost_scale
    cheapest = int(np.argmin(C_DISPATCH)) + 1        # 1-indexed lab action
    tau_rep  = TAU_MAX / 2.0
    exog_sm  = exog[:min(500, len(exog))]

    # Representative state: mixed-age depot (weighted toward fresher stock),
    # labs empty. Built from L_AGE dynamically -- [1.0]*(L_AGE-1) + [2.0] --
    # rather than hardcoding exactly 3 age classes, so this generalizes to
    # any L_AGE. Reduces to the original [1.0, 1.0, 2.0] exactly at L_AGE=3.
    s_rep               = np.zeros(N_STATE)
    s_rep[:L_AGE]        = [1.0] * (L_AGE - 1) + [2.0]
    s_rep[mdp.TAU_IDX]  = tau_rep

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


def run_psmd(mdp, alp, lb_calc, ub_calc, cfg, run_seed=0):
    """
    Main PSMD loop.
    Returns: thetabar, Thetabar, LB_history, LB_best, UB_history, UB_best, best_thetabar

    run_seed: offsets every internal np.random.seed(...) reset by
    run_seed*1000, so different run_seed values give genuinely different
    (but each individually reproducible) training trajectories. Defaults
    to 0, which reproduces the exact seed values (0, 1, 2) the production
    functions/psmd.py hardcodes.
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

    seed_base = run_seed * 1000

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
        alp._phi_sigma_report = sigma
        alp._cost_scale_report = cscale

    dual    = Dual(mdp, alp, T=T, N_mh=N_mh)
    grad_   = Gradient(mdp, alp)
    sampler = MHSampler(mdp, dual,
                        proposal_std=cfg.MH_PROPOSAL_STD,
                        n_total=n_total, n_keep=n_keep)

    np.random.seed(seed_base + 1)
    DSampleHist = np.vstack([mdp.sample_exog(N_mh) for _ in range(T)])
    dual.set_exog_history(DSampleHist)

    np.random.seed(seed_base + 2)
    S_init, A_init = sample_initial_points(mdp, n_init)
    exog_large     = mdp.sample_exog(2000)
    Alp_init, blp_init = build_alp_constraints(mdp, alp, S_init, A_init, exog_large)

    # ── LP warm start (initialisation only) ──────────────────────────────────
    theta = lp_warm_start(mdp, alp, cfg.LP_N_INIT, cfg.LP_N_EXOG,
                          theta_lb=_lb_static, theta_ub=_ub_static,
                          seed=seed_base + 0)

    for b in range(B):
        if _lb_static[b] is not None:
            theta[b] = max(theta[b], float(_lb_static[b]))
        if _ub_static[b] is not None:
            theta[b] = min(theta[b], float(_ub_static[b]))
    print(f"\nLP warm start theta (sign-clipped): {np.round(theta, 4)}")

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

        wc_sum, wt_sum = dual.weight_sums()
        scores  = violation_score(Alp_init, blp_init, wc_sum, wt_sum, gamma)
        tempid  = int(np.argmax(scores))
        start_s = S_init[tempid]
        start_a = A_init[tempid]

        sa_chain, _, _ = sampler.sample(start_s, start_a)

        exog_grad = mdp.sample_exog(N)
        grad_vec  = grad_.compute_batch(sa_chain[:H, :], exog_grad)

        theta_t = theta.copy()
        theta   = primal.update(theta, grad_vec, eta)

        dual.update(eta, lam, theta_t)

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

    ell_hat = ell_hat_se = None
    final_lambdabar = lam_sum / eta_sum_acc if eta_sum_acc > 0 else 0.0
    if getattr(cfg, 'COMPUTE_SAA_LB', True):
        from classes.bounds import SAALowerBound
        saa_lb = SAALowerBound(mdp, alp)
        ell_hat, ell_hat_se = saa_lb.compute(
            best_thetabar, final_lambdabar,
            H_prime = getattr(cfg, 'SAA_H_PRIME', 100),
            N_prime = getattr(cfg, 'SAA_N_PRIME', 500),
        )
        print(f"\nCertified SAA lower bound ell_hat(theta_bar) "
              f"= {ell_hat:.4f} ± {ell_hat_se:.4f}  (kappa_bar={final_lambdabar:.2e}, "
              f"Cbar={alp.Cbar:.3e})")
        print(f"  Note: this bound is VALID (per Section 5.3's theory, now that "
              f"Cbar is properly derived) but typically much looser than the "
              f"ad hoc LP-based LB above — see SAALowerBound's docstring.")

    return (thetabar, Thetabar, LB_history, LB_best,
            UB_history, UB_best, best_thetabar, ell_hat, ell_hat_se)