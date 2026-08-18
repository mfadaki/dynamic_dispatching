"""
evaluate_approximation.py
===========================
Validates the PSMD/VFA machinery against a ground-truth exact solution, on
an instance small enough to solve exactly.

What it does
------------
1. Loads inputs_evaluate_approximation.py (a small instance -- see that
   file's docstring for sizing) in place of the production inputs.py, via
   a sys.modules swap performed BEFORE any inputs.inputs-dependent module
   is imported. Uses ISOLATED copies of MDP/ALP
   (classes/mdp_evaluate_approximation.py, classes/alp_evaluate_approximation.py)
   rather than the production classes.mdp / classes.alp, so nothing this
   script does can affect production training runs through main.py -- see
   those two files' own docstrings for exactly what differs and why.
2. Enumerates the small instance's ENTIRE state space exhaustively.
3. Builds and solves the EXACT linear program (one variable per state,
   Eq. (26) restricted to this finite S, using the EXACT one-step
   transition distribution -- no basis functions, no sampling) to get
   V*(s) for every state and the exact optimal policy pi*(s).
4. Trains a VFA via the SAME PSMD procedure used for the production
   instance (functions.psmd.run_psmd), on this small instance.
5. Extracts the VFA-greedy policy pi_VFA(s), again using the EXACT
   transition distribution (not Monte Carlo), so the comparison isn't
   contaminated by its own sampling noise.
6. Evaluates pi_VFA's TRUE value V^{pi_VFA}(s) exactly, by solving the
   linear system (I - gamma P^{pi_VFA}) V = r^{pi_VFA} -- not simulation.
7. Reports: policy agreement rate, value-function gap stats, and the
   EXACT suboptimality of the VFA policy under several initial-state
   weightings.

Usage
-----
    python evaluate_approximation.py

Run from the project root (the directory containing classes/, functions/,
inputs/), same as main.py.
"""

import sys
import os
import time as time_module
import itertools
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _clean_argv():
    """Extract real positional arguments from sys.argv, ignoring anything
    that looks like a flag -- same reasoning as post.py's _clean_argv:
    Jupyter/IPython kernels populate sys.argv with their OWN launch
    arguments (e.g. '--f=/path/to/kernel-xxxx.json'), which must never be
    mistaken for a config-module name or seed."""
    return [a for a in sys.argv[1:] if not a.startswith('-')]


# ── Swap in the small-instance config BEFORE importing anything that reads
# inputs.inputs at module load time (classes.mdp, classes.alp, functions.psmd
# all do `from inputs.inputs import ...`). This must happen first. The config
# module and an optional run seed are both chosen dynamically, so this same
# script can be run against multiple instance sizes AND multiple seeds --
# e.g.  python evaluate_approximation.py inputs_evaluate_approximation_small 3
# runs the small instance with run_seed=3. Each (config, seed) combination
# writes its own, separately-named results/ folder (see compare_instances.py
# and run_multiseed.py, which read those folders back). ────────────────────
_argv = _clean_argv()
_CONFIG_MODULE_NAME = _argv[0] if len(_argv) >= 1 else 'inputs_evaluate_approximation'
_RUN_SEED = int(_argv[1]) if len(_argv) >= 2 else 0

small_cfg = importlib.import_module(f'inputs.{_CONFIG_MODULE_NAME}')
sys.modules['inputs.inputs'] = small_cfg

import inputs.inputs as cfg           # noqa: E402  (now resolves to small_cfg)
# Uses ISOLATED copies of MDP/ALP (see classes/mdp_evaluate_approximation.py
# and classes/alp_evaluate_approximation.py for exactly what differs and
# why) -- NOT the production classes.mdp / classes.alp, so this script's
# small-instance fixes never touch the production training pipeline.
from classes.mdp_evaluate_approximation import MDP   # noqa: E402
from classes.alp_evaluate_approximation import ALP   # noqa: E402
from classes.bounds import LowerBound, UpperBound   # noqa: E402
# Uses the ISOLATED copy of run_psmd (see functions/psmd_evaluate_approximation.py
# for exactly what differs and why: an L_AGE generalization fix, and a
# run_seed parameter for genuine multi-seed comparison) -- NOT the
# production functions.psmd, so this script's changes never touch the
# production training pipeline.
from functions.psmd_evaluate_approximation import run_psmd   # noqa: E402
from functions.functions import saveResultsFn, RunTime   # noqa: E402


# =============================================================================
# 1. Exhaustive state enumeration
# =============================================================================

def enumerate_states(mdp, cfg):
    """All states reachable within the small instance's box bounds:
    every inventory cell in {0,...,K_CAPACITY}, every tau on the
    epochs_per_day grid. Returns (states, index, tau_grid): index maps a
    state key -> its row in `states`; tau_grid is passed through so
    exact_transitions() can snap transitioned-to tau values onto it."""
    n_cells = cfg.N_INV
    cell_values = range(0, cfg.N_MAX + 1)
    tau_grid = [cfg.TAU_MAX - k * cfg.DELTA_T for k in range(cfg.epochs_per_day)]

    states = []
    for combo in itertools.product(cell_values, repeat=n_cells):
        n = np.array(combo, dtype=float).reshape(cfg.N_LABS + 1, cfg.L_AGE)
        for tau in tau_grid:
            states.append(mdp.build_state(n, tau))
    states = np.array(states)

    index = {_state_key(s, tau_grid): i for i, s in enumerate(states)}
    return states, index, tau_grid


def _snap_tau(tau, tau_grid, tol=1e-6):
    """Snap a computed tau onto the nearest value in tau_grid, raising if
    none is close enough. This -- not independently rounding two floats
    that may differ by ~1e-9 and land on opposite sides of a rounding
    boundary -- is what keeps state keys consistent between enumeration
    and transition."""
    diffs = [abs(tau - g) for g in tau_grid]
    j = int(np.argmin(diffs))
    if diffs[j] > tol:
        raise KeyError(
            f"tau={tau} is not within {tol} of any enumerated grid value "
            f"{tau_grid} — check DELTA_T/TAU_MAX consistency in "
            f"inputs_evaluate_approximation.py."
        )
    return tau_grid[j]


def _state_key(s, tau_grid, ndigits=6):
    n_part = tuple(round(float(v), ndigits) for v in s[:-1])
    tau_part = round(_snap_tau(float(s[-1]), tau_grid), ndigits)
    return n_part + (tau_part,)


# =============================================================================
# 2. Exact transition model (no sampling -- the uniformized chain's own
#    exact event probabilities, applied deterministically per event)
# =============================================================================

def exact_transitions(mdp, s, a, index, tau_grid):
    """Returns list of (next_state_row_index, probability) for every one of
    the N_EVENT_TYPES possible events at (s,a) -- exact, not sampled."""
    out = []
    for event in range(mdp.N_EVENT_TYPES):
        p = mdp._probs[event]
        if p <= 0:
            continue
        s_next = mdp._transition_single(s, a, event)
        key = _state_key(s_next, tau_grid)
        if key not in index:
            raise KeyError(
                "Transition left the enumerated state space -- the small "
                "instance's bounds are inconsistent with its own dynamics "
                "(e.g. K_CAPACITY too small for the arrival/processing "
                "rates chosen). Increase K_CAPACITY in "
                "inputs_evaluate_approximation.py."
            )
        out.append((index[key], p))
    return out


# =============================================================================
# 3. Exact LP:  max_V E_nu[V(s)]  s.t.  V(s) <= r(s,u) + gamma E[V(s')|s,u]
# =============================================================================

def solve_exact_lp(mdp, states, index, tau_grid):
    M = len(states)
    actions = mdp.action_set

    print(f"Exact LP: {M} states x {len(actions)} actions "
          f"= {M * len(actions)} constraints...")

    # Precompute cost and transitions for every (s,u)
    rhs_const = np.zeros((M, len(actions)))          # r(s,u)
    trans = [[None] * len(actions) for _ in range(M)]  # list of (j, p)

    for i, s in enumerate(states):
        for ai, a in enumerate(actions):
            rhs_const[i, ai] = mdp._cost_single_raw(s, a) / mdp.cost_scale
            trans[i][ai] = exact_transitions(mdp, s, a, index, tau_grid)

    # Build sparse-ish constraint matrix: for each (i, ai), a row with
    # V(s_i) coefficient +1 and -gamma*Pr(j|i,ai) for each reachable j.
    from scipy.sparse import lil_matrix
    n_constraints = M * len(actions)
    A_ub = lil_matrix((n_constraints, M))
    b_ub = np.zeros(n_constraints)

    row = 0
    for i in range(M):
        for ai, a in enumerate(actions):
            A_ub[row, i] += 1.0
            for j, p in trans[i][ai]:
                A_ub[row, j] += -mdp.gamma * p
            b_ub[row] = rhs_const[i, ai]
            row += 1

    A_ub = A_ub.tocsr()
    c_obj = -np.ones(M) / M   # maximize (uniform) mean V(s) == minimize -mean

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * M, method='highs')
    if not res.success:
        raise RuntimeError(f"Exact LP failed to solve: {res.message}")

    V_star = res.x
    return V_star, rhs_const, trans


def extract_exact_policy(mdp, states, rhs_const, trans, V_star):
    """pi*(s) = argmin_u { r(s,u) + gamma * E[V*(s') | s, u] }, recomputed
    directly from the solved V* (robust, rather than relying on which LP
    constraint happens to bind)."""
    M, n_actions = rhs_const.shape
    policy = np.zeros(M, dtype=int)
    for i in range(M):
        best_val, best_a = np.inf, 0
        for ai in range(n_actions):
            cont = sum(p * V_star[j] for j, p in trans[i][ai])
            val = rhs_const[i, ai] + mdp.gamma * cont
            if val < best_val:
                best_val, best_a = val, ai
        policy[i] = best_a
    return policy


# =============================================================================
# 4. VFA-greedy policy, using the SAME exact transition model
# =============================================================================

def extract_vfa_policy(mdp, alp, theta, states, index):
    """pi_VFA(s) = argmin_u { c(s,u) + gamma * sum_b theta_b E[phi_b(s')|s,u] },
    using the exact event distribution (no Monte Carlo) for an apples-to-
    apples comparison against the exact policy."""
    M = len(states)
    n_actions = len(mdp.action_set)
    policy = np.zeros(M, dtype=int)
    Vhat = np.zeros(M)

    phi_s_all = alp.phi(states)          # (M, B)
    Vhat = phi_s_all @ theta

    for i, s in enumerate(states):
        best_val, best_a = np.inf, 0
        for ai, a in enumerate(mdp.action_set):
            c = mdp._cost_single_raw(s, a) / mdp.cost_scale
            cont = 0.0
            for event in range(mdp.N_EVENT_TYPES):
                p = mdp._probs[event]
                if p <= 0:
                    continue
                s_next = mdp._transition_single(s, a, event)
                phi_next = alp.phi(s_next)
                cont += p * float(phi_next @ theta)
            val = c + mdp.gamma * cont
            if val < best_val:
                best_val, best_a = val, ai
        policy[i] = best_a

    return policy, Vhat


# =============================================================================
# 5. Exact policy evaluation: solve (I - gamma P^pi) V = r^pi directly
# =============================================================================

def evaluate_policy_exactly(mdp, states, index, policy, tau_grid, rhs_const=None, trans=None):
    M = len(states)
    if rhs_const is None or trans is None:
        rhs_const = np.zeros(M)
        trans = [None] * M
        for i, s in enumerate(states):
            a = mdp.action_set[policy[i]]
            rhs_const[i] = mdp._cost_single_raw(s, a) / mdp.cost_scale
            trans[i] = exact_transitions(mdp, s, a, index, tau_grid)
        r_pi = rhs_const
        trans_pi = trans
    else:
        r_pi = np.array([rhs_const[i, policy[i]] for i in range(M)])
        trans_pi = [trans[i][policy[i]] for i in range(M)]

    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    Amat = lil_matrix((M, M))
    for i in range(M):
        Amat[i, i] += 1.0
        for j, p in trans_pi[i]:
            Amat[i, j] += -mdp.gamma * p
    Amat = Amat.tocsr()

    V_pi = spsolve(Amat, r_pi)
    return V_pi


# =============================================================================
# Result-saving helper
# =============================================================================

def capture_all_inputs(cfg_module):
    """Snapshot every constant-like attribute of the small-instance config,
    same convention as main.py's capture_all_inputs, for reproducibility."""
    import types
    snapshot = {}
    for name in dir(cfg_module):
        if name.startswith('_'):
            continue
        val = getattr(cfg_module, name)
        if isinstance(val, types.ModuleType) or callable(val):
            continue
        snapshot[name] = val
    return snapshot


def feature_names_for(n_labs, l_age):
    """Same feature-set logic as classes/alp_evaluate_approximation.py's
    _phi_raw (the canonical formula), so plot legends/labels always match
    what was actually trained regardless of N_LABS."""
    names = ['const'] + [f'depot_age{a+1}' for a in range(l_age)]
    names += ['lab_shortfall_agg']
    if n_labs >= 2:
        names += ['imbalance', 'exp_risk']
    else:
        names += ['exp_risk']
    return names


def plot_theta_convergence(Thetabar_trajectory, n_labs, l_age, instance_tag, seed):
    """One convergence plot per run: every theta_bar coefficient's value
    across all T training iterations, to visually confirm the coefficients
    have actually flattened out by the end of training rather than still
    trending -- the qualitative check requested alongside increasing T."""
    T, B = Thetabar_trajectory.shape
    names = feature_names_for(n_labs, l_age)
    assert len(names) == B, (
        f"feature_names_for produced {len(names)} names but Thetabar has "
        f"B={B} columns -- update feature_names_for to match the actual "
        f"feature set for this N_LABS/L_AGE combination."
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Top: const separately, since its scale usually dwarfs the others
    axes[0].plot(range(T), Thetabar_trajectory[:, 0], label=names[0], color='black')
    axes[0].set_ylabel('theta_bar value')
    axes[0].set_title(f'const coefficient -- {instance_tag}, seed={seed}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Bottom: every other coefficient together, since they're on a
    # comparable scale to each other
    for b in range(1, B):
        axes[1].plot(range(T), Thetabar_trajectory[:, b], label=names[b])
    axes[1].set_xlabel('PSMD iteration t')
    axes[1].set_ylabel('theta_bar value')
    axes[1].set_title('All other coefficients')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    fname = f'theta_convergence_{instance_tag}_seed{seed}.png'
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nTheta convergence plot written: {fname}")

    # Quantitative convergence check alongside the visual one: how much did
    # each coefficient move in the LAST 10% of training vs its own range
    # over the full run -- small values here confirm the plot's "flat at
    # the end" impression rather than relying on eyeballing alone.
    last_10pct = max(1, T // 10)
    late_drift = np.abs(Thetabar_trajectory[-1, :] - Thetabar_trajectory[-last_10pct, :])
    full_range = Thetabar_trajectory.max(axis=0) - Thetabar_trajectory.min(axis=0)
    print(f"\nConvergence check (movement in the last {last_10pct} iterations, "
          f"as %% of that coefficient's full-training range):")
    for b in range(B):
        pct = (late_drift[b] / full_range[b] * 100) if full_range[b] > 1e-8 else 0.0
        flag = "" if pct < 5 else "  <-- still moving noticeably late in training"
        print(f"  {names[b]:<22} moved {pct:5.1f}% of its range in the last "
              f"{last_10pct} iters{flag}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    start_time = time_module.time()

    print(f"Small instance: N_LABS={cfg.N_LABS}, L_AGE={cfg.L_AGE}, "
          f"K_CAPACITY={cfg.N_MAX}, epochs/day={cfg.epochs_per_day}, "
          f"gamma={cfg.GAMMA}")

    mdp = MDP()
    alp = ALP()
    # Match run_psmd's internal (re-)calibration exactly (same n_samples,
    # same seed) -- otherwise V_star (computed here, before run_psmd) and
    # V_pi_vfa (computed after run_psmd returns) end up on two slightly
    # different cost/feature scales, since run_psmd unconditionally
    # recalibrates using cfg.NORM_N_SAMPLES/cfg.NORM_SEED regardless of
    # what was used here. Confirmed directly: with cfg.NORM_N_SAMPLES=5000
    # (this instance) vs the previous default of 20000, cost_scale differed
    # by ~0.79% between the two calls -- a real, systematic inconsistency,
    # not LP-solver floating-point noise.
    alp.calibrate_normalization(mdp, n_samples=getattr(cfg, 'NORM_N_SAMPLES', 20000),
                                seed=getattr(cfg, 'NORM_SEED', 12345))
    mdp.calibrate_cost_scale(n_samples=getattr(cfg, 'NORM_N_SAMPLES', 20000),
                             seed=getattr(cfg, 'NORM_SEED', 12345))

    states, index, tau_grid = enumerate_states(mdp, cfg)
    print(f"Enumerated {len(states)} states.")

    # --- Exact solution ------------------------------------------------------
    V_star, rhs_const, trans = solve_exact_lp(mdp, states, index, tau_grid)
    pi_star = extract_exact_policy(mdp, states, rhs_const, trans, V_star)
    print(f"Exact LP solved. E[V*] over enumerated states "
          f"(uniform weighting) = {V_star.mean():.4f}")

    # --- Train the VFA on this same small instance ---------------------------
    print("\nTraining PSMD on the small instance...")
    lb = LowerBound(mdp, alp)
    ub = UpperBound(mdp, alp)
    out = run_psmd(mdp, alp, lb, ub, cfg, run_seed=_RUN_SEED)
    thetabar, Thetabar_trajectory, best_thetabar = out[0], out[1], out[6]
    theta = best_thetabar
    print(f"\nTrained theta (best UB iterate): {np.round(theta, 4)}")

    # --- VFA-greedy policy, evaluated exactly on the same state space -------
    pi_vfa, Vhat = extract_vfa_policy(mdp, alp, theta, states, index)
    V_pi_vfa = evaluate_policy_exactly(mdp, states, index, pi_vfa, tau_grid)

    # =========================================================================
    # Comparison report
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXACT vs. VFA-GREEDY POLICY COMPARISON")
    print("=" * 70)

    agree = (pi_star == pi_vfa)
    print(f"\nPolicy agreement: {agree.mean()*100:.2f}% of states "
          f"({agree.sum()}/{len(states)})")

    diffs = V_pi_vfa - V_star
    print(f"\nExact suboptimality of the VFA policy, V^pi_VFA(s) - V*(s):")
    print(f"  mean = {diffs.mean():.4f}   max = {diffs.max():.4f}   "
          f"min = {diffs.min():.4f}  (should be >= 0 everywhere, since "
          f"pi* is optimal)")
    frac_gap = (diffs / np.maximum(np.abs(V_star), 1e-8))
    print(f"  mean relative gap = {frac_gap.mean()*100:.2f}%   "
          f"max relative gap = {frac_gap.max()*100:.2f}%")

    print(f"\nUniformly-weighted average over all {len(states)} states:")
    print(f"  E[V*]        = {V_star.mean():.4f}")
    print(f"  E[V^pi_VFA]  = {V_pi_vfa.mean():.4f}")
    print(f"  Gap          = {(V_pi_vfa.mean()-V_star.mean()):.4f}  "
          f"({(V_pi_vfa.mean()-V_star.mean())/abs(V_star.mean())*100:.2f}%)")

    # Near-empty-depot initial states specifically (closer to how the
    # production model's sample_initial_state() actually starts trajectories)
    near_empty = np.array([
        i for i, s in enumerate(states)
        if s[:cfg.N_INV].sum() <= 2
    ])
    near_empty_stats = {}
    if len(near_empty) > 0:
        print(f"\nRestricted to {len(near_empty)} near-empty-depot states "
              f"(total inventory <= 2, closer to typical trajectory starts):")
        print(f"  E[V*]        = {V_star[near_empty].mean():.4f}")
        print(f"  E[V^pi_VFA]  = {V_pi_vfa[near_empty].mean():.4f}")
        gap = V_pi_vfa[near_empty].mean() - V_star[near_empty].mean()
        print(f"  Gap          = {gap:.4f}  "
              f"({gap/abs(V_star[near_empty].mean())*100:.2f}%)")
        near_empty_stats = {
            'n_states': int(len(near_empty)),
            'E_V_star': float(V_star[near_empty].mean()),
            'E_V_pi_vfa': float(V_pi_vfa[near_empty].mean()),
            'gap': float(gap),
            'gap_pct': float(gap / abs(V_star[near_empty].mean()) * 100),
        }

    print("\n" + "=" * 70)

    # =========================================================================
    # Save everything to ./results/, same convention as main.py
    # =========================================================================
    run_time_str = RunTime(start_time)

    comparison_summary = {
        'config_module': _CONFIG_MODULE_NAME,
        'run_seed': int(_RUN_SEED),
        'K_CAPACITY': int(cfg.N_MAX),
        'epochs_per_day': int(cfg.epochs_per_day),
        'n_states': int(len(states)),
        'policy_agreement_pct': float(agree.mean() * 100),
        'policy_agreement_count': int(agree.sum()),
        'suboptimality_mean': float(diffs.mean()),
        'suboptimality_max': float(diffs.max()),
        'suboptimality_min': float(diffs.min()),
        'relative_gap_mean_pct': float(frac_gap.mean() * 100),
        'relative_gap_max_pct': float(frac_gap.max() * 100),
        'E_V_star_uniform': float(V_star.mean()),
        'E_V_pi_vfa_uniform': float(V_pi_vfa.mean()),
        'gap_uniform': float(V_pi_vfa.mean() - V_star.mean()),
        'gap_uniform_pct': float((V_pi_vfa.mean() - V_star.mean()) / abs(V_star.mean()) * 100),
        'near_empty_depot': near_empty_stats,
    }

    model_param = {
        'V_star': V_star,
        'pi_star': pi_star,
        'theta': theta,
        'Vhat': Vhat,
        'pi_vfa': pi_vfa,
        'V_pi_vfa': V_pi_vfa,
        'states': states,
        'comparison_summary': comparison_summary,
        'Thetabar_trajectory': Thetabar_trajectory,
    }

    all_inputs = capture_all_inputs(cfg)

    save_dict = {
        'format_shelve': {
            'model_param': model_param,
            'inputs': all_inputs,
        },
        'format_json': {
            'model_param': model_param,
            'inputs': all_inputs,
            'comparison_summary': comparison_summary,
            'run_time': run_time_str,
        },
        'format_cplex_model': {},
    }

    settings_md = (
        f"# Exact vs. VFA Approximation-Quality Evaluation\n\n"
        f"Run time: {run_time_str}\n\n"
        f"## Small instance\n"
        f"- N_LABS={cfg.N_LABS}, L_AGE={cfg.L_AGE}, K_CAPACITY={cfg.N_MAX}, "
        f"epochs/day={cfg.epochs_per_day}, gamma={cfg.GAMMA}\n"
        f"- {len(states)} states enumerated exactly\n\n"
        f"## Results\n"
        f"- Policy agreement: {comparison_summary['policy_agreement_pct']:.2f}% "
        f"({comparison_summary['policy_agreement_count']}/{len(states)})\n"
        f"- Mean/max exact suboptimality: "
        f"{comparison_summary['suboptimality_mean']:.4f} / "
        f"{comparison_summary['suboptimality_max']:.4f}\n"
        f"- Mean/max relative gap: "
        f"{comparison_summary['relative_gap_mean_pct']:.2f}% / "
        f"{comparison_summary['relative_gap_max_pct']:.2f}%\n"
        f"- E[V*] vs E[V^pi_VFA] (uniform): "
        f"{comparison_summary['E_V_star_uniform']:.4f} vs "
        f"{comparison_summary['E_V_pi_vfa_uniform']:.4f} "
        f"({comparison_summary['gap_uniform_pct']:.2f}% gap)\n"
        f"- Trained theta: {[round(float(x), 4) for x in theta]}\n"
    )

    short_tag = f"K{cfg.N_MAX}_E{cfg.epochs_per_day}_n{len(states)}"
    plot_theta_convergence(Thetabar_trajectory, cfg.N_LABS, cfg.L_AGE,
                           short_tag, _RUN_SEED)

    instance_tag = f"{short_tag}_seed{_RUN_SEED}"
    results_path = saveResultsFn(
        save_dict, settings_md,
        filename=f"evaluate_approximation_{instance_tag}"
    )
    print(f"\nResults saved to: {results_path}")
