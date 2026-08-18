"""
main.py  — Dynamic Dispatching Problem
=======================================
Run PSMD for the dispatching problem. Wire everything together.

Usage:  python main.py

To reproduce a run with different training randomness (e.g. to check
whether an outcome like "the trained policy barely dispatches" is specific
to one training seed or reproduces across seeds), edit RUN_SEED below and
rerun. RUN_SEED=0 reproduces the exact previous behaviour byte-for-byte;
any other integer gives an independently-reproducible training run (see
functions/psmd.py's run_psmd docstring for exactly what this does and does
not change).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from time import time

# Make sure imports resolve correctly when running from dispatching/
sys.path.insert(0, os.path.dirname(__file__))

import inputs.inputs as cfg
from classes.mdp     import MDP
from classes.alp     import ALP
from classes.bounds  import LowerBound, UpperBound
from functions.psmd  import run_psmd
from functions.functions import saveResultsFn, RunTime, convert_keys_to_strings

RUN_SEED = 0   # <-- change this to get an independent, reproducible training run


def feature_names_for(n_labs, l_age):
    """Human-readable basis-function names, matching the index map
    documented in inputs/inputs.py (B = L_AGE + 4):
        0: const
        1..L: depot_age1..depot_ageL
        L+1: lab_shortfall
        L+2: imbalance
        L+3: exp_risk
    Built from L_AGE/N_LABS rather than hardcoded so this still lines up
    correctly if the case-study instance size changes. Mirrors the
    identically-purposed helper in evaluate_approximation.py, kept
    separate rather than imported since main.py and evaluate_approximation.py
    are deliberately isolated from each other (see classes/*_evaluate_approximation.py's
    module docstrings)."""
    names = ['const'] + [f'depot_age{a+1}' for a in range(l_age)]
    names += ['lab_shortfall']
    if n_labs >= 2:
        names += ['imbalance', 'exp_risk']
    else:
        names += ['exp_risk']
    return names


# =============================================================================
# Greedy policy display
# =============================================================================

def print_policy(mdp, alp, thetabar, n_exog=100):
    """Print greedy action for a sample of states across varied tau/inventory."""
    ub   = UpperBound(mdp, alp)
    exog = mdp.sample_exog(n_exog)

    print("\nGreedy policy (sample states):")
    print(f"  {'State summary':40s}  Action")
    print("  " + "-"*55)

    # Sample states with varied tau to show policy coverage across the day
    # (no day index — the model is infinite-horizon and stationary, so tau
    # alone determines where a state sits within its (repeating) day).
    test_cases = []
    np.random.seed(99)
    for tau in [1.5, 4.0, 7.0]:
        for _ in range(3):
            s = mdp.sample_initial_state()
            s[mdp.TAU_IDX] = tau
            test_cases.append(s)
    # also add 2 pure random states
    for _ in range(2):
        test_cases.append(mdp.sample_initial_state())

    for s in test_cases[:10]:
        a = ub.greedy_action(s, thetabar, exog)
        n, tau = mdp.parse_state(s)
        depot_total = int(n[0].sum())
        lab_totals  = [int(n[p+1].sum()) for p in range(cfg.N_LABS)]
        lab_str     = " ".join([f"lab{p+1}={lab_totals[p]}"
                                for p in range(cfg.N_LABS)])
        action_str  = "no dispatch" if a == 0 else f"dispatch → lab {a}"
        print(f"  depot={depot_total:2d}  {lab_str}  τ={tau:.1f}  "
              f"→  {action_str}")


# =============================================================================
# Plotting
# =============================================================================

def plot_results(Thetabar, LB_history, LB_best, UB_history, UB_best):
    """Four-panel figure."""
    B   = Thetabar.shape[1]
    fig = plt.figure(figsize=(16, 10))

    # 1. First 6 thetabars (to avoid crowding)
    ax1 = fig.add_subplot(2, 2, 1)
    for b in range(min(B, 6)):
        ax1.plot(Thetabar[:, b], label=f"θ_{b}", alpha=0.8)
    ax1.set_title("Thetabar convergence (first 6)")
    ax1.set_xlabel("Iteration"); ax1.legend(fontsize=7); ax1.grid(True)

    # 2. Current bounds
    ax2 = fig.add_subplot(2, 2, 2)
    if LB_history:
        ax2.plot(*zip(*[(i, v) for i, v in LB_history]),
                 'b-o', markersize=4, label="Lower bound")
    if UB_history:
        ub_i = [r[0] for r in UB_history]
        ub_v = [r[1] for r in UB_history]
        ub_e = [r[2] for r in UB_history]
        ax2.errorbar(ub_i, ub_v, yerr=ub_e, fmt='r-s',
                     markersize=4, capsize=3, label="Policy cost")
    ax2.set_title("Bounds at current iteration")
    ax2.set_xlabel("Iteration"); ax2.legend(); ax2.grid(True)

    # 3. Best historic bounds
    ax3 = fig.add_subplot(2, 2, 3)
    if LB_best:
        ax3.plot(*zip(*LB_best), 'b-o', markersize=4, label="Best LB")
    if UB_best:
        ax3.plot(*zip(*UB_best), 'r-s', markersize=4, label="Best UB")
    ax3.set_title("Best historic bounds")
    ax3.set_xlabel("Iteration"); ax3.legend(); ax3.grid(True)

    # 4. Optimality gap
    ax4 = fig.add_subplot(2, 2, 4)
    if LB_history and UB_history:
        n = min(len(LB_history), len(UB_history))
        iters = [LB_history[i][0] for i in range(n)]
        gaps  = [(UB_history[i][1] - LB_history[i][1]) / abs(UB_history[i][1]) * 100
                 for i in range(n) if UB_history[i][1] != 0]
        ax4.plot(iters[:len(gaps)], gaps, 'g-^', markersize=6, label="Gap % (LP LB)")
        ax4.axhline(y=0, color='k', ls='--', alpha=0.3)
    ax4.set_title("Optimality gap % (LP lower bound, same thetabar)")
    ax4.set_xlabel("Iteration"); ax4.legend(); ax4.grid(True)

    plt.tight_layout()
    plt.savefig("dispatching_bounds.png", dpi=150)
    plt.show()


def plot_thetas(Thetabar):
    """One subplot per basis function."""
    B    = Thetabar.shape[1]
    cols = min(B, 7)
    rows = (B + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).ravel()

    for b in range(B):
        axes[b].plot(Thetabar[:, b], color=f"C{b % 10}", linewidth=1.0)
        axes[b].set_title(f"θ_{b}", fontsize=9)
        axes[b].grid(True, alpha=0.4)
        axes[b].set_xlabel("Iter", fontsize=7)
        final = Thetabar[-1, b]
        axes[b].text(0.98, 0.05, f"{final:.2f}",
                     transform=axes[b].transAxes, ha="right", fontsize=7)

    for b in range(B, len(axes)):
        axes[b].set_visible(False)

    plt.suptitle("Thetabar convergence (one panel per basis function)", fontsize=11)
    plt.tight_layout()
    plt.savefig("thetabar_each.png", dpi=150)
    plt.show()


def myopic_policy_cost(mdp, alp, num_traj=30, num_stage=200, n_exog=50):
    """Baseline: always dispatch to cheapest lab (lab 1) when depot non-empty."""
    from inputs.inputs import C_DISPATCH
    cheapest_lab = int(np.argmin(C_DISPATCH)) + 1
    class MyopicUB(UpperBound):
        def greedy_action(self, s, thetabar, exog):
            n, _ = mdp.parse_state(s)
            return cheapest_lab if n[0].sum() > 0 else 0
    mub = MyopicUB(mdp, alp)
    return mub.compute(np.zeros(alp.B), num_stage=num_stage,
                       num_traj=num_traj, n_exog_policy=n_exog)


def print_summary(LB_history, UB_history, LB_best, UB_best, mdp=None, alp=None,
                   ell_hat=None, ell_hat_se=None):
    """Prints the training summary AND returns it as a dict, so callers (e.g.
    the entry point's save_dict construction) can reuse these values without
    re-running the (Monte Carlo, non-trivial cost) myopic baseline a second
    time."""
    summary = {}
    if not LB_history or not UB_history:
        return summary
    n = min(len(LB_history), len(UB_history))
    # Paired, same-iteration gaps only — this is the quantity that actually
    # certifies LB <= UB held at every checkpoint's own thetabar.
    gaps = [(UB_history[i][1] - LB_history[i][1]) / abs(UB_history[i][1]) * 100
            for i in range(n) if UB_history[i][1] != 0]
    best_gap_idx = int(np.argmin(gaps)) if gaps else None
    best_gap     = gaps[best_gap_idx] if gaps else float('nan')

    # Independent running extrema (kept for reference), clearly labelled as
    # NOT necessarily from the same iteration — do not compare these two
    # directly as if they were a matched bound pair.
    best_ub_indep = min(v for _, v, _ in UB_history)
    best_lb_indep = max(v for _, v in LB_history)

    print(f"\n{'='*60}")
    if best_gap_idx is not None:
        tight_iter, tight_ub, _ = UB_history[best_gap_idx]
        _,          tight_lb    = LB_history[best_gap_idx]
        print(f"Tightest same-iteration bound pair (iter {tight_iter}):")
        print(f"  LB={tight_lb:.4f}  UB={tight_ub:.4f}  gap={best_gap:.2f}%")
        summary['tight_iter'] = tight_iter
        summary['tight_lb']   = tight_lb
        summary['tight_ub']   = tight_ub
        summary['tight_gap_pct'] = best_gap
    print(f"Best LB across all iterations (independent) : {best_lb_indep:.4f}")
    print(f"Best UB across all iterations (independent) : {best_ub_indep:.4f}")
    summary['best_lb_independent'] = best_lb_indep
    summary['best_ub_independent'] = best_ub_indep
    if ell_hat is not None:
        print(f"Certified SAA lower bound (Section 5.3)     : "
              f"{ell_hat:.4f} ± {ell_hat_se:.4f}")
        print(f"  Note: this bound is VALID (properly-derived Cbar) but much")
        print(f"  looser than the LP-based LB above, which is not certified.")
        summary['ell_hat']    = ell_hat
        summary['ell_hat_se'] = ell_hat_se
    print(f"  Note: the two lines above are running extrema from possibly")
    print(f"  DIFFERENT iterations and are not a matched bound pair — compare")
    print(f"  the tightest same-iteration pair above instead. LP LB is a")
    print(f"  valid but weak, sample-dependent lower bound on V*.")
    best_ub = best_ub_indep  # kept for the myopic-comparison line below
    if mdp is not None and alp is not None:
        m_val, m_se = myopic_policy_cost(mdp, alp)
        impr = (m_val - best_ub) / m_val * 100
        print(f"Myopic policy cost   : {m_val:.2f} ± {m_se:.2f}")
        print(f"PSMD vs myopic       : {impr:+.2f}% improvement")
        summary['myopic_cost']         = m_val
        summary['myopic_cost_se']      = m_se
        summary['psmd_vs_myopic_pct']  = impr
    print(f"{'='*60}")
    return summary


# =============================================================================
# Result-saving helpers
# =============================================================================

def capture_all_inputs(cfg_module):
    """Snapshot every constant-like attribute of inputs.inputs for
    reproducibility, so a saved run always carries the exact configuration
    it was produced under. Skips imported modules and callables (there
    shouldn't be either in inputs.inputs, but this guards against picking
    up e.g. 'np' from `import numpy as np`)."""
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


def save_csv_exports(results_path, model_param, summary, cfg):
    """Write the raw arrays behind every manuscript figure/table as plain
    CSV files, alongside the existing JSON/shelve saves in the same
    results_path folder.

    model_param.json already contains this data, but nested inside one
    JSON blob -- pulling a (T, B) trajectory or a (checkpoint, iter/lb/ub/se)
    history back out of that structure for a plotting tool outside this
    codebase (R, Excel, a LaTeX pgfplots/tikz pipeline) means writing
    custom parsing code every time. Flat CSVs with a header row are usable
    directly by any of those tools, at essentially no cost -- every array
    written here already exists in memory at the point this is called.

    Writes four files:
      thetabar_trajectory.csv : (T, B) Polyak-Ruppert average per iteration,
                                 one named column per basis function --
                                 drives plot_thetas/thetabar_each.png.
      bounds_history.csv      : LB, LB_best, UB, UB_se, UB_best, and a
                                 computed gap_pct, merged on their shared
                                 iteration checkpoints (verified identical
                                 across all four saved lists) -- drives
                                 plot_results/dispatching_bounds.png.
      feature_normalization.csv : per-basis-function sigma_b (Sec 5.2).
      scalars_summary.csv     : every single-value quantity the manuscript's
                                 tables cite -- geometry constants and Cbar
                                 (Sec 5.3), the certified SAA bound, the
                                 tightest same-iteration bound pair, the two
                                 independent running extrema, and the
                                 myopic-baseline comparison -- as tidy
                                 quantity/value pairs.
    """
    import csv

    names = feature_names_for(cfg.N_LABS, cfg.L_AGE)
    Thetabar = np.asarray(model_param['Thetabar_trajectory'])
    T, B = Thetabar.shape
    assert len(names) == B, (
        f"feature_names_for produced {len(names)} names but "
        f"Thetabar_trajectory has B={B} columns -- update feature_names_for "
        f"to match this instance's actual N_LABS/L_AGE before trusting the "
        f"column headers below."
    )

    # 1. Thetabar trajectory -- one row per PSMD iteration (1-indexed, so
    #    row 200 lines up with the iteration=200 checkpoint elsewhere).
    path = os.path.join(results_path, 'thetabar_trajectory.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration'] + [f'theta_{n}' for n in names])
        for t in range(T):
            w.writerow([t + 1] + list(Thetabar[t, :]))

    # 2. Bounds history -- LB/UB (+running-best variants) merged on their
    #    shared iteration checkpoints.
    lb_h, lb_b = model_param['LB_history'], model_param['LB_best']
    ub_h, ub_b = model_param['UB_history'], model_param['UB_best']
    iters_match = ([r[0] for r in lb_h] == [r[0] for r in lb_b]
                  == [r[0] for r in ub_h] == [r[0] for r in ub_b])
    assert iters_match, (
        "LB_history/LB_best/UB_history/UB_best don't share identical "
        "iteration checkpoints -- merging them positionally (as done below) "
        "would silently misalign values. Merge by iteration value instead "
        "before trusting bounds_history.csv."
    )
    path = os.path.join(results_path, 'bounds_history.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'LB', 'LB_best', 'UB', 'UB_se', 'UB_best', 'gap_pct'])
        for i in range(len(lb_h)):
            it = lb_h[i][0]
            lb, lb_best = lb_h[i][1], lb_b[i][1]
            ub, ub_se = ub_h[i][1], ub_h[i][2]
            ub_best = ub_b[i][1]
            gap_pct = (ub - lb) / abs(ub) * 100 if ub != 0 else float('nan')
            w.writerow([it, lb, lb_best, ub, ub_se, ub_best, gap_pct])

    # 3. Per-basis-function feature normalization scale.
    sigma = model_param.get('feature_sigma')
    if sigma is not None:
        path = os.path.join(results_path, 'feature_normalization.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['basis_index', 'feature_name', 'feature_sigma'])
            for b, s in enumerate(sigma):
                w.writerow([b, names[b] if b < len(names) else f'theta_{b}', s])

    # 4. Every other scalar the manuscript's tables cite, as key/value pairs.
    scalar_rows = [
        ('cost_scale',          model_param.get('cost_scale')),
        ('geometry_n_dim',      model_param.get('geometry_n_dim')),
        ('geometry_R',          model_param.get('geometry_R')),
        ('geometry_diam',       model_param.get('geometry_diam')),
        ('geometry_p_bar',      model_param.get('geometry_p_bar')),
        ('geometry_L',          model_param.get('geometry_L')),
        ('geometry_Cbar',       model_param.get('geometry_Cbar')),
        ('ell_hat',             model_param.get('ell_hat')),
        ('ell_hat_se',          model_param.get('ell_hat_se')),
        ('tight_iter',          summary.get('tight_iter')),
        ('tight_lb',            summary.get('tight_lb')),
        ('tight_ub',            summary.get('tight_ub')),
        ('tight_gap_pct',       summary.get('tight_gap_pct')),
        ('best_lb_independent', summary.get('best_lb_independent')),
        ('best_ub_independent', summary.get('best_ub_independent')),
        ('myopic_cost',         summary.get('myopic_cost')),
        ('myopic_cost_se',      summary.get('myopic_cost_se')),
        ('psmd_vs_myopic_pct',  summary.get('psmd_vs_myopic_pct')),
    ]
    path = os.path.join(results_path, 'scalars_summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['quantity', 'value'])
        for k, v in scalar_rows:
            if v is not None:
                w.writerow([k, v])

    print(f"\nCSV exports written to {results_path}:")
    print("  thetabar_trajectory.csv, bounds_history.csv, "
          "feature_normalization.csv, scalars_summary.csv")


def build_settings_markdown(cfg, thetabar, best_thetabar, summary, run_time_str):
    """Human-readable settings.md companion to the saved run — the same
    file saveResultsFn writes alongside the shelve/json artifacts."""
    lines = []
    lines.append("# Dispatching PSMD — Run Settings and Summary")
    lines.append(f"\nRun time: {run_time_str}\n")

    lines.append("## Problem size")
    lines.append(f"- Labs (P): {cfg.N_LABS}")
    lines.append(f"- Age classes (L): {cfg.L_AGE}")
    lines.append(f"- Basis functions (B): {cfg.NO_BASIS_FN}")
    lines.append(f"- State dimension: {cfg.N_STATE}")
    lines.append(f"- Discount factor (gamma): {cfg.GAMMA}\n")

    lines.append("## PSMD hyperparameters")
    lines.append(f"- Iterations (T): {cfg.T}")
    lines.append(f"- eta0: {cfg.ETA0}, lam0: {cfg.LAM0}")
    lines.append(f"- H (primal mini-batch): {cfg.H_GRAD}, "
                 f"N (transition samples): {cfg.N_SAMPLES}")
    lines.append(f"- N_MH: {cfg.N_MH}, N_MH_TOTAL: {cfg.N_MH_TOTAL}, "
                 f"N_MH_KEEP: {cfg.N_MH_KEEP}")
    lines.append(f"- LP warm-start: N_INIT={cfg.N_INIT}, "
                 f"LP_N_INIT={cfg.LP_N_INIT}, LP_N_EXOG={cfg.LP_N_EXOG}\n")

    lines.append("## Final results")
    lines.append(f"- Final thetabar: {[round(float(x), 4) for x in thetabar]}")
    lines.append(f"- Best thetabar (lowest UB): "
                 f"{[round(float(x), 4) for x in best_thetabar]}")
    if 'tight_lb' in summary:
        lines.append(f"- Tightest same-iteration bound pair (iter "
                     f"{summary['tight_iter']}): LB={summary['tight_lb']:.4f}, "
                     f"UB={summary['tight_ub']:.4f}, gap={summary['tight_gap_pct']:.2f}%")
    if 'ell_hat' in summary:
        lines.append(f"- Certified SAA lower bound: {summary['ell_hat']:.4f} "
                     f"± {summary['ell_hat_se']:.4f}")
    if 'myopic_cost' in summary:
        lines.append(f"- Myopic baseline cost: {summary['myopic_cost']:.4f} "
                     f"± {summary['myopic_cost_se']:.4f}")
        lines.append(f"- PSMD vs myopic improvement: "
                     f"{summary['psmd_vs_myopic_pct']:+.2f}%")
    return "\n".join(lines)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    start_time = time()

    print(f"Problem: {cfg.N_LABS} labs, {cfg.L_AGE} age classes, "
          f"infinite horizon (gamma={cfg.GAMMA})")
    print(f"State dim: {cfg.N_STATE},  B = {cfg.NO_BASIS_FN} basis functions")
    print(f"Action set: {cfg.ACTION_SET}\n")

    mdp = MDP()
    alp = ALP()
    lb  = LowerBound(mdp, alp)
    ub  = UpperBound(mdp, alp)

    thetabar, Thetabar, LB_history, LB_best, UB_history, UB_best, best_thetabar = \
        run_psmd(mdp, alp, lb, ub, cfg, run_seed=RUN_SEED)
    # run_psmd returns exactly these 7 values (see its own docstring/return
    # statement) -- it does not compute a certified SAA lower bound itself.
    # ell_hat/ell_hat_se are left as None here rather than guessed at; if you
    # have a certified-bound computation elsewhere (a different psmd.py, or
    # a separate script), wire its output in here instead of leaving these
    # as None. print_summary/save_csv_exports both already handle None
    # gracefully -- the certified-bound line is just omitted, everything
    # else still runs and saves correctly.
    ell_hat, ell_hat_se = None, None

    print_policy(mdp, alp, best_thetabar)
    summary = print_summary(LB_history, UB_history, LB_best, UB_best, mdp, alp,
                            ell_hat=ell_hat, ell_hat_se=ell_hat_se)

    # Export Excel report using the BEST thetabar (at lowest UB iteration)
    try:
        from functions.export_policy_excel import *
        export_policy_excel(mdp, alp, best_thetabar, n_days=5, seed=7)
    except ImportError:
        print("export_policy_excel.py not found — skipping Excel export")
    plot_results(Thetabar, LB_history, LB_best, UB_history, UB_best)
    plot_thetas(Thetabar)

    run_time_str = RunTime(start_time)

    # ── Save all inputs + all important results ──────────────────────────────
    all_inputs = capture_all_inputs(cfg)

    model_param = {
        'thetabar_final'      : thetabar,
        'thetabar_best'       : best_thetabar,
        'Thetabar_trajectory' : Thetabar,           # (T, B) full PSMD iterate history
        'LB_history'          : LB_history,          # [(iter, lb), ...]
        'LB_best'             : LB_best,             # [(iter, best_lb_so_far), ...]
        'UB_history'          : UB_history,          # [(iter, ub, ub_se), ...]
        'UB_best'             : UB_best,             # [(iter, best_ub_so_far), ...]
        'ell_hat'             : ell_hat,             # certified SAA lower bound (Sec 5.3)
        'ell_hat_se'          : ell_hat_se,
        # normalization constants the run was calibrated under (Sec 5.2)
        'feature_sigma'       : getattr(alp, '_phi_sigma_report', None),
        'cost_scale'          : getattr(alp, '_cost_scale_report', None),
        # geometry / Lipschitz / Cbar (Sec 5.3, eq:Cbar) actually used this run
        'geometry_n_dim'      : getattr(alp, 'n_dim', None),
        'geometry_R'          : getattr(alp, 'R', None),
        'geometry_diam'       : getattr(alp, 'diam', None),
        'geometry_p_bar'      : getattr(alp, 'p_bar', None),
        'geometry_L'          : getattr(alp, 'L', None),
        'geometry_Cbar'       : getattr(alp, 'Cbar', None),
    }

    save_dict = {
        'format_shelve': {
            # exact-dtype copies (numpy arrays, tuples) for precise reloading
            'model_param': model_param,
            'inputs'     : all_inputs,
        },
        'format_json': {
            'model_param': model_param,   # NumpyArrayEncoder handles ndarray/np scalars
            'inputs'     : all_inputs,
            'bound_summary': convert_keys_to_strings(summary),
            'run_time'   : run_time_str,
        },
        'format_cplex_model': {},   # this pipeline uses scipy.optimize.linprog, not CPLEX
    }

    settings_md = build_settings_markdown(cfg, thetabar, best_thetabar, summary,
                                          run_time_str)
    results_path = saveResultsFn(save_dict, settings_md, filename="dispatching_psmd")
    print(f"\nResults saved to: {results_path}")

    save_csv_exports(results_path, model_param, summary, cfg)