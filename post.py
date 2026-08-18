"""
post.py
=======
Post-processing / reporting for a saved PSMD run (i.e., the output of
functions.functions.saveResultsFn, called at the end of main.py). Does NOT
retrain or re-simulate anything — everything printed and plotted here comes
straight out of the saved model_param / inputs / bound_summary / run_time
files.

What it does
------------
1. Prints a full report: config snapshot, final/best thetabar, geometry /
   Lipschitz / Cbar (Section 5.3), all three bound quantities (LP-based LB,
   certified SAA LB, Monte Carlo UB), and the myopic-baseline comparison.
2. Reproduces the two standard training figures exactly as main.py produces
   them (bounds-vs-iteration 4-panel figure, and the per-basis-function
   thetabar convergence figure) — from the saved Thetabar/LB/UB arrays, not
   from a rerun.
3. Adds two new summary figures: a feature-scale bar chart (sigma_b per
   basis function), and a bound/baseline comparison bar chart (LP LB vs
   certified SAA LB vs MC UB vs myopic).

Usage
-----
    python post.py                              # auto-detects latest ./results/ run
    python post.py results/<run_folder>/         # or point at a specific run's folder
    python post.py results/<run_folder>/model_param.json   # or a file inside it
"""

import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions.functions import lastFolder_fn
from main import plot_results, plot_thetas


# =============================================================================
# Locating and loading a saved run
# =============================================================================

def resolve_results_folder(arg_path=None):
    """Accepts a folder, a file inside it, or nothing (auto-detects the most
    recently modified run under ./results/, via functions.functions.lastFolder_fn)."""
    if arg_path is not None:
        if os.path.isfile(arg_path):
            return os.path.dirname(os.path.abspath(arg_path))
        return arg_path.rstrip('/')

    latest = lastFolder_fn()
    if latest is None:
        raise FileNotFoundError(
            "No run found under ./results/ — pass a run folder or file path explicitly."
        )
    return os.path.join("./results", latest)


def _load_json(folder, name):
    """Load `name`.json from folder. JSON-only, by design — saveResultsFn
    also writes a shelve (.db) copy of the same data, but that path pulls in
    Python's dbm backend, which is both unnecessary here (the JSON copy is
    already a complete, round-trippable record — see NumpyArrayEncoder in
    functions.functions) and a source of platform-dependent fragility
    (dbm.dumb in particular behaves inconsistently across systems)."""
    json_path = os.path.join(folder, f"{name}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found. Was this run saved with an older version "
            f"of saveResultsFn, or with a different filename?"
        )
    with open(json_path, 'r') as f:
        return json.load(f)


def load_run(folder):
    """Load everything saveResultsFn wrote for one run (JSON only)."""
    model_param = _load_json(folder, 'model_param')
    inputs_snap = _load_json(folder, 'inputs')

    bound_summary_path = os.path.join(folder, 'bound_summary.json')
    bound_summary = {}
    if os.path.exists(bound_summary_path):
        with open(bound_summary_path) as f:
            bound_summary = json.load(f)

    run_time_path = os.path.join(folder, 'run_time.json')
    run_time = None
    if os.path.exists(run_time_path):
        with open(run_time_path) as f:
            run_time = json.load(f)

    return model_param, inputs_snap, bound_summary, run_time


# =============================================================================
# Printing
# =============================================================================

def print_full_report(model_param, inputs_snap, bound_summary, run_time, folder):
    print("=" * 70)
    print(f"POST-RUN REPORT — {folder}")
    print("=" * 70)

    print(f"\nRun time: {run_time}")

    print("\n--- Problem size / config (from inputs.inputs snapshot) ---")
    for key in ['N_LABS', 'L_AGE', 'NO_BASIS_FN', 'N_STATE', 'GAMMA',
                'TAU_MAX', 'K_CAPACITY', 'T', 'ETA0', 'LAM0',
                'H_GRAD', 'N_SAMPLES', 'N_MH', 'N_INIT', 'EVAL_EVERY']:
        if key in inputs_snap:
            print(f"  {key:15s}: {inputs_snap[key]}")

    print("\n--- Theta (final = raw Polyak-Ruppert average at end of training;")
    print("           best  = the checkpoint iterate with the lowest simulated UB) ---")
    theta_final = np.array(model_param['thetabar_final'])
    theta_best  = np.array(model_param['thetabar_best'])
    for b in range(len(theta_final)):
        print(f"  theta_{b}: final={theta_final[b]: .6f}   best_ub_iterate={theta_best[b]: .6f}")

    print("\n--- Normalization / geometry (Sections 5.2 / 5.3) ---")
    for key, label in [('feature_sigma', 'feature sigma (per basis fn)'),
                        ('cost_scale',    'cost scale'),
                        ('geometry_n_dim','n = dim(S x U)'),
                        ('geometry_R',    'R (inscribed-ball radius)'),
                        ('geometry_diam', 'diam(S x U)'),
                        ('geometry_p_bar','p_bar (uniform ref. density)'),
                        ('geometry_L',    'L (Lipschitz constant)'),
                        ('geometry_Cbar', 'Cbar')]:
        val = model_param.get(key)
        print(f"  {label:32s}: {val}")

    print("\n--- Bounds ---")
    if 'tight_lb' in bound_summary:
        print(f"  Tightest same-iteration pair (iter {bound_summary['tight_iter']}): "
              f"LB={bound_summary['tight_lb']:.4f}  UB={bound_summary['tight_ub']:.4f}  "
              f"gap={bound_summary['tight_gap_pct']:.2f}%")
    if 'best_lb_independent' in bound_summary:
        print(f"  Best LB (independent, LP-based, uncertified) : "
              f"{bound_summary['best_lb_independent']:.4f}")
    if 'best_ub_independent' in bound_summary:
        print(f"  Best UB (independent, Monte Carlo)            : "
              f"{bound_summary['best_ub_independent']:.4f}")
    if 'ell_hat' in bound_summary:
        print(f"  Certified SAA lower bound (Sec 5.3)           : "
              f"{bound_summary['ell_hat']:.4f} +/- {bound_summary['ell_hat_se']:.4f}")
        print(f"    Note: valid (properly-derived Cbar) but typically looser")
        print(f"    than the LP-based LB above, which is not certified.")

    print("\n--- Policy quality ---")
    if 'myopic_cost' in bound_summary:
        print(f"  Myopic baseline cost : {bound_summary['myopic_cost']:.4f} "
              f"+/- {bound_summary['myopic_cost_se']:.4f}")
        print(f"  PSMD vs myopic       : "
              f"{bound_summary['psmd_vs_myopic_pct']:+.2f}% improvement")

    print("=" * 70)


# =============================================================================
# Plotting: reproduce the standard training figures from saved data
# =============================================================================

def replot_standard_figures(model_param):
    """Regenerates dispatching_bounds.png and thetabar_each.png exactly as
    main.py's plot_results / plot_thetas do — from the saved arrays, no
    retraining involved."""
    Thetabar   = np.array(model_param['Thetabar_trajectory'])
    LB_history = [tuple(r) for r in model_param['LB_history']]
    LB_best    = [tuple(r) for r in model_param['LB_best']]
    UB_history = [tuple(r) for r in model_param['UB_history']]
    UB_best    = [tuple(r) for r in model_param['UB_best']]

    plot_results(Thetabar, LB_history, LB_best, UB_history, UB_best)
    plot_thetas(Thetabar)


# =============================================================================
# Plotting: additional summary figures
# =============================================================================

def plot_feature_scale(model_param):
    sigma = model_param.get('feature_sigma')
    if sigma is None:
        return
    sigma = np.asarray(sigma, dtype=float)
    B = len(sigma)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(B), sigma, color=[f"C{b % 10}" for b in range(B)])
    ax.set_xticks(range(B))
    ax.set_xticklabels([f"θ_{b}" for b in range(B)])
    ax.set_ylabel(r"$\sigma_b$ (feature std under $\nu$)")
    ax.set_title("Feature normalization scale per basis function")
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig("post_feature_scale.png", dpi=150)
    plt.show()


def plot_bound_comparison(bound_summary):
    """Two panels: the certified SAA bound (Sec 5.3) is typically orders of
    magnitude more negative than the LP-based LB / MC UB / myopic cost —
    sharing one linear axis would squash those three into an unreadable
    sliver near the top, so it gets its own panel instead."""
    practical_labels, practical_values, practical_errs, practical_colors = [], [], [], []
    if 'best_lb_independent' in bound_summary:
        practical_labels.append("Best LB\n(LP, uncertified)")
        practical_values.append(bound_summary['best_lb_independent']); practical_errs.append(0)
        practical_colors.append('tab:blue')
    if 'best_ub_independent' in bound_summary:
        practical_labels.append("Best UB\n(Monte Carlo)")
        practical_values.append(bound_summary['best_ub_independent']); practical_errs.append(0)
        practical_colors.append('tab:red')
    if 'myopic_cost' in bound_summary:
        practical_labels.append("Myopic\nbaseline")
        practical_values.append(bound_summary['myopic_cost'])
        practical_errs.append(bound_summary.get('myopic_cost_se', 0))
        practical_colors.append('tab:gray')

    have_saa = 'ell_hat' in bound_summary
    if not practical_labels and not have_saa:
        print("No bound/baseline values found in bound_summary — skipping comparison plot.")
        return

    ncols = 2 if have_saa else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    axes = np.atleast_1d(axes)

    ax0 = axes[0]
    if practical_labels:
        ax0.bar(practical_labels, practical_values, yerr=practical_errs,
                capsize=4, color=practical_colors)
    ax0.axhline(0, color='k', lw=0.7)
    ax0.set_ylabel("Discounted cost")
    ax0.set_title("Practical comparison\n(LP LB, MC UB, myopic — comparable scale)")
    ax0.grid(True, axis='y', alpha=0.4)

    if have_saa:
        ax1 = axes[1]
        ax1.bar(["Certified SAA LB\n(Sec 5.3)"], [bound_summary['ell_hat']],
                yerr=[bound_summary.get('ell_hat_se', 0)], capsize=4, color='tab:purple')
        ax1.axhline(0, color='k', lw=0.7)
        ax1.set_ylabel("Discounted cost")
        ax1.set_title("Certified bound\n(separate axis — typically orders of\n"
                      "magnitude looser than the practical trio)")
        ax1.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig("post_bound_comparison.png", dpi=150)
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

def main(folder=None):
    """Callable entry point — use this directly from a notebook/Jupyter cell,
    where sys.argv is NOT reliable (see note in __main__ below):

        import post
        post.main()                          # auto-detect the latest run
        post.main("results/<run_folder>/")   # or a specific run
    """
    folder = resolve_results_folder(folder)

    model_param, inputs_snap, bound_summary, run_time = load_run(folder)

    print_full_report(model_param, inputs_snap, bound_summary, run_time, folder)

    replot_standard_figures(model_param)
    plot_feature_scale(model_param)
    plot_bound_comparison(bound_summary)

    print("\nFigures written: dispatching_bounds.png, thetabar_each.png "
          "(reproduced), post_feature_scale.png, post_bound_comparison.png (new)")


def _clean_argv():
    """Extract a real positional argument from sys.argv, ignoring anything
    that looks like a flag. This is necessary because Jupyter/IPython
    kernels populate sys.argv with their OWN launch arguments (typically
    something like '--f=/path/to/kernel-xxxx.json') whenever a script is
    run inside a notebook (e.g. via %run) — that is not a user-supplied
    argument to this script, and treating it as a results-folder path (as
    a naive `sys.argv[1] if len(sys.argv) > 1 else None` would) causes
    exactly the FileNotFoundError this function exists to avoid.
    If you're calling this from a notebook, prefer post.main(folder=...)
    directly over relying on sys.argv at all.
    """
    positional = [a for a in sys.argv[1:] if not a.startswith('-')]
    return positional[0] if positional else None


if __name__ == "__main__":
    main(_clean_argv())