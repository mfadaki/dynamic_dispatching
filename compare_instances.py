"""
compare_instances.py
======================
Reads MULTIPLE saved evaluate_approximation.py runs (one per instance size)
and compares policy agreement between the exact optimal policy and the
VFA-greedy policy across them.

Does NOT run anything itself -- purely extracts and compares from what's
already in ./results/. Run evaluate_approximation.py once per instance size
FIRST, e.g.:

    python evaluate_approximation.py inputs_evaluate_approximation_small
    python evaluate_approximation.py inputs_evaluate_approximation
    python evaluate_approximation.py inputs_evaluate_approximation_large

each of which writes its own results/evaluate_approximation_K{...}_E{...}_n{...}/
folder (see evaluate_approximation.py's saveResultsFn call). Then run this
script to compare them.

Usage
-----
    python compare_instances.py
        Auto-detects every results/evaluate_approximation_*/ folder and
        compares all of them, sorted by instance size (n_states).

    python compare_instances.py results/foo/ results/bar/ results/baz/
        Compares exactly the folders given, in the order given.
"""

import sys
import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt


def find_result_folders():
    """Every results/evaluate_approximation_*/ folder, sorted by n_states
    (smallest instance first) using each folder's own comparison_summary.json."""
    candidates = sorted(glob.glob("./results/*evaluate_approximation_*/"))
    if not candidates:
        # fall back to the single un-tagged folder name used before instance
        # sizing was added (filename="evaluate_approximation", no K/E/n tag)
        candidates = sorted(glob.glob("./results/*evaluate_approximation}*/"))
    return candidates


def load_summary(folder):
    path = os.path.join(folder, "comparison_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- is this a results folder written by "
            f"evaluate_approximation.py's saveResultsFn?"
        )
    with open(path) as f:
        summary = json.load(f)
    summary['_folder'] = folder
    return summary


def print_comparison_table(summaries):
    print("=" * 100)
    print("POLICY COMPLIANCE ACROSS INSTANCE SIZES (exact pi* vs VFA-greedy pi_VFA)")
    print("=" * 100)

    header = (f"{'Instance':<14} {'Seed':>5} {'n_states':>9} {'Agreement %':>12} "
              f"{'Mean gap %':>11} {'Max gap %':>10} {'Uniform E[gap] %':>17}")
    print(header)
    print("-" * len(header))

    for s in summaries:
        K = s.get('K_CAPACITY', '?')
        E = s.get('epochs_per_day', '?')
        seed = s.get('run_seed', '?')
        label = f"K={K},E={E}"
        print(f"{label:<14} {seed:>5} {s['n_states']:>9d} "
              f"{s['policy_agreement_pct']:>11.2f}% "
              f"{s['relative_gap_mean_pct']:>10.2f}% "
              f"{s['relative_gap_max_pct']:>9.2f}% "
              f"{s['gap_uniform_pct']:>16.2f}%")

    print("=" * 100)
    print("\nNote: if several rows share the same Instance and n_states but different")
    print("Seed, those are repeated runs of ONE instance (see run_multiseed.py for the")
    print("proper mean+/-std aggregation across them) -- not genuinely different problems.")
    print("\nInterpretation:")
    print("  Agreement %      : fraction of states where pi*(s) == pi_VFA(s)")
    print("  Mean/Max gap %   : per-state relative suboptimality (V^pi_VFA - V*)/|V*|,")
    print("                     averaged / maxed over all states")
    print("  Uniform E[gap] % : gap between the uniformly-averaged value functions,")
    print("                     (E[V^pi_VFA] - E[V*]) / |E[V*]|")
    print("  If agreement stays roughly flat (or improves) and the gaps don't grow")
    print("  as n_states increases, that's evidence the 7-parameter VFA scales")
    print("  reasonably rather than only working by coincidence on the very")
    print("  smallest instance.")


def plot_comparison(summaries, out_path="instance_comparison.png"):
    n_states = [s['n_states'] for s in summaries]
    agreement = [s['policy_agreement_pct'] for s in summaries]
    mean_gap = [s['relative_gap_mean_pct'] for s in summaries]
    max_gap = [s['relative_gap_max_pct'] for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    ax0.plot(n_states, agreement, 'o-', color='tab:blue')
    ax0.set_xlabel("Number of states (instance size)")
    ax0.set_ylabel("Policy agreement %")
    ax0.set_title("Exact vs. VFA-greedy policy agreement")
    ax0.grid(True, alpha=0.4)
    ax0.set_ylim(0, 100)

    ax1 = axes[1]
    ax1.plot(n_states, mean_gap, 'o-', color='tab:green', label='mean relative gap')
    ax1.plot(n_states, max_gap, 's--', color='tab:red', label='max relative gap')
    ax1.set_xlabel("Number of states (instance size)")
    ax1.set_ylabel("Relative value gap %")
    ax1.set_title("VFA suboptimality vs. instance size")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nFigure written: {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('-')]

    folders = args if args else find_result_folders()
    if not folders:
        print("No evaluate_approximation results found under ./results/.")
        print("Run evaluate_approximation.py at least once first, e.g.:")
        print("    python evaluate_approximation.py inputs_evaluate_approximation_small")
        print("    python evaluate_approximation.py inputs_evaluate_approximation")
        print("    python evaluate_approximation.py inputs_evaluate_approximation_large")
        sys.exit(1)

    summaries = [load_summary(f) for f in folders]
    summaries.sort(key=lambda s: s['n_states'])

    print(f"Found {len(summaries)} instance(s):")
    for s in summaries:
        print(f"  {s['_folder']}  (n_states={s['n_states']})")
    print()

    print_comparison_table(summaries)

    if len(summaries) >= 2:
        plot_comparison(summaries)
    else:
        print("\n(Only one instance found -- run at least two different sizes "
              "to get a comparison plot.)")
