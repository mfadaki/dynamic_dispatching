"""
diagnose_policy_disagreement_multiseed.py
============================================
Same idea as diagnose_policy_disagreement.py, but aggregated across ALL
seeds of one instance config, not just one run. Two things this adds that
a single-seed diagnosis can't:

1. Statistically correct aggregation: disagreement-by-bucket rates are
   computed by summing raw (n_disagree, n_total) counts across seeds
   BEFORE dividing, not by averaging per-seed percentages (which would
   silently mis-weight buckets that happen to differ in size... though
   here bucket sizes are identical across seeds since the state space is
   the same instance -- the real reason this matters is variance
   reduction: pooling counts across 3x the samples per bucket gives a
   much less noisy rate estimate than trusting any single seed's).

2. Cross-seed consistency: since every seed enumerates the SAME state
   space in the SAME order (see evaluate_approximation.py's
   enumerate_states -- deterministic itertools.product, independent of
   training randomness), state index i means the same physical state in
   every seed's saved results. This lets us ask, for each state, "in how
   many of the N seeds did pi_VFA disagree with pi* here?" -- a state
   that disagrees in every independently-trained seed is much stronger
   evidence of a structural VFA limitation than one that only disagreed
   once, which could just as easily be that seed's training noise.

Usage
-----
Just open this file in VS Code and click "Run Python File" (top right) or
press Ctrl+F5 -- no terminal, no arguments needed. Edit the variable below
first if you want a different instance:

    CONFIG_NAME = 'inputs_evaluate_approximation_L2'

Running from a terminal with explicit arguments still works exactly as
before, and OVERRIDES the variable above when given:
    python diagnose_policy_disagreement_multiseed.py inputs_evaluate_approximation_L2
        Aggregates every saved seed of this config found under ./results/.

    python diagnose_policy_disagreement_multiseed.py inputs_evaluate_approximation_L2 0,1,2
        Aggregates exactly the seeds listed (skips others even if present).
"""

import sys
import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# EDIT THIS LINE, then just click "Run Python File" in VS Code
# (only used when no command-line arguments are given -- running from a
# terminal with an argument, e.g. `python diagnose_policy_disagreement_multiseed.py foo`,
# overrides it).
# =============================================================================
CONFIG_NAME = 'inputs_evaluate_approximation_L2'   # which inputs_evaluate_approximation*.py to aggregate


# =============================================================================
# Loading
# =============================================================================

def find_runs(config_name, seeds=None):
    """All results/ folders for this config (optionally restricted to
    specific seeds), sorted by seed."""
    runs = []
    for folder in sorted(glob.glob("./results/*evaluate_approximation_*/")):
        summary_path = os.path.join(folder, "comparison_summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            s = json.load(f)
        if s.get('config_module') != config_name:
            continue
        if seeds is not None and s.get('run_seed') not in seeds:
            continue
        runs.append((s.get('run_seed'), folder))
    runs.sort(key=lambda x: x[0])
    return runs


def load_run(folder):
    with open(os.path.join(folder, "model_param.json")) as f:
        model_param = json.load(f)
    with open(os.path.join(folder, "inputs.json")) as f:
        cfg = json.load(f)
    return model_param, cfg


def parse_states(states, cfg):
    N_LABS, L_AGE, N_INV = cfg['N_LABS'], cfg['L_AGE'], cfg['N_INV']
    states = np.asarray(states, dtype=float)
    n = states[:, :N_INV].reshape(-1, N_LABS + 1, L_AGE)
    tau = states[:, N_INV]
    return n, tau


def action_label(a, n_labs):
    return "hold" if a == 0 else f"->lab{a}"


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    config_name = args[0] if len(args) >= 1 else CONFIG_NAME
    seeds = [int(x) for x in args[1].split(',')] if len(args) >= 2 else None

    runs = find_runs(config_name, seeds)
    if not runs:
        print(f"No saved runs found for config={config_name}"
              f"{f', seeds={seeds}' if seeds else ''} under ./results/.")
        sys.exit(1)

    print(f"Aggregating {len(runs)} seed(s) for {config_name}: "
          f"{[s for s, _ in runs]}")

    # Load everything; verify the state space is identical (same order)
    # across all seeds before pooling anything index-wise.
    all_pi_star, all_pi_vfa, all_V_star, all_V_vfa = [], [], [], []
    ref_states, cfg = None, None
    for seed, folder in runs:
        model_param, this_cfg = load_run(folder)
        states = np.asarray(model_param['states'])
        if ref_states is None:
            ref_states, cfg = states, this_cfg
        elif not np.allclose(states, ref_states):
            raise ValueError(
                f"seed={seed}'s state enumeration doesn't match the first "
                f"seed's -- these results aren't from the same instance "
                f"config, or enumerate_states() isn't as deterministic as "
                f"assumed. Aborting rather than silently misaligning states."
            )
        all_pi_star.append(np.asarray(model_param['pi_star'], dtype=int))
        all_pi_vfa.append(np.asarray(model_param['pi_vfa'], dtype=int))
        all_V_star.append(np.asarray(model_param['V_star']))
        all_V_vfa.append(np.asarray(model_param['V_pi_vfa']))

    n_seeds = len(runs)
    M = len(ref_states)
    n_actions = cfg['N_LABS'] + 1
    n_labs = cfg['N_LABS']
    n_parsed, tau = parse_states(ref_states, cfg)

    pi_star_mat = np.stack(all_pi_star)   # (n_seeds, M)
    pi_vfa_mat = np.stack(all_pi_vfa)     # (n_seeds, M)
    V_star_mat = np.stack(all_V_star)
    V_vfa_mat = np.stack(all_V_vfa)
    disagree_mat = pi_star_mat != pi_vfa_mat   # (n_seeds, M)

    print(f"\n{M} states x {n_seeds} seeds = {M * n_seeds} (state, seed) pairs")

    # ---- 1. Pooled confusion matrix (sum of counts, not average of rates) --
    print("\n" + "=" * 70)
    print("1. POOLED CONFUSION MATRIX (summed across all seeds)")
    print("=" * 70)
    Mconf = np.zeros((n_actions, n_actions), dtype=int)
    for k in range(n_seeds):
        for a_star, a_vfa in zip(pi_star_mat[k], pi_vfa_mat[k]):
            Mconf[a_star, a_vfa] += 1
    labels = [action_label(a, n_labs) for a in range(n_actions)]
    header = " " * 14 + "".join(f"{l:>10}" for l in labels)
    print(header)
    print("pi* \\ pi_VFA" + "-" * (len(header) - 13))
    for i, row_label in enumerate(labels):
        row = "".join(f"{Mconf[i, j]:>10d}" for j in range(n_actions))
        print(f"{row_label:<14}{row}")
    total = Mconf.sum()
    diag = np.trace(Mconf)
    print(f"\nPooled agreement: {diag}/{total} = {diag/total*100:.2f}%")
    lab_confusion = sum(Mconf[i, j] for i in range(1, n_actions)
                        for j in range(1, n_actions) if i != j)
    hold_confusion = total - diag - lab_confusion
    print(f"Of {total - diag} total disagreements: {lab_confusion} "
          f"({lab_confusion/(total-diag)*100:.1f}%) are lab-vs-lab "
          f"confusions, {hold_confusion} ({hold_confusion/(total-diag)*100:.1f}%) "
          f"are hold-vs-dispatch confusions")

    # ---- 2. Pooled disagreement-by-bucket -----------------------------------
    print("\n" + "=" * 70)
    print("2. POOLED DISAGREEMENT RATE BY BUCKET (counts summed across seeds)")
    print("=" * 70)

    def pooled_bucket_rates(values, label):
        uniq = sorted(set(np.round(values, 6).tolist()))
        print(f"\n{label}:")
        for u in uniq:
            mask = np.isclose(values, u)          # (M,) which states are in this bucket
            n_total = mask.sum() * n_seeds
            n_dis = disagree_mat[:, mask].sum()
            print(f"  {label}={u:<8.4g}  n=({mask.sum()} states x {n_seeds} seeds)"
                  f"={n_total:<5d}  disagreement={n_dis}/{n_total} "
                  f"= {n_dis/n_total*100:5.1f}%")

    depot_total = n_parsed[:, 0, :].sum(axis=1)
    lab_total = n_parsed[:, 1:, :].sum(axis=(1, 2))
    pooled_bucket_rates(depot_total, "depot total inventory")
    pooled_bucket_rates(lab_total, "combined lab inventory")
    pooled_bucket_rates(tau, "tau")

    # ---- 3. Cross-seed consistency: which states disagree in EVERY seed? ---
    print("\n" + "=" * 70)
    print("3. CROSS-SEED CONSISTENCY (the actually interesting part)")
    print("=" * 70)
    disagree_count_per_state = disagree_mat.sum(axis=0)   # (M,) how many seeds disagreed here

    for k in range(n_seeds, -1, -1):
        n_states_at_k = (disagree_count_per_state == k).sum()
        if n_states_at_k > 0:
            print(f"  Disagree in exactly {k}/{n_seeds} seeds: "
                  f"{n_states_at_k} states")

    always_disagree = np.where(disagree_count_per_state == n_seeds)[0]
    if len(always_disagree) > 0 and n_seeds >= 2:
        avg_gap = (V_vfa_mat - V_star_mat).mean(axis=0)
        print(f"\n{len(always_disagree)} states disagree in ALL {n_seeds} "
              f"independently-trained seeds -- these are the strongest")
        print(f"candidates for a genuine, structural VFA limitation rather "
              f"than training noise:")
        print(f"{'depot':<18}{'lab totals':<16}{'tau':>7}  {'pi*':>7}"
              f"  avg_gap_across_seeds")
        idx_sorted = always_disagree[np.argsort(-avg_gap[always_disagree])]
        for i in idx_sorted[:10]:
            depot_str = str(n_parsed[i, 0].astype(int).tolist())
            lab_str = str(n_parsed[i, 1:].sum(axis=1).astype(int).tolist())
            # pi* should agree across seeds since it's from the same exact LP;
            # report it once, plus each seed's pi_VFA choice
            vfa_choices = [action_label(int(pi_vfa_mat[k, i]), n_labs)
                           for k in range(n_seeds)]
            print(f"{depot_str:<18}{lab_str:<16}{tau[i]:>7.3f}  "
                  f"{action_label(int(pi_star_mat[0, i]), n_labs):>7}  "
                  f"{avg_gap[i]:>8.3f}   (pi_VFA per seed: {vfa_choices})")
    else:
        print("\nNo states disagree in every seed (or fewer than 2 seeds "
              "given) -- either disagreement is fairly seed-dependent, or "
              "you don't have enough seeds loaded yet to tell.")

    # ---- Plot: pooled disagreement rate by bucket ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, values, label in [
        (axes[0], depot_total, "Depot total inventory"),
        (axes[1], lab_total, "Combined lab inventory"),
        (axes[2], tau, "tau (time of day)"),
    ]:
        uniq = sorted(set(np.round(values, 6).tolist()))
        rates = []
        for u in uniq:
            mask = np.isclose(values, u)
            n_total = mask.sum() * n_seeds
            n_dis = disagree_mat[:, mask].sum()
            rates.append(n_dis / max(n_total, 1) * 100)
        ax.bar(range(len(uniq)), rates, color='tab:blue')
        ax.set_xticks(range(len(uniq)))
        ax.set_xticklabels([f"{u:.2g}" for u in uniq], rotation=45)
        ax.set_ylabel("Pooled disagreement rate %")
        ax.set_title(f"{label}\n(pooled across {n_seeds} seeds)")
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig("policy_disagreement_multiseed.png", dpi=150)
    plt.show()
    print(f"\nFigure written: policy_disagreement_multiseed.png")