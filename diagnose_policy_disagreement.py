"""
diagnose_policy_disagreement.py
=================================
State-by-state comparison of the exact optimal policy pi* against the
VFA-greedy policy pi_VFA, for ONE already-saved evaluate_approximation.py
run. Answers "WHERE do they disagree, and does it matter?" rather than
just reporting one aggregate agreement percentage.

Pure post-hoc analysis -- reads model_param.json + inputs.json from a
results/ folder already written by evaluate_approximation.py. Does not
retrain or re-solve anything.

What it reports
----------------
1. Confusion matrix: for every state, (pi*(s), pi_VFA(s)) -- so you can see
   not just THAT they disagree, but which specific mistake is most common
   (e.g. "exact says dispatch to lab 2, VFA says hold" vs the reverse).
2. Disagreement rate broken down by depot inventory level, by each lab's
   inventory level, and by tau (time of day) -- to see whether mismatches
   cluster in a particular region of the state space rather than being
   spread uniformly.
3. Whether disagreement states carry a bigger value cost than agreement
   states (they should, if disagreement is actually costing you
   something rather than being incidental -- e.g. ties where several
   actions are all close to optimal).
4. The concrete states with the LARGEST value gap under disagreement --
   the specific situations most worth understanding.

Usage
-----
    python diagnose_policy_disagreement.py
        Auto-detects the most recently modified results/evaluate_approximation_*/
        folder.

    python diagnose_policy_disagreement.py results/<folder>/
        Diagnoses a specific saved run.
"""

import sys
import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Loading
# =============================================================================

def find_latest_folder():
    candidates = glob.glob("./results/*evaluate_approximation_*/")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_run(folder):
    with open(os.path.join(folder, "model_param.json")) as f:
        model_param = json.load(f)
    with open(os.path.join(folder, "inputs.json")) as f:
        cfg = json.load(f)
    return model_param, cfg


def parse_states(states, cfg):
    """states: (M, N_STATE) array -> (n: (M, N_LABS+1, L_AGE), tau: (M,))"""
    N_LABS, L_AGE, N_INV = cfg['N_LABS'], cfg['L_AGE'], cfg['N_INV']
    states = np.asarray(states, dtype=float)
    n = states[:, :N_INV].reshape(-1, N_LABS + 1, L_AGE)
    tau = states[:, N_INV]
    return n, tau


def action_label(a, n_labs):
    return "hold" if a == 0 else f"->lab{a}"


# =============================================================================
# Diagnostics
# =============================================================================

def confusion_matrix(pi_star, pi_vfa, n_actions):
    M = np.zeros((n_actions, n_actions), dtype=int)
    for a_star, a_vfa in zip(pi_star, pi_vfa):
        M[a_star, a_vfa] += 1
    return M


def print_confusion_matrix(M, n_labs):
    labels = [action_label(a, n_labs) for a in range(M.shape[0])]
    header = " " * 14 + "".join(f"{l:>10}" for l in labels)
    print(header)
    print("pi* \\ pi_VFA" + "-" * (len(header) - 13))
    for i, row_label in enumerate(labels):
        row = "".join(f"{M[i, j]:>10d}" for j in range(M.shape[1]))
        print(f"{row_label:<14}{row}")
    total = M.sum()
    diag = np.trace(M)
    print(f"\nDiagonal (agreement): {diag}/{total} = {diag/total*100:.2f}%")
    off_diag_by_row = M.sum(axis=1) - np.diag(M)
    for i, row_label in enumerate(labels):
        if off_diag_by_row[i] > 0:
            worst_col = np.argmax(M[i] - (np.eye(M.shape[0])[i] * M[i, i]))
            print(f"  When pi* says '{row_label}' and they disagree "
                  f"({off_diag_by_row[i]} states), pi_VFA most often says "
                  f"'{action_label(worst_col, n_labs)}' ({M[i, worst_col]}x)")


def disagreement_by_bucket(values, disagree_mask, label, bucket_fn=None):
    """values: (M,) some scalar characteristic of each state (e.g. depot
    total inventory, or tau). Reports disagreement RATE within each
    distinct value of `values` (or of bucket_fn(values) if given)."""
    keys = bucket_fn(values) if bucket_fn else values
    uniq = sorted(set(np.round(keys, 6).tolist()))
    print(f"\nDisagreement rate by {label}:")
    for k in uniq:
        mask = np.isclose(keys, k)
        n_total = mask.sum()
        n_dis = (mask & disagree_mask).sum()
        print(f"  {label}={k:<8.4g}  n_states={n_total:<5d}  "
              f"disagreement={n_dis}/{n_total} = {n_dis/n_total*100:5.1f}%")


def value_cost_of_disagreement(V_star, V_pi_vfa, disagree_mask):
    gap = V_pi_vfa - V_star
    agree_gap = gap[~disagree_mask]
    disagree_gap = gap[disagree_mask]
    print(f"\nValue gap (V_pi_VFA - V*), agreement vs disagreement states:")
    print(f"  Agreement states    (n={len(agree_gap):3d}): "
          f"mean={agree_gap.mean():.4f}  max={agree_gap.max():.4f}")
    print(f"  Disagreement states (n={len(disagree_gap):3d}): "
          f"mean={disagree_gap.mean():.4f}  max={disagree_gap.max():.4f}")
    if agree_gap.mean() > 0:
        ratio = disagree_gap.mean() / max(agree_gap.mean(), 1e-8)
        print(f"  -> disagreement states cost {ratio:.1f}x more, on average, "
              f"than agreement states")
    else:
        print("  -> agreement states have ~zero value gap, as expected "
              "(pi_VFA happens to pick the optimal action there)")


def print_worst_examples(states, n, tau, pi_star, pi_vfa, V_star, V_pi_vfa,
                         disagree_mask, cfg, top_k=10):
    gap = V_pi_vfa - V_star
    idx = np.where(disagree_mask)[0]
    idx = idx[np.argsort(-gap[idx])][:top_k]

    n_labs = cfg['N_LABS']
    print(f"\nTop {min(top_k, len(idx))} costliest disagreement states "
          f"(largest V_pi_VFA - V* among mismatches):")
    print(f"{'depot':<18}{'lab totals':<16}{'tau':>7}  {'pi*':>7}{'pi_VFA':>9}"
          f"{'V*':>8}{'V_VFA':>8}{'gap':>8}")
    for i in idx:
        depot_str = str(n[i, 0].astype(int).tolist())
        lab_totals = n[i, 1:].sum(axis=1).astype(int).tolist()
        print(f"{depot_str:<18}{str(lab_totals):<16}{tau[i]:>7.3f}  "
              f"{action_label(int(pi_star[i]), n_labs):>7}"
              f"{action_label(int(pi_vfa[i]), n_labs):>9}"
              f"{V_star[i]:>8.3f}{V_pi_vfa[i]:>8.3f}{gap[i]:>8.3f}")


def plot_diagnostics(n, tau, disagree_mask, out_path="policy_disagreement.png"):
    depot_total = n[:, 0, :].sum(axis=1)
    lab_total = n[:, 1:, :].sum(axis=(1, 2))

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
            rates.append((disagree_mask & mask).sum() / max(mask.sum(), 1) * 100)
        ax.bar(range(len(uniq)), rates, color='tab:orange')
        ax.set_xticks(range(len(uniq)))
        ax.set_xticklabels([f"{u:.2g}" for u in uniq], rotation=45)
        ax.set_ylabel("Disagreement rate %")
        ax.set_title(label)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nFigure written: {out_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    folder = args[0] if args else find_latest_folder()
    if folder is None:
        print("No evaluate_approximation results found under ./results/.")
        sys.exit(1)

    print(f"Diagnosing: {folder}")
    model_param, cfg = load_run(folder)

    states = np.asarray(model_param['states'])
    pi_star = np.asarray(model_param['pi_star'], dtype=int)
    pi_vfa = np.asarray(model_param['pi_vfa'], dtype=int)
    V_star = np.asarray(model_param['V_star'])
    V_pi_vfa = np.asarray(model_param['V_pi_vfa'])

    n, tau = parse_states(states, cfg)
    disagree_mask = pi_star != pi_vfa
    n_actions = cfg['N_LABS'] + 1

    print(f"\n{len(states)} states, {disagree_mask.sum()} disagreements "
          f"({disagree_mask.mean()*100:.2f}%)")

    print("\n" + "=" * 70)
    print("1. CONFUSION MATRIX (rows = pi*, columns = pi_VFA)")
    print("=" * 70)
    M = confusion_matrix(pi_star, pi_vfa, n_actions)
    print_confusion_matrix(M, cfg['N_LABS'])

    print("\n" + "=" * 70)
    print("2. WHERE DO DISAGREEMENTS CLUSTER?")
    print("=" * 70)
    depot_total = n[:, 0, :].sum(axis=1)
    lab_totals_combined = n[:, 1:, :].sum(axis=(1, 2))
    disagreement_by_bucket(depot_total, disagree_mask, "depot total inventory")
    disagreement_by_bucket(lab_totals_combined, disagree_mask, "combined lab inventory")
    disagreement_by_bucket(tau, disagree_mask, "tau")

    print("\n" + "=" * 70)
    print("3. DOES DISAGREEMENT ACTUALLY COST VALUE?")
    print("=" * 70)
    value_cost_of_disagreement(V_star, V_pi_vfa, disagree_mask)

    print("\n" + "=" * 70)
    print("4. COSTLIEST SPECIFIC MISMATCHES")
    print("=" * 70)
    print_worst_examples(states, n, tau, pi_star, pi_vfa, V_star, V_pi_vfa,
                         disagree_mask, cfg)

    plot_diagnostics(n, tau, disagree_mask)