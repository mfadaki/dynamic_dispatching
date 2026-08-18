"""
inspect_disagreements_linebyline.py
=====================================
For every state where pi* != pi_VFA, shows not just WHAT they disagree on
but WHY: the VFA's per-action value estimate (c(s,u) + gamma*theta.E[phi(s')|s,u])
for every candidate action, broken down feature-by-feature, so you can see
exactly which feature's contribution is what's tipping the VFA's decision
away from the exact-optimal action.

Pure post-hoc analysis of an already-saved run -- reconstructs the greedy
policy's own reasoning using the saved theta and the exact transition
model (classes/mdp_evaluate_approximation.py, classes/alp_evaluate_approximation.py),
no retraining.

Usage
-----
    python inspect_disagreements_linebyline.py
        Diagnoses the most recently modified results/evaluate_approximation_*/ folder.

    python inspect_disagreements_linebyline.py results/<folder>/
        Diagnoses a specific saved run.

    python inspect_disagreements_linebyline.py results/<folder>/ 20
        Show at most 20 disagreement states (default: all of them).
"""

import sys
import os
import glob
import json
import importlib

import numpy as np


def find_latest_folder():
    candidates = glob.glob("./results/*evaluate_approximation_*/")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def action_label(a, n_labs):
    return "hold" if a == 0 else f"->lab{a}"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    folder = args[0] if len(args) >= 1 else find_latest_folder()
    max_states = int(args[1]) if len(args) >= 2 else None
    if folder is None:
        print("No evaluate_approximation results found under ./results/.")
        sys.exit(1)

    with open(os.path.join(folder, "model_param.json")) as f:
        model_param = json.load(f)
    with open(os.path.join(folder, "inputs.json")) as f:
        cfg_snapshot = json.load(f)

    config_module = model_param['comparison_summary']['config_module']
    print(f"Diagnosing: {folder}")
    print(f"Config: {config_module}")

    # Load the SAME small instance this run used, so mdp/alp behave
    # identically to how they did during training/comparison.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    small_cfg = importlib.import_module(f'inputs.{config_module}')
    sys.modules['inputs.inputs'] = small_cfg
    import inputs.inputs as cfg
    from classes.mdp_evaluate_approximation import MDP
    from classes.alp_evaluate_approximation import ALP

    mdp = MDP()
    alp = ALP()
    # Feature normalization must match what training used, or phi() values
    # (and hence theta.phi) won't line up with the saved theta -- recompute
    # with the SAME seed/sample count the run itself used (NORM_SEED,
    # NORM_N_SAMPLES), matching functions/psmd_evaluate_approximation.py.
    alp.calibrate_normalization(mdp, n_samples=getattr(cfg, 'NORM_N_SAMPLES', 20000),
                                seed=getattr(cfg, 'NORM_SEED', 12345))
    mdp.calibrate_cost_scale(n_samples=getattr(cfg, 'NORM_N_SAMPLES', 20000),
                             seed=getattr(cfg, 'NORM_SEED', 12345))

    states = np.asarray(model_param['states'])
    pi_star = np.asarray(model_param['pi_star'], dtype=int)
    pi_vfa = np.asarray(model_param['pi_vfa'], dtype=int)
    V_star = np.asarray(model_param['V_star'])
    theta = np.asarray(model_param['theta'])

    N_LABS, L_AGE, N_INV = cfg.N_LABS, cfg.L_AGE, cfg.N_INV
    n_all = states[:, :N_INV].reshape(-1, N_LABS + 1, L_AGE)
    tau_all = states[:, N_INV]

    feature_names = (['const'] +
                     [f'depot_age{a+1}' for a in range(L_AGE)] +
                     ['lab_shortfall_agg'] +
                     [f'lab_shortfall_lab{p+1}' for p in range(N_LABS)] +
                     ['imbalance', 'exp_risk'])
    assert len(feature_names) == alp.B, (
        f"feature_names has {len(feature_names)} entries but alp.B={alp.B} "
        f"-- update feature_names above to match this run's actual feature set."
    )

    disagree_idx = np.where(pi_star != pi_vfa)[0]
    if max_states:
        disagree_idx = disagree_idx[:max_states]

    print(f"\n{len(states)} states total, {(pi_star != pi_vfa).sum()} disagreements, "
          f"showing {len(disagree_idx)}\n")
    print("=" * 100)

    # A small, fixed exog sample for evaluating E_phi_next -- same role as
    # during training/comparison, just a fresh draw here since we only need
    # this for DIAGNOSTIC value estimates, not for training itself.
    np.random.seed(0)
    exog = mdp.sample_exog(500)

    for i in disagree_idx:
        s = states[i]
        n_s = n_all[i]
        tau = tau_all[i]
        depot = n_s[0].astype(int).tolist()
        labs = [n_s[1 + p].astype(int).tolist() for p in range(N_LABS)]

        print(f"\nSTATE idx={i}: depot={depot}  labs={labs}  tau={tau:.4f}")
        print(f"  pi*={action_label(int(pi_star[i]), N_LABS):<8}  "
              f"pi_VFA={action_label(int(pi_vfa[i]), N_LABS):<8}  "
              f"V*={V_star[i]:.4f}")

        # Recompute the VFA's own greedy-policy reasoning at this state:
        # for each candidate action, c(s,u) + gamma*theta.E[phi(s')|s,u],
        # broken down feature by feature so you can see exactly which
        # feature's contribution differs between the chosen and correct action.
        print(f"  {'action':<10}{'cost c(s,u)':>13}{'  + gamma*theta.E[phi]':>24}"
              f"{'  = greedy value':>18}")
        per_action_detail = {}
        for a in range(N_LABS + 1):
            c = float(mdp.cost(s[None, :], np.array([a]), exog)[0])
            E_phi_next = alp.E_phi_next(mdp, s[None, :], np.array([a]), exog)[0]
            contributions = mdp.gamma * theta * E_phi_next   # (B,) per-feature contribution
            continuation = float(contributions.sum())
            total = c + continuation
            per_action_detail[a] = (c, contributions, total)
            marker = ""
            if a == int(pi_star[i]):
                marker += " <- pi* (should be lowest)"
            if a == int(pi_vfa[i]):
                marker += " <- pi_VFA chose this"
            print(f"  {action_label(a, N_LABS):<10}{c:>13.4f}{continuation:>24.4f}"
                  f"{total:>18.4f}{marker}")

        # Feature-by-feature breakdown, chosen action vs correct action
        a_star, a_vfa = int(pi_star[i]), int(pi_vfa[i])
        c_star, contrib_star, _ = per_action_detail[a_star]
        c_vfa, contrib_vfa, _ = per_action_detail[a_vfa]
        diff = contrib_vfa - contrib_star
        print(f"  Feature contribution DIFFERENCE (VFA-chosen minus pi*-optimal action):")
        order = np.argsort(-np.abs(diff))
        for b in order:
            if abs(diff[b]) < 1e-6:
                continue
            print(f"    {feature_names[b]:<20} theta={theta[b]:>8.4f}  "
                  f"contributes {diff[b]:>+9.4f} to why VFA prefers "
                  f"{action_label(a_vfa, N_LABS)} over {action_label(a_star, N_LABS)}")

        print("-" * 100)