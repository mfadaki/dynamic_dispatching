from inputs.inputs import *
from classes.mdp import *
from classes.alp import *
from classes.gradient import *
from classes.primalupdater import *
from classes.dualupdater import *
from classes.mhsampler import *

mdp      = Mdp(inputs)
alp      = Alp(mdp)
gradient = Gradient(mdp, alp)

# ============================================================
# LP warm start
# ============================================================
def lp_warm_start(mdp, alp, n_init=200, N_lp=5000):
    from scipy.optimize import linprog
    np.random.seed(0)
    S = np.random.uniform(mdp.s_min, mdp.s_max, n_init)
    A = np.random.uniform(mdp.a_min, mdp.a_max, n_init)
    G = mdp.drawDemandSamples(N_lp)
    B = alp.B
    Alp_ = np.zeros((n_init, B))
    blp_ = np.zeros(n_init)
    for i in range(n_init):
        s_raw  = S[i] + A[i] - G
        s_next = np.clip(s_raw, mdp.s_min, mdp.s_max)
        cost   = (mdp.cp*A[i]
                  + mdp.ch*np.mean(np.maximum(s_next, 0))
                  + mdp.cb*np.mean(np.maximum(-s_next, 0))
                  + mdp.cd*np.mean(np.maximum(s_raw - mdp.s_max, 0))
                  + mdp.cl*np.mean(np.maximum(mdp.s_min - s_raw, 0)))
        E_phi_next = alp.expected_phi_next(S[i], A[i], G)
        phi_s      = alp.phi(S[i])
        Alp_[i, 0] = 1.0 - mdp.gamma
        Alp_[i, 1] = phi_s[1] - mdp.gamma * E_phi_next[1]
        Alp_[i, 2] = phi_s[2] - mdp.gamma * E_phi_next[2]
        blp_[i]    = cost
    res = linprog(-alp.E_phi, A_ub=Alp_, b_ub=blp_,
                  bounds=[(None, None)]*B, method='highs')
    if res.success:
        print(f"LP warm start: theta={np.round(res.x, 4)}, obj={-res.fun:.4f}")
        return res.x
    print("LP failed, using zeros")
    return np.zeros(B)


# ============================================================
# PSMD — parameters matching paper Section 7.1 and 7.2
#
# Paper settings (Section 7.1):
#   eta0   = 0.1       (not 0.01)
#   lam0   = 0.0001    (not 1.0)
#   H      = 10        (sa_samples per gradient, not 200)
#   N      = 50        (demand samples in f_hat, not 1000)
#   N_mh   = 50        (DSampleHist row size) — same as N
#   n_total= 400, n_keep= 200  (MH chain, matches MATLAB)
#
# Paper lower bound evaluation (Section 7.1):
#   H_lb   = 1000
#   N_lb   = 10000
# ============================================================
def run_psmd(mdp, alp, gradient,
             T       = 1000,
             eta0    = 0.1,      # paper value
             lam0    = 0.0001,   # paper value
             H       = 10,       # paper value for gradient
             N       = 50,       # paper value for f_hat
             N_mh    = 50,       # DSampleHist row size
             n_total = 400,
             n_keep  = 200,
             n_init  = 200):

    B     = alp.B
    gamma = mdp.gamma

    # LP warm start
    theta    = lp_warm_start(mdp, alp)
    thetabar = theta.copy()

    # Pre-generate DSampleHist (T, N_mh) — fixed demand rows
    np.random.seed(1)
    DSampleHist = np.vstack([mdp.drawDemandSamples(N_mh) for _ in range(T)])

    # InitialSample — grid + random for full coverage
    np.random.seed(2)
    s_grid = np.linspace(mdp.s_min, mdp.s_max, 14)
    a_grid = np.linspace(mdp.a_min, mdp.a_max, 14)
    ss, aa = np.meshgrid(s_grid, a_grid)
    n_grid = min(n_init // 2, len(ss.ravel()))
    S_grid = ss.ravel()[:n_grid]
    A_grid = aa.ravel()[:n_grid]
    S_rand = np.random.uniform(mdp.s_min, mdp.s_max, n_init - n_grid)
    A_rand = np.random.uniform(mdp.a_min, mdp.a_max, n_init - n_grid)
    S_init = np.concatenate([S_grid, S_rand])
    A_init = np.concatenate([A_grid, A_rand])

    # Alp_init, blp_init
    G_large  = mdp.drawDemandSamples(2000)
    Alp_init = np.zeros((n_init, B))
    blp_init = np.zeros(n_init)
    for i in range(n_init):
        s_raw  = S_init[i] + A_init[i] - G_large
        s_next = np.clip(s_raw, mdp.s_min, mdp.s_max)
        cost   = (mdp.cp*A_init[i]
                  + mdp.ch*np.mean(np.maximum(s_next, 0))
                  + mdp.cb*np.mean(np.maximum(-s_next, 0))
                  + mdp.cd*np.mean(np.maximum(s_raw - mdp.s_max, 0))
                  + mdp.cl*np.mean(np.maximum(mdp.s_min - s_raw, 0)))
        E_phi_next     = alp.expected_phi_next(S_init[i], A_init[i], G_large)
        phi_s          = alp.phi(S_init[i])
        Alp_init[i, 0] = 1.0 - gamma
        Alp_init[i, 1] = phi_s[1] - gamma * E_phi_next[1]
        Alp_init[i, 2] = phi_s[2] - gamma * E_phi_next[2]
        blp_init[i]    = cost

    # Dual with full history
    dual    = Dual(alp, T=T, N_mh=N_mh)
    dual.set_demand_history(DSampleHist)
    sampler = MHSampler(mdp, dual, proposal_std=0.2,
                        n_total=n_total, n_keep=n_keep)
    primal  = Primal(gradient)

    Theta    = np.zeros((T, B))
    Thetabar = np.zeros((T, B))

    for t in range(T):
        eta = eta0 / np.sqrt(t + 1)
        lam = lam0 / np.sqrt(t + 1)

        # Dual update
        dual.update(eta, lam, theta)

        # MH from most-violated constraint
        wc_sum = dual.weightcost[:t+1].sum()
        wt_sum = dual.weighttheta[:t+1, :].sum(axis=0)
        temprecord = (blp_init * wc_sum
                      - Alp_init[:, 1] * wt_sum[1]
                      - Alp_init[:, 2] * wt_sum[2]) / (1.0 - gamma)
        tempid     = np.argmax(temprecord)
        sa_samples, _, _ = sampler.sample(S_init[tempid], A_init[tempid])

        # Gradient — H samples from MH, N demand samples
        # Paper uses H=10 (s,a) samples and N=50 demand samples
        sa_for_grad = sa_samples[:H, :]          # first H of the n_keep kept
        demand_grad = mdp.drawDemandSamples(N)   # N demand samples
        grad        = gradient.compute_batch(sa_for_grad, demand_grad)
        theta       = primal.update(theta, grad, eta)

        Theta[t, :] = theta

        # Cumulative average
        if t == 0:
            thetabar = theta.copy()
        else:
            thetabar = (thetabar * (t + 1) + theta) / (t + 2)

        Thetabar[t, :] = thetabar

        if t % 50 == 0:
            print(f"t={t:4d} | theta={np.round(theta,3)} | thetabar={np.round(thetabar,3)}")

    print(f"\nFinal thetabar: {thetabar}")
    return thetabar, Theta, Thetabar


# ============================================================
# Run
# ============================================================
thetabar, Theta, Thetabar = run_psmd(
    mdp, alp, gradient,
    T       = 1000,
    eta0    = 0.1,
    lam0    = 0.0001,
    H       = 10,
    N       = 50,
    N_mh    = 50,
    n_total = 400,
    n_keep  = 200,
    n_init  = 200
)


# ============================================================
# Greedy policy
# ============================================================
def greedy_policy(mdp, alp, theta, s, num_actions=200):
    actions  = mdp.ActionSpace(num_actions)
    best_a, best_val = None, float("inf")
    G = mdp.drawDemandSamples(1000)
    for a in actions:
        c          = mdp.ContributionFn(s, a, G)
        E_phi_next = alp.expected_phi_next(s, a, G)
        value      = c + mdp.gamma * np.dot(theta, E_phi_next)
        if value < best_val:
            best_val, best_a = value, a
    return best_a


print("\nGreedy policy:")
for k in range(-10, 11):
    order = greedy_policy(mdp, alp, thetabar, k)
    print(f"  s={k:3d},  a={order:.4f}")


# ============================================================
# Convergence plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for b, ax in enumerate(axes):
    ax.plot(Thetabar[:, b], label=f"thetabar_{b}")
    ax.set_title(f"thetabar_{b}")
    ax.set_xlabel("Iteration")
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.savefig("theta_convergence.png", dpi=150)
plt.show()
