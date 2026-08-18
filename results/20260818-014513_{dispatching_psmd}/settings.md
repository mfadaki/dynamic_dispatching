# Dispatching PSMD — Run Settings and Summary

Run time: 00:33:60

## Problem size
- Labs (P): 2
- Age classes (L): 3
- Basis functions (B): 7
- State dimension: 10
- Discount factor (gamma): 0.95

## PSMD hyperparameters
- Iterations (T): 5000
- eta0: 0.001, lam0: 0.0001
- H (primal mini-batch): 10, N (transition samples): 50
- N_MH: 50, N_MH_TOTAL: 400, N_MH_KEEP: 200
- LP warm-start: N_INIT=200, LP_N_INIT=300, LP_N_EXOG=2000

## Final results
- Final thetabar: [4.7684, 0.1017, 0.1016, 0.0507, 0.04, 0.1261, 0.3462]
- Best thetabar (lowest UB): [4.7684, 0.1017, 0.1016, 0.0509, 0.0472, 0.2595, 0.3556]
- Tightest same-iteration bound pair (iter 1200): LB=12.2794, UB=12.0432, gap=-1.96%
- Myopic baseline cost: 14.6853 ± 0.5763
- PSMD vs myopic improvement: +17.99%