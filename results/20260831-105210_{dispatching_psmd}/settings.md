# Dispatching PSMD — Run Settings and Summary

Run time: 00:14:12

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
- Final thetabar: [5.0807, 0.0175, 0.0221, 0.0097, 0.0078, 0.1142, 0.0886]
- Best thetabar (lowest UB): [5.0807, 0.0175, 0.0223, 0.0097, 0.0078, 0.1191, 0.0893]
- Tightest same-iteration bound pair (iter 3600): LB=10.2496, UB=83.6485, gap=87.75%
- Certified SAA lower bound: -252.1802 ± 0.0000
- Myopic baseline cost: 110.3387 ± 11.6382
- PSMD vs myopic improvement: +34.89%