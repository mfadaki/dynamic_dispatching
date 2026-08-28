# Dispatching PSMD — Run Settings and Summary

Run time: 08:13:00

## Problem size
- Labs (P): 2
- Age classes (L): 3
- Basis functions (B): 7
- State dimension: 10
- Discount factor (gamma): 0.95

## PSMD hyperparameters
- Iterations (T): 50000
- eta0: 0.001, lam0: 0.0001
- H (primal mini-batch): 10, N (transition samples): 50
- N_MH: 50, N_MH_TOTAL: 400, N_MH_KEEP: 200
- LP warm-start: N_INIT=200, LP_N_INIT=300, LP_N_EXOG=2000

## Final results
- Final thetabar: [5.0807, 0.0175, 0.0183, 0.0088, 0.0075, 0.0413, 0.0782]
- Best thetabar (lowest UB): [5.0807, 0.0175, 0.0189, 0.0088, 0.0077, 0.0666, 0.0818]
- Tightest same-iteration bound pair (iter 31800): LB=18.3794, UB=65.9678, gap=72.14%
- Certified SAA lower bound: 0.3246 ± 0.0085
- Myopic baseline cost: 110.3387 ± 11.6382
- PSMD vs myopic improvement: +69.05%