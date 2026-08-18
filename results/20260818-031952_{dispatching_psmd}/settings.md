# Dispatching PSMD — Run Settings and Summary

Run time: 00:26:27

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
- Final thetabar: [1.7549, 0.7444, 0.7431, 0.3699, 0.3127, 0.4375, 3.0717]
- Best thetabar (lowest UB): [1.7549, 0.7444, 0.7431, 0.3699, 0.3157, 0.5161, 3.0717]
- Tightest same-iteration bound pair (iter 2800): LB=6.3470, UB=2089.9797, gap=99.70%
- Myopic baseline cost: 2553.2446 ± 363.0840
- PSMD vs myopic improvement: +18.14%