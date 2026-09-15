# Exact vs. VFA Approximation-Quality Evaluation

Run time: 00:27:43

## Small instance
- N_LABS=2, L_AGE=2, K_CAPACITY=2, epochs/day=3, gamma=0.95
- 2187 states enumerated exactly

## Results
- Policy agreement: 86.24% (1886/2187)
- Mean/max exact suboptimality: 0.0438 / 0.1687
- Mean/max relative gap: 1.41% / 8.32%
- E[V*] vs E[V^pi_VFA] (uniform): 3.2719 vs 3.3157 (1.34% gap)
- Trained theta: [0.4344, 0.0244, 0.0229, 0.0163, 0.0354, 0.1429]
