# Experiment 1 single-seed pilot

Seed 0; full-IID train/validation/test = 50000/10000/10000; AdamW lr=0.001, weight_decay=0.0001, batch=512, updates=5000; lambda_cr=1.0, lambda_bal=0.01, num_directions=1. Checkpoints are selected only by validation prediction MSE.

| Model | Best Val Prediction MSE | Test Prediction MSE | 25-step Rollout MSE | Directional Response MSE | Jacobian Frobenius Error | Subspace Similarity | Projection Error | Routing Entropy | Effective Expert Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 6.47232e-06 | 6.62772e-06 | 1.64479e-05 | 3.44235e-05 | 6.9205e-05 | N/A | N/A | N/A | N/A |
| moe_a | 1.62486e-06 | 1.65006e-06 | 8.46034e-06 | 1.01145e-05 | 2.026e-05 | 0.994708 | 0.0104893 | 1.0986 | 2.99996 |
| moe_s | 9.59056e-07 | 1.00524e-06 | 2.59593e-06 | 5.37996e-06 | 1.06913e-05 | 0.998471 | 0.0030531 | 1.0985 | 2.99967 |
| ours | 3.85485e-07 | 3.91312e-07 | 2.58689e-06 | 1.30613e-06 | 2.60472e-06 | 0.999334 | 0.00132955 | 1.09855 | 2.99981 |

## Routing usage

- dense: N/A
- moe_a: 0.333692, 0.33112, 0.335188
- moe_s: 0.326468, 0.33587, 0.337662
- ours: 0.328193, 0.337235, 0.334572

## Ours validation loss trajectory

| Point | L_pred | L_CR | L_balance | Total |
|---|---:|---:|---:|---:|
| initial | 0.0544749 | 0.0172224 | 0.00754002 | 0.0717727 |
| final | 3.85485e-07 | 1.28637e-06 | 0.000179022 | 3.46208e-06 |

Best Ours checkpoint: step 5000.
