# Experiment 1 single-seed pilot

Seed 0; full-IID train/validation/test = 50000/10000/10000; AdamW lr=0.001, weight_decay=0.0001, batch=512, updates=10000; lambda_cr=1.0, lambda_bal=0.01, num_directions=1. Checkpoints are selected only by validation prediction MSE.

| Model | Best Val Prediction MSE | Test Prediction MSE | 25-step Rollout MSE | Directional Response MSE | Jacobian Frobenius Error | Subspace Similarity | Projection Error | Routing Entropy | Effective Expert Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2.29588e-06 | 2.36345e-06 | 9.39582e-06 | 2.20076e-05 | 4.39184e-05 | N/A | N/A | N/A | N/A |
| moe_a | 6.38589e-07 | 6.5326e-07 | 2.74652e-06 | 3.25808e-06 | 6.5528e-06 | 0.998475 | 0.00304481 | 1.09861 | 2.99999 |
| moe_s | 2.73496e-07 | 2.74459e-07 | 1.06079e-06 | 1.90953e-06 | 3.81888e-06 | 0.999328 | 0.00134186 | 1.09861 | 2.99999 |
| ours | 2.09783e-07 | 2.13938e-07 | 1.37124e-06 | 8.5145e-07 | 1.69488e-06 | 0.999574 | 0.000852039 | 1.09859 | 2.99992 |

## Routing usage

- dense: N/A
- moe_a: 0.33258, 0.333162, 0.334259
- moe_s: 0.332566, 0.333046, 0.334388
- ours: 0.333785, 0.330149, 0.336066

## Ours validation loss trajectory

| Point | L_pred | L_CR | L_balance | Total |
|---|---:|---:|---:|---:|
| initial | 0.0544749 | 0.0172224 | 0.00754002 | 0.0717727 |
| final | 3.40744e-07 | 1.47557e-06 | 3.35507e-05 | 2.15182e-06 |

Best Ours checkpoint: step 8500.
