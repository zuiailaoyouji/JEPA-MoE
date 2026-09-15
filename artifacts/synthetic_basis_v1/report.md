# Synthetic basis-dynamics recovery

Protocol: 5000 updates, batch 512, AdamW lr=0.001, lambda_jac=0.1, 5 seeds. Checkpoints selected only by validation MSE.

## Aggregate results

| Model | IID one-step MSE | Held-out one-step MSE | IID rollout-25 MSE | Held-out rollout-25 MSE | Redundancy | Basis recovery |
|---|---:|---:|---:|---:|---:|---:|
| dense | 8.75034e-06 +/- 2.26e-06 | 6.97081e-06 +/- 2.64e-06 | 2.75525e-05 +/- 1.16e-05 | 5.85457e-05 +/- 3.68e-05 | n/a | n/a |
| vanilla_moe | 6.25015e-07 +/- 2.16e-07 | 8.80313e-07 +/- 4.69e-07 | 4.83934e-06 +/- 2.46e-06 | 8.04886e-06 +/- 5.98e-06 | 0.197452 +/- 0.121 | 0.746756 +/- 0.245 |
| jacobian_moe | 1.75476e-06 +/- 7.74e-07 | 2.82703e-06 +/- 2.39e-06 | 7.9127e-06 +/- 2.7e-06 | 3.21978e-05 +/- 3.54e-05 | 0.0976982 +/- 0.0338 | 0.583418 +/- 0.173 |

## Aggregate MoE diagnostics

| Model | abs cos(0,1) | abs cos(0,2) | abs cos(1,2) | route 0 | route 1 | route 2 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_moe | 0.196107 +/- 0.136 | 0.156406 +/- 0.0687 | 0.239842 +/- 0.19 | 0.330706 +/- 0.0727 | 0.34592 +/- 0.0604 | 0.323374 +/- 0.0818 |
| jacobian_moe | 0.102503 +/- 0.0369 | 0.0938475 +/- 0.0286 | 0.0967444 +/- 0.0455 | 0.369842 +/- 0.111 | 0.316169 +/- 0.115 | 0.313989 +/- 0.144 |

## Per-seed results

| Model | Seed | Best step | IID MSE | Held-out MSE | IID rollout | Held-out rollout | Redundancy | Recovery | Routing weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dense | 0 | 5000 | 9.16565e-06 | 8.9799e-06 | 2.20665e-05 | 9.79486e-05 | nan | nan | n/a |
| dense | 1 | 4750 | 1.05686e-05 | 8.93116e-06 | 3.11602e-05 | 8.23335e-05 | nan | nan | n/a |
| dense | 2 | 5000 | 5.49975e-06 | 3.35734e-06 | 1.5165e-05 | 1.50545e-05 | nan | nan | n/a |
| dense | 3 | 5000 | 1.09845e-05 | 8.63929e-06 | 4.56483e-05 | 7.34232e-05 | nan | nan | n/a |
| dense | 4 | 5000 | 7.53318e-06 | 4.94636e-06 | 2.37225e-05 | 2.39688e-05 | nan | nan | n/a |
| jacobian_moe | 0 | 5000 | 2.93372e-06 | 1.82923e-06 | 1.0187e-05 | 1.50572e-05 | 0.1369 | 0.5468 | [0.344, 0.475, 0.181] |
| jacobian_moe | 1 | 4750 | 1.02116e-06 | 1.32852e-06 | 4.441e-06 | 1.16906e-05 | 0.0971 | 0.4554 | [0.481, 0.198, 0.320] |
| jacobian_moe | 2 | 5000 | 1.08989e-06 | 1.36063e-06 | 9.71684e-06 | 1.57574e-05 | 0.0455 | 0.8879 | [0.333, 0.213, 0.455] |
| jacobian_moe | 3 | 4750 | 1.87051e-06 | 2.60776e-06 | 9.67115e-06 | 2.35134e-05 | 0.0942 | 0.5140 | [0.215, 0.327, 0.458] |
| jacobian_moe | 4 | 5000 | 1.85853e-06 | 7.00901e-06 | 5.54747e-06 | 9.49704e-05 | 0.1148 | 0.5130 | [0.476, 0.368, 0.156] |
| vanilla_moe | 0 | 5000 | 8.37323e-07 | 1.60722e-06 | 8.69679e-06 | 1.794e-05 | 0.3543 | 0.8863 | [0.322, 0.372, 0.306] |
| vanilla_moe | 1 | 5000 | 8.342e-07 | 1.0544e-06 | 5.89197e-06 | 8.15511e-06 | 0.2112 | 0.4562 | [0.365, 0.415, 0.219] |
| vanilla_moe | 2 | 4750 | 6.34559e-07 | 6.69517e-07 | 3.36071e-06 | 4.41591e-06 | 0.2594 | 0.5068 | [0.380, 0.252, 0.368] |
| vanilla_moe | 3 | 4750 | 3.56335e-07 | 6.72306e-07 | 2.84428e-06 | 7.26716e-06 | 0.1210 | 0.9079 | [0.208, 0.358, 0.435] |
| vanilla_moe | 4 | 4250 | 4.62656e-07 | 3.98128e-07 | 3.40296e-06 | 2.4661e-06 | 0.0414 | 0.9765 | [0.378, 0.333, 0.288] |

## Preliminary-support criterion

IID mean MSE no more than 5% worse, lower mean redundancy, higher mean basis recovery, and lower mean held-out one-step and rollout MSE.

- FAIL: `iid_not_more_than_5_percent_worse`
- PASS: `lower_mean_expert_redundancy`
- FAIL: `higher_mean_basis_recovery`
- FAIL: `lower_mean_heldout_one_step_mse`
- FAIL: `lower_mean_heldout_rollout_mse`

Paired-seed wins: held-out one-step 0/5; held-out rollout 1/5.

Overall: does not meet the preliminary-support criterion.

Full pairwise cosines, routing weights, recovery matrices, and Hungarian matches are preserved in `summary.json` and `results/*.json`.
