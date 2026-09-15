# Full-IID synthetic dynamics-basis experiment

Protocol: x ~ Uniform(-0.5, 0.5)^4 and a ~ Uniform(-1, 1)^2 without exclusions; 50000/10000/10000 train/validation/test samples, 5000 updates, batch 512, AdamW lr=0.001, lambda_jac=0.1, 5 seeds. Checkpoints selected only by validation MSE.

## Aggregate results

| Model | IID one-step MSE | IID rollout-25 MSE | Redundancy | Subspace principal similarity | Projection error | Sample routing entropy | Usage entropy | Min usage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 8.37696e-06 +/- 2.03e-06 | 2.91313e-05 +/- 5.83e-06 | n/a | n/a | n/a | n/a | n/a | n/a |
| vanilla_moe | 6.0411e-07 +/- 3e-07 | 3.57194e-06 +/- 1.42e-06 | 0.185256 +/- 0.0784 | 0.81682 +/- 0.185 | 0.251023 +/- 0.229 | 0.74642 +/- 0.0746 | 0.978902 +/- 0.02 | 0.252459 +/- 0.0465 |
| jacobian_moe | 1.98101e-06 +/- 1.06e-06 | 8.07677e-06 +/- 4.38e-06 | 0.104904 +/- 0.0103 | 0.693405 +/- 0.136 | 0.410742 +/- 0.157 | 0.786747 +/- 0.0726 | 0.934125 +/- 0.0326 | 0.196999 +/- 0.0301 |

## Aggregate MoE diagnostics

| Model | abs cos(0,1) | abs cos(0,2) | abs cos(1,2) | route 0 | route 1 | route 2 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_moe | 0.19117 +/- 0.0871 | 0.14445 +/- 0.0725 | 0.220147 +/- 0.166 | 0.334021 +/- 0.0801 | 0.339801 +/- 0.0608 | 0.326178 +/- 0.0906 |
| jacobian_moe | 0.0838962 +/- 0.0236 | 0.135161 +/- 0.0269 | 0.0956562 +/- 0.025 | 0.345853 +/- 0.132 | 0.317872 +/- 0.0935 | 0.336274 +/- 0.182 |

Hungarian one-to-one recovery remains an auxiliary diagnostic only: Vanilla 0.741056 +/- 0.256; Jacobian 0.564979 +/- 0.185.

## Per-seed results

| Model | Seed | Best step | IID MSE | IID rollout | Redundancy | Principal similarity | Projection error | Sample entropy | Usage entropy | Min usage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0 | 5000 | 9.40107e-06 | 3.47979e-05 | n/a | n/a | n/a | n/a | n/a | n/a |
| dense | 1 | 5000 | 9.62861e-06 | 2.76123e-05 | n/a | n/a | n/a | n/a | n/a | n/a |
| dense | 2 | 5000 | 5.29661e-06 | 2.27717e-05 | n/a | n/a | n/a | n/a | n/a | n/a |
| dense | 3 | 5000 | 1.02002e-05 | 3.56467e-05 | n/a | n/a | n/a | n/a | n/a | n/a |
| dense | 4 | 5000 | 7.35831e-06 | 2.4828e-05 | n/a | n/a | n/a | n/a | n/a | n/a |
| jacobian_moe | 0 | 5000 | 3.54629e-06 | 1.49769e-05 | 0.1191 | 0.6177 | 0.5082 | 0.8801 | 0.9405 | 0.1756 |
| jacobian_moe | 1 | 5000 | 1.27813e-06 | 6.48986e-06 | 0.1099 | 0.5582 | 0.5568 | 0.7567 | 0.9655 | 0.2352 |
| jacobian_moe | 2 | 4750 | 8.05245e-07 | 2.92099e-06 | 0.0972 | 0.9148 | 0.1524 | 0.6849 | 0.8842 | 0.2007 |
| jacobian_moe | 3 | 4250 | 2.37367e-06 | 8.09228e-06 | 0.0933 | 0.6617 | 0.4391 | 0.7933 | 0.9582 | 0.2140 |
| jacobian_moe | 4 | 5000 | 1.90172e-06 | 7.90387e-06 | 0.1050 | 0.7146 | 0.3973 | 0.8187 | 0.9223 | 0.1596 |
| vanilla_moe | 0 | 5000 | 6.41495e-07 | 3.46865e-06 | 0.2302 | 0.9555 | 0.0828 | 0.7596 | 0.9928 | 0.2916 |
| vanilla_moe | 1 | 5000 | 7.76e-07 | 3.75614e-06 | 0.2165 | 0.6536 | 0.4681 | 0.8582 | 0.9652 | 0.2144 |
| vanilla_moe | 2 | 4750 | 9.85586e-07 | 5.76589e-06 | 0.2541 | 0.5823 | 0.5287 | 0.7298 | 0.9902 | 0.2662 |
| vanilla_moe | 3 | 4500 | 3.76189e-07 | 3.01297e-06 | 0.1690 | 0.9153 | 0.1314 | 0.7337 | 0.9505 | 0.1932 |
| vanilla_moe | 4 | 3750 | 2.41281e-07 | 1.85603e-06 | 0.0565 | 0.9774 | 0.0441 | 0.6509 | 0.9959 | 0.2970 |

## Preliminary-support criterion

IID one-step and rollout mean MSE no more than 5% worse than Vanilla, lower mean redundancy, higher principal-angle subspace similarity, lower projection reconstruction error, and no routing collapse (minimum mean usage >= 0.05 and normalized usage entropy >= 0.8).

- FAIL: `iid_one_step_not_more_than_5_percent_worse`
- FAIL: `iid_rollout_not_more_than_5_percent_worse`
- PASS: `lower_mean_expert_redundancy`
- FAIL: `higher_mean_subspace_principal_similarity`
- FAIL: `lower_mean_subspace_projection_error`
- PASS: `no_obvious_routing_collapse`

Paired-seed wins: redundancy 4/5; principal similarity 1/5; projection error 1/5.

Overall: does not meet the preliminary-support criterion.

Full principal-angle spectra, pairwise cosines, routing statistics, projection errors, and auxiliary Hungarian matches are preserved in `summary.json` and `results/*.json`.
