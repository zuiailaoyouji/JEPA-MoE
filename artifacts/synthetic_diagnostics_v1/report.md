# Synthetic diagnostic experiments

Protocol: 5000 updates, batch 512, AdamW lr=0.001, lambda_jac=0.1, 5 seeds. Checkpoints are selected only by validation prediction MSE.

The oracle-router regime fixes alpha to alpha_true and trains only experts. The oracle-expert regime fixes F_k to F_gt_k and trains only the original router. Joint-learning trains the original router and experts together.

## Aggregate results

| Condition | IID MSE | Held-out MSE | IID rollout-25 | Held-out rollout-25 | Redundancy | Recovery | Router MSE | Matched router MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_router_vanilla | 1.45271e-07 +/- 2.66e-08 | 1.64906e-07 +/- 4.49e-08 | 5.10565e-07 +/- 1.87e-07 | 1.23285e-06 +/- 6.33e-07 | 0.0618733 +/- 0.016 | 0.990258 +/- 0.00401 | 0 +/- 0 | 0 +/- 0 |
| oracle_router_jacobian | 1.79635e-07 +/- 2.24e-08 | 1.51505e-07 +/- 4.78e-08 | 5.87127e-07 +/- 1.82e-07 | 9.87317e-07 +/- 4.8e-07 | 0.0438703 +/- 0.0151 | 0.992627 +/- 0.00296 | 0 +/- 0 | 0 +/- 0 |
| oracle_expert | 3.1828e-09 +/- 6.32e-10 | 1.16424e-09 +/- 4.22e-10 | 1.01861e-08 +/- 4.19e-09 | 2.2727e-09 +/- 7.52e-10 | 0 +/- 0 | 1 +/- 0 | 2.59744e-07 +/- 6.34e-08 | 2.59744e-07 +/- 6.34e-08 |
| joint_vanilla | 6.25015e-07 +/- 2.16e-07 | 8.80313e-07 +/- 4.69e-07 | 4.83934e-06 +/- 2.46e-06 | 8.04886e-06 +/- 5.98e-06 | 0.197452 +/- 0.121 | 0.746756 +/- 0.245 | 0.111025 +/- 0.0368 | 0.0251437 +/- 0.0201 |
| joint_jacobian | 1.75476e-06 +/- 7.74e-07 | 2.82703e-06 +/- 2.39e-06 | 7.9127e-06 +/- 2.7e-06 | 3.21978e-05 +/- 3.54e-05 | 0.0976982 +/- 0.0338 | 0.583418 +/- 0.173 | 0.0947708 +/- 0.0368 | 0.0543023 +/- 0.0364 |

## Routing and expert diagnostics

| Condition | abs cos(0,1) | abs cos(0,2) | abs cos(1,2) | route 0 | route 1 | route 2 |
|---|---:|---:|---:|---:|---:|---:|
| oracle_router_vanilla | 0.054431 +/- 0.0332 | 0.0817989 +/- 0.0275 | 0.0493901 +/- 0.0247 | 0.318536 +/- 0 | 0.318165 +/- 0 | 0.363299 +/- 0 |
| oracle_router_jacobian | 0.0445634 +/- 0.0311 | 0.048693 +/- 0.011 | 0.0383543 +/- 0.0117 | 0.318536 +/- 0 | 0.318165 +/- 0 | 0.363299 +/- 0 |
| oracle_expert | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0.318541 +/- 0.000215 | 0.31806 +/- 0.00017 | 0.363399 +/- 0.00022 |
| joint_vanilla | 0.196107 +/- 0.136 | 0.156406 +/- 0.0687 | 0.239842 +/- 0.19 | 0.330706 +/- 0.0727 | 0.34592 +/- 0.0604 | 0.323374 +/- 0.0818 |
| joint_jacobian | 0.102503 +/- 0.0369 | 0.0938475 +/- 0.0286 | 0.0967444 +/- 0.0455 | 0.369842 +/- 0.111 | 0.316169 +/- 0.115 | 0.313989 +/- 0.144 |

## Per-seed results

| Condition | Seed | Best step | IID MSE | Held-out MSE | Held-out rollout | Redundancy | Recovery | Router MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_jacobian | 0 | 5000 | 2.93372e-06 | 1.82923e-06 | 1.50572e-05 | 0.1369 | 0.5468 | 0.0997697 |
| joint_jacobian | 1 | 4750 | 1.02116e-06 | 1.32852e-06 | 1.16906e-05 | 0.0971 | 0.4554 | 0.0728361 |
| joint_jacobian | 2 | 5000 | 1.08989e-06 | 1.36063e-06 | 1.57574e-05 | 0.0455 | 0.8879 | 0.15656 |
| joint_jacobian | 3 | 4750 | 1.87051e-06 | 2.60776e-06 | 2.35134e-05 | 0.0942 | 0.5140 | 0.0790139 |
| joint_jacobian | 4 | 5000 | 1.85853e-06 | 7.00901e-06 | 9.49704e-05 | 0.1148 | 0.5130 | 0.065674 |
| joint_vanilla | 0 | 5000 | 8.37323e-07 | 1.60722e-06 | 1.794e-05 | 0.3543 | 0.8863 | 0.142106 |
| joint_vanilla | 1 | 5000 | 8.342e-07 | 1.0544e-06 | 8.15511e-06 | 0.2112 | 0.4562 | 0.0636598 |
| joint_vanilla | 2 | 4750 | 6.34559e-07 | 6.69517e-07 | 4.41591e-06 | 0.2594 | 0.5068 | 0.136327 |
| joint_vanilla | 3 | 4750 | 3.56335e-07 | 6.72306e-07 | 7.26716e-06 | 0.1210 | 0.9079 | 0.0789711 |
| joint_vanilla | 4 | 4250 | 4.62656e-07 | 3.98128e-07 | 2.4661e-06 | 0.0414 | 0.9765 | 0.13406 |
| oracle_expert | 0 | 5000 | 3.9239e-09 | 1.19958e-09 | 3.10415e-09 | 0.0000 | 1.0000 | 3.16653e-07 |
| oracle_expert | 1 | 5000 | 3.76735e-09 | 1.80465e-09 | 2.39099e-09 | 0.0000 | 1.0000 | 3.0429e-07 |
| oracle_expert | 2 | 4750 | 2.60236e-09 | 1.05376e-09 | 2.41831e-09 | 0.0000 | 1.0000 | 2.42466e-07 |
| oracle_expert | 3 | 5000 | 2.5957e-09 | 6.27747e-10 | 1.04135e-09 | 0.0000 | 1.0000 | 1.58382e-07 |
| oracle_expert | 4 | 4250 | 3.0247e-09 | 1.13544e-09 | 2.40868e-09 | 0.0000 | 1.0000 | 2.7693e-07 |
| oracle_router_jacobian | 0 | 3000 | 1.82135e-07 | 1.86066e-07 | 1.68581e-06 | 0.0328 | 0.9933 | 0 |
| oracle_router_jacobian | 1 | 3500 | 1.71633e-07 | 1.50889e-07 | 7.9554e-07 | 0.0503 | 0.9889 | 0 |
| oracle_router_jacobian | 2 | 3750 | 1.69957e-07 | 1.75176e-07 | 1.19775e-06 | 0.0297 | 0.9970 | 0 |
| oracle_router_jacobian | 3 | 4750 | 1.57786e-07 | 6.92061e-08 | 4.11857e-07 | 0.0398 | 0.9918 | 0 |
| oracle_router_jacobian | 4 | 3000 | 2.16666e-07 | 1.76189e-07 | 8.45627e-07 | 0.0669 | 0.9920 | 0 |
| oracle_router_vanilla | 0 | 2500 | 1.60565e-07 | 2.18595e-07 | 2.09877e-06 | 0.0679 | 0.9927 | 0 |
| oracle_router_vanilla | 1 | 2500 | 1.17764e-07 | 1.65943e-07 | 1.38114e-06 | 0.0529 | 0.9864 | 0 |
| oracle_router_vanilla | 2 | 2500 | 1.34612e-07 | 1.43767e-07 | 1.2055e-06 | 0.0391 | 0.9944 | 0 |
| oracle_router_vanilla | 3 | 2500 | 1.2974e-07 | 1.02528e-07 | 3.24111e-07 | 0.0696 | 0.9923 | 0 |
| oracle_router_vanilla | 4 | 2000 | 1.83674e-07 | 1.93695e-07 | 1.15471e-06 | 0.0799 | 0.9856 | 0 |

## Diagnostic reading

With the true router fixed, Jacobian specialization changes the IID MSE by 1.237x, held-out MSE by 0.919x, held-out rollout MSE by 0.801x, redundancy by 0.709x, and recovery by 1.002x.

Joint Vanilla has 4.30x the oracle-router IID MSE and 5.34x the held-out MSE. Joint Jacobian has 9.77x and 18.66x, respectively.

The oracle-expert router MSE is 2.6e-07; after permutation matching, joint Vanilla/Jacobian router MSEs are 0.0251 and 0.0543.

Experts and router are each recoverable in isolation; the large joint-to-oracle error and router-MSE gaps identify joint co-adaptation/identifiability as the dominant failure mode under this protocol.

Interpretation order: oracle-router isolates expert learning; oracle-expert isolates router learning; joint-learning shows whether degradation appears only when both components co-adapt.

Full routing matrices, Hungarian matches, pairwise cosine values, and checkpoint provenance are preserved in `summary.json` and `results/*.json`.
