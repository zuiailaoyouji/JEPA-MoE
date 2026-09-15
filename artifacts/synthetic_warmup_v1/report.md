# Delayed Jacobian specialization on full IID

Shared protocol: 50k/10k/10k full-IID splits, 10k rollout-25 trajectories, 5000 updates, batch 512, AdamW lr=0.001, five matched seeds, validation prediction MSE-only checkpoint selection. Vanilla reuses the identical full-IID baseline; all other conditions are retrained.

Warm-up conditions use prediction alone through W, ramp lambda linearly from zero after W to 0.1 at W+500, then keep 0.1. Immediate uses 0.1 at every optimizer update.

## Aggregate results

| Condition | IID one-step MSE | Rollout-25 MSE | Redundancy | Principal similarity | Projection error | Usage entropy | Minimum usage |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 6.0411e-07 +/- 3e-07 | 3.57194e-06 +/- 1.42e-06 | 0.185256 +/- 0.0784 | 0.81682 +/- 0.185 | 0.251023 +/- 0.229 | 0.978902 +/- 0.02 | 0.252459 +/- 0.0465 |
| immediate | 1.98101e-06 +/- 1.06e-06 | 8.07677e-06 +/- 4.38e-06 | 0.104904 +/- 0.0103 | 0.693405 +/- 0.136 | 0.410742 +/- 0.157 | 0.934125 +/- 0.0326 | 0.196999 +/- 0.0301 |
| warmup_250 | 7.96948e-07 +/- 3.01e-07 | 5.16281e-06 +/- 1.02e-06 | 0.101133 +/- 0.0268 | 0.802788 +/- 0.166 | 0.272763 +/- 0.201 | 0.980391 +/- 0.00836 | 0.249483 +/- 0.0223 |
| warmup_500 | 7.41096e-07 +/- 4.36e-07 | 3.99856e-06 +/- 2.14e-06 | 0.0995702 +/- 0.0258 | 0.802644 +/- 0.171 | 0.271553 +/- 0.207 | 0.980321 +/- 0.00898 | 0.249734 +/- 0.0267 |
| warmup_1000 | 7.12408e-07 +/- 2.89e-07 | 4.72702e-06 +/- 2.77e-06 | 0.0994427 +/- 0.0287 | 0.80745 +/- 0.175 | 0.261823 +/- 0.217 | 0.982619 +/- 0.00958 | 0.254879 +/- 0.0325 |

## Per-seed results

| Condition | Seed | Best step | IID MSE | Rollout MSE | Redundancy | Principal similarity | Projection error | Usage entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| immediate | 0 | 5000 | 3.54629e-06 | 1.49769e-05 | 0.1191 | 0.6177 | 0.5082 | 0.9405 |
| immediate | 1 | 5000 | 1.27813e-06 | 6.48986e-06 | 0.1099 | 0.5582 | 0.5568 | 0.9655 |
| immediate | 2 | 4750 | 8.05245e-07 | 2.92099e-06 | 0.0972 | 0.9148 | 0.1524 | 0.8842 |
| immediate | 3 | 4250 | 2.37367e-06 | 8.09228e-06 | 0.0933 | 0.6617 | 0.4391 | 0.9582 |
| immediate | 4 | 5000 | 1.90172e-06 | 7.90387e-06 | 0.1050 | 0.7146 | 0.3973 | 0.9223 |
| vanilla | 0 | 5000 | 6.41495e-07 | 3.46865e-06 | 0.2302 | 0.9555 | 0.0828 | 0.9928 |
| vanilla | 1 | 5000 | 7.76e-07 | 3.75614e-06 | 0.2165 | 0.6536 | 0.4681 | 0.9652 |
| vanilla | 2 | 4750 | 9.85586e-07 | 5.76589e-06 | 0.2541 | 0.5823 | 0.5287 | 0.9902 |
| vanilla | 3 | 4500 | 3.76189e-07 | 3.01297e-06 | 0.1690 | 0.9153 | 0.1314 | 0.9505 |
| vanilla | 4 | 3750 | 2.41281e-07 | 1.85603e-06 | 0.0565 | 0.9774 | 0.0441 | 0.9959 |
| warmup_1000 | 0 | 4750 | 7.59878e-07 | 7.1944e-06 | 0.0920 | 0.9454 | 0.0991 | 0.9896 |
| warmup_1000 | 1 | 4750 | 8.98714e-07 | 5.79585e-06 | 0.1131 | 0.6228 | 0.5064 | 0.9757 |
| warmup_1000 | 2 | 4750 | 9.05453e-07 | 3.30758e-06 | 0.1167 | 0.6181 | 0.4772 | 0.9771 |
| warmup_1000 | 3 | 3750 | 7.89582e-07 | 6.75959e-06 | 0.1230 | 0.8710 | 0.1872 | 0.9748 |
| warmup_1000 | 4 | 4250 | 2.08413e-07 | 5.77649e-07 | 0.0524 | 0.9800 | 0.0392 | 0.9959 |
| warmup_250 | 0 | 4000 | 1.06483e-06 | 4.77192e-06 | 0.0930 | 0.8870 | 0.1946 | 0.9750 |
| warmup_250 | 1 | 4750 | 8.94651e-07 | 5.51571e-06 | 0.1045 | 0.6325 | 0.4844 | 0.9776 |
| warmup_250 | 2 | 4250 | 1.06455e-06 | 6.70522e-06 | 0.1179 | 0.6184 | 0.4866 | 0.9787 |
| warmup_250 | 3 | 4500 | 5.4484e-07 | 4.86249e-06 | 0.1300 | 0.9003 | 0.1511 | 0.9755 |
| warmup_250 | 4 | 4500 | 4.15876e-07 | 3.95871e-06 | 0.0602 | 0.9758 | 0.0471 | 0.9951 |
| warmup_500 | 0 | 4000 | 1.20217e-06 | 7.02923e-06 | 0.0908 | 0.9077 | 0.1622 | 0.9800 |
| warmup_500 | 1 | 4750 | 7.03896e-07 | 2.50589e-06 | 0.1058 | 0.6256 | 0.4969 | 0.9753 |
| warmup_500 | 2 | 4500 | 1.15777e-06 | 5.28918e-06 | 0.1152 | 0.6099 | 0.4896 | 0.9785 |
| warmup_500 | 3 | 4500 | 4.20641e-07 | 3.38655e-06 | 0.1263 | 0.8988 | 0.1531 | 0.9723 |
| warmup_500 | 4 | 3750 | 2.21003e-07 | 1.78193e-06 | 0.0597 | 0.9712 | 0.0559 | 0.9955 |

## Paired warm-up comparisons

Ratios use aggregate means (below 1 is better for errors and redundancy). Deltas are candidate minus reference. Wins count matched seeds in the favorable direction.

### Relative to immediate

| Condition | IID ratio (wins) | Rollout ratio (wins) | Redundancy ratio (wins) | Principal delta (wins) | Projection ratio (wins) | Entropy delta (wins) |
|---|---:|---:|---:|---:|---:|---:|
| warmup_250 | 0.402 (4/5) | 0.639 (4/5) | 0.964 (3/5) | +0.1094 (4/5) | 0.664 (4/5) | +0.0463 (5/5) |
| warmup_500 | 0.374 (4/5) | 0.495 (4/5) | 0.949 (3/5) | +0.1092 (4/5) | 0.661 (4/5) | +0.0462 (5/5) |
| warmup_1000 | 0.360 (4/5) | 0.585 (4/5) | 0.948 (2/5) | +0.1140 (4/5) | 0.637 (4/5) | +0.0485 (5/5) |

### Relative to vanilla

| Condition | IID ratio (wins) | Rollout ratio (wins) | Redundancy ratio (wins) | Principal delta (wins) | Projection ratio (wins) | Entropy delta (wins) |
|---|---:|---:|---:|---:|---:|---:|
| warmup_250 | 1.319 (0/5) | 1.445 (0/5) | 0.546 (4/5) | -0.0140 (1/5) | 1.087 (1/5) | +0.0015 (2/5) |
| warmup_500 | 1.227 (2/5) | 1.119 (3/5) | 0.537 (4/5) | -0.0142 (1/5) | 1.082 (1/5) | +0.0014 (2/5) |
| warmup_1000 | 1.179 (2/5) | 1.323 (2/5) | 0.537 (5/5) | -0.0094 (2/5) | 1.043 (2/5) | +0.0037 (3/5) |

## Expert gradient ratios during training

`r_grad = ||lambda(step) * grad_expert(L_spec)|| / (||grad_expert(L_pred)|| + 1e-12)`; measured before the update on the same batch. Values are mean +/- sample std across completed seeds.

| Condition | Step 0 | Step 1 | Step 250 | Step 500 | Step 750 | Step 1000 | Step 1250 | Step 1500 | Step 2000 | Step 5000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 0 by design | 0 by design | 0 by design | 0 by design | 0 by design | 0 by design | 0 by design | 0 by design | 0 by design | 0 by design |
| immediate | 4.26 +/- 4.9 | 3.96 +/- 3.9 | 0.0747 +/- 0.055 | 0.0186 +/- 0.022 | 0.013 +/- 0.014 | 0.00738 +/- 0.01 | 0.0154 +/- 0.021 | 0.0249 +/- 0.023 | 0.0217 +/- 0.031 | 0.00423 +/- 0.0064 |
| warmup_250 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0.0244 +/- 0.019 | 0.022 +/- 0.018 | 0.00971 +/- 0.0086 | 0.00742 +/- 0.0055 | 0.00645 +/- 0.008 | 0.0181 +/- 0.012 | 0.00117 +/- 0.0013 |
| warmup_500 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0.0227 +/- 0.019 | 0.0112 +/- 0.0089 | 0.00855 +/- 0.0056 | 0.00711 +/- 0.0081 | 0.0156 +/- 0.014 | 0.0011 +/- 0.0019 |
| warmup_1000 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0 +/- 0 | 0.019 +/- 0.0081 | 0.00847 +/- 0.0071 | 0.0175 +/- 0.015 | 0.000693 +/- 0.00076 |

`gradients/*.jsonl` contains the pre-update expert gradient ratio on the same shuffled training batch at step 1 and every 250 updates; step 0 is an unshuffled, shared fixed-batch initialization probe. Unweighted specialization gradient is also recorded while lambda is zero. The probe never writes optimizer gradients. Vanilla has algebraically zero weighted specialization gradient and reuses prior checkpoints.

Hungarian matching remains an auxiliary metric in `results/*.json`, not a basis target or success criterion.
