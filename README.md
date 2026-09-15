# JEPA-MoE

第一阶段的最小低维 MoE dynamics predictor。当前实现有意固定以下实验条件：

- `state_dim = 4`
- `action_dim = 2`
- `num_experts = 3`
- 唯一输入为 `concat([state, action])`
- 每个 expert 为相互独立的 `6 -> 128 -> 128 -> 4` SiLU MLP
- router 为独立的 `6 -> 64 -> 64 -> 3` SiLU MLP，输出经过 softmax

项目包含三个具有相同 `forward(state, action)` 接口的模型：

- `DensePredictor`
- `VanillaMoEPredictor`
- `JacobianMoEPredictor`

`forward` 返回 `PredictorOutput(pred, alpha, expert_outputs)`。Dense 模型没有 routing，因此后两项为 `None`；MoE 模型中它们的 shape 分别为 `[batch, 3]` 和 `[batch, 3, 4]`。

## 安装与测试

```bash
uv venv --python 3.12 .venv
uv sync --extra test
.venv/bin/python -m pytest
```

依赖配置固定使用官方 `PyTorch 2.5.1 + CUDA 12.1` wheel，以适配本机 RTX 3090 和 NVIDIA 535 驱动。`uv sync --extra test` 会读取 `pyproject.toml` 中的专用 PyTorch index，并按 `uv.lock` 重建同一环境。

## Jacobian 分解

`control_jacobian_decomposition` 计算：

```text
B_k      = d F_k(x, a) / d a
J_within = sum_k alpha_k B_k
J_switch = sum_k outer(F_k, d alpha_k / d a)
d pred / d a = J_within + J_switch
```

所有模型使用同一个标准 next-state MSE，目标是绝对 next state，不使用 residual 或 multi-step loss。`JacobianMoEPredictor.training_loss` 的默认总目标为：

```text
prediction_loss = mean((pred - x_next) ** 2)
pair_loss_ij = relu(abs(cosine(B_i, B_j)) - 0.3) ** 2
weight_ij = detach(alpha_i * alpha_j)
jacobian_diversity_loss = sum(weight_ij * pair_loss_ij) / (sum(weight_ij) + eps)
activity_loss = mean(relu(0.05 - FrobeniusNorm(B_k)) ** 2)
specialization_loss = jacobian_diversity_loss + 0.1 * activity_loss
total_loss = prediction_loss + 0.1 * specialization_loss
```

`lambda_jac` 可在调用 `training_loss` 时改为 `0.01`、`0.1` 或 `1.0`；`margin`、`min_jacobian_norm` 和 `beta_activity` 也以关键字参数暴露。`VanillaMoEPredictor.training_loss` 严格等于 prediction loss。

`TrainingLoss.logging_metrics()` 返回独立的 prediction/diversity/activity loss、三个 expert pair 的 batch-mean Jacobian cosine similarity、每个 expert 的平均 Jacobian norm 和平均 routing weight，供训练 logger 逐项记录。Vanilla MoE 也以 `create_graph=False` 计算同口径诊断，但这些量不参与它的 `total_loss`；只有 Jacobian MoE 会对 specialization loss 反向传播。

第一阶段没有 encoder、Transformer、shared trunk/shared expert、residual prediction、Top-k/hard routing 或 history/context 输入。

## Experiment 1: full-IID dynamics basis quality

实验入口：

```bash
synthetic-basis-experiment --device cuda:0 --output-dir artifacts/synthetic_iid_v1
```

该实验是 deterministic direct next-state prediction。三个 `F_gt_k(x, a)` 直接输出 next state，`x_next = sum_k alpha_true_k F_gt_k(x, a)`；实现中不存在 vector field、`dt`、Euler 离散化、residual target 或噪声。

默认协议固定为 50,000 个训练样本、10,000 个 validation 样本、10,000 个 IID test 样本、长度 25 的 10,000 条 IID rollout，以及 5 个 model seeds。所有 split 都使用 `x ~ Uniform(-0.5, 0.5)^4`、`a ~ Uniform(-1, 1)^2`，不排除任何 action 区域。所有模型使用相同的 AdamW、batch 512、5,000 updates 和 batch 顺序，checkpoint 只依据 validation prediction MSE。Vanilla/Jacobian 在每个 seed 上具有相同初始化，唯一训练差别是后者加入 Control-Jacobian specialization。

主指标是 IID one-step MSE、rollout-25 MSE、expert Jacobian redundancy、routing entropy/usage 和 dynamics subspace quality。对每个样本将三个 `[4, 2]` Jacobian flatten 后组成 learned dictionary `L in R^(3x8)` 和 ground-truth dictionary `G in R^(3x8)`：

- principal-angle similarity 是两个 dictionary span 的三个 principal-angle cosine 的均值；rank 不足会以零 cosine 计入。
- projection reconstruction error 为 `||G - G L^+ L||_F^2 / ||G||_F^2`，其中 `L^+` 是 Moore-Penrose 伪逆。它表示每个 GT Jacobian 用 learned Jacobian 线性组合进行最小二乘重构后的归一化残差。
- Hungarian one-to-one cosine matching 只作为辅助诊断，不进入成功标准。
- routing collapse 判据为最小平均 usage `< 0.05` 或 normalized aggregate usage entropy `< 0.8`。

方案 E 的完整成功条件是：Jacobian MoE 的 IID one-step/rollout mean MSE 均不超过 Vanilla 的 `1.05x`，redundancy 更低，principal-angle similarity 更高，projection error 更低，并且不发生 routing collapse。

服务器共享时最多使用两张 GPU，例如：

```bash
synthetic-basis-experiment --device cuda:0 --run-seeds 0 2 4
synthetic-basis-experiment --device cuda:1 --run-seeds 1 3
synthetic-basis-experiment --device cpu  # resume 已完成结果并统一汇总
```

当前 `lambda_jac=0.1` 的 5-seed 正式结果保存在 `artifacts/synthetic_iid_v1`。Jacobian MoE 降低了 redundancy 且未发生 routing collapse，但 prediction/rollout 明显变差，principal-angle similarity 更低、projection error 更高，因此不满足方案 E 的初步支持条件。

### Delayed-specialization timing 诊断

为区分性能下降来自 specialization 的施加时机还是目标本身，新增入口：

```bash
synthetic-warmup-experiment \
  --device cuda:0 \
  --output-dir artifacts/synthetic_warmup_v1 \
  --source-dir artifacts/synthetic_iid_v1
```

该实验复用完全相同的 full-IID 数据协议和 Vanilla checkpoint，并重新训练 Immediate、Warmup-250、Warmup-500、Warmup-1000。每个 warm-up 在前 `W` steps 只训练 prediction objective，随后用 500 steps 将 `lambda_jac` 从 0 线性增加至 0.1。每 250 steps 还在同一训练 batch、参数更新前记录
`r_grad = ||lambda_jac * grad_expert(L_spec)|| / (||grad_expert(L_pred)|| + 1e-12)`；探针不写入 optimizer gradient。

5-seed 正式均值如下：

| Condition | IID MSE | Rollout-25 MSE | Redundancy | Principal similarity | Projection error | Usage entropy |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 6.041e-7 | 3.572e-6 | 0.1853 | 0.8168 | 0.2510 | 0.9789 |
| Immediate | 1.981e-6 | 8.077e-6 | 0.1049 | 0.6934 | 0.4107 | 0.9341 |
| Warmup-250 | 7.969e-7 | 5.163e-6 | 0.1011 | 0.8028 | 0.2728 | 0.9804 |
| Warmup-500 | 7.411e-7 | 3.999e-6 | 0.0996 | 0.8026 | 0.2716 | 0.9803 |
| Warmup-1000 | 7.124e-7 | 4.727e-6 | 0.0994 | 0.8075 | 0.2618 | 0.9826 |

Immediate 的 `r_grad` 在 step 0/1 分别为 `4.26 +/- 4.94` 和 `3.96 +/- 3.88`，到 step 250 已降至 `0.0747 +/- 0.0554`；warm-up ramp 中的典型比例约为 `0.02`。三个 warm-up 都显著修复了 Immediate 的 prediction、subspace 和 router usage，同时保持更低 redundancy，说明过强的初始化阶段 gradient 是主要问题之一。结论仍不是“目标本身已无害”：最佳 warm-up 的 one-step 和 rollout 均值分别仍为 Vanilla 的 `1.18x` 和 `1.12x`（来自不同 warm-up），未达到原先 `1.05x` 无明显退化门槛。当前证据因此支持“timing 是主要但并非唯一原因”；完整配对结果、逐 seed 指标和梯度曲线数据见 `artifacts/synthetic_warmup_v1/report.md`、`summary.json` 和 `gradients/*.jsonl`。

### 历史 oracle 诊断

此前使用“中心 action 区域排除/held-out”协议的结果不再属于主实验，只保留为定位 joint co-adaptation 的历史诊断：

```bash
synthetic-diagnostic-experiments \
  --device cuda:0 \
  --output-dir artifacts/synthetic_diagnostics_v1 \
  --reuse-joint-from artifacts/synthetic_basis_v1
```

- `oracle_router_vanilla`：固定 `alpha = alpha_true`，只用 prediction MSE 训练 experts。
- `oracle_router_jacobian`：同一个 oracle router，训练 experts 时额外加入 Control-Jacobian specialization。
- `oracle_expert`：固定三个 `F_gt_k`，只用 prediction MSE 训练原结构 router。
- `joint_vanilla` / `joint_jacobian`：router 与 experts 都自由学习，用作原 joint-learning 对照。

默认会验证协议后复用 `synthetic_basis_v1` 的 joint checkpoints，并用新增的 router 指标重新评估，而不是重复训练。传入 `--train-joint` 可从头训练 joint 条件。除原有预测、rollout、redundancy 和 basis recovery 外，诊断还记录 router 相对 `alpha_true` 的 raw MSE，以及按 expert 的 Hungarian 匹配重排后的 router MSE。

并行运行时，`--seeds` 定义完整共享协议，`--run-seeds` 只选择当前进程负责的 seed；因此不同 GPU 可以写入同一个 output directory，最后再运行一次完整命令汇总所有已完成结果。

当前 5-seed 正式结果保存在 `artifacts/synthetic_diagnostics_v1`。诊断显示 router 和 experts 在各自的 oracle 条件下都能高精度恢复，而 joint-learning 的预测误差和 matched router MSE 显著增大，当前主要问题因此定位为两者自由协同学习时的 co-adaptation / identifiability，而不是单独的网络容量不足。具体均值、标准差和逐 seed 数据见该目录的 `report.md` 与 `summary.json`。
