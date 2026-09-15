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

## Experiment 1: synthetic basis recovery

实验入口：

```bash
synthetic-basis-experiment --device cuda:0 --output-dir artifacts/synthetic_basis_v1
```

该实验是 deterministic direct next-state prediction。三个 `F_gt_k(x, a)` 直接输出 next state，`x_next = sum_k alpha_true_k F_gt_k(x, a)`；实现中不存在 vector field、`dt`、Euler 离散化、residual target 或噪声。

默认协议固定为 50,000 个训练样本、10,000 个 validation 样本、10,000 个 IID test 样本、10,000 个中心 held-out-composition 样本、长度 25 的 10,000 条 IID/held-out rollout，以及 5 个 model seeds。所有条件使用相同的 AdamW、batch 512、5,000 updates 和 batch 顺序，checkpoint 只依据 validation prediction MSE。

Basis recovery 在固定的 10,000-sample probe 上评估，其中 IID 与 held-out samples 各占一半。计算 learned-vs-ground-truth `3 x 3` mean absolute Jacobian cosine matrix，再通过 Hungarian matching 获得 permutation-invariant score。完整配置、checkpoint、逐步日志、per-seed 结果与汇总报告写入指定 output directory。

### 三个诊断实验

在调整主实验前，可运行严格共享原数据与优化协议的诊断组：

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
