# JEPA Basis Dynamics · Research Plan v0.4

**方案 E：JEPA 中的可组合 Basis Dynamics**

**Identity-Free Basis-Composition Predictor with Control-Response Factorization**

> Codex 工作版。内容依据原始实验方案整理；保留研究定义与实验顺序，移除 Word 排版噪音。

方案 E：JEPA 中的可组合 Basis Dynamics

Identity-Free Basis-Composition Predictor with Control-Response Factorization

研究方案 · v0.4

## 一句话摘要

在 action-conditioned JEPA 中，将 MoE 仅用于 dynamics predictor：每个 expert F_k(z,a) 承载一个可复用的局部受控转移机制，state-conditioned router R(z) 仅估计这些 basis dynamics 的组合权重。通过 expert control Jacobian B_k=∂F_k/∂a 定义局部 basis control-response，并利用真实局部 action-response 监督其组合覆盖，从而学习可跨系统复用、可重新组合的 dynamics dictionary。

## 研究定位

目标不是单纯提高 MoE 或 JEPA 的模型容量，而是验证并利用一个更结构化的假设：heterogeneous controlled dynamics 中存在一组可复用的局部 dynamics bases，不同系统、状态与交互阶段可通过不同权重组合这些 bases 来产生整体转移。

首阶段聚焦 deterministic、fully observed、action space 一致的设定，先验证 basis discovery、coverage、reuse 与 recombination，再扩展到视觉 JEPA、不同机器人与真实 manipulation。

## 方案结构

| 模块 | 核心设计 |
| --- | --- |
| JEPA 外壳 | Context/target encoder 保持标准 joint-embedding predictive learning。 |
| Dynamics Experts | F_k(z,a) 直接建模候选 next-latent transition。 |
| Basis Router | α=R(z)，只根据当前 state/latent 估计 basis mixture coefficients。 |
| Basis 定义 | B_k(z,a)=∂F_k/∂a，表示第 k 个 expert 的局部 control-response operator。 |
| 核心约束 | 真实 local action-response 应可由 Σ_k α_k B_k 的组合解释。 |
| 核心收益 | 跨系统复用、unseen combination recombination、few-shot router adaptation、long-horizon prediction/planning。 |

## 1. 研究问题与核心假设

设多个机器人、物体交互或局部 dynamics regime 具有不同的整体 transition function。我们假设这些复杂 dynamics 并非完全独立，而是共享一组局部受控转移机制。模型应该学习这些机制本身，并在不同状态下重新组合，而不是为每个 system/task 建立独立的完整 dynamics。

```text
F_true(z,a) ≈ Σ_k α_k(z) F_k(z,a) F_k 为可复用 transition experts；α_k 为当前 state 下的 basis composition coefficient。
```

### 核心假设

H1：heterogeneous controlled dynamics 可以被少量 reusable basis dynamics 组合表示。 H2：MoE 的 conditional computation 可以作为发现与组合这些 basis dynamics 的建模框架。 H3：好的 basis 不要求彼此正交，而要求它们的组合能够覆盖 task-relevant control-response，并在新系统/新组合中被复用。

该分解不要求唯一。只要 learned dictionary 能以低误差表示真实 dynamics family，并在不同 systems/regimes 中重复调用同一组 bases，就满足本方案的结构目标。

## 2. JEPA 基本设定

视觉观测 o_t 经 context encoder 得到当前 latent state，下一帧由 target encoder 产生 stop-gradient target latent。Dynamics predictor 只预测下一时刻的 target representation，不直接预测像素。

```text
z_t = E_θ(o_t)
```

```text
z*_{t+1} = sg(E_θ̄(o_{t+1}))
```

```text
ẑ_{t+1} = P_MoE(z_t,a_t)
```

```text
L_pred = d(ẑ_{t+1}, z*_{t+1})
```

MoE 仅发生在 dynamics predictor 内；encoder 负责状态表征，predictor 负责 action-conditioned latent transition。

## 3. Basis-Composition MoE Predictor

Router 被定义为纯粹的 basis weight estimator。它不读取当前 action，只根据当前 latent state 输出非负并归一化的 mixture coefficients：

```text
α = R(z), α_k ≥ 0, Σ_k α_k = 1
```

每个 dynamics expert 仍同时读取 state 和 action：

```text
F_k = F_k(z,a), k = 1,…,K
```

最终预测是 experts 的加权组合：

```text
F(z,a) = Σ_k α_k(z) F_k(z,a)
```

### 角色分工

Router 回答“当前 state 应该如何组合已有 dynamics bases”；Experts 回答“给定 action 后，各局部 dynamics mechanism 如何产生 next-state response”。Router 不承担 action-dependent dynamics generation。

## 4. Control Jacobian 与 Basis Dynamics 定义

对于第 k 个 transition expert，定义其对 action 的局部敏感度为：

```text
B_k(z,a) := ∂F_k(z,a) / ∂a
```

F_k 是神经网络 transition expert；B_k 是该 expert 在当前 state-action 处诱导的 local basis control dynamics / control-response operator。

因为 α=R(z) 不依赖当前 action，所以 predictor 的完整 action Jacobian 直接为：

```text
J_pred(z,a) = ∂F(z,a)/∂a = Σ_k α_k(z) B_k(z,a)
```

### 关键结构性质

模型的 local action-response 无法由 router 的 action derivative 产生；所有 control sensitivity 必须由 expert Jacobian dictionary 及其 mixture 表达。因此“整体 dynamics 准确”与“expert basis 覆盖真实 response”在该结构下被直接关联。

这一设定故意先不建模 action-induced discrete switching。free→contact、stick→slip、release 等 hybrid mode transitions 留到后续扩展，以保证第一阶段的 basis-composition 假设能够被干净验证。

## 5. Control-Response Factorization Objective

Prediction loss 负责整体 transition accuracy；为了进一步约束局部 action-response 与 learned basis composition 对齐，引入 control-response consistency。对于单位 action direction v，真实 directional response 定义为：

```text
r* = J_true(z,a) v
```

```text
在 synthetic 或可 reset 的 simulator 中，不需要显式构造 J_true，可通过中心有限差分获得：
```

```text
r* ≈ [ z*(a+εv) − z*(a−εv) ] / (2ε)
```

每个 expert 对同一 action direction 的 response 为：

```text
r_k = B_k(z,a) v
```

由于 router 不依赖 action，模型的 directional response 为：

```text
r_pred = Σ_k α_k(z) r_k = Σ_k α_k(z) B_k(z,a) v
```

据此定义 Control-Response Consistency loss：

```text
L_CR = || r* − Σ_k α_k(z) B_k(z,a) v ||²
```

### L_CR 的含义

该目标不指定“第几个 expert 应对应哪个真实 basis”，也不要求 experts 两两正交。它只要求 learned dictionary 的组合能够解释真实 local control-response，因此允许 basis transformation 与非唯一分解。

## 6. 总训练目标与优化方式

```text
L = L_pred + λ_CR L_CR + λ_bal L_balance
```

| 损失项 | 作用 |
| --- | --- |
| L_pred | 保证整体 next-state / next-latent prediction 正确。 |
| L_CR | 使 expert Jacobian dictionary 的组合覆盖真实 local action-response。 |
| L_balance | 标准 MoE 负载均衡，避免长期只使用少数 experts。 |

第一版采用标准 end-to-end joint training：router 与 experts 在同一次反向传播中更新。暂不引入 alternating optimization、显式 coefficient solver 或 GT basis supervision，以便先验证最简结构是否已经能产生 reusable basis dynamics。

## 7. 实验路线总览

实验采用“机制验证 → 跨系统复用 → 视觉 JEPA”三级路线。每一级都围绕同一条证据链：basis coverage 是否成立、同一 dictionary 是否被多个 dynamics 复用、未见组合是否可以通过重新加权已有 bases 表示。

| 阶段 | 目的 | 关键结论 |
| --- | --- | --- |
| Experiment 1: Synthetic Basis Recovery | 验证结构与训练目标能否学到覆盖真实 dynamics space 的 dictionary。 | 不是恢复 GT expert identity，而是验证 coverage、local response accuracy 与稳定训练。 |
| Experiment 2: Multi-System Reuse | 验证同一组 experts 是否跨 systems/regimes 复用。 | 重点测试 unseen composition 与 freeze-experts / adapt-router。 |
| Experiment 3: Visual JEPA | 在视觉 latent 中验证同样的 factorization。 | 证明方法不仅适用于低维 state，也能服务 long-horizon prediction / planning。 |

## 8. Experiment 1：Synthetic Basis-Composition Mechanism Study

### 8.1 数据生成

保持低维 deterministic next-state prediction，建议 state_dim=4、action_dim=2、num_experts=3。三个 ground-truth transition experts F_gt,k(x,a) 均为平滑非线性函数，并具有不同但不必正交的 control-response。

真实 mixture coefficients 只由 state 决定，例如：

```text
α_true(x) = softmax( 2x₀, 2x₁, −2(x₀+x₁) )
```

```text
x_{t+1} = Σ_k α_true,k(x_t) F_gt,k(x_t,a_t)
```

因此真实 action Jacobian严格满足：

```text
J_true(x,a) = Σ_k α_true,k(x) B_gt,k(x,a)
```

Train / validation / IID test 统一从完整分布采样，不人为挖掉 action 区域。建议首轮规模 50k / 10k / 10k，5 个随机种子，checkpoint 仅按 validation prediction MSE 选择。

### 8.2 模型对照

| 模型 | Router | 训练目标 | 目的 |
| --- | --- | --- | --- |
| Dense Predictor | 无 | L_pred | 非 MoE 容量基线。 |
| Vanilla MoE-A | R(x,a) | L_pred + balance | 允许 action-dependent routing 的强 MoE 基线。 |
| Vanilla MoE-S | R(x) | L_pred + balance | 与 Ours 相同结构，检验 state-only router 本身。 |
| Ours | R(x) | L_pred + λ_CR L_CR + balance | 检验 control-response factorization 的增益。 |

### 8.3 评价指标

- IID one-step prediction MSE。
- 25-step multi-step rollout MSE。
- Local response error：E_v[||J_pred v − J_true v||²]；synthetic 中可额外报告完整 Jacobian Frobenius error。
- Basis subspace quality：将 B_k flatten 后比较 learned dictionary 与 GT dictionary 的 principal-angle similarity / projection reconstruction error。
- Router usage entropy、expert utilization 与 load balance。
- Pairwise Jacobian similarity 仅作为 diagnostic，不作为成功判据。
### 8.4 Experiment 1 成功标准

Ours 应在 prediction / rollout 上至少保持与 Vanilla MoE-S 同一量级，同时显著降低 local response error，并提高或保持 dynamics-subspace coverage；不同 experts 不要求一一匹配 GT experts。若 Vanilla MoE-S 已能达到同样的 control-response coverage，则说明结构本身足够，L_CR 的额外价值需要通过后续 OOD/reuse 实验判断。

## 9. Experiment 2：Cross-System Reuse 与 Unseen Recombination

这是验证“为什么要学习 basis dynamics，而不只追求整体 prediction”的核心实验。构造多个 systems 或 dynamics contexts，共享同一组潜在 basis mechanisms，但使用不同 state-conditioned composition patterns。

```text
System s: F^(s)(x,a) = Σ_k α_k^(s)(x) F_k^basis(x,a)
```

数据划分采用 factorial hold-out：训练集覆盖部分 system × dynamics-combination，测试集故意包含训练期间未出现的组合。Action 分布尽量保持一致，使主要 distribution shift 来自 basis composition 而不是单纯 action extrapolation。

### 9.1 核心测试

## 1. Seen-system IID prediction：确认结构化 factorization 不牺牲基本建模能力。

## 2. Unseen basis combination：测试已学 experts 是否能通过新的 α 组合表达未见 dynamics。

## 3. Cross-system reuse：统计同一 expert / basis 在多个 systems 中的调用频率与局部 response similarity。

## 4. Freeze experts + adapt router：固定 dynamics dictionary，仅用少量新系统数据训练 R(z)，测 sample efficiency。

## 5. Full fine-tuning baseline：与 Dense / Vanilla MoE 的全模型适配比较。

### 9.2 关键指标

| 指标 | 解释 |
| --- | --- |
| Unseen-combination MSE / rollout | basis recombination 能否改善 OOD dynamics prediction。 |
| Few-shot adaptation curve | 使用 1%、5%、10%、25% 新系统数据时的性能。 |
| Frozen-expert gap | 只调 router 与全模型 fine-tune 的差距；越小说明 bases 越可复用。 |
| Cross-system basis reuse | 相似 local response 是否由同一批 experts 承担，而非一 system 一 expert。 |
| Dictionary coverage | 新系统 J_true 是否仍落在 learned expert Jacobian span 内。 |

### Experiment 2 的核心判断

如果 structured basis factorization 的价值成立，主要优势应出现在 unseen composition、few-shot new-system adaptation 与 frozen-expert transfer，而不一定体现为 seen-distribution IID MSE 的大幅提升。

## 10. Experiment 3：Visual JEPA World Model

在视觉环境中使用 context encoder / target encoder 将 observation 映射到 latent state，再在 predictor 中使用 Basis-Composition MoE。优先选择可 reset、未来相对确定、action 维度一致的 manipulation / control simulator，以便构造局部 action perturbation pair。

Control-response target 在 latent space 中通过 target encoder 得到：

```text
r* ≈ [ E_target(o_{t+1}^{a+εv}) − E_target(o_{t+1}^{a−εv}) ] / (2ε)
```

核心评估包括 next-latent prediction、long-horizon latent rollout、unseen interaction composition、few-shot new-system adaptation，以及 goal-conditioned planning / action optimization 成功率。

## 11. Ablation 与诊断实验

| Ablation | 问题 |
| --- | --- |
| 去掉 L_CR | local control-response supervision 是否真正贡献 coverage / OOD reuse？ |
| R(z) vs R(z,a) | 允许 action-dependent gate 后，性能是否提高，但 reusable basis 结构是否变弱？ |
| 不同 K | dictionary size 对 coverage、reuse 与过度参数化的影响。 |
| 不同 perturbation ε / directions v | 有限差分 target 的数值稳定性与计算成本。 |
| JVP sampling 数量 | 每个 batch 使用 1 / 2 / 4 个方向时的效率-性能权衡。 |
| 标准 load balance on/off | expert collapse 是否来自普通 MoE optimization。 |
| Joint vs staged optimization | 若 joint training 不稳定，再评估先学 dictionary、后联合微调。 |

## 12. 工程实现建议

### 12.1 第一阶段低维实现

- 每个 expert 独立 MLP，输入 concat([state, action])，直接输出 next state candidate。
- Router MLP 只输入 state，并通过 softmax 输出 α。
- Prediction：pred = Σ_k α_k * expert_k(state, action)。
- B_k v 优先使用 JVP 计算，避免显式构造完整 Jacobian。
- Synthetic 中 r* 可由解析 GT Jacobian 或 finite difference 计算；正式方法不使用 GT expert labels。
- 同一 seed 下保证 baseline 与 Ours 的 initialization、batch order、optimizer 和训练步数尽可能一致。
### 12.2 Visual JEPA 实现

- Attention / shared trunk 保持共享，MoE 可以只放在 predictor 后若干 FFN blocks。
- 需要明确定义 expert branch 对最终 transition prediction 的贡献，以便计算 expert-specific B_k v。
- 如完整 counterfactual perturbation 数据成本过高，可在 simulator 中低频采样 response pairs，或仅对部分 batch 计算 L_CR。
- 第一版不处理 heterogeneous action dimensions；后续可加入 canonical action latent / action adapter。
## 13. 预期证据链

## 1. 整体 prediction：模型能准确预测 seen dynamics。

## 2. Local response：Σ_k α_k B_k 能重建真实 control-response。

## 3. Subspace coverage：learned dictionary 覆盖真实 dynamics response space，而无需逐 expert GT matching。

## 4. Cross-system reuse：相同局部机制跨 systems 复用同一批 experts。

## 5. Unseen recombination：未见 dynamics combination 可通过新的 α 组合表示。

## 6. Few-shot transfer：冻结 experts，仅适配 router 即可快速适应新 system。

## 7. Long-horizon / planning：结构化 factorization 最终改善 rollout 或控制性能。

## 14. 论文叙述草案

### Problem

Accurate aggregate prediction does not by itself reveal whether a heterogeneous world model has discovered reusable dynamics structure. A model may fit every observed transition while still representing each system or regime as an entangled mapping that is difficult to recombine or adapt.

### Hypothesis

Heterogeneous controlled dynamics can be factorized into a small dictionary of reusable local control-response bases. Different systems and regimes are represented by state-dependent compositions of the same bases.

### Method

We use an identity-free state-conditioned router to estimate basis coefficients and action-conditioned transition experts to carry the dynamics. Each expert induces a control-response basis B_k=∂F_k/∂a. Because the router does not depend on the current action, the model control Jacobian is exactly the weighted sum of expert basis Jacobians. A control-response consistency objective trains this dictionary to explain observed local action responses without imposing pairwise orthogonality or ground-truth expert identities.

### Benefit

The learned dynamics dictionary is intended to be reused across systems and recombined under new coefficients, enabling unseen-combination generalization, data-efficient adaptation by updating only the router, and more reliable long-horizon prediction or planning.

## 15. 主要风险与开放问题

| 风险 / 问题 | 处理思路 |
| --- | --- |
| State-only router 是否限制 contact switching？ | 第一阶段有意简化；若核心 basis hypothesis 成立，再增加独立 switching mechanism。 |
| L_CR 是否只是提高 Jacobian accuracy，而未改善 transfer？ | 以 Experiment 2 的 unseen recombination / few-shot adaptation 作为必要性检验。 |
| 有限差分 response pair 成本较高 | 使用 JVP 方向采样、稀疏 batch supervision、simulator reset 并行采样。 |
| Basis decomposition 非唯一 | 评价 dictionary span / coverage / reuse，而非 one-to-one GT matching。 |
| Latent Jacobian 几何是否稳定 | 先低维 state 验证，再分析视觉 latent 中 response consistency 与 planning correlation。 |
| Experts 可能仍产生冗余 | 先依赖 task loss + load balance；只有出现实质 collapse 时再引入针对 coverage 的结构约束。 |

## 16. 最小可行版本（MVP）

## 1. 低维 deterministic synthetic environment，α_true 仅依赖 state。

## 2. Dense、Vanilla MoE-A、Vanilla MoE-S、Ours 四组严格对照。

## 3. 标准 prediction loss + state-only router + expert transition MLPs。

## 4. 基于 JVP / finite difference 的 L_CR。

## 5. 5 seeds，报告 prediction、rollout、local response error、subspace coverage 与 routing usage。

## 6. 在机制验证通过后立即进入 multi-system factorial hold-out，而不是长期停留在 IID synthetic 调参。

### MVP 通过条件

如果 Ours 能在保持基本 prediction 能力的同时提高 local response / dictionary coverage，并且在后续 multi-system 实验中通过 frozen reusable experts 实现更强的 unseen-combination 与 few-shot adaptation，则核心“reusable basis dynamics”假设得到实证支持。

## 附录 A：统一符号表

| 符号 | 含义 |
| --- | --- |
| o_t | 时刻 t 的视觉观测。 |
| a_t | 时刻 t 的 action / control。 |
| z_t | JEPA context latent state。 |
| z*_{t+1} | Target encoder 产生的下一时刻 latent target。 |
| R(z) | State-conditioned basis weight estimator。 |
| α_k | 第 k 个 expert 的 mixture coefficient。 |
| F_k(z,a) | 第 k 个 neural transition expert。 |
| B_k=∂F_k/∂a | 第 k 个 expert 的 local basis control dynamics。 |
| J_pred | 整个 predictor 的 action Jacobian；本方案中 J_pred=Σ_k α_k B_k。 |
| v | 用于 JVP / local perturbation 的 action direction。 |
| r* | 真实 directional control-response J_true v。 |
| L_CR | Control-Response Consistency loss。 |

---

## Codex 使用约定

- 本文档是当前实验实现的设计依据。
- 若代码行为与本文档冲突，应先报告冲突，不要擅自改写实验定义。
- 第一阶段优先完成低维 deterministic synthetic Experiment 1。
- 在 Experiment 1 机制验证完成前，不进入 Visual JEPA。
- Ground-truth expert identity 不是评价目标；重点是 local response、dictionary coverage、reuse 与 recombination。
- Router 在第一阶段只依赖 state，不读取当前 action。
- 未确认前不要删除 legacy 代码；先做可复用性审计。
