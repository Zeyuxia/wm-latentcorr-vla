# ACT_LatentCorr 阶段性总结（截至 2026-05-11）

本文档面向阶段性复盘，重点总结当前 ACT 多任务单视角潜空间纠错主线的研究动机、路线演化、算法原理、当前实现语义、代表性实验观察以及下一步值得投入的方向。范围上只覆盖 policy/ACT_LatentCorr 这条 ACT 代码线，不覆盖已经分开的 PI0.5 训练线。

## 1. 一页摘要

- 当前主线已经从早期的 closed-loop / stage2 / bridge 路线，收缩为“基础 ACT 权重 + ACT-aligned explore + unified stage1 + pred 推理”的更干净范式。
- 当前真正的理论核心不是 latent 对齐本身，而是把训练时依赖未来观测的 condition token，改造成推理时可由当前观测直接预测的 deployable token。
- 当前 clean restart 的真实语义已经固定为：normal batch 的未来 latent 用 EVAC VAE 编码 GT future frame；failure batch 的未来 latent 用当前错误观测加纠错动作前缀做 WM rollout。
- align loss 虽然仍被计算和记录，但权重已经设为 0；wm action decoder 和 bridge future 相关损失也关闭，主目标集中在 action / pred-conditioned action / token / dynamics 四项。
- 已有可行性验证说明“当前输入到 future condition token”的映射并非严重 one-to-many，简单 probe 就能显著优于均值基线，因此 predictor 路线在统计上是站得住的。
- 当前最大的瓶颈更像是 failure table 质量与分布、failure batch 在线构造成本、multitask 任务干扰、以及 predictor 可能丢失空间细节。
- 代表性结果上，open_laptop 的 pred 推理在一轮 50-seed 无视频评测里，450 epoch 权重达到 86.0%，但整体趋势并不单调。

## 2. 研究动机

这条工作的起点不是“再做一个更大的行为克隆模型”，而是要解决 open-loop ACT 在真实偏差状态上的恢复能力不足问题。基础 ACT 在成功分布上可以很强，但进入失败态、偏位态或抓取姿态错误态后，往往会继续沿着错误轨迹前进，而不是主动回到正确轨道。

因此，我们希望引入世界模型和潜空间监督，让策略不仅学会“看到当前画面输出动作”，还学会“当前状态相对于未来目标状态缺了什么，以及怎样恢复”。从本质上讲，这是一条把行为克隆向可恢复控制推进的路线。

- 动机 1：open-loop 成功轨迹训练无法覆盖偏差状态。
- 动机 2：多任务训练下，单纯增加数据不一定提升恢复能力，反而可能放大任务间干扰。
- 动机 3：EVAC / WM 提供了未来状态表示，可以作为“未来该到哪里”的教师信号。
- 动机 4：真正可用的方法必须在推理时可部署，不能依赖未来图像、未来动作或双次推理。

## 3. 路线演化

1. 第一阶段是纯 open-loop ACT 多任务基线。这一阶段给了我们一个非常重要的事实：基础 ACT 已经不弱，因此任何新方法都不能只在训练 loss 上看起来更复杂，而必须在真实评测上证明自己。
2. 第二阶段曾经尝试更完整的 closed-loop / stage2 / bridge 路线，希望显式生成未来动作，再用未来动作驱动 latent predictor。但这条路线后来被证明工程上过重、推理上割裂，并且与新的 stage1 逐渐重叠。
3. 第三阶段把焦点转移到 unified stage1：不再单独维护 stage2，而是在同一个训练器里混合 normal batch 和 failure batch，让模型同时看到正常样本与纠错样本。
4. 第四阶段识别并解决了最关键的理论缺陷：原来的 conditioned path 依赖 teacher future token，训练时可用，推理时不可得，因此 train/test mismatch 很严重。于是我们引入 FutureTokenPredictor，把 condition token 变成当前观测可直接预测的 deployable token。
5. 第五阶段进一步修正 failure batch 的监督语义：normal batch 的 future latent 应该仍来自 VAE 编码的 GT future frame，而 failure batch 的 future latent 不能继续用编码器直取，而应来自“当前错误观测 + 纠错动作前缀”的 WM rollout。
6. 第六阶段是数据与工程修补：新的 EVAC 权重、完整 explore、task_name 回填、failure table merge 与无效项 pruning，让训练入口尽量干净稳定。

## 4. 当前端到端 workflow

1. 先训练五任务单视角的基础 ACT 多任务权重，作为所有 LatentCorr 训练的初始化起点。
2. 运行 ACT-aligned explore，对关键任务相位施加扰动，评估策略是否能自恢复，并把“恢复成功率低于阈值”的 failure unit 保留下来。
3. 对 explore 结果做 task_name 回填、merge 和无效项 pruning，得到统一的 failure_table.json。
4. 启动 unified stage1 训练。每个 step 混合 normal batch 与 failure batch；normal batch 负责保持基础行为克隆能力，failure batch 负责引入纠错恢复监督。
5. 训练时在当前 ACT 视觉特征上额外挂载 latent projector、WM adapter、future token predictor 和 action-conditioned predictor，但尽量少改原始 ACT 主体。
6. 推理与评测时默认走 pred 路径：由当前观测直接预测 deployable condition token，在同一次 ACT forward 中消费它。

## 5. 当前训练配置快照

- 训练主线: ACT 基础权重 + ACT-aligned explore + unified stage1 + pred 推理
- 任务数: 5 个多任务单视角任务
- 相机: cam_high（单视角）
- 动作 horizon H: 50
- future offset Δ: 16
- 每卡 batch: normal=4, failure=2
- 全局 batch: 24
- 基础初始化权重: policy_epoch_2000_seed_0.ckpt
- failure table: failure_table.json
- EVAC checkpoint: epoch=333-step=10000.ckpt
- EVAC config: train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml
- lambda_action: 1.0
- lambda_action_conditioned: 0.5
- conditioned loss schedule: true
- lambda_condition_token: 0.5
- lambda_align: 0.0（当前关闭，只记录不优化）
- beta_dynamics_max: 1.0
- failure future latent: rollout
- 当前主动优化的核心项: L_action + scheduled L_cond-pred + scheduled L_token + scheduled L_dyn
- 当前关闭/不启用的历史项: L_align=0, L_wm_action_current=0, L_wm_action_future=0, L_bridge_future=0

## 6. Explore 与 failure table 语义

当前 explore 的高层逻辑是：先人为生成偏差状态，再判断模型是否能从这个状态自己恢复。如果某个 failure unit 自恢复成功率过高，就说明这个错误模式对当前策略并不“致命”，没有必要放进统一训练里占用 failure batch 配额。反过来，如果某个 unit 多次尝试后仍很难恢复，它才值得进入训练。

```text
r(u) = (# successful recoveries of unit u) / k
```

```text
keep(u) = 1 if r(u) < tau, else 0
```

当前统一口径是 k = 4，阈值 tau = 0.5。也就是说，每个 failure unit 会被测试 4 次，如果恢复成功率小于 0.5，则该 unit 被保留。

当前新 EVAC explore 产出的 failure table 总计 7985 个条目。按任务统计：

- put_bottles_dustbin: 4512
- handover_block: 1238
- open_laptop: 916
- place_burger_fries: 915
- pick_dual_bottles: 404

这张表已经清楚说明 failure data 分布高度不均衡，put_bottles_dustbin 明显占主导。

训练启动时还出现了额外的无效 entry pruning：

- put_bottles_dustbin: 986
- place_burger_fries: 36
- handover_block: 507

## 7. 算法原理：旧方法的问题

```text
x_t = (o_t, q_t)
```

```text
z_t^{act} = f_act(o_t)
```

```text
z_{t+\Delta}^{wm} = E_{wm}(o_{t+\Delta})
```

```text
\tilde{z}_{t+\Delta}^{wm} = A(z_{t+\Delta}^{wm})
```

```text
c_t^* = P(pool(\tilde{z}_{t+\Delta}^{wm}))
```

```text
\hat{a}_{t:t+H-1}^{cond} = \pi_{ACT}(x_t; c_t^*)
```

问题在于 c_t^* 依赖未来图像，因此训练时优化的是 π_ACT(x_t; c_t^*)，推理时真正能跑的却是 π_ACT(x_t)。这就是旧 conditioned path 的核心割裂点。

## 8. 当前 deployable condition token 路线

```text
z_t^{proj} = \Pi(f_act(o_t))
```

```text
h_t = [pool(z_t^{proj}); q_t]
```

```text
\hat{c}_t = g_\phi(h_t)
```

```text
\hat{a}_{t:t+H-1}^{pred} = \pi_{ACT}(x_t; \hat{c}_t)
```

teacher future token 仍然存在，但只作为监督目标，不再直接充当部署路径输入。真正部署时使用的是从当前观测直接预测出的 token。

## 9. 当前 loss 组成与调度

```text
\mathcal{L}_{base} = \ell(\pi_{ACT}(x_t), a_{t:t+H-1})
```

```text
\mathcal{L}_{cond-pred} = \ell(\pi_{ACT}(x_t; \hat{c}_t), a_{t:t+H-1})
```

```text
\mathcal{L}_{token} = ||\hat{c}_t - c_t^*||_2^2
```

```text
\mathcal{L}_{dyn} = ||F(z_t^{proj}, a_{t:t+\Delta-1}) - \tilde{z}_{t+\Delta}^{wm}||_2^2
```

```text
\mathcal{L} = \lambda_a \mathcal{L}_{base} + \lambda_c(t) \mathcal{L}_{cond-pred} + \beta(t) \lambda_{tok} \mathcal{L}_{token} + \beta(t) \mathcal{L}_{dyn}
```

当前 schedule_action_conditioned=true，因此 conditioned loss 权重会渐进增长；token loss 与 dynamics loss 也都由同一个 warmup β(t) 调度。align loss 当前权重为 0，只记录不优化。

## 10. normal batch 与 failure batch 的监督语义

1. normal batch：future latent 来自 GT future frame 的 EVAC VAE encode。
2. failure batch：先构造当前错误观测与纠错动作 chunk，再取前缀 16 步纠错动作做 rollout，得到 failure future latent。
3. 因此 failure future latent 表达的是“执行纠错后应该到达的未来状态”，而不是“当前错误状态的编码”。

```text
normal: z_{t+\Delta}^{teacher} = E_{wm}(o_{t+\Delta}^{gt})
```

```text
failure: z_{t+\Delta}^{teacher} = Rollout_{wm}(o_t^{err}, q_t^{err}, a_{corr,t:t+\Delta-1})
```

## 11. 为什么现在不是双次推理

```text
x_t -> z_t^{act} -> z_t^{proj} -> \hat{c}_t
```

```text
(x_t, \hat{c}_t) -> \hat{a}_{t:t+H-1}^{pred}
```

pred 路径只是当前观测图上的一个额外 predictor head，不需要两次 ACT 解码。

## 12. condition token 可行性分析

```text
x_t -> c_t^*
```

```text
\hat{c}_t = g_\phi(x_t)
```

现有分析使用 pool(projector latent) 与 qpos 的拼接作为特征，维度为 18；目标 token 维度为 512。指标如下：

- feature_dim: 18
- token_dim: 512
- linear_probe_mse: 3.391e-06
- mean_baseline_mse: 7.468e-06
- linear_probe_gain: 2.20
- knn_future_mse: 3.347e-06
- global_adjacent_pair_mse: 1.504e-05
- exact_duplicate_groups: 1
- exact_duplicate_target_mse: 4.099e-07

结论是：当前没有证据表明 predictor 会被严重 one-to-many 直接否掉；它在统计上是可学的。

## 13. 当前实现与代码职责

- train_stage1_unified_failure_multitask_latent.py: 统一训练入口；构造 normal/failure batch；在线调用 correction builder；为 failure batch 生成 rollout future latent。
- latent_policy.py: ACTLatentStage1 主体；定义 projector、predictor、token loss、dynamics loss、pred-conditioned action path；包含 pred 推理接口。
- latent_modules.py: 定义 FutureTokenPredictor、LatentProjector、ResidualLatentAdapter、ActionConditionedPredictor 等模块。
- deploy_policy.py: 部署包装；支持 base / pred / teacher / bridge 模式；当前 pred 模式直接调用 predicted-conditioned 路径。
- act_aligned_correction.py: 在线构造 failure 样本；根据当前模型、错误状态和 planner 逻辑生成纠错轨迹。
- merge_failure_tables.py: 多 rank explore 结果合并。
- backfill_explore_task_names.py: 为老 explore 结果回填 task_name，保证 failure unit 能被训练器正确解析。

## 14. 当前训练状态与代表性实验观察

- 当前 clean restart 训练日志的最新已完成 epoch 为 194。
- 最新日志摘要：total loss=1.3398，action=0.0630，cond=0.0634，ctoken=0.0000，dyn=1.2450。
- 对应 warmup 已经拉满：beta=1.000，conditioned loss 实际权重=0.500。
- 日志显示的 failure batch 有效条目数约为 2.00，skip 约为 0.00。
- 每个 epoch 的墙钟时间约 80.0 秒，日志里 builder=3.01 秒、rollout=1.82 秒，说明 failure 样本构造仍是重要开销来源。

open_laptop 的 pred 推理 50-seed 无视频结果：

- epoch 350: 39/50 = 78.0%
- epoch 400: 40/50 = 80.0%
- epoch 450: 43/50 = 86.0%
- epoch 600: 40/50 = 80.0%

450 epoch 在当前已检查结果里最好，但趋势并不单调。

## 15. 当前遇到的核心问题

1. 多任务收益不一致。
2. checkpoint 表现强烈非单调。
3. failure data 分布严重失衡。
4. failure batch 在线构造昂贵，且 correction builder 依赖当前模型，难以静态 cache。
5. FutureTokenPredictor 目前使用 GAP，可能丢失微小但关键的空间偏差。
6. ctoken 日志可观测性不足。
7. pred 评测链路历史上比 base 更脆弱。
8. EVAC 更新会改变 explore 与 failure table 质量，旧表会部分过时。
9. 辅助 loss 是否真正稳定转化为最终成功率增益，目前仍缺少足够干净的 ablation。

## 16. 已经可以确认的结论

- 统一到 unified stage1 是对的。
- condition token 的 train-test mismatch 必须显式解决。
- FutureTokenPredictor 路线在统计上可行。
- failure batch 必须使用 rollout future latent，而不是简单的错误图像 VAE encode。
- 当前瓶颈更多来自数据分布与训练动态，而不是单纯的模型容量不足。
- checkpoint 选择必须依赖固定评测协议，不能只看 latest。

## 17. 值得优先尝试的新思路或新方案

1. 先做 data-first 修正：基于最新 EVAC 完成更干净的 explore，并对 failure table 做按任务、按 phase 的重平衡。
2. 升级 token predictor 的信息汇聚方式，用 attention pooling 或 token learner 取代单纯 GAP。
3. 系统做 loss ablation，重点比较 token loss 权重与 conditioned loss ramp。
4. 在同一 checkpoint 上严格比较 pred 与 base 推理，区分训练问题和推理路径问题。
5. 考虑让 correction builder 脱离当前在线训练模型，例如使用 lagged EMA teacher，从而释放缓存空间。
6. 若多任务干扰持续明显，考虑 task-balanced failure sampling 或 task-specific predictor adapter。
7. 探索更弱约束的 condition 目标，例如 low-rank condition code、condition residual，或直接蒸馏动作改进而不是逐维模仿 token。

## 18. 推荐的下一步实验顺序

1. 固定一个干净参考线。
2. 先做 token / ramp / pred-vs-base 三类小规模 ablation。
3. 若仍不稳定，优先动数据与 predictor 汇聚方式，而不是继续堆 epoch。
4. 只有单任务确认 pred 确实优于 base 后，再继续大规模多任务 sweep。

## 19. 关键文件与路径

- condition token 理论分析: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/condition_token_方法与可行性分析_20260429.md
- 算法上下文: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/ALGORITHM_CONTEXT.txt
- 实现规划: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/CODE_IMPLEMENTATION_PLAN.txt
- 统一训练入口: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_unified_failure_multitask_latent.py
- 模型主体: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py
- 模块定义: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_modules.py
- 部署入口: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/deploy_policy.py
- 当前 clean restart launcher: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_stage1_unified_failure_multitask_fulltable_lctok05_gpu0123.sh
- 当前训练日志: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage1_unified_failure_multitask_fulltable_lctok05_gpu0123_20260511_163455/train.log
- 当前 failure table: /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/stage2_multitask_failure_explore_new_evac_gpu0123_20260429_004924/failure_explore/failure_table.json

## 20. 总结

如果把整个阶段压缩成一句话，那么当前 ACT_LatentCorr 的核心贡献并不是“把 WM 强行接进 ACT”，而是逐步把一条原本训练时成立、部署时断裂的 condition 路线，改造成了在训练和推理两端都闭合的可部署路径。这个方向本身是成立的，但它距离稳定、普适地提升五任务多任务成功率，还差最后一段最难的路：数据分布、failure 质量、任务干扰和 predictor 信息瓶颈。

换句话说，我们现在已经不再处于“方法是否完全错误”的阶段，而是处于“主框架已经成型，但还没有把收益稳定释放出来”的阶段。接下来最值得做的事情，不是继续无差别加大训练，而是围绕数据质量、损失耦合和部署路径做更克制、更干净的验证。
