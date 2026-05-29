# ACT_LatentCorr 项目移交文档（供新对话直接接手）

更新时间：2026-04-10
项目根目录：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr`

## 1. 项目当前阶段的核心结论

这条项目线的核心目标，是在 ACT 策略上加入“闭环纠错”能力：策略不仅要会模仿专家轨迹，还要在动作发生偏差、状态偏离任务轨迹时，能够生成恢复动作并回到任务流程。我们已经完成了从方法实现、训练、并行评测、自动汇总到问题定位的一整套基础设施，当前阶段最重要的结论有三个。第一，旧版闭环方法虽然能拿到不错的结果，但其本质更接近“往 GT 最近点回拉”，方法定义并不干净；第二，ACT-aligned 结构化纠错动作生成已经被完整接入到我们自己的 `ACT_LatentCorr` 目录中，不再依赖外部 `policy/ACT` 目录；第三，在统一的 `100 seed bridge` 协议下，当前 ACT-aligned 扰动版的最好结果是 `ep600 = 83/100 = 83.0%`，这是当前这条主线最有代表性的 checkpoint。

## 2. 当前项目中我们真正做过的三条方法线

### 2.1 旧版 legacy 闭环（历史参考，不建议作为后续主线）

旧版方法的核心是：用当前动作末状态去 GT 局部轨迹里找最近点，然后直接构造一条“往回拉”的纠错前缀。它能训练、也能出结果，但后来我们确认它并不是真正意义上的“先检测异常、再触发纠错”，而更像一种 GT manifold pullback，因此方法解释上存在缺口。它的最好 bridge 结果来自更早那条旧版路线，在 `150 seed` 协议下有一个代表性结果：`ep200 = 123/150 = 82.0%`。这一结果可以作为历史上界或早期参考，但不建议再继续沿这条线推进。

### 2.2 ACT-aligned 扰动版（当前最稳定、结果最好的一条主线）

这条路线已经把 ACT 那边“结构化纠错动作生成”的关键逻辑迁到了我们自己的目录下，并且训练和评测都已经完整跑通。它的基本机制是：先人为构造错误动作或错误情形，再判断后续能否恢复，并生成更结构化的纠错动作轨迹，而不是简单最近点回拉。当前这条线是整个项目里最稳定、最好用的一条，因为它既能持续产生 correction sample，又有一整套多卡训练与多卡评测链路支撑。虽然从方法定义上看，它仍然带有“人为构造偏差样本”的成分，不是我们理想中的最终形态，但在实验上它是当前最强、最成熟的一条线。

### 2.3 trigger-only 真实偏差触发版（方法定义更干净，但工程上仍未完全稳定）

这条路线的目标是：不再人为加扰动，而是只在模型真实产生偏差时才触发纠错动作生成。从研究定义上讲，这条线更符合我们真正想要的闭环训练目标。我们已经做过多轮修复，包括真实偏差阈值触发、多卡全局 skip、样本级超时保护、分支统计、timeout/error 统计等，并且单卡和 4 卡 smoke test 都证明它可以产生有效训练信号。但在长时间多卡正式训练里，这条线仍然容易因为在线 planner / rollout / correction builder 的样本耗时差异而出现卡死或 DDP 不稳定，所以现阶段不建议直接把它作为主线继续推；它更适合作为后续重构和深入研究的方向。

## 3. 目前 ACT-aligned 主线的关键实现文件

以下这些文件是当前项目最关键、最需要在新对话里继续理解和沿用的代码入口。

- 训练主入口：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_latent.py`
- stage2 启动封装：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2.sh`
- DDP 启动封装：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_ddp.sh`
- 模型主逻辑：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py`
- ACT-aligned builder 适配层：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_correction.py`
- 本地化 ACT-aligned 纠错逻辑：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_pkg/correction.py`
- 本地化 ACT-aligned 扰动逻辑：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_pkg/perturbation.py`
- 并行评测 launcher：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh`
- 固定 `100 seed` 评测 launcher：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_eval_g1g2_100.sh`
- 单组评测 summary 写出：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_parallel_eval_summary.py`
- 多组结果汇总 summary 写出：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_grouped_eval_summary.py`
- 顺序评测 `300~800` 的总控脚本：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_eval_series_actaligned_bridge100_300to800.sh`
- 顺序评测 `300~800` 的执行脚本：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_eval_series_actaligned_bridge100_300to800.sh`
- 评测实际执行包装：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/eval_success.sh`

## 4. 当前统一的评测协议

我们后期已经把评测协议统一成固定的 `100 seed bridge` 口径，这比最早的 `150 seed` 更适合快速比较多个 checkpoint。具体做法是固定只跑 `g1 + g2` 两组，总共 `100` 个有效 seed；这两组由 `launch_eval_g1g2_100.sh` 自动并行切 shard 并生成汇总。历史上我们也用过 `150 seed` 协议，但后面主要实验和横向比较都已经切到 `100 seed bridge`，因此新对话里如果要继续复现实验结果，应优先沿用这套 `100 seed` 协议，避免和早期 `150 seed` 数字混淆。

## 5. ACT-aligned 扰动版训练权重链路

当前主线的 checkpoint 是连续接起来的，分为几段。

第一段，初始 `0 -> 50 epoch`：
`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_free0123/20260405_220904/`
其中关键权重包括：
- `stage2_epoch_0025.pt`
- `stage2_epoch_0050.pt`

第二段，`50 -> 250 epoch`：
`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume50_to250_free01234567/20260405_235020/`
其中关键权重包括：
- `stage2_epoch_0075.pt`
- `stage2_epoch_0100.pt`
- `stage2_epoch_0125.pt`
- `stage2_epoch_0150.pt`
- `stage2_epoch_0175.pt`
- `stage2_epoch_0200.pt`
- `stage2_epoch_0225.pt`
- `stage2_epoch_0250.pt`

第三段，`250 -> 700 epoch`：
`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357/20260406_165413/`
其中关键权重包括：
- `stage2_epoch_0275.pt`
- `stage2_epoch_0300.pt`
- `stage2_epoch_0325.pt`
- `stage2_epoch_0350.pt`
- `stage2_epoch_0375.pt`
- `stage2_epoch_0400.pt`

第四段，`400 -> 1000 epoch` 夜间续训：
`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_resume400_to1000_free01234567/20260407_005658/`
目前已经保存到：
- `stage2_epoch_0425.pt`
- `stage2_epoch_0450.pt`
- `stage2_epoch_0475.pt`
- `stage2_epoch_0500.pt`
- `stage2_epoch_0525.pt`
- `stage2_epoch_0550.pt`
- `stage2_epoch_0575.pt`
- `stage2_epoch_0600.pt`
- `stage2_epoch_0625.pt`
- `stage2_epoch_0650.pt`
- `stage2_epoch_0675.pt`
- `stage2_epoch_0700.pt`
- `stage2_epoch_0725.pt`
- `stage2_epoch_0750.pt`
- `stage2_epoch_0775.pt`
- `stage2_epoch_0800.pt`
- `stage2_epoch_0825.pt`
- `stage2_epoch_0850.pt`
- `stage2_epoch_0875.pt`

## 6. 目前最有代表性的 ACT-aligned 扰动版结果

### 6.1 100-seed bridge 主结果

当前这条主线最有代表性的横向结果，已经在顺序评测总表里统一汇总：
`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/actaligned_bridge100_300to800_20260407_091220/summary.txt`

结果如下：
- `ep300 = 70/100 = 70.0%`
- `ep400 = 71/100 = 71.0%`
- `ep500 = 77/100 = 77.0%`
- `ep600 = 83/100 = 83.0%`
- `ep700 = 77/100 = 77.0%`
- `ep800 = 73/100 = 73.0%`

这一串结果说明，**ACT-aligned 扰动版当前的最好 checkpoint 是 `ep600`，在统一 `100 seed bridge` 协议下达到 `83.0%`**。如果新对话需要快速知道“当前最好的 ACT-aligned 主结果是什么”，优先直接引用这组数字。

### 6.2 更早的关键参考点

在 `50 -> 250 epoch` 这一段里，我们也单独测过：
- `ep200 = 80/100 = 80.0%`
  - 汇总：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage2_ep200_actaligned_bridge100_summary_20260406/summary.txt`
- `ep250 = 68/100 = 68.0%`
  - 汇总：`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/stage2_ep250_actaligned_bridge100_summary_20260406/summary.txt`

这两个点的重要性在于，它们表明 ACT-aligned 扰动版在继续训练过程中并不是单调上升的，而是在后期重新上升，并最终在 `ep600` 达到最优。因此，如果新对话要分析训练动态，可以用 `ep200 -> ep250 -> ep500 -> ep600` 这几个点作为关键拐点。

## 7. 历史 legacy 结果（仅作参考，不作为当前主线）

更早那条旧版闭环 / bridge 路线的历史最好结果之一，是在 `150 seed` 协议下得到的 `ep200 = 123/150 = 82.0%`。相关历史总表可以参考：
`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/bridge_eval_resume_from_ep400_20260405_112640/summary.tsv`
其中同一条历史线里还有：
- `ep300 = 113/150 = 75.3%`
- `ep400 = 119/150 = 79.3%`
- `ep500 = 117/150 = 78.0%`
- `ep600 = 116/150 = 77.3%`
- `ep700 = 121/150 = 80.7%`
- `ep800 = 116/150 = 77.3%`

这些结果的意义主要是帮助新对话理解：旧版最近点回拉确实曾经拿过不错数字，但我们后来不再把它当作方法定义上最干净的路线。

## 8. trigger-only 路线当前状态

trigger-only 的目标是：不再做人为扰动，而是只在真实偏差发生时才触发结构化纠错动作生成。从研究定义上，它比扰动版更符合“真正闭环”的要求。我们已经为这条线做过以下关键修复：真实偏差阈值触发、builder 超时保护、多卡全局 skip、timeout/error 统计、本地化纠错逻辑封装、4 卡 smoke test 等。单卡和短 smoke test 都证明它能产生有效训练信号，但在长时间多卡正式训练上，仍然容易因为在线 planner / rollout / correction builder 的样本耗时差异而卡住。因此，这条线目前不建议直接作为生产实验主线，而是建议保留为后续继续重构和深入研究的方向。

和 trigger-only 相关的关键文件是：
- `act_aligned_correction.py`
- `act_aligned_pkg/correction.py`
- `train_stage2_latent.py`
- `train_stage2.sh`
- `train_stage2_ddp.sh`

如果新对话后面要重启这条线，建议优先围绕“样本生成与训练解耦”“异步 correction 生成”“planner 开销分离”这几个方向来做，而不是继续硬把在线 planner 版直接塞进 DDP 长训。

## 9. 弱策略筛选实验与结论

为了给 trigger-only 找到一个更合理的“真实偏差样本产生器”，我们做了一轮弱策略筛选实验：在训练集 `50 seed / bridge` 协议下，测量较早 checkpoint 的能力水平。对应结果如下：
- `ep5 = 43/50 = 86.0%`
- `ep10 = 43/50 = 86.0%`
- `ep15 = 33/50 = 66.0%`
- `ep20 = 42/50 = 84.0%`
- `ep25 = 46/50 = 92.0%`
- `ep50 = 41/50 = 82.0%`

这些 summary 分别在：
- `probe25_train50_bridge_ep005_20260406_214352/summary.txt`
- `probe25_train50_bridge_ep010_20260406_220022/summary.txt`
- `probe25_train50_bridge_ep015_20260406_221503/summary.txt`
- `probe25_train50_bridge_ep020_20260406_223143/summary.txt`
- `ep25_train50_bridge_20260406_203000/summary.txt`
- `ep50_train50_bridge_20260406_201321/summary.txt`

这一轮实验最重要的结论是：**`ep15` 是当前最像“弱但还可用”的策略**，因为它在 `train50 bridge` 上只有 `66.0%`，明显比 `ep5/ep10/ep20/ep25/ep50` 更弱，但还没有弱到完全不会做任务。因此，如果后续还要继续研究“弱策略产生真实偏差样本”的路线，`ep15` 是最合适的候选起点。相应弱策略探测训练得到的权重在：
`/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_from_unified_ep400_actalignedcorr_probe25_free01234567/20260406_205807/`
其中关键 checkpoint 包括：
- `stage2_epoch_0005.pt`
- `stage2_epoch_0010.pt`
- `stage2_epoch_0015.pt`
- `stage2_epoch_0020.pt`
- `stage2_epoch_0025.pt`

## 10. 目前项目的阶段性结论

如果只看“当前最成熟、最值得保留的主线”，答案很明确：**ACT-aligned 扰动版是当前最稳定、最好用的路线，它的最佳 `100 seed bridge` 结果是 `ep600 = 83.0%`**。如果只看“后续最值得继续研究的方法方向”，答案也很明确：**trigger-only 真实偏差触发版在定义上更干净，但还需要继续解决多卡稳定性与在线样本生成代价的问题**。如果只看“未来最可能帮助 trigger-only 真正产生足够训练信号的设计”，当前最有价值的实验发现是：**可以考虑用 `ep15` 这样的弱策略来产生更多真实偏差样本**。

因此，新对话如果要“无损接手”当前项目，最重要的是牢记这三条：
1. 当前最好的已跑通主线结果：`ep600 = 83/100 = 83.0%`。
2. 当前最干净但未完全稳定的方法方向：trigger-only。
3. 当前最有潜力支撑 trigger-only 的弱策略候选：`ep15`。

## 11. 如果新对话要切到 PI0，该怎么接

如果下一个方向是“把 ACT 换成 PI0”，建议新对话不要重新从零理解整个项目，而是把这份文档作为唯一上下文入口。新对话里需要知道的核心不是所有历史细节，而是以下三点。第一，当前项目的核心问题不是“能不能训练闭环”，而是“纠错样本从哪里来、如何触发”；第二，现有代码里关于 ACT-aligned 纠错动作生成、并行评测、结果汇总、多卡训练封装这些基础设施都已经比较成熟，可以直接借给 PI0 这条新线；第三，PI0 迁移时最值得保留的不是 ACT 头本身，而是“闭环训练的样本构造逻辑、评测协议、权重管理与结果汇总方式”。因此，新对话如果开始做 PI0，建议优先回答三个问题：PI0 当前的动作输出接口是否也是 chunk-based、PI0 的训练入口能否接收类似的 correction sample、PI0 是否可以复用现有 `100 seed bridge` 评测协议。如果这三个问题能接上，后面的迁移就会顺很多。

## 12. 新对话建议先读哪些文件

如果新对话一上来时间有限，建议按下面顺序读：
- 第一步读结果总表：
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs/actaligned_bridge100_300to800_20260407_091220/summary.txt`
- 第二步读训练主入口与模型主逻辑：
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_latent.py`
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py`
- 第三步读本地化纠错 builder：
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_correction.py`
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_pkg/correction.py`
- 第四步读评测链路：
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_eval_g1g2_100.sh`
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh`
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_parallel_eval_summary.py`
  - `/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/write_grouped_eval_summary.py`

做到这一步，新对话基本就已经能无损接管整个项目，并直接开始新的实验设计。
