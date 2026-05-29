# ACT / PI0 LatentCorr 技术报告

## 1. 报告目的

这份报告用于在切换对话后，无损继续 `PI0_LatentCorr` 实验。报告分为两部分：第一部分描述当前已经完整实现并多轮实验验证的 `ACT_LatentCorr` 两阶段算法、训练流程与推理流程；第二部分描述 `PI0_LatentCorr` 当前已经完成的迁移内容、与 ACT 对齐到什么程度、以及目前还没有补齐的部分。报告只写当前工作区中可以直接核对的内容；凡是当前目录里不存在、或明显依赖外部未跟踪文件的地方，都会明确指出。

## 2. 总体算法目标

整套方法的核心目标，是把“开环行为策略”扩展成“可感知偏差、可在错误状态上生成纠错动作的闭环策略”。方法分成两个阶段。`Stage 1` 先学习一个跨模型的潜空间桥接：把策略网络内部的视觉表征投影到 EVAC / world model 的 latent 空间，并通过动作条件预测器学习“给定当前图像 latent 和动作前缀，预测未来错误/目标状态 latent”。`Stage 2` 再利用闭环探索或 failure table 采样出来的错误模式，构造“错误状态 -> 纠错动作”的训练样本，并同时保留三个关键约束：第一，错误状态确实来自错误动作执行后的状态；第二，用 dynamics loss 把策略内部预测到的 future latent 对齐到 EVAC rollout 出来的 latent；第三，用 retain / anchor loss 把策略拉回到较好的开环策略附近，避免闭环训练把原始行为能力训漂。

## 3. ACT_LatentCorr：当前完整可运行的参考实现

### 3.1 Stage 1 算法过程

ACT 版 `Stage 1` 的核心代码在 [`train_stage1_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_latent.py) 和 [`latent_policy.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py)。训练时，数据由 `utils_latent` 中的 stage1 数据构造函数提供，输入至少包含当前时刻图像 `image_t`、未来时刻图像 `image_t1`、当前关节 `qpos_t`、未来关节或未来条件关节 `qpos_future_norm`、ACT 的动作 chunk `act_action_chunk`、动作 padding 掩码 `act_is_pad`、以及动作前缀 `action_prefix`。模型主体类是 `ACTLatentStage1`。它保留原始 `ACTPolicy` 作为 `base_act`，然后额外挂接五个 latent 相关模块：`projector`、`wm_adapter`、`readout_adapter`、`predictor`、`action_decoder`。其中 projector 把 ACT 的视觉特征图投影到 world model latent 空间；predictor 接收投影后的 latent 和动作前缀，预测未来 latent；action_decoder 则把 latent 解码回动作前缀。具体训练时，先从当前图像中抽取 ACT backbone 的视觉特征，再让 EVAC teacher 从当前图像和未来图像分别编码出 `z_wm_t` 与 `z_wm_t1`。然后通过 projector 产生 `z_proj`，用 `loss_align` 约束 `z_proj` 与 teacher latent 对齐；再用 predictor 产生 `z_hat_next`，用 `loss_dynamics` 约束其对齐到 `z_wm_t1`。同时还保留了原始 ACT 的动作监督 `loss_action`，并可以在开启 `use_act_head_conditioning` 时，把预测的 latent 通过 `act_condition_proj` 映射成额外 token 注入 ACT decoder，形成条件动作损失 `loss_action_conditioned`。除此之外，ACT 版 stage1 还实现了三项和潜空间动作读出有关的辅助损失：`loss_wm_action_current`、`loss_wm_action_future`、`loss_bridge_future`，它们分别监督当前 teacher latent、未来 teacher latent、以及 predictor 预测 future latent 能否被 decoder 解码成正确动作前缀。所有这些权重由 [`config_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/config_latent.py) 中的 `LatentLossConfig` 和 `DynamicsWarmupConfig` 控制，dynamics 权重通过 warmup 调度器逐步升高，而不是一开始就全力约束 latent dynamics。

### 3.2 Stage 1 训练文件与职责

ACT 版 `Stage 1` 的训练主入口是 [`train_stage1_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_latent.py)。这个文件负责解析训练参数、初始化 DDP、解析任务对应的数据路径、构造 ACT 配置、创建 EVAC teacher、构造 stage1 数据集和 dataloader、调用 `ACTLatentStage1.forward_stage1(...)` 计算损失、保存 checkpoint，并在需要时恢复 resume checkpoint。潜空间网络结构和 stage1 / stage2 前向逻辑都集中在 [`latent_policy.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py)。loss 与 latent 结构超参数定义在 [`config_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/config_latent.py)。训练时使用的 EVAC latent teacher 封装在 `evac_interface.py`，而数据构造与原始 episode 读取则依赖 `utils_latent.py`。实验记录和 wandb 相关逻辑在 `wandb_utils.py`。因此，如果要修改 ACT 的 stage1 训练语义，优先查看的文件顺序是：`train_stage1_latent.py -> latent_policy.py -> config_latent.py -> utils_latent.py -> evac_interface.py`。

### 3.3 Stage 2 算法过程

ACT 版 `Stage 2` 的核心代码在 [`train_stage2_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_latent.py)、[`latent_policy.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py)、[`stage2_failure_dataset.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/stage2_failure_dataset.py)、[`act_aligned_correction.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_correction.py)。Stage 2 的一个 batch 实际上由两类样本拼成：原始正常样本 batch，以及 correction batch。正常样本 batch 仍然提供原始图像、原始状态和 GT 动作 chunk；correction batch 则来自 failure table 采样或探索路径生成的错误模式。对于每个 correction sample，算法先在某个起点 `start_ts` 取当前图像和状态，然后得到一个“错误动作前缀”。这个错误动作前缀有两种来源：一种是旧版 planner / 人为构造逻辑；另一种是现在主要使用的 `ACT-aligned` 构造逻辑，它会基于 failure table 抽样的错误模式和扰动参数，生成更贴近 ACT 原始闭环探索语义的错误动作。得到错误动作前缀后，再调用 EVAC teacher 的 `rollout_latent_from_actions(...)` 把这个错误动作执行后的未来错误状态 rollout 成 latent，即 `z_wm_sim`。与此同时，当前策略也从当前图像和当前状态预测自己的开环动作 chunk，取前 `prefix_steps` 作为 `action_dev_norm`，再通过 predictor 预测 future latent `z_hat_next`。这样，Stage 2 至少有三条训练约束：第一条是 `loss_correct`，即“错误状态 latent 条件下的纠错动作损失”；第二条是 `loss_dynamics`，即 `z_hat_next` 对齐 `z_wm_sim` 的 latent dynamics 约束；第三条是 `loss_retain`，即使用开环 anchor 或 stage1 anchor 策略在当前正常状态上的输出做蒸馏，防止闭环训练把开环策略训漂。如果开启 bridge 相关项，还会有 `loss_bridge`。在当前正式 ACT 闭环配置里，常见做法是 `use_act_head_correction=true`，也就是先直接用 teacher rollout latent 作为条件 token 去训练 ACT 头生成纠错动作，再通过 dynamics loss 逐步把 predictor 学到和 teacher latent 对齐。这样训练前期条件更稳，后期再逐渐依赖 predictor latent，符合“先学会对 teacher latent 纠错，再学会 internal bridge”的思路。

### 3.4 Stage 2 中 failure table、纠错样本与 retain 的关系

ACT 当前闭环路线强调三个不能丢的语义。第一，纠错目标必须来自“错误动作已经执行后的错误状态”，而不是凭空在当前状态上做动作重写。第二，错误状态必须有潜空间对齐损失，也就是 `loss_dynamics`：让策略内部预测的 future latent 去对齐 EVAC rollout 出来的 latent。第三，必须保留 retain / anchor 约束，防止闭环纠错阶段把开环主能力训漂。对应代码里，failure table 与起点采样由 [`stage2_failure_dataset.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/stage2_failure_dataset.py) 提供；ACT 对齐的错误动作生成器封装在 [`act_aligned_correction.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_correction.py)，其中 `ACTAlignedCorrectionBuilder.build(...)` 会返回 correction 图像、纠错状态、纠错动作 chunk、padding 掩码、以及错误动作前缀等中间量；真正的 stage2 损失计算在 [`latent_policy.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py) 的 `prepare_stage2_context(...)`、`compute_stage2_loss(...)` 和 `forward_stage2(...)` 中；retain 权重的调度则由 `train_stage2_latent.py` 和 shell 启动器传入，如当前正式跑法中 retain 会从 `1.0` 逐渐降到 `0.1`，而 dynamics 相关权重在训练前期逐步升高。

### 3.5 Stage 2 训练文件与职责

ACT 版 `Stage 2` 的核心训练入口是 [`train_stage2_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_latent.py)，而真正的多卡启动入口是 [`train_stage2_ddp.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_ddp.sh)。`train_stage2_ddp.sh` 负责把 shell 层面的实验配置整理成大量环境变量与命令行参数，包括 `batch_size`、`correction_batch_size`、retain 调度、bridge 权重、failure table 路径、`ACT_ALIGNED_*` 全套扰动与纠错参数，然后用 `torchrun` 调起 `policy.ACT_LatentCorr.train_stage2_latent`。`train_stage2_latent.py` 负责组装 world model teacher、stage2 dataset、correction builder、rollout cache、anchor 模型、optimizer 和 DDP。正式训练时，`train_stage2_latent.py` 会先读取一批正常样本，再按 `correction_batch_size` 从 failure-aware dataset 或 correction builder 中构造纠错样本，把两者拼成训练 batch，然后调用 `ACTLatentStage1.forward_stage2(...)` 计算 stage2 loss。围绕 workflow 的辅助脚本包括 [`run_stage2_failure_workflow.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_stage2_failure_workflow.sh) 和 [`launch_stage2_failure_workflow_4gpu.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_stage2_failure_workflow_4gpu.sh)，它们更偏“先 explore 构表，再 train”的完整工作流封装，而不只是单次 stage2 训练。

### 3.6 ACT 推理 / 评测过程

ACT 当前使用的推理评测入口主要是 [`eval_success.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/eval_success.sh) 和 [`launch_parallel_eval.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh)，最终都调用统一的评测驱动 [`script/eval_policy.py`](/data/zhenyangfan/RoboTwin/script/eval_policy.py)。`eval_success.sh` 是单卡评测入口，它设置 `policy_name=ACT_LatentCorr`、指定 `latent_ckpt_path`、`inference_mode`、seed 或 seed_file、EVAC checkpoint 与 URDF，然后调用 `script/eval_policy.py --config policy/ACT_LatentCorr/deploy_policy.yml`。`launch_parallel_eval.sh` 则把 seed 文件按 shard 分片，给每个 GPU 开一个 tmux session，并在所有 shard 结束后自动调用 `write_parallel_eval_summary.py` 汇总结果。需要特别注意的一点是：按当前工作区快照，我没有在 `policy/ACT_LatentCorr` 目录里找到被 `eval_success.sh` 引用的 `deploy_policy.yml` 或 `deploy_policy.py` 文件，也就是说 ACT 当前的评测入口依赖一个没有出现在当前工作区快照里的 deploy 配置文件。这个细节在新对话里必须优先核对，否则会出现“训练文件都在，但评测入口复现不出来”的问题。

## 4. PI0_LatentCorr：当前迁移状态与算法过程

### 4.1 PI0 开环训练的定位

PI0 这条线当前分成“开环基础策略训练”和“latent-correction 两阶段训练迁移”两部分。开环基础策略的主入口是 [`train_openloop.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_openloop.py)，shell 启动入口是 [`train_openloop.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_openloop.sh)。`train_openloop.py` 支持 `jax` 和 `pytorch` 两条后端；如果选择 `jax`，会构造 OpenPI 的训练配置并直接调用官方 `openpi/scripts/train.py`；如果选择 `pytorch`，则转到 [`train_openloop_pytorch.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_openloop_pytorch.py)。这一层的目标不是纠错，而是先得到任务级 open-loop PI0 权重，后续 `Stage 1` 再在这个任务开环权重基础上继续训练 latent 头。当前实现已经支持优先从本地 `model.safetensors`、或任务 checkpoint 目录导出的 `model.safetensors` 初始化，而不是强依赖远程 Orbax / gs params 下载，这一修改是为了对齐 ACT 那种“先有任务开环，再做 stage1”的实验范式。

### 4.2 PI0 Stage 1 算法过程

PI0 版 `Stage 1` 的核心入口是 [`train_stage1.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage1.py)，模型主体是 [`stage1_model.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_model.py)，checkpoint 恢复工具是 [`stage1_checkpoint.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_checkpoint.py)，数据集是 [`stage1_dataset.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_dataset.py)。PI0 版 stage1 的总体思想与 ACT 保持一致：先让 PI0 自己的视觉特征对齐到 EVAC latent，再学习动作条件的 future latent predictor，再把预测 latent 作为额外条件 token 注入 PI0 的生成头中。与 ACT 不同的是，PI0 没有原生的 ACT decoder 接口，因此条件注入是在 PI0 的 prefix embedding 上额外拼接一个 latent token 实现的。训练时，`PI0Stage1Dataset` 从处理好的 episode 中读出 `image_t`、`image_t1`、`qpos_t_norm`、`qpos_t1_norm`、规范化后的 `act_action_chunk`、`action_prefix` 与 padding 掩码。`PI0LatentStage1.forward_stage1(...)` 先通过 `_extract_visual_feature(...)` 把当前图像编码成 PI0 图像 token map，再用 projector 投影到 EVAC latent 维度；然后用 predictor 结合动作前缀预测 `z_hat_next`；接着保留正常 open-loop 动作损失 `loss_action`，再可选地把 `z_hat_next` 投成 latent token 注入 PI0 条件分支，形成 `loss_action_conditioned`；最后用 `loss_dynamics` 约束 `z_hat_next` 对齐未来 teacher latent，用 `loss_align` 约束当前投影 latent 对齐当前 teacher latent。也就是说，PI0 stage1 目前已经完成了 ACT 风格的 latent projection、future latent prediction、latent-conditioned action generation、以及动态 warmup 调度。

### 4.3 PI0 Stage 1 训练文件与职责

PI0 版 stage1 训练主入口是 [`train_stage1.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage1.py)，多卡 shell 入口是 [`train_stage1_ddp.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage1_ddp.sh)，单卡 shell 入口是 [`train_stage1.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage1.sh)。其中 `train_stage1.py` 负责解析参数、选择 base PI0 初始化来源、构造 norm stats 路径、创建 `PI0Stage1Dataset`、创建 EVAC teacher、bootstrap latent heads、进入训练循环并保存 `stage1_step_*.pt`。模型结构和 stage1 / stage2 核心前向逻辑都在 [`stage1_model.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_model.py)。checkpoint 恢复和部署复原都由 [`stage1_checkpoint.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_checkpoint.py) 完成，它会根据 checkpoint 中记录的 `train_config_name` 和初始化权重路径，重建完整的 `PI0LatentStage1`。因此，PI0 版 stage1 的训练主线已经是完整的。

### 4.4 PI0 推理 / 评测过程

PI0 当前的推理部署入口是 [`deploy_policy.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/deploy_policy.py)，配置文件是 [`deploy_policy.yml`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/deploy_policy.yml)，评测入口是 [`eval.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/eval.sh)、[`eval_success.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/eval_success.sh) 和 [`eval_success_stage1.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/eval_success_stage1.sh)。`deploy_policy.py` 当前实现了两种模型形态。第一种是 `PI0OpenLoopModel`，直接加载 open-loop checkpoint，通过 OpenPI 官方 policy 接口做普通动作生成。第二种是 `PI0LatentStage1Deploy`，它会加载 `stage1_ckpt`，支持三种 `inference_mode`：`base`、`teacher`、`bridge`。`base` 模式只调用 open-loop PI0；`teacher` 模式先用 open-loop PI0 预测动作前缀，再用 EVAC 对这个动作前缀 rollout 得到 future teacher latent，并把 teacher latent 转成条件 token 注入 PI0，再生成 conditioned chunk；`bridge` 模式则同样先取开环动作前缀，但条件 latent 不再直接使用 teacher rollout，而是用 `predict_future_latent(...)` 预测出的 latent，再喂给 `predict_pi0_chunk_conditioned(...)`。因此，PI0 当前推理链路在 stage1 范围内已经完整具备了 ACT 对应的 base / teacher / bridge 三种语义。对于正式评测，`eval.sh` 最终仍然调用统一的 [`script/eval_policy.py`](/data/zhenyangfan/RoboTwin/script/eval_policy.py)，只是 policy 名称换成 `PI0_LatentCorr`，并通过 overrides 传入 `train_config_name`、`model_name`、`checkpoint_id` 或 `stage1_ckpt` 等参数。

### 4.5 PI0 Stage 2：当前已经完成的部分

PI0 原本只有“stage2 的核心模型语义”，但到 2026-04-18 晚上的最新代码状态，已经进一步补上了训练胶水层。具体来说，除 [`stage1_model.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_model.py) 中的 `Stage2LossOutput`、`Stage2PreparedContext`、`prepare_stage2_context(...)`、`compute_stage2_loss(...)` 和 `forward_stage2(...)` 之外，现在还新增了 [`evac_interface.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/evac_interface.py)、[`utils_latent.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/utils_latent.py)、[`train_stage2_latent.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage2_latent.py) 与 [`train_stage2_ddp.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage2_ddp.sh)。这条新链路已经把 ACT 的 failure-table 采样、ACT-aligned correction builder、EVAC rollout latent、retain 调度、多卡 `torchrun` 启动器都接到了 PI0 backbone 上，同时显式加入了一层“ACT 归一化空间 <-> PI0/OpenPI 归一化空间”的转换，避免直接把 ACT 的 `mean/std` 归一化误喂给 PI0。也就是说，PI0 现在不再只是“stage2 损失算子已迁移”，而是已经具备了一条可继续 smoke / debug / 正式训练的 stage2 训练入口；不过这条入口刚补完，当前仅完成了语法和导入级别检查，还没有在真实 checkpoint 上完成完整长程 smoke 训练与闭环评测验证，因此运行稳定性和最终效果仍需要下一轮实验进一步确认。

### 4.6 PI0 Stage 2：当前已迁移的辅助模块与未完成部分

为了继续迁移 PI0 stage2，我已经把 ACT 里与 failure-aware 数据和 ACT-aligned correction 直接相关、且和 backbone 弱耦合的三个模块复制到了 PI0 目录：[`stage2_failure_dataset.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage2_failure_dataset.py)、[`failure_utils.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/failure_utils.py)、[`act_aligned_correction.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/act_aligned_correction.py)。这意味着 PI0 目录里已经有了 failure table 的字段解析、error mode 离散化、和 ACT-aligned correction builder 的代码骨架。与此同时，迁移进度说明文件是 [`ACT_TO_PI0_MIGRATION_NOTES.md`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/ACT_TO_PI0_MIGRATION_NOTES.md)。但必须明确指出：截至当前快照，PI0 目录里还没有完成以下几项，因此“PI0 stage2 还不能像 ACT 那样直接开正式闭环训练”：第一，没有完整的 `train_stage2_latent.py`；第二，没有对应的 `train_stage2_ddp.sh`；第三，没有专门适配 PI0 数据目录和 raw episode 读取方式的 `utils_latent.py` 等 glue 代码；第四，还没有 stage2 版本的正式 eval success pipeline。因此，当前 PI0 目录的状态是：`stage1 + stage1 deploy 已经可用，stage2 核心模型语义已实现，但完整的 stage2 训练 / 评测工作流尚未打通`。

## 5. 训练与推理文件清单

### 5.1 ACT_LatentCorr 关键文件

- Stage 1 训练主入口：[`train_stage1_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1_latent.py)
- Stage 2 训练主入口：[`train_stage2_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_latent.py)
- Stage 2 多卡启动器：[`train_stage2_ddp.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage2_ddp.sh)
- 模型与损失核心：[`latent_policy.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/latent_policy.py)
- latent 超参数：[`config_latent.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/config_latent.py)
- failure-aware dataset：[`stage2_failure_dataset.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/stage2_failure_dataset.py)
- ACT 对齐纠错生成：[`act_aligned_correction.py`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/act_aligned_correction.py)
- 完整 failure workflow 启动：[`run_stage2_failure_workflow.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/run_stage2_failure_workflow.sh)
- 4 卡 workflow 封装：[`launch_stage2_failure_workflow_4gpu.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_stage2_failure_workflow_4gpu.sh)
- 单卡评测入口：[`eval_success.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/eval_success.sh)
- 并行分片评测入口：[`launch_parallel_eval.sh`](/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/launch_parallel_eval.sh)
- 通用评测驱动：[`script/eval_policy.py`](/data/zhenyangfan/RoboTwin/script/eval_policy.py)

### 5.2 PI0_LatentCorr 关键文件

- 开环训练主入口：[`train_openloop.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_openloop.py)
- 开环 PyTorch 训练实现：[`train_openloop_pytorch.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_openloop_pytorch.py)
- 开环 shell 启动器：[`train_openloop.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_openloop.sh)
- Stage 1 训练主入口：[`train_stage1.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage1.py)
- Stage 1 单卡启动器：[`train_stage1.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage1.sh)
- Stage 1 多卡启动器：[`train_stage1_ddp.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/train_stage1_ddp.sh)
- Stage 1 数据集：[`stage1_dataset.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_dataset.py)
- Stage 1 / Stage 2 模型核心：[`stage1_model.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_model.py)
- Stage 1 checkpoint 复原：[`stage1_checkpoint.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage1_checkpoint.py)
- 部署与推理：[`deploy_policy.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/deploy_policy.py)
- 部署配置：[`deploy_policy.yml`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/deploy_policy.yml)
- 开环评测入口：[`eval.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/eval.sh)、[`eval_success.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/eval_success.sh)
- Stage 1 latent 评测入口：[`eval_success_stage1.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/eval_success_stage1.sh)
- 并行评测封装：[`launch_parallel_eval.sh`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/launch_parallel_eval.sh)
- 已迁移的 stage2 failure / correction 模块：[`stage2_failure_dataset.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/stage2_failure_dataset.py)、[`failure_utils.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/failure_utils.py)、[`act_aligned_correction.py`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/act_aligned_correction.py)
- 迁移进度说明：[`ACT_TO_PI0_MIGRATION_NOTES.md`](/data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr/ACT_TO_PI0_MIGRATION_NOTES.md)

## 6. 当前最重要的事实结论

如果新对话要继续推进 `PI0.5` 实验，最重要的事实有三条。第一，`ACT_LatentCorr` 仍然是完整的、已经经过多轮真实闭环训练和评测验证的参考实现；任何 PI0 迁移都应该以它为语义参照，而不是只看日志猜测。第二，`PI0_LatentCorr` 目前已经完整具备了开环训练、stage1 latent 训练、以及 stage1 推理三种模式的能力，并且已经实现了 stage2 的核心模型语义，但外层完整工作流还没有接好。第三，当前如果要真正开始 PI0 的闭环阶段，不应该直接“开训”，而应该优先补齐 `train_stage2_latent.py + 数据 glue + DDP 启动器 + 正式 eval pipeline` 这四件事；否则即便模型层逻辑已经有了，也没有办法像 ACT 那样稳定、可复现地完成闭环训练和正式 100-seed 评测。
