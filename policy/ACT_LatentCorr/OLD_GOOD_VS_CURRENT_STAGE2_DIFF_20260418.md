# ACT_LatentCorr 旧好版本 vs 当前坏版本 对比

## 1. 先说明证据来源

`policy/ACT_LatentCorr` 的关键训练文件（如 `train_stage2_latent.py`、`latent_policy.py`）目前没有被 git 正式追踪，所以**无法直接用 git 恢复出 2026-04-05 当天的完整源码快照**。因此，这份对比只使用下面这些可以直接核验的代码级证据：

1. 旧好主线仍然保留的 launch 脚本：
   - `launch_stage2_from_unified_ep400_actalignedcorr_free0123.sh`
   - `launch_stage2_from_unified_ep400_actalignedcorr_resume50_to250_free01234567.sh`
   - `launch_stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357.sh`
2. 当前坏版本使用的 launcher / workflow：
   - `launch_stage2_failure_workflow_4gpu.sh`
   - `run_stage2_failure_workflow.sh`
   - `train_stage2_ddp.sh`
3. 当前训练主干代码：
   - `train_stage2_latent.py`
   - `latent_policy.py`
   - `act_aligned_correction.py`
4. 现有 handoff 文档中记录的历史最好结果：
   - `HANDOFF_FOR_NEW_CHAT_PI0.md`

## 2. 可以确认的“旧好版本”是哪条线

目前能被代码和文档同时指向的旧好主线，是 `actalignedcorr` 这一串：

- `stage2_from_unified_ep400_actalignedcorr_free0123`
- `stage2_from_unified_ep400_actalignedcorr_resume50_to250_free01234567`
- `stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357`
- `stage2_from_unified_ep400_actalignedcorr_resume400_to1000_free01234567`

`HANDOFF_FOR_NEW_CHAT_PI0.md` 明确写了：

- 最好结果是 `ep600 = 83/100 = 83.0%`
- 这条线就是 **ACT-aligned 扰动版**

因此，当前排查应该以这条 `actalignedcorr` 训练语义为旧好参考，而不是以现在的 `failure_workflow` 为参考。

## 3. 旧好版本和当前坏版本最核心的训练语义差异

### 3.1 旧好版本根本没有启用 failure-table 训练混合

旧好脚本里：

- `BATCH_SIZE=1`
- `CORRECTION_BATCH_SIZE=0`
- 没有传 `FAILURE_MODE=train`
- 没有传 `FAILURE_TABLE_PATH`

见：

- `launch_stage2_from_unified_ep400_actalignedcorr_free0123.sh`
- `launch_stage2_from_unified_ep400_actalignedcorr_resume50_to250_free01234567.sh`
- `launch_stage2_from_unified_ep400_actalignedcorr_resume250_to700_free012357.sh`

这意味着旧好版本训练时只走普通 batch，不会额外拼一份 failure-table 采样出来的 correction batch。

而当前坏版本：

- `run_stage2_failure_workflow.sh` 在训练阶段强制设置：
  - `FAILURE_MODE=train`
  - `CORRECTION_BATCH_SIZE=${TRAIN_CORRECTION_BATCH_SIZE}`
  - `FAILURE_TABLE_PATH=${FAILURE_TABLE_PATH}`
- `train_stage2_latent.py` 中会把一个 step 拆成：
  - `("off", batch)`
  - `("train", corr_batch)`
  然后拼在一起训练（见 `train_stage2_latent.py:1199-1206`）

也就是说，**当前坏版本已经不是旧好版本那种“纯扰动纠错训练”语义，而是“正常样本 + failure-table 采样样本混训”语义**。

### 3.2 旧好版本训练的是 ACT 头纠错；当前坏版本实际切到了 latent decoder 纠错

旧好脚本中都明确写着：

- `USE_ACT_HEAD_CORRECTION=true`

而当前 `train_stage2_ddp.sh` 默认值是：

- `USE_ACT_HEAD_CORRECTION=${USE_ACT_HEAD_CORRECTION:-false}`

当前坏版本的 `stage2_config.txt` 也确实显示：

- `use_act_head_correction: False`

这会在 `latent_policy.py:487-533` 触发完全不同的损失分支：

- `use_act_head_correction=True`：
  - 走 `self.base_act(... external_latent_input=corr_token ...)`
  - 直接训练 ACT 头输出整个纠错 chunk
- `use_act_head_correction=False`：
  - 走 `corr_hat = self.decode_action_latent(z_wm_sim_shared.detach())`
  - 只训练 latent decoder 去拟合 prefix

这不是小改动，而是**纠错监督对象变了**。

### 3.3 旧好版本的 stage2 几乎不启用 future latent / bridge 损失；当前坏版本启用了

旧好 `actalignedcorr` launcher 里固定写的是：

- `LAMBDA_WM_ACTION_CURRENT=0.0`
- `LAMBDA_WM_ACTION_FUTURE=0.0`
- `LAMBDA_BRIDGE_FUTURE=0.0`
- `DETACH_ACT_FEATURE_FOR_LATENT=true`
- `DYN_RAMP_STEPS=200`

当前坏版本 `stage2_config.txt` 显示：

- `lambda_wm_action_current: 0.5`
- `lambda_wm_action_future: 1.0`
- `lambda_bridge_future: 0.25`
- `detach_act_feature_for_latent: False`
- `dyn_ramp_steps: 1000`

这说明当前坏版本不是在复现旧好版本的“ACT 头纠错主导”训练，而是把 stage2 再次推回了更重的 latent 约束训练范式。

### 3.4 旧好版本 retain 很强；当前坏版本 retain 很弱

旧好 `actalignedcorr` launcher：

- `RETAIN_WEIGHT=1.0`
- `BASE_ANCHOR_CKPT=${STAGE1_CKPT}`

当前坏版本 `stage2_config.txt`：

- `retain_weight: 0.1`
- `base_anchor_ckpt: None`

在 `latent_policy.py:502-512` 和 `518-527` 里，retain 是通过再次喂 `base_act` 去约束动作头不要漂掉。当前坏版本把 retain 大幅削弱，而且没有显式 anchor ckpt，这会直接改变 stage2 训练稳定性。

### 3.5 当前 failure workflow 默认还改了若干纠错/恢复细节超参

`launch_stage2_failure_workflow_4gpu.sh` 和旧好 `actalignedcorr` 脚本相比，还额外改了：

- `TRAIN_CORRECTION_BATCH_SIZE=1`
- `FAILURE_EXPLORE_K=4`
- `ACT_ALIGNED_SAMPLE_PREGRASP_PHASE_WINDOW_LEN=20`（旧好是 30）
- explore 阶段 `FAILURE_TRANSLATION_MAG_BINS=2`、`FAILURE_ROTATION_MAG_BINS=2`
- 训练阶段 `FAILURE_TRANSLATION_MAG_BINS=3`、`FAILURE_ROTATION_MAG_BINS=3`
- `ACT_ALIGNED_PERTURB_EEF_FAIL_GAIN=0.12`（旧好是 0.10）
- `ACT_ALIGNED_PERTURB_ROT_MAX_DEG=24`（旧好是 15）
- `RECOVER_EVAL_ENABLE=true`

这些改动会让当前的错误模式分布、可恢复样本筛选口径、扰动幅度都和旧好版本不同。

## 4. 当前代码里这些差异具体会落到哪里

### 4.1 训练 batch 组成已经变了

`train_stage2_latent.py:1199-1206`

当前代码每个 step 会先放一份普通 batch：

- `batch_groups = [("off", batch)]`

如果 `corr_iter` 存在，再加一份 correction batch：

- `batch_groups.append((args.failure_mode, corr_batch))`

所以只要 `correction_batch_size > 0`，训练语义就已经不是旧好版本。

### 4.2 纠错目标来源已经变了

`train_stage2_latent.py:1289-1413`

- `batch_failure_mode == "off"` 时，纠错目标直接退化成原始 ACT chunk
- `batch_failure_mode != "off"` 时，会进入 `correction_builder.build(...)`，从 failure-table 采样的错误模式去构造纠错目标

因此当前坏版本里，真正参与训练的“纠错目标分布”与旧好版本不再一致。

### 4.3 Stage2 的监督分支已经变了

`latent_policy.py:487-533`

- 旧好语义：`use_act_head_correction=True`
- 当前坏语义：`use_act_head_correction=False`

这会把纠错训练从 “ACT 头直接学 chunk” 切成 “latent decoder 学 prefix”，这是最值得优先回退的差异之一。

## 5. 现在能下的结论

基于现存代码证据，可以明确说：

1. **当前坏版本并没有严格沿用旧好 `actalignedcorr` 那条训练语义。**
2. 差异不是只有一个，而是至少同时发生了下面四件事：
   - 从 `CORRECTION_BATCH_SIZE=0` 变成 `>0`
   - 从 `FAILURE_MODE=off` 变成 `train`
   - 从 `USE_ACT_HEAD_CORRECTION=true` 变成 `false`
   - 从弱 latent / 强 retain，变成强 latent / 弱 retain
3. 因此，当前效果崩掉时，不能把原因简单归结为“新的 failure table 思路不行”；更准确地说，是**你把训练对象、训练样本组成、latent 约束强度、retain 强度，一起改了**。

## 6. 最合理的回滚基线

如果目标是“严格回到旧好版本语义，再逐项加新逻辑做消融”，那第一步应该回到下面这组条件：

- `FAILURE_MODE=off`
- `CORRECTION_BATCH_SIZE=0`
- `USE_ACT_HEAD_CORRECTION=true`
- `RETAIN_WEIGHT=1.0`
- `BASE_ANCHOR_CKPT=${STAGE1_CKPT}`
- `DETACH_ACT_FEATURE_FOR_LATENT=true`
- `LAMBDA_WM_ACTION_CURRENT=0.0`
- `LAMBDA_WM_ACTION_FUTURE=0.0`
- `LAMBDA_BRIDGE_FUTURE=0.0`
- `DYN_RAMP_STEPS=200`
- 其余扰动参数对齐旧好 `actalignedcorr` 脚本

然后再逐项只打开一项新逻辑，比如：

1. 只打开 failure-table 采样，但仍保持 `USE_ACT_HEAD_CORRECTION=true`
2. 或者只改成 trigger-only，但不改 loss 语义
3. 或者只改成新的 recover/check 逻辑，但 batch 组成不变

这样才能知道究竟是哪一环把效果打坏了。
