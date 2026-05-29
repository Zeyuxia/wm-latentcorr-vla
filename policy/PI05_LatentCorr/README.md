# PI0_LatentCorr

独立的 PI0/PI0.5 开环实验目录。它不改动 `policy/ACT_LatentCorr` 和 `policy/pi05` 的源码，而是在这个目录里提供一条平行的 RoboTwin 开环训练与评测链路。

当前范围：
- 已接通 PI0/PI0.5 开环数据准备
- 已接通 PI0/PI0.5 开环训练 wrapper
- 已接通 RoboTwin 标准 `eval_policy.py` 的部署入口
- 已接通 `100 seed bridge` 风格并行评测 launcher
- 默认按 `RobotWin + PI0.5 + tri-view + 8 GPU` 运行
- 尚未接入闭环 correction / latent correction

默认目录：
- 资产目录：`policy/PI0_LatentCorr/outputs/assets`
- checkpoint 目录：`policy/PI0_LatentCorr/outputs/checkpoints`
- 日志目录：`policy/PI0_LatentCorr/outputs/logs`
- `uv` 工程目录：`policy/PI0_LatentCorr`

推荐流程：
先进入这个项目目录：
`cd /data/zhenyangfan/RoboTwin/policy/PI0_LatentCorr`

1. 初始化独立 uv 环境
   `bash ./setup_uv_env.sh`
1. 准备 LeRobot 数据
   `bash ./prepare_openloop_data.sh`
2. 训练开环 PI0.5
   `bash ./train_openloop.sh`
3. 单卡评测
   `bash ./eval.sh`
4. 100-seed bridge 并行评测
   `bash ./launch_eval_g1g2_100.sh`

常用环境变量：
- `TRAIN_CONFIG_NAME`
  默认是 `pi05_aloha_full_base`。如果你要切回 PI0，可以手动设成 `pi0_base_aloha_robotwin_full`。
- `REPO_ID`
  LeRobot 数据集 repo id。默认单视角会使用 `robotwin/open_laptop_demo_clean_50_headonly`。
- `CAMERA_MODE`
  默认是 `head_only`。这是为了和你当前 EVAC 的单视角设定保持一致。
- `CUDA_VISIBLE_DEVICES`
  默认是 `0,1,2,3,4,5,6,7`。
- `FSDP_DEVICES`
  默认是 `8`。这条 OpenPI 训练线底层走的是 JAX/FSDP 多卡并行，不是 PyTorch DDP，但会实际吃满 8 张 4090。
- `EXP_NAME`
  训练实验名。评测时的 `MODEL_NAME` 就是这个值。
- `CHECKPOINT_ID`
  具体 checkpoint step 目录名；设成 `latest` 会自动取最新一步。

说明：
- 这里底层仍然复用 `policy/pi05` 里的 OpenPI 代码和 RobotWin/Aloha 配置，因为它已经带了 `pi0_base_aloha_robotwin_*` 和 `pi0_fast_aloha_robotwin_*`。
- 脚本默认把 `OPENPI_DATA_HOME` 指到 `policy/PI0_LatentCorr/outputs/openpi_cache`，避免把基础权重缓存到 `~/.cache/openpi` 挤爆系统盘。
- 现在这个目录本身已经有独立 `uv` 项目定义，训练和评测脚本默认通过 `uv run --project policy/PI0_LatentCorr` 执行。
- 你可以直接在 `policy/PI0_LatentCorr` 目录里运行这些脚本；脚本内部会在需要时自动切到 RoboTwin 根目录，以兼容 `script/eval_policy.py` 这类依赖根目录相对路径的模块。
