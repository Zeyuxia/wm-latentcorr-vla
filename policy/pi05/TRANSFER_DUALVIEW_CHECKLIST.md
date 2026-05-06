# PI0.5 Dual-View Transfer Checklist

这个清单对应当前 RoboTwin 上的 `PI0.5` 双视角训练链路。

当前双视角定义：

- `camera_mode=dual_view`
- 默认 `secondary_camera=right_wrist`
- 也支持 `secondary_camera=left_wrist`
- 当前实现的双视角是 `cam_high + 单个 wrist camera`

## 必传

下面这些传到新服务器上，就可以重新准备数据并训练 5 任务的 PI0.5 双视角开环权重。

- `policy/__init__.py`
- `policy/pi05/`
- `data/open_laptop/demo_clean/`
- `data/place_burger_fries/demo_clean/`
- `data/handover_block/demo_clean/`
- `data/pick_dual_bottles/demo_clean/`
- `data/put_bottles_dustbin/demo_clean/`

## 可选

下面这些不是“重新训练”必须的，但会很有帮助。

- `policy/pi05/outputs/assets/`
  已经算好的 `norm_stats.json`，传过去后可以少跑一步。
- `policy/pi05/outputs/processed_data/`
  已经转好的 LeRobot 中间数据，传过去后可以不重新导出 episode。
- `policy/pi05/outputs/checkpoints/`
  如果你要续训或直接评测旧权重。

## 如果你后面还想评测

只训练不需要这些；如果你后面还要在新服务器上跑 RoboTwin 环境评测，再额外传：

- `script/`
- `envs/`
- `assets/`
- `task_config/`

## 推荐传输方式

建议在 RoboTwin 根目录执行 `rsync`，并使用本目录下的：

- `transfer_dualview_files.txt`

这个文件已经是按 RoboTwin 根目录写好的相对路径。

## 训练命令

默认右腕双视角：

```bash
bash policy/pi05/prepare_robotwin_multitask_dualview.sh
bash policy/pi05/train_robotwin_multitask_dualview.sh
```

左腕双视角：

```bash
SECONDARY_CAMERA=left_wrist bash policy/pi05/prepare_robotwin_multitask_dualview.sh
SECONDARY_CAMERA=left_wrist bash policy/pi05/train_robotwin_multitask_dualview.sh
```
