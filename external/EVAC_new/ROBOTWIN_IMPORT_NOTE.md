# EVAC_new Source Snapshot for WM LatentCorr

This directory is a lightweight source/config snapshot copied from:

`/data/yujieyang/EVAC_new`

It is included because the WM latent / rollout path used by ACT_LatentCorr and SmolVLA_LatentCorr depends on EVAC model code, configs, dataset statistics, and inference utilities.

Large artifacts are intentionally excluded from git:

- `runs/`
- checkpoints such as `.ckpt`, `.pt`, `.pth`, `.safetensors`
- rendered videos/images
- caches and `__pycache__`

The EVAC checkpoint required by the experiments is staged separately for Hugging Face upload:

`hf_export_latentcorr_20260529/evac/evac_robotwin_new_mixed50p12_plus_pi05_rollout_epoch333_step10000.ckpt`

The most important config is:

`configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml`

## Included experiment config snapshot

The generated run configs from the important EVAC run are preserved under:

`experiment_configs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/`

The original `runs/` directory is intentionally excluded because it contains logs, tensorboard files, caches, and checkpoints.
