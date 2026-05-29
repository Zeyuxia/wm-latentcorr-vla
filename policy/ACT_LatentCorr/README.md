# ACT_LatentCorr

Independent experimental policy package for latent-space closed-loop correction.
This package is isolated from `policy/ACT` to avoid code pollution.

## Step-1 (implemented)

- New stage-1 training entry: `train_stage1_latent.py`
- New latent modules:
  - `LatentProjector` (2D topology-preserving projection)
  - `ActionConditionedPredictor` (AdaLN residual predictor)
  - `LatentActionDecoder`
  - `DynamicsWarmup` (step-based dynamics loss warmup)
- Stage-1 detach behavior implemented:
  - Predictor input uses `z_proj.detach()`
  - Action decoder input uses `z_proj.detach()`
- EVAC teacher interface (real frame -> final VAE latent):
  - `EvacLatentTeacher.encode_image(...)`

## Not implemented yet

- Closed-loop phase-2 (`z_wm_sim` via DiT and `L_correct`)
- Planner-triggered latent correction pipeline
- `L_bridge` distillation in late phase
- Full deployment path for ACT_LatentCorr (currently wrapper to original ACT)

## Run stage-1

```bash
bash /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/train_stage1.sh
```

