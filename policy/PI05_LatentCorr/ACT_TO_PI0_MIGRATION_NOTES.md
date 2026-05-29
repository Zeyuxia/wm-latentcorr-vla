# ACT to PI0 Migration Notes

This file tracks the staged migration from `policy/ACT_LatentCorr` to
`policy/PI0_LatentCorr`.

## Goal

Reproduce the validated ACT two-stage workflow on PI0.5 while keeping PI0's
own policy backbone and deployment path.

## Current Status

### Stage 1

Aligned pieces already ported into `PI0_LatentCorr`:

- `config_latent.py`
  - `DynamicsWarmupConfig`
  - `LatentLossConfig`
  - `LatentModelConfig`
- `train_stage1.py`
  - ACT-style latent/loss config construction
  - `lambda_action`
  - `lambda_action_conditioned`
  - `beta_dynamics_max`
  - `dyn_schedule_unit`
  - `use_act_head_conditioning`
  - `use_projector_detach_for_predictor`
  - `detach_act_feature_for_latent`
  - `future_teacher_latent`-compatible forward API
- `stage1_model.py`
  - conditioned branch uses predictor output as condition token
  - conditioned branch can be toggled by `use_act_head_conditioning`
  - dynamics weight uses warmup scheduler
  - projector detach behavior matches ACT stage1 semantics
  - runtime helpers added for later deploy/stage2 reuse:
    - `initialize_latent_heads(...)`
    - `project_current_latent(...)`
    - `predict_future_latent(...)`
    - `predict_pi0_chunk(...)`
    - `predict_pi0_chunk_conditioned(...)`
- `stage1_checkpoint.py`
  - reconstructs a full `PI0LatentStage1` model directly from a saved stage1 checkpoint
  - restores ACT-aligned latent configs from checkpoint args
  - intended as the shared loader for future deploy/stage2 scripts

Known differences vs ACT stage1 that are still expected:

- PI0 does not yet have a direct equivalent of ACT's
  `wm_action_current / wm_action_future / bridge_future` auxiliary losses.
- PI0 stage1 deployment path is still open-loop oriented.

### Stage 2

Core model semantics are now partially migrated inside `stage1_model.py`:

- `Stage2LossOutput`
- `Stage2PreparedContext`
- `prepare_stage2_context(...)`
  - PI0 chunk prediction
  - error-prefix raw/normalized action reconstruction
  - EVAC rollout latent target hookup
  - predicted future latent generation
- `compute_stage2_loss(...)`
  - correction loss
  - dynamics rollout alignment loss
  - retain/anchor loss
  - optional bridge-side supervision hook
- `forward_stage2(...)`
  - unified stage2 forward path over the prepared context

Still not migrated yet. The following ACT modules remain the source of truth:

- `train_stage2_latent.py`
- `train_stage2_ddp.sh`
- `act_aligned_correction.py`
- `stage2_failure_dataset.py`
- `failure_utils.py`
- `stage2_latent_cache.py`
- `eval_success.sh`

## Recommended Migration Order

1. Finish PI0 stage1 deployment semantics
   - base / teacher / bridge inference modes
   - checkpoint argument compatibility
2. Add PI0 stage2 latent policy model
   - retain anchor
   - correction target construction
   - dynamics rollout target
3. Port ACT-aligned correction builder to PI0
   - failure table sampling
   - perturbation-driven error action generation
   - correction batch assembly
4. Add PI0 stage2 training / eval scripts
   - DDP launcher
   - eval success pipeline
   - grouped summary scripts

## File Mapping

- `ACT_LatentCorr/config_latent.py`
  -> `PI0_LatentCorr/config_latent.py`
- `ACT_LatentCorr/train_stage1_latent.py`
  -> `PI0_LatentCorr/train_stage1.py`
- `ACT_LatentCorr/latent_policy.py`
  -> split across `PI0_LatentCorr/stage1_model.py` and future `stage2` model files
- `ACT_LatentCorr/deploy_policy.py`
  -> future extension of `PI0_LatentCorr/deploy_policy.py`
- `ACT_LatentCorr/train_stage2_latent.py`
  -> not yet created in `PI0_LatentCorr`
