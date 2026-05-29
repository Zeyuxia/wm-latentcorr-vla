#!/bin/bash
set -euo pipefail

cd /data/zhenyangfan/RoboTwin

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_TAG="stage1_unified_failure_multitask_resume650_dp_gpu0123_${TIMESTAMP}"
OUTPUT_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_ROOT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/logs"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
LOG_FILE="${LOG_DIR}/train.log"
SESSION="stage1_unified_resume650_dp_gpu0123"

RESUME_CKPT="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage1_unified_failure_multitask/stage1_unified_failure_multitask_resume450_gpu0123_20260512_134044/stage1_unified_epoch_0650.pt"
FAILURE_TABLE_PATH="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/outputs/formal_runs/stage2_multitask_failure_explore/stage2_multitask_failure_explore_new_evac_gpu0123_20260429_004924/failure_explore/failure_table.json"
EVAC_CKPT="/data/yujieyang/EVAC_new/runs/evac_robotwin_new_mixed50p12_plus_pi05_rollout_2026-04-22T17-46-59/checkpoints/epoch=333-step=10000.ckpt"
EVAC_CONFIG="/data/yujieyang/EVAC_new/configs/robotwin/train_config_robotwin_new_mixed50p12_plus_pi05_rollout.yaml"
GPU_IDS="0,1,2,3"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${LOG_ROOT}" "${LOG_DIR}"

cat > "${OUTPUT_DIR}/launch_info.txt" <<INFO
run_tag=${RUN_TAG}
output_dir=${OUTPUT_DIR}
log_dir=${LOG_DIR}
log_file=${LOG_FILE}
session=${SESSION}
gpus=${GPU_IDS}
resume_ckpt=${RESUME_CKPT}
failure_table_path=${FAILURE_TABLE_PATH}
evac_ckpt=${EVAC_CKPT}
evac_config=${EVAC_CONFIG}
normal_batch_size=4
failure_batch_size=2
num_epochs=1000
save_freq=50
ddim_steps=4
lr=3e-5
lambda_action=1.0
lambda_teacher_max=0.5
lambda_pred_max=0.5
lambda_latent_max=0.3
lambda_token_init=0.1
lambda_token_late=0.02
normal_condition_keep_prob=0.5
beta_dynamics_max=1.0
failure_future_latent_mode=rollout
balance_failure_tasks=true
rebalance_failure_groups=true
mode=dataparallel_4gpu
note=resume_650_on_4gpu_dataparallel
INFO

tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"
VENV_BIN_DIR="/data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin"
tmux new-session -d -s "${SESSION}" "cd /data/zhenyangfan/RoboTwin && PATH='${VENV_BIN_DIR}:${PATH}' PYTHONPATH='/data/zhenyangfan/RoboTwin:/data/zhenyangfan/RoboTwin/envs/curobo/src:/data/zhenyangfan/RoboTwin/envs/robot' CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONUNBUFFERED=1 /data/zhenyangfan/RoboTwin/policy/ACT_LatentCorr/.venv-act-latent/bin/python -m policy.ACT_LatentCorr.train_stage1_unified_failure_multitask_latent \
  --output_dir '${OUTPUT_DIR}' \
  --evac_ckpt '${EVAC_CKPT}' \
  --evac_config '${EVAC_CONFIG}' \
  --resume_ckpt '${RESUME_CKPT}' \
  --multi_task_names sim-open_laptop-demo_clean-50 sim-pick_dual_bottles-demo_clean-50 sim-put_bottles_dustbin-demo_clean-50 sim-place_burger_fries-demo_clean-50 sim-handover_block-demo_clean-50 \
  --failure_table_paths '${FAILURE_TABLE_PATH}' \
  --urdf_path '/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf' \
  --curobo_left_yml '/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_left.yml' \
  --curobo_right_yml '/data/zhenyangfan/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml' \
  --device cuda:0 \
  --num_epochs 1000 \
  --max_steps -1 \
  --normal_batch_size 4 \
  --failure_batch_size 2 \
  --num_workers 4 \
  --lr 3e-5 \
  --base_act_lr_scale 1.0 \
  --weight_decay 1e-4 \
  --save_freq 50 \
  --ddim_steps 4 \
  --prefix_steps 16 \
  --act_chunk_size 50 \
  --future_offset 16 \
  --lambda_action 1.0 \
  --normal_condition_keep_prob 0.5 \
  --beta_dynamics_max 1.0 \
  --lambda_wm_action_current 0.0 \
  --lambda_wm_action_future 0.0 \
  --lambda_bridge_future 0.0 \
  --lambda_teacher_max 0.5 \
  --lambda_pred_max 0.5 \
  --lambda_latent_max 0.3 \
  --lambda_token_init 0.1 \
  --lambda_token_late 0.02 \
  --latent_loss_type normalized_mse \
  --token_loss_type mse \
  --teacher_decay_start_ratio 0.20 \
  --teacher_decay_end_ratio 0.90 \
  --pred_warmup_start_ratio 0.10 \
  --pred_warmup_end_ratio 0.70 \
  --latent_warmup_end_ratio 0.20 \
  --token_decay_start_ratio 0.20 \
  --token_decay_end_ratio 0.90 \
  --pred_only_finetune_start_ratio 0.90 \
  --stopgrad_wm_teacher true \
  --stopgrad_teacher_token true \
  --stopgrad_adapter_input_for_token_loss false \
  --freeze_base_act false \
  --freeze_readout_decoder true \
  --detach_act_feature_for_latent false \
  --use_raw_wm_targets false \
  --dyn_zero_steps 0 \
  --dyn_ramp_steps 1000 \
  --dyn_warmup_curve cosine \
  --reference_global_batch_size 24 \
  --predictor_num_blocks 3 \
  --projector_mid_channels 256 \
  --wm_adapter_mid_channels 128 \
  --readout_adapter_mid_channels 128 \
  --predictor_mlp_hidden 512 \
  --action_decoder_hidden 512 \
  --token_adapter_hidden_dim 512 \
  --token_adapter_num_layers 2 \
  --token_adapter_dropout 0.1 \
  --backbone resnet18 \
  --hidden_dim 512 \
  --state_dim 14 \
  --action_dim 14 \
  --max_rollout_steps 1 \
  --planner_target_mode backward \
  --planner_target_lookahead_steps 6 \
  --planner_orient_weight 0.0573 \
  --planner_gripper_penalty 1.0 \
  --planner_nearest_window_radius 12 \
  --planner_active_joint_delta_thresh 0.01 \
  --planner_active_gripper_delta_thresh 0.05 \
  --act_aligned_rollout_exec_steps 16 \
  --act_aligned_min_dist_fallback_force_correction true \
  --act_aligned_min_dist_recover_ratio 0.75 \
  --act_aligned_real_error_trigger_enable true \
  --act_aligned_real_error_min_dist_thresh 0.01 \
  --act_aligned_real_error_min_dist_delta_thresh 0.005 \
  --act_aligned_debug_recover_eval_rollout false \
  --act_aligned_debug_correction_evac_rollout false \
  --recover_eval_enable false \
  --recover_eval_save_video false \
  --recover_eval_gripper_open_thresh 0.8 \
  --recover_eval_pos_thresh_m 0.03 \
  --recover_eval_rot_thresh_deg 10.0 \
  --recover_eval_nearest_window_radius 16 \
  --recover_eval_video_bridge_steps 16 \
  --act_aligned_correction_interp_nearest_enable false \
  --act_aligned_correction_interp_prefix_ratio 0.6 \
  --act_aligned_correction_planner_prefix_ratio 0.5 \
  --act_aligned_correction_gripper_close_prefix_ratio 0.32 \
  --act_aligned_correction_compose_gt_tail_enable true \
  --act_aligned_correction_gripper_switch_ratio 0.5 \
  --act_aligned_recover_gripper_penalty 0.0 \
  --act_aligned_enable_perturb true \
  --act_aligned_perturb_prob 1.0 \
  --act_aligned_perturb_error_mode open_laptop_pregrasp \
  --act_aligned_perturb_open_laptop_pregrasp_close_prob 0.5 \
  --act_aligned_perturb_open_laptop_pregrasp_translation_prob 0.0 \
  --act_aligned_perturb_open_laptop_pregrasp_rotation_prob 0.0 \
  --act_aligned_perturb_eef_fail_gain 0.10 \
  --act_aligned_perturb_rot_max_deg 15.0 \
  --act_aligned_perturb_mag_random false \
  --act_aligned_perturb_mag_rand_min 1.0 \
  --act_aligned_perturb_mag_rand_max 1.4 \
  --act_aligned_perturb_reject_sampling_enable true \
  --act_aligned_perturb_reject_max_trials 4 \
  --act_aligned_perturb_reject_dir_jitter_eps 0.2 \
  --act_aligned_perturb_gripper_close_min 0.10 \
  --act_aligned_perturb_gripper_open_max 0.90 \
  --act_aligned_perturb_gripper_fast_ratio 0.20 \
  --act_aligned_sample_pregrasp_phase_window_len 30 \
  --act_aligned_sample_timeout_sec 30 \
  --failure_phase_bins 3 \
  --failure_translation_dir_bins 6 \
  --failure_translation_mag_bins 3 \
  --failure_rotation_dir_bins 6 \
  --failure_rotation_mag_bins 3 \
  --failure_explore_k 4 \
  --failure_sample_skip_head_ratio 0.6 \
  --failure_future_latent_mode rollout \
  --balance_failure_tasks true \
  --rebalance_failure_groups true \
  --use_wandb false \
  --wandb_mode disabled \
  --use_data_parallel true \
  --data_parallel_device_ids 0,1,2,3 \
  2>&1 | tee -a '${LOG_FILE}'"

echo "started ${RUN_TAG}"
echo "session=${SESSION}"
echo "log=${LOG_FILE}"
echo "output=${OUTPUT_DIR}"
