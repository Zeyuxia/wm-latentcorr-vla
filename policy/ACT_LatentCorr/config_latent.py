from dataclasses import dataclass


@dataclass
class DynamicsWarmupConfig:
    zero_steps: int = 5000
    ramp_steps: int = 20000
    max_weight: float = 1.0
    curve: str = "cosine"  # {"linear", "cosine"}
    unit: str = "step"  # {"step", "epoch"}


@dataclass
class LatentLossConfig:
    lambda_action: float = 1.0
    normal_condition_keep_prob: float = 1.0
    beta_dynamics_max: float = 1.0
    lambda_wm_action_current: float = 0.0
    lambda_wm_action_future: float = 0.0
    lambda_bridge_future: float = 0.0
    use_projector_detach_for_predictor: bool = False
    use_projector_detach_for_action_decoder: bool = True
    detach_act_feature_for_latent: bool = False
    use_raw_wm_targets: bool = False
    lambda_teacher_max: float = 0.5
    lambda_pred_max: float = 0.5
    lambda_latent_max: float = 0.3
    lambda_token_init: float = 0.1
    lambda_token_late: float = 0.02
    latent_loss_type: str = "normalized_mse"
    token_loss_type: str = "mse"
    teacher_decay_start_ratio: float = 0.20
    teacher_decay_end_ratio: float = 0.90
    pred_warmup_start_ratio: float = 0.10
    pred_warmup_end_ratio: float = 0.70
    latent_warmup_end_ratio: float = 0.20
    token_decay_start_ratio: float = 0.20
    token_decay_end_ratio: float = 0.90
    pred_only_finetune_start_ratio: float = 0.90
    stopgrad_wm_teacher: bool = True
    stopgrad_teacher_token: bool = True
    stopgrad_adapter_input_for_token_loss: bool = False


@dataclass
class LatentModelConfig:
    projector_mid_channels: int = 256
    wm_adapter_mid_channels: int = 128
    readout_adapter_mid_channels: int = 128
    predictor_num_blocks: int = 3
    predictor_mlp_hidden: int = 512
    action_decoder_hidden: int = 512
    action_dim: int = 14
    state_dim: int = 14
    prefix_steps: int = 16
    token_adapter_hidden_dim: int = 512
    token_adapter_num_layers: int = 2
    token_adapter_dropout: float = 0.1
