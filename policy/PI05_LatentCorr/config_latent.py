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
    lambda_action_conditioned: float = 0.0
    lambda_align: float = 0.0
    beta_dynamics_max: float = 1.0
    use_projector_detach_for_predictor: bool = True
    detach_act_feature_for_latent: bool = False
    use_raw_wm_targets: bool = True


@dataclass
class LatentModelConfig:
    projector_mid_channels: int = 256
    predictor_hidden_dim: int = 512
    predictor_num_blocks: int = 3
    action_dim: int = 14
    model_action_dim: int = 32
    action_horizon: int = 50
    prefix_steps: int = 16
