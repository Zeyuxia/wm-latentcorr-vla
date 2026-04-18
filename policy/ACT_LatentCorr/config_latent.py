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
    lambda_align: float = 0.1
    beta_dynamics_max: float = 1.0
    lambda_wm_action_current: float = 0.0
    lambda_wm_action_future: float = 0.1
    lambda_bridge_future: float = 0.0
    use_projector_detach_for_predictor: bool = True
    use_projector_detach_for_action_decoder: bool = True
    detach_act_feature_for_latent: bool = False
    use_raw_wm_targets: bool = False


@dataclass
class LatentModelConfig:
    projector_mid_channels: int = 256
    wm_adapter_mid_channels: int = 128
    readout_adapter_mid_channels: int = 128
    predictor_num_blocks: int = 3
    predictor_mlp_hidden: int = 512
    action_decoder_hidden: int = 512
    action_dim: int = 14
    prefix_steps: int = 16
