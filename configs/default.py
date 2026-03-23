from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Config:
    # Raster
    raster_channels: int = 3
    raster_size: int = 64
    aux_dim: int = 1  # speed only

    # Model
    latent_dim: int = 64
    encoder_channels: Tuple[int, ...] = (32, 64, 128)
    predictor_hidden: int = 256
    action_dim: int = 3
    action_emb_dim: int = 32

    # Planner (CEM)
    planner_candidates: int = 512  # N
    planner_horizon: int = 8       # H
    cem_iterations: int = 3
    cem_top_k: int = 64
    score_offtrack_weight: float = 5.0

    # Training
    lr: float = 3e-4
    batch_size: int = 64
    rollout_horizon: int = 4
    ema_decay: float = 0.99
    num_epochs: int = 50
    data_collect_steps: int = 10000

    # Environment
    max_speed: float = 50.0
    dt: float = 0.1
    max_steps: int = 5000
    max_steer_rate: float = 3.5
    off_track_tolerance: int = 10  # steps off-track before done

    # Track
    track_csv: str = "tracks/Monza.csv"
    pixels_per_meter: float = 3.0

    # Reward
    progress_reward: float = 0.02
    off_track_penalty: float = 0.5
    step_penalty: float = 0.005
    lap_bonus: float = 1.0
