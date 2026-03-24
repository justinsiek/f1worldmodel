import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """Embeds the low-dimensional [steer, throttle, brake] action vector."""
    def __init__(self, action_dim=3, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ELU(),
            nn.Linear(64, emb_dim)
        )
        
    def forward(self, action):
        return self.net(action)


class Predictor(nn.Module):
    """
    Transition dynamics model.
    Predicts next latent z_t+1 given current latent z_t and action embedding.
    """
    def __init__(self, latent_dim=64, action_emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, z_t, a_emb):
        x = torch.cat([z_t, a_emb], dim=-1)
        return self.net(x)


class ProgressHead(nn.Module):
    """Auxiliary head predicting the lap progress delta scalar to score actions."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, z):
        return self.net(z).squeeze(-1)


class OffTrackHead(nn.Module):
    """Auxiliary head predicting the probability of being off-track (wall mask)."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)  # Returns logits for BCEWithLogitsLoss
        )
        
    def forward(self, z):
        return self.net(z).squeeze(-1)
