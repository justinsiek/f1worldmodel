import torch
import torch.nn as nn

from models.encoder import Encoder
from models.predictor import ActionEncoder, Predictor, ProgressHead, OffTrackHead


class WorldModel(nn.Module):
    """
    Wraps the Encoder, Predictor, and Auxiliary Heads.
    Maintains an Exponential Moving Average (EMA) target encoder for stable latent targets.
    """
    def __init__(self, raster_channels=3, aux_dim=16, action_dim=3, latent_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Online networks (trained via gradient descent)
        self.encoder = Encoder(raster_channels, aux_dim, latent_dim)
        self.action_encoder = ActionEncoder(action_dim, emb_dim=32)
        self.predictor = Predictor(latent_dim, action_emb_dim=32)
        
        # Auxiliary heads for planner scoring
        self.progress_head = ProgressHead(latent_dim)
        self.offtrack_head = OffTrackHead(latent_dim)
        
        # Target network
        self.target_encoder = Encoder(raster_channels, aux_dim, latent_dim)
        
        # Initialize target explicitly to match online
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        # Freeze target params
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def update_target_ema(self, tau=0.01):
        """Soft updates the target encoder weights."""
        with torch.no_grad():
            for online_param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
                
    def encode(self, raster, aux):
        """Encode raw observation to online latent variable z_t."""
        return self.encoder(raster, aux)
        
    def get_target(self, raster, aux):
        """Encode raw observation to target latent variable z_target using EMA weights."""
        with torch.no_grad():
            return self.target_encoder(raster, aux)
            
    def predict(self, z_t, action):
        """Predict the next latent state given a latent state and raw action."""
        a_emb = self.action_encoder(action)
        return self.predictor(z_t, a_emb)
