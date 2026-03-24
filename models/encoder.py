import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    CNN + MLP encoder for 2D F1 World Model.
    Inputs:
        raster: (B, 3, 64, 64)
        aux: (B, A)
    Outputs:
        latent: (B, latent_dim)
    """

    def __init__(self, raster_channels=3, aux_dim=1, latent_dim=64):
        super().__init__()
        
        # CNN to process 64x64 spatial rasters
        self.cnn = nn.Sequential(
            nn.Conv2d(raster_channels, 32, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),             # -> 16x16
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # -> 8x8
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),           # -> 4x4
            nn.ELU(),
            nn.Flatten()                                                       # -> 256*4*4 = 4096
        )
        
        # Aux MLP to process continuous 1D features (like speed)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ELU()
        )
        
        # Projection to Latent vector
        self.fc = nn.Sequential(
            nn.Linear(4096 + 32, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, raster, aux):
        """
        raster: (B, C, H, W) float32 tensor
        aux: (B, A) float32 tensor
        """
        c_feat = self.cnn(raster)
        a_feat = self.aux_mlp(aux)
        
        x = torch.cat([c_feat, a_feat], dim=-1)
        z = self.fc(x)
        
        return z
