import torch
import torch.nn.functional as F

class WorldModelTrainer:
    """
    Handles Multi-Step Unrolling and Loss calculations for the Latent World Model.
    """
    def __init__(self, model, lr=3e-4, seq_len=8, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.seq_len = seq_len
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        # Batch contains sequential chunks of length `H == seq_len`
        # shape: (B, H, ...)
        rasters = batch["raster"].to(self.device).float()
        auxs = batch["aux"].to(self.device).float()
        actions = batch["action"].to(self.device).float()
        prog_targets = batch["progress_target"].to(self.device).float()
        off_targets = batch["offtrack_target"].to(self.device).float()

        B, H = rasters.shape[:2]

        loss_z = 0.0
        loss_prog = 0.0
        loss_off = 0.0

        # Encode the very first step in the sequence
        z = self.model.encode(rasters[:, 0], auxs[:, 0])

        # Autoregressively unroll the predictor in Latent Space
        # Since we have H states, we can predict h=1 to h=H-1
        unrolls = H - 1
        for t in range(unrolls):
            # Predict state at t+1 using action executed at t
            z = self.model.predict(z, actions[:, t])

            # Target is the true observation at t+1 encoded by the stable EMA Target Encoder
            z_target = self.model.get_target(rasters[:, t + 1], auxs[:, t + 1])
            
            # Latent Consistency Loss (MSE)
            loss_z += F.mse_loss(z, z_target.detach())

            # Auxiliary Head Losses
            prog_pred = self.model.progress_head(z)
            off_pred = self.model.offtrack_head(z)
            
            loss_prog += F.mse_loss(prog_pred, prog_targets[:, t])
            # Binary Cross Entropy with Logits for off-track boolean bounds
            loss_off += F.binary_cross_entropy_with_logits(off_pred, off_targets[:, t])

        # Aggregate multi-step losses
        # Weightings: Progress scales roughly the same, OffTrack is slightly higher priority
        total_loss = loss_z + 1.0 * loss_prog + 0.5 * loss_off
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Polyak averaging update parameter EMA
        self.model.update_target_ema(tau=0.01)

        return {
            "loss": total_loss.item() / unrolls,
            "loss_z": loss_z.item() / unrolls,
            "loss_prog": loss_prog.item() / unrolls,
            "loss_off": loss_off.item() / unrolls,
        }
