import torch
import numpy as np


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) Latent Model Predictive Control.
    Optimized explicitly for FULL THROTTLE, NO-BRAKE 50m/s Lap Times!
    """
    def __init__(self, model, num_candidates: int = 400, horizon: int = 30, 
                 iterations: int = 5, num_elites: int = 40, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.iterations = iterations
        self.num_elites = num_elites
        self.device = device
        
        # Pure Speed Matrix
        self.progress_reward = 100.0

        self.action_ub = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.action_lb = torch.tensor([-1.0, 0.0, 0.0], device=device)
        
        self.mu = torch.zeros((self.horizon, 3), device=self.device)
        self.mu[:, 1] = 1.0 # 100% Throttle Warm-Start
        
    def __call__(self, obs, car_state=None):
        raster = torch.from_numpy(obs["raster"]).float().unsqueeze(0).to(self.device)
        aux = torch.from_numpy(obs["aux"]).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            z_0 = self.model.encode(raster, aux)
            
            self.mu[:-1] = self.mu[1:].clone()
            self.mu[-1] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            
            var = torch.ones((self.horizon, 3), device=self.device) * 0.5
            var[:, 2] = 0.0 # THE CAR DOES NOT NEED TO BRAKE! Completely eliminate brake drag
            # Steering variance remains 0.5 so it can mathematically guess 100% lock angles!
            
            for i in range(self.iterations):
                noise = torch.randn((self.num_candidates, self.horizon, 3), device=self.device)
                actions = self.mu.unsqueeze(0) + noise * torch.sqrt(var).unsqueeze(0)
                
                actions = torch.max(torch.min(actions, self.action_ub), self.action_lb)
                
                throttle_mask = actions[:, :, 1] > actions[:, :, 2]
                actions[:, :, 1] = actions[:, :, 1] * throttle_mask
                actions[:, :, 2] = actions[:, :, 2] * (~throttle_mask)
                
                z = z_0.expand(self.num_candidates, -1)
                scores = torch.zeros(self.num_candidates, device=self.device)
                
                for t in range(self.horizon):
                    z = self.model.predict(z, actions[:, t, :])
                    
                    prog = self.model.progress_head(z)
                    off_prob = torch.sigmoid(self.model.offtrack_head(z))
                    
                    scores += self.progress_reward * prog
                    
                    scores -= (off_prob ** 2) * 200.0 # Soft alignment

                # Jitter penalty to keep the car perfectly smooth on straights while allowing 100% lock at hairpins
                steer_diffs = torch.abs(actions[:, 1:, 0] - actions[:, :-1, 0])
                steer_penalty = 5.0 * steer_diffs.sum(dim=1)
                scores -= steer_penalty

                elite_idxs = torch.topk(scores, self.num_elites).indices
                elite_actions = actions[elite_idxs]
                
                self.mu = elite_actions.mean(dim=0)
                var = elite_actions.var(dim=0) + 1e-5
                
        best_action = self.mu[0].cpu().numpy()
        
        if abs(best_action[0]) < 0.1:
            best_action[0] = 0.0
            
        if obs["aux"][0] < 0.1 and best_action[1] < 0.1:
            best_action[1] = 0.4
            
        return best_action
