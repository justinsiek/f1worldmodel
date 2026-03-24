import torch
import numpy as np


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) Latent Model Predictive Control.
    Iteratively refines a Gaussian action distribution to discover smooth, 
    high-reward driving trajectories in imagination. Uses a receding horizon 
    warm-start to maintain extreme temporal smoothness (no steering jitter).
    """
    def __init__(self, model, num_candidates: int = 400, horizon: int = 15, 
                 iterations: int = 4, num_elites: int = 40, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.num_candidates = num_candidates
        self.horizon = horizon
        self.iterations = iterations
        self.num_elites = num_elites
        self.device = device
        
        # Scoring hyperparameters (Uncertainty decoupled)
        self.progress_reward = 10.0

        # Action limits: [steer, throttle, brake]
        self.action_ub = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.action_lb = torch.tensor([-1.0, 0.0, 0.0], device=device)
        
        # Maintain the optimal plan sequence from the previous timestep (Warm Starting)
        self.mu = torch.zeros((self.horizon, 3), device=self.device)
        self.mu[:, 1] = 0.5 # Warm-start with 50% baseline throttle
        
    def __call__(self, obs, car_state=None):
        # 1. Structure raw observation
        raster = torch.from_numpy(obs["raster"]).float().unsqueeze(0).to(self.device)
        aux = torch.from_numpy(obs["aux"]).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 2. Encode true world state to latent seed `z_0`
            z_0 = self.model.encode(raster, aux)
            
            # 3. Receding Horizon: Shift the previous optimal mean plan forward by 1 step
            self.mu[:-1] = self.mu[1:].clone()
            self.mu[-1] = torch.tensor([0.0, 0.5, 0.0], device=self.device) # reset terminal step to baseline throttle
            
            # Expansive search variance so the solver can discover 100% braking!
            var = torch.ones((self.horizon, 3), device=self.device) * 0.5
            var[:, 0] = 0.25
            
            # CEM Iteration Loop
            for i in range(self.iterations):
                # Sample N action sequences from the current N(mu, var) belief
                noise = torch.randn((self.num_candidates, self.horizon, 3), device=self.device)
                actions = self.mu.unsqueeze(0) + noise * torch.sqrt(var).unsqueeze(0)
                
                # Clip to valid bounds
                actions = torch.max(torch.min(actions, self.action_ub), self.action_lb)
                
                # Enforce mutually exclusive Throttle/Brake to prevent OOD action locking
                throttle_mask = actions[:, :, 1] > actions[:, :, 2]
                actions[:, :, 1] = actions[:, :, 1] * throttle_mask
                actions[:, :, 2] = actions[:, :, 2] * (~throttle_mask)
                
                # Rollout Latent Imagination
                z = z_0.expand(self.num_candidates, -1)
                scores = torch.zeros(self.num_candidates, device=self.device)
                
                for t in range(self.horizon):
                    z = self.model.predict(z, actions[:, t, :])
                    
                    # Evaluate imagined state
                    prog = self.model.progress_head(z)
                    off_logits = self.model.offtrack_head(z)
                    off_prob = torch.sigmoid(off_logits)
                    
                    scores += self.progress_reward * prog
                    
                    # Hard prune trajectories that mathematically predict a grass touch.
                    # Restored threshold to 0.5 since the new balanced dataset will cleanly remove statistical pessimism.
                    death_mask = off_prob > 0.5
                    scores[death_mask] = -1e6

                # Relaxed steering penalty allow the optimizer to naturally carve through corners smoothly
                steer_diffs = torch.abs(actions[:, 1:, 0] - actions[:, :-1, 0])
                steer_penalty = 1.0 * steer_diffs.sum(dim=1)
                scores -= steer_penalty

                # Select Elites
                elite_idxs = torch.topk(scores, self.num_elites).indices
                elite_actions = actions[elite_idxs] # (num_elites, H, 3)
                
                # Refit Gaussian belief to the Elites (tighten the search)
                self.mu = elite_actions.mean(dim=0)
                var = elite_actions.var(dim=0) + 1e-5 # add small epsilon to prevent collapse
                
        # Final optimized action for the immediate timestep
        best_action = self.mu[0].cpu().numpy()
        
        # Deadzone to perfectly center the wheel on straights (kills baseline drift)
        if abs(best_action[0]) < 0.05:
            best_action[0] = 0.0
            
        # Deadlock prevention heuristic (brake lockup when stationary)
        if obs["aux"][0] < 0.1 and best_action[1] < 0.1:
            best_action[1] = 0.4
            best_action[2] = 0.0
            
        return best_action
