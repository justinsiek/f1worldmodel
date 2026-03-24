import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Loads compressed .npz offline trajectories.
    Yields overlapping sequences of length `seq_len` for multi-step prediction training.
    """
    def __init__(self, data_dir="data/trajectories", seq_len=8):
        self.seq_len = seq_len
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        if not self.files:
            raise FileNotFoundError(f"No .npz trajectory files found in {data_dir}")
            
        print(f"Loading {len(self.files)} trajectory files...")
        
        self.rasters = []
        self.auxs = []
        self.actions = []
        self.progresses = []
        self.on_tracks = []
        self.dones = []
        
        total_transitions = 0
        for f in self.files:
            data = np.load(f)
            
            # Extract
            r = data["rasters"]      # [T, 3, 64, 64]   float32
            ax = data["auxs"]        # [T, A]           float32
            ac = data["actions"]     # [T, 3]           float32
            p = data["progresses"]   # [T]              float32
            ot = data["on_tracks"]   # [T]              bool -> float32
            d = data["dones"]        # [T]              bool
            
            self.rasters.append(r)
            self.auxs.append(ax)
            self.actions.append(ac)
            self.progresses.append(p)
            self.on_tracks.append(ot.astype(np.float32))
            self.dones.append(d)
            
            total_transitions += len(r)
            
        # Concatenate everything into massive contiguous arrays
        # (This uses ~500MB of RAM for 10k steps, which is fine)
        self.rasters = np.concatenate(self.rasters, axis=0)
        self.auxs = np.concatenate(self.auxs, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.progresses = np.concatenate(self.progresses, axis=0)
        self.on_tracks = np.concatenate(self.on_tracks, axis=0)
        self.dones = np.concatenate(self.dones, axis=0)
        
        print(f"Loaded {total_transitions} total transitions.")
        
        # Build valid starting indices for sequence chunks
        # A valid start index `i` is one where `i` to `i + seq_len` does not cross an episode boundary (done == True)
        self.valid_starts = []
        T = len(self.dones)
        
        for i in range(T - self.seq_len):
            # If any step in the sequence [i, i + seq_len - 1] is a terminal 'done' state,
            # this chunk crosses an episode boundary and is invalid for continuous unrolling.
            if not np.any(self.dones[i:i + self.seq_len]):
                self.valid_starts.append(i)
                
        print(f"Generated {len(self.valid_starts)} valid sequence chunks of length {self.seq_len}.")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len
        
        # Slicing creates extremely fast views into the numpy memory buffers
        raster_seq = self.rasters[start:end]
        aux_seq = self.auxs[start:end]
        action_seq = self.actions[start:end]
        
        # Targets for heads (shifted by 1 -> predict state at t+1)
        # Sequence at `t` predicts targets at `t+1`
        progress_target = self.progresses[start+1:end+1]
        offtrack_target = 1.0 - self.on_tracks[start+1:end+1] # 1.0 if off track, 0.0 if on track
        
        return {
            "raster": torch.from_numpy(raster_seq),          # [H, 3, 64, 64]
            "aux": torch.from_numpy(aux_seq),                # [H, A]
            "action": torch.from_numpy(action_seq),          # [H, 3]
            "progress_target": torch.from_numpy(progress_target), # [H]
            "offtrack_target": torch.from_numpy(offtrack_target)  # [H]
        }
