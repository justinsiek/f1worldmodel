"""
Entry point for training the Latent World Model.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.world_model import WorldModel
from data.dataset import TrajectoryDataset
from train.trainer import WorldModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5, help="Unroll sequence length")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data_dir", type=str, default="data/trajectories")
    args = parser.parse_args()

    # Hardware acceleration setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon M1/M2/M3
    else:
        device = "cpu"
        
    print(f"Starting training on device: {device}")

    # Initialize Dataset
    dataset = TrajectoryDataset(data_dir=args.data_dir, seq_len=args.horizon)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )

    # Initialize Architecture and Optimizer
    model = WorldModel()
    trainer = WorldModelTrainer(model, lr=args.lr, seq_len=args.horizon, device=device)

    # Training Loop
    total_steps = 0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        for i, batch in enumerate(loader):
            stats = trainer.step(batch)
            total_steps += 1
            
            if i % 10 == 0:
                print(f"Step {total_steps:5d} | "
                      f"Total Loss: {stats['loss']:.4f} | "
                      f"Z-MSE: {stats['loss_z']:.4f} | "
                      f"Prog: {stats['loss_prog']:.4f} | "
                      f"Off: {stats['loss_off']:.4f}")

    # Save finalized representation model
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/world_model_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
