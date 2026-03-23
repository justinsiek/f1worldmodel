"""
Collect trajectory data from the F1 environment.

Usage:
    python collect.py --num_steps 5000 --policy scripted
    python collect.py --num_steps 5000 --policy random
    python collect.py --num_steps 5000 --policy noisy
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default import Config
from env.f1_env import F1Env
from data.controllers import RandomPolicy, ScriptedPolicy, NoisyScriptedPolicy
from data.collector import collect_trajectories


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory data")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--policy", type=str, default="scripted",
                        choices=["random", "scripted", "noisy"])
    parser.add_argument("--save_dir", type=str, default="data/trajectories")
    parser.add_argument("--track", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    if args.track:
        config.track_csv = args.track

    print(f"Creating environment with track: {config.track_csv}")
    env = F1Env.from_config(config)

    if args.policy == "random":
        policy = RandomPolicy()
        needs_car_state = False
    elif args.policy == "scripted":
        policy = ScriptedPolicy(env.track, lookahead=5)
        needs_car_state = True
    elif args.policy == "noisy":
        policy = NoisyScriptedPolicy(env.track, lookahead=5)
        needs_car_state = True
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    print(f"Collecting {args.num_steps} steps with {args.policy} policy...")
    save_path = collect_trajectories(
        env, policy, args.num_steps,
        save_dir=args.save_dir,
        policy_name=args.policy,
        needs_car_state=needs_car_state,
    )
    print(f"Done! Saved to {save_path}")


if __name__ == "__main__":
    main()
