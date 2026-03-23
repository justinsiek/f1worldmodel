"""
Run the F1 environment with a scripted controller and Pygame visualization.

Usage:
    python run_env.py                     # scripted controller
    python run_env.py --policy noisy      # noisy scripted
    python run_env.py --policy random     # random actions
    python run_env.py --policy keyboard   # manual keyboard control
"""
import argparse
import sys
import os
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default import Config
from env.f1_env import F1Env
from data.controllers import RandomPolicy, ScriptedPolicy, NoisyScriptedPolicy
from viz.renderer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Run F1 environment with visualization")
    parser.add_argument("--policy", type=str, default="scripted",
                        choices=["random", "scripted", "noisy", "keyboard"])
    parser.add_argument("--track", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    import glob

    config = Config()
    if args.track:
        config.track_csv = args.track
    else:
        # Track Picker
        tracks = sorted(glob.glob("tracks/*.csv"))
        if not tracks:
            print("No tracks found in tracks/ directory!")
            sys.exit(1)
            
        print("\n--- SELECT TRACK ---")
        for i, t in enumerate(tracks):
            name = os.path.basename(t).replace('.csv', '')
            print(f"[{i+1}] {name}")
            
        while True:
            try:
                choice = input(f"Enter track number (1-{len(tracks)}) [1]: ").strip()
                if choice == "":
                    idx = 0
                    break
                idx = int(choice) - 1
                if 0 <= idx < len(tracks):
                    break
                print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")
                
        config.track_csv = tracks[idx]

    print(f"Creating environment with track: {config.track_csv}")
    env = F1Env.from_config(config)

    # Create policy
    if args.policy == "random":
        policy = RandomPolicy()
        needs_car_state = False
    elif args.policy == "scripted":
        policy = ScriptedPolicy(env.track, lookahead=5)
        needs_car_state = True
    elif args.policy == "noisy":
        policy = NoisyScriptedPolicy(env.track, lookahead=5)
        needs_car_state = True
    elif args.policy == "keyboard":
        policy = None
        needs_car_state = False

    print(f"Starting visualization with {args.policy} policy...")
    if args.policy == "keyboard":
        print("Controls: Arrow keys (Up=throttle, Down=brake, Left/Right=steer)")

    viz = Visualizer(env.track)

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            # Get action
            if args.policy == "keyboard":
                action = get_keyboard_action()
            elif needs_car_state:
                action = policy(obs, car_state=env.get_car_state())
            else:
                action = policy(obs)

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Render
            car_state = env.get_car_state()
            if not viz.render_frame(car_state, action=action, info=info, obs=obs):
                print("Window closed")
                viz.close()
                return

        print(f"Episode {ep+1}/{args.episodes}: "
              f"reward={episode_reward:.2f}, "
              f"progress={info.get('progress', 0):.3f}, "
              f"steps={step_count}, "
              f"{'LAP!' if info.get('lap_complete') else info.get('termination', '')}")

    viz.close()
    print("Done!")


def get_keyboard_action():
    """Get action from keyboard input."""
    keys = pygame.key.get_pressed()
    steer = 0.0
    throttle = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steer = -0.5
    if keys[pygame.K_RIGHT]:
        steer = 0.5
    if keys[pygame.K_UP]:
        throttle = 0.8
    if keys[pygame.K_DOWN]:
        brake = 0.5

    return np.array([steer, throttle, brake], dtype=np.float32)


if __name__ == "__main__":
    main()
