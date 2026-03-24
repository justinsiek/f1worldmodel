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
    parser.add_argument("--policy", type=str, default="all",
                        choices=["random", "scripted", "noisy", "kamikaze", "latebrake", "wobble", "drift", "braketest", "recovery", "apexcut", "rightbias", "leftbias", "sinewave", "coast", "snake", "brakepump", "donut", "reverse", "panic", "all"])
    parser.add_argument("--save_dir", type=str, default="data/trajectories")
    parser.add_argument("--track", type=str, default=None)
    parser.add_argument("--all_tracks", action="store_true", help="Split collection equally across all tracks")
    args = parser.parse_args()

    config = Config()
    
    import glob
    if args.all_tracks:
        tracks = sorted(glob.glob("tracks/*.csv"))
        steps_per_track = max(10, args.num_steps // len(tracks))
    else:
        tracks = [args.track] if args.track else ["tracks/Monza.csv"]
        steps_per_track = args.num_steps

    for track_path in tracks:
        config.track_csv = track_path
        track_name = os.path.basename(track_path).replace(".csv", "")
        
        print(f"\\n--- {track_name} ---")
        env = F1Env.from_config(config)

        policies_to_run = [
            "scripted", "noisy", "random", "kamikaze", "latebrake", 
            "wobble", "drift", "braketest", "recovery", "apexcut",
            "rightbias", "leftbias", "sinewave", "coast", "snake",
            "brakepump", "donut", "reverse", "panic"
        ] if args.policy == "all" else [args.policy]
        steps_per_policy = steps_per_track // len(policies_to_run)
        
        for pol_name in policies_to_run:
            print(f"  -> Profiling {pol_name} policy ({steps_per_policy} steps)")
            if pol_name == "random":
                policy = RandomPolicy()
                needs_car_state = False
            elif pol_name == "scripted":
                policy = ScriptedPolicy(env.track, lookahead=5)
                needs_car_state = True
            elif pol_name == "noisy":
                policy = NoisyScriptedPolicy(env.track, lookahead=5)
                needs_car_state = True
            elif pol_name == "kamikaze":
                from data.controllers import KamikazePolicy
                policy = KamikazePolicy()
                needs_car_state = False
            elif pol_name == "wobble":
                from data.controllers import WobblePolicy
                policy = WobblePolicy()
                needs_car_state = False
            elif pol_name == "latebrake":
                from data.controllers import LateBrakePolicy
                policy = LateBrakePolicy(env.track, lookahead=5)
                needs_car_state = True
            elif pol_name == "drift":
                from data.controllers import DriftPolicy
                policy = DriftPolicy()
                needs_car_state = False
            elif pol_name == "braketest":
                from data.controllers import BrakeTestPolicy
                policy = BrakeTestPolicy()
                needs_car_state = False
            elif pol_name == "recovery":
                from data.controllers import RecoveryPolicy
                policy = RecoveryPolicy(env.track, lookahead=8)
                needs_car_state = True
            elif pol_name == "apexcut":
                from data.controllers import ApexCutPolicy
                policy = ApexCutPolicy(env.track, lookahead=40)
                needs_car_state = True
            elif pol_name == "rightbias":
                from data.controllers import RightBiasPolicy
                policy = RightBiasPolicy(env.track, lookahead=5)
                needs_car_state = True
            elif pol_name == "leftbias":
                from data.controllers import LeftBiasPolicy
                policy = LeftBiasPolicy(env.track, lookahead=5)
                needs_car_state = True
            elif pol_name == "sinewave":
                from data.controllers import SineWavePolicy
                policy = SineWavePolicy()
                needs_car_state = False
            elif pol_name == "coast":
                from data.controllers import CoastPolicy
                policy = CoastPolicy(env.track, lookahead=5)
                needs_car_state = True
            elif pol_name == "snake":
                from data.controllers import SnakePolicy
                policy = SnakePolicy()
                needs_car_state = False
            elif pol_name == "brakepump":
                from data.controllers import BrakePumpPolicy
                policy = BrakePumpPolicy()
                needs_car_state = False
            elif pol_name == "donut":
                from data.controllers import DonutPolicy
                policy = DonutPolicy()
                needs_car_state = False
            elif pol_name == "reverse":
                from data.controllers import ReverseCornerPolicy
                policy = ReverseCornerPolicy(env.track, lookahead=5)
                needs_car_state = True
            elif pol_name == "panic":
                from data.controllers import PanicSwervePolicy
                policy = PanicSwervePolicy(env.track, lookahead=5)
                needs_car_state = True
            else:
                raise ValueError(f"Unknown policy: {pol_name}")

            collect_trajectories(
                env, policy, steps_per_policy,
                save_dir=args.save_dir,
                policy_name=pol_name,
                track_name=track_name,
                needs_car_state=needs_car_state,
            )

if __name__ == "__main__":
    main()
