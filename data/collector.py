import numpy as np
import os


def collect_trajectories(env, policy, num_steps: int,
                         save_dir: str = "data/trajectories",
                         policy_name: str = "default",
                         needs_car_state: bool = False):
    """
    Collect transitions from rollouts.

    Returns list of transitions and saves to disk as .npz.
    """
    os.makedirs(save_dir, exist_ok=True)

    rasters = []
    auxs = []
    actions = []
    next_rasters = []
    next_auxs = []
    rewards = []
    dones = []
    progresses = []
    on_tracks = []

    obs = env.reset()
    steps_collected = 0

    while steps_collected < num_steps:
        if needs_car_state:
            action = policy(obs, car_state=env.get_car_state())
        else:
            action = policy(obs)

        next_obs, reward, done, info = env.step(action)

        rasters.append(obs["raster"])
        auxs.append(obs["aux"])
        actions.append(action)
        next_rasters.append(next_obs["raster"])
        next_auxs.append(next_obs["aux"])
        rewards.append(reward)
        dones.append(done)
        progresses.append(info.get("progress", 0.0))
        on_tracks.append(info.get("on_track", True))

        steps_collected += 1
        obs = next_obs

        if done:
            obs = env.reset()

        if steps_collected % 1000 == 0:
            print(f"  Collected {steps_collected}/{num_steps} steps")

    # Save to disk
    save_path = os.path.join(save_dir, f"{policy_name}_{num_steps}.npz")
    np.savez_compressed(
        save_path,
        rasters=np.array(rasters),
        auxs=np.array(auxs),
        actions=np.array(actions),
        next_rasters=np.array(next_rasters),
        next_auxs=np.array(next_auxs),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=np.bool_),
        progresses=np.array(progresses, dtype=np.float32),
        on_tracks=np.array(on_tracks, dtype=np.bool_),
    )
    print(f"Saved {steps_collected} transitions to {save_path}")
    return save_path
