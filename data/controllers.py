import numpy as np


class RandomPolicy:
    """Uniform random actions."""

    def __call__(self, obs):
        return np.array([
            np.random.uniform(-1, 1),   # steer
            np.random.uniform(0, 1),    # throttle
            np.random.uniform(0, 0.3),  # brake (less aggressive)
        ], dtype=np.float32)


class ScriptedPolicy:
    """
    PD controller that steers toward a lookahead point on the centerline
    and adjusts throttle based on upcoming curvature.
    """

    def __init__(self, track, lookahead: int = 20):
        self.track = track
        self.lookahead = lookahead

    def __call__(self, obs, car_state=None):
        if car_state is None:
            raise ValueError("ScriptedPolicy requires car_state")

        x, y, theta, velocity = (
            car_state["x"], car_state["y"],
            car_state["theta"], car_state["velocity"]
        )

        # Find nearest centerline point
        idx = self.track.get_nearest_centerline_idx(x, y)

        # Dynamic lookahead: farther when fast, shorter when slow
        speed_factor = max(0.3, velocity / 50.0)
        dynamic_la = max(5, int(self.lookahead * speed_factor))
        target_idx = (idx + dynamic_la) % self.track.num_points
        target = self.track.centerline[target_idx]

        # Compute angle to target
        dx = target[0] - x
        dy = target[1] - y
        target_angle = np.arctan2(dy, dx)

        # Heading error
        angle_error = target_angle - theta
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # PD steer — strong gain for responsive turning
        steer = np.clip(angle_error * 3.0, -1.0, 1.0)

        # Look further ahead to anticipate curvature
        future_idx = (idx + dynamic_la * 2) % self.track.num_points
        future = self.track.centerline[future_idx]
        fdx = future[0] - target[0]
        fdy = future[1] - target[1]
        future_angle = np.arctan2(fdy, fdx)
        upcoming_curve = abs((future_angle - target_angle + np.pi) % (2 * np.pi) - np.pi)

        # Throttle/brake based on current error AND upcoming curvature
        curvature = max(abs(angle_error), upcoming_curve)

        if curvature > 0.5:
            throttle = 0.15
            brake = 0.4
        elif curvature > 0.3:
            throttle = 0.3
            brake = 0.2
        elif curvature > 0.15:
            throttle = 0.5
            brake = 0.0
        else:
            throttle = 0.85
            brake = 0.0

        return np.array([steer, throttle, brake], dtype=np.float32)


class NoisyScriptedPolicy:
    """Scripted policy with Gaussian noise for exploration."""

    def __init__(self, track, lookahead: int = 5,
                 steer_noise: float = 0.2, throttle_noise: float = 0.1):
        self.scripted = ScriptedPolicy(track, lookahead)
        self.steer_noise = steer_noise
        self.throttle_noise = throttle_noise

    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        noise = np.array([
            np.random.normal(0, self.steer_noise),
            np.random.normal(0, self.throttle_noise),
            np.random.normal(0, 0.05),
        ], dtype=np.float32)
        action = action + noise
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)
        return action
