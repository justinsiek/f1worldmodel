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

class KamikazePolicy:
    """Holds 100% throttle, zero steering, guaranteed high-speed wall collision."""
    def __call__(self, obs, car_state=None):
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

class WobblePolicy:
    """Holds 100% throttle but yanks the wheel violently left and right."""
    def __call__(self, obs, car_state=None):
        steer = np.random.choice([-1.0, 0.0, 1.0])
        return np.array([steer, 1.0, 0.0], dtype=np.float32)

class LateBrakePolicy:
    """Follows the expert racing line but randomly 'forgets' to brake 30% of the time, causing apex crashes."""
    def __init__(self, track, lookahead: int = 5):
        self.scripted = ScriptedPolicy(track, lookahead)
        
    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        # 30% chance to completely ignore the brake and pin the throttle
        if np.random.rand() < 0.3:
            action[1] = 1.0
            action[2] = 0.0
        return action

class DriftPolicy:
    """Deliberately induces extreme oversteer slides by holding throttle and pinning the steering wheel."""
    def __call__(self, obs, car_state=None):
        steer = 1.0 if np.random.rand() > 0.5 else -1.0
        return np.array([steer, 1.0, 0.0], dtype=np.float32)

class BrakeTestPolicy:
    """Accelerates to high speed, then locks the brakes abruptly to gather strict linear kinematic data."""
    def __init__(self):
        self.step_counter = 0
        
    def __call__(self, obs, car_state=None):
        self.step_counter += 1
        # Accelerate for 3 seconds (30 steps), brake for 3 seconds, repeat
        if (self.step_counter % 60) < 30:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

class RecoveryPolicy:
    """Drives off the racing line toward the wall, then attempts a frantic last-second recovery."""
    def __init__(self, track, lookahead: int = 8):
        self.scripted = ScriptedPolicy(track, lookahead)
        
    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        # Add a massive, smooth bias that pulls the car off the racing line
        if np.random.rand() > 0.15:
            action[0] = np.clip(action[0] + 0.6, -1.0, 1.0)
        # 15% of the time, the scripted policy takes over perfectly, snapping the car back
        return action

class ApexCutPolicy:
    """Aggressively turns into corners far too early to gather data on inner-wall collisions."""
    def __init__(self, track, lookahead: int = 40): # Huge lookahead causes early turn-in
        self.scripted = ScriptedPolicy(track, lookahead)
    
    def __call__(self, obs, car_state=None):
        return self.scripted(obs, car_state=car_state)

class RightBiasPolicy:
    """Maintains a heavy aerodynamic offset to the right side of the track."""
    def __init__(self, track, lookahead: int = 5):
        self.scripted = ScriptedPolicy(track, lookahead)
        
    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        action[0] = np.clip(action[0] + 0.4, -1.0, 1.0)
        return action

class LeftBiasPolicy:
    """Maintains a heavy aerodynamic offset to the left side of the track."""
    def __init__(self, track, lookahead: int = 5):
        self.scripted = ScriptedPolicy(track, lookahead)
        
    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        action[0] = np.clip(action[0] - 0.4, -1.0, 1.0)
        return action

class SineWavePolicy:
    """Smoothly weaves completely across the full width of the track, mapping symmetrical lateral aerodynamics."""
    def __init__(self):
        self.step_counter = 0
        
    def __call__(self, obs, car_state=None):
        self.step_counter += 1
        steer = np.sin(self.step_counter / 10.0) 
        return np.array([steer, 0.7, 0.0], dtype=np.float32)

class CoastPolicy:
    """Sits completely neutral on the pedals to gather perfect rolling-resistance deceleration curves."""
    def __init__(self, track, lookahead: int = 5):
        self.scripted = ScriptedPolicy(track, lookahead)
        
    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        action[1] = 0.0  # Zero throttle
        action[2] = 0.0  # Zero brake
        return action

class SnakePolicy:
    """Aggressively swerves back and forth in sharp jagged lines to map rapid lateral G-force changes."""
    def __init__(self):
        self.step_counter = 0
        
    def __call__(self, obs, car_state=None):
        self.step_counter += 1
        steer = 1.0 if (self.step_counter % 30) < 15 else -1.0
        return np.array([steer, 0.8, 0.0], dtype=np.float32)

class BrakePumpPolicy:
    """Simulates an active Anti-Lock Braking System (ABS) by rapidly fluttering the brake pedal during hard stops."""
    def __init__(self):
        self.step_counter = 0
        
    def __call__(self, obs, car_state=None):
        self.step_counter += 1
        if (self.step_counter % 30) < 20:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            brake = 1.0 if (self.step_counter % 2) == 0 else 0.0
            return np.array([0.0, 0.0, brake], dtype=np.float32)

class DonutPolicy:
    """Pins the throttle and steering wheel to map 360-degree rotational burnouts."""
    def __init__(self):
        self.steer_dir = 1.0 if np.random.rand() > 0.5 else -1.0
        
    def __call__(self, obs, car_state=None):
        return np.array([self.steer_dir, 1.0, 0.0], dtype=np.float32)

class ReverseCornerPolicy:
    """Queries the expert racing line target, and intentionally steers the exact opposite direction off the track."""
    def __init__(self, track, lookahead: int = 5):
        self.scripted = ScriptedPolicy(track, lookahead)
        
    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        action[0] = -action[0] # Flip steering to opposite lock
        return action

class PanicSwervePolicy:
    """Follows the line smoothly, but occasionally yanks the steering wheel to maximum lock for 5 consecutive frames."""
    def __init__(self, track, lookahead: int = 5):
        self.scripted = ScriptedPolicy(track, lookahead)
        self.panic_counter = 0
        self.panic_dir = 0.0
        
    def __call__(self, obs, car_state=None):
        action = self.scripted(obs, car_state=car_state)
        if self.panic_counter > 0:
            self.panic_counter -= 1
            action[0] = self.panic_dir
            return action
            
        if np.random.rand() < 0.02:
            self.panic_counter = 5
            self.panic_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            
        return action
