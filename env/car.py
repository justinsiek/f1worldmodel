import numpy as np


class Car:
    """Kinematic bicycle-model car."""

    def __init__(self, x=0.0, y=0.0, theta=0.0, velocity=0.0,
                 max_speed=50.0, max_steer_rate=3.5, dt=0.1):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity
        self.max_speed = max_speed
        self.max_steer_rate = max_steer_rate
        self.dt = dt

    def step(self, steer: float, throttle: float, brake: float):
        """
        Update car state.
        steer:    [-1, 1]
        throttle: [0, 1]
        brake:    [0, 1]
        """
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # Heading update — direct steering, no speed damping
        self.theta += steer * self.max_steer_rate * self.dt

        # Normalise theta to [-pi, pi]
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # Velocity update (fixed physical rates, not scaled to max_speed)
        accel = throttle * 12.0    # ~12 m/s² at full throttle (0-50 in ~4s)
        decel = brake * 25.0       # ~25 m/s² at full brake (strong braking)
        self.velocity += (accel - decel) * self.dt
        self.velocity = np.clip(self.velocity, 0.0, self.max_speed)

        # Position update
        self.x += self.velocity * np.cos(self.theta) * self.dt
        self.y += self.velocity * np.sin(self.theta) * self.dt

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "velocity": self.velocity,
        }

    def set_state(self, x, y, theta, velocity=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity
