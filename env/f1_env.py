import numpy as np
from env.car import Car
from env.track import Track


class F1Env:
    """
    Gym-style environment for 2D F1 racing.
    """

    def __init__(self, track_csv: str = "tracks/Monza.csv",
                 max_speed: float = 10.0, max_steer_rate: float = 2.0,
                 dt: float = 0.1, max_steps: int = 500,
                 raster_size: int = 64, pixels_per_meter: float = 3.0,
                 off_track_tolerance: int = 10,
                 progress_reward: float = 0.02,
                 off_track_penalty: float = 0.5,
                 step_penalty: float = 0.005,
                 lap_bonus: float = 1.0):
        self.track = Track.load(track_csv, pixels_per_meter)
        self.car = Car(max_speed=max_speed, max_steer_rate=max_steer_rate, dt=dt)
        self.max_steps = max_steps
        self.off_track_tolerance = off_track_tolerance
        self.progress_reward = progress_reward
        self.off_track_penalty = off_track_penalty
        self.step_penalty = step_penalty
        self.lap_bonus = lap_bonus

        self.steps = 0
        self.off_track_count = 0
        self.last_progress = 0.0
        self.crossed_halfway = False

        self.lap_count = 0
        self.lap_start_step = 0
        self.fastest_lap_time = None
        self.last_lap_time = None

    @classmethod
    def from_config(cls, config):
        return cls(
            track_csv=config.track_csv,
            max_speed=config.max_speed,
            max_steer_rate=config.max_steer_rate,
            dt=config.dt,
            max_steps=config.max_steps,
            raster_size=config.raster_size,
            pixels_per_meter=config.pixels_per_meter,
            off_track_tolerance=config.off_track_tolerance,
            progress_reward=config.progress_reward,
            off_track_penalty=config.off_track_penalty,
            step_penalty=config.step_penalty,
            lap_bonus=config.lap_bonus,
        )

    def reset(self):
        """Reset environment. Returns first observation."""
        x, y, theta = self.track.get_start_state()
        self.car.set_state(x, y, theta, velocity=0.0)
        self.steps = 0
        self.off_track_count = 0
        self.last_progress = 0.0
        self.crossed_halfway = False

        self.lap_count = 0
        self.lap_start_step = 0
        return self._get_obs()

    def step(self, action):
        """
        Take one step.
        action: dict or array-like with (steer, throttle, brake)
        Returns: obs, reward, done, info
        """
        if isinstance(action, dict):
            steer = action["steer"]
            throttle = action["throttle"]
            brake = action["brake"]
        else:
            steer, throttle, brake = action[0], action[1], action[2]

        # Step car
        self.car.step(steer, throttle, brake)
        self.steps += 1

        # Check track status
        on_track = self.track.is_on_track(self.car.x, self.car.y)

        # Progress
        progress = self.track.get_progress(self.car.x, self.car.y)
        progress_delta = progress - self.last_progress

        # Handle lap wraparound
        if progress_delta < -0.5:
            progress_delta += 1.0
        elif progress_delta > 0.5:
            progress_delta -= 1.0

        # Track halfway for lap detection
        if progress > 0.5:
            self.crossed_halfway = True

        # Reward
        reward = self.progress_reward * progress_delta * 1000  # scale progress
        if not on_track:
            reward -= self.off_track_penalty
            self.off_track_count += 1
        else:
            self.off_track_count = 0
        reward -= self.step_penalty

        # Check done conditions
        done = False
        info = {"progress": progress, "on_track": on_track, "lap_complete": False}

        # Lap complete: crossed halfway and back near start
        if self.crossed_halfway and progress < 0.1 and self.last_progress > 0.9:
            reward += self.lap_bonus
            info["lap_complete"] = True
            self.crossed_halfway = False
            
            # Lap time calculation
            lap_time = (self.steps - self.lap_start_step) * self.car.dt
            self.last_lap_time = lap_time
            if self.fastest_lap_time is None or lap_time < self.fastest_lap_time:
                self.fastest_lap_time = lap_time
            
            self.lap_start_step = self.steps
            self.lap_count += 1
            print(f"LAP {self.lap_count} COMPLETED: {lap_time:.2f}s (Fastest: {self.fastest_lap_time:.2f}s)")

        info["current_lap_time"] = (self.steps - self.lap_start_step) * self.car.dt
        info["last_lap_time"] = self.last_lap_time
        info["fastest_lap_time"] = self.fastest_lap_time
        info["lap_count"] = self.lap_count

        # Off track too long
        if self.off_track_count >= self.off_track_tolerance:
            done = True
            info["termination"] = "off_track"

        # Max steps
        if self.steps >= self.max_steps:
            done = True
            info["termination"] = "max_steps"

        self.last_progress = progress
        obs = self._get_obs()

        return obs, reward, done, info

    def _get_obs(self):
        """Get current observation."""
        aux = np.array([self.car.velocity / self.car.max_speed], dtype=np.float32)
        return {"aux": aux}

    def get_car_state(self):
        return self.car.get_state()

    def get_progress(self):
        return self.track.get_progress(self.car.x, self.car.y)
