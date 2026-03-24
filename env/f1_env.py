import numpy as np
from env.car import Car
from env.track import Track
from env.renderer import Renderer
from env.lidar import Lidar


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
        self.renderer = Renderer(raster_size=raster_size, pixels_per_meter=pixels_per_meter)
        self.lidar = Lidar(self.track, num_rays=15)
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

        old_x, old_y = self.car.x, self.car.y

        # Step car
        self.car.step(steer, throttle, brake)
        self.steps += 1

        # Check track status
        on_track = self.track.is_on_track(self.car.x, self.car.y)

        # 1. Global track progress (0 to 1) for Lap Tracking
        progress = self.track.get_progress(self.car.x, self.car.y)
        
        # Only log halfway marker if actually near the halfway point (prevents start-line reverse wraparound bugs)
        if 0.4 < progress < 0.6:
            self.crossed_halfway = True

        # 2. Local projected progress for AI Reward (Dot Product vs Centerline)
        idx = self.track.get_nearest_centerline_idx(old_x, old_y)
        target_idx = (idx + 3) % self.track.num_points
        target = self.track.centerline[target_idx]
        
        tx = target[0] - old_x
        ty = target[1] - old_y
        t_len = np.sqrt(tx**2 + ty**2) + 1e-6
        tx, ty = tx / t_len, ty / t_len
        
        dx = self.car.x - old_x
        dy = self.car.y - old_y
        
        projected_progress = dx * tx + dy * ty

        # Reward
        reward = self.progress_reward * projected_progress * 1000  # scale progress
        if not on_track:
            reward -= self.off_track_penalty
            self.off_track_count += 1
        else:
            self.off_track_count = 0
        reward -= self.step_penalty

        # Check done conditions
        done = False
        info = {"progress": projected_progress, "on_track": on_track, "lap_complete": False}

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
        raster = self.renderer.render(
            self.car.x, self.car.y, self.car.theta, self.track
        )
        speed_norm = np.array([self.car.velocity / self.car.max_speed], dtype=np.float32)
        ray_dists = self.lidar.scan(self.car.x, self.car.y, self.car.theta)
        
        aux = np.concatenate([speed_norm, ray_dists]).astype(np.float32)
        return {"raster": raster, "aux": aux}

    def get_car_state(self):
        return self.car.get_state()

    def get_progress(self):
        return self.track.get_progress(self.car.x, self.car.y)
