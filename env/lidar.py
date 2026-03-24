import numpy as np


class Lidar:
    """
    Computes mathematical ray-polygon intersections to simulate LiDAR rays.
    Fully vectorized in NumPy to run thousands of intersection checks in < 1ms.
    """
    def __init__(self, track, num_rays: int = 15, max_range: float = 150.0, fov: float = np.pi):
        self.track = track
        self.num_rays = num_rays
        self.max_range = max_range
        self.fov = fov
        
        # Pre-build all line segments (A -> B) for the track boundaries
        # We append the track boundaries seamlessly
        A_l = track.boundary_left[:-1]
        B_l = track.boundary_left[1:]
        A_l = np.vstack((A_l, track.boundary_left[-1]))
        B_l = np.vstack((B_l, track.boundary_left[0]))
        
        A_r = track.boundary_right[:-1]
        B_r = track.boundary_right[1:]
        A_r = np.vstack((A_r, track.boundary_right[-1]))
        B_r = np.vstack((B_r, track.boundary_right[0]))
        
        # Master segment lists
        self.A = np.vstack((A_l, A_r))  # (M, 2)
        self.B = np.vstack((B_l, B_r))  # (M, 2)

    def scan(self, x: float, y: float, theta: float):
        """
        Casts `num_rays` and returns distance to nearest wall normalized to [0, 1].
        """
        # Angles spread from -fov/2 to fov/2, aligned to car heading
        angles = np.linspace(-self.fov/2, self.fov/2, self.num_rays) + theta
        
        # Ray directions D: (N, 2)
        D = np.column_stack((np.cos(angles), np.sin(angles)))
        
        P = np.array([x, y])
        
        # Vectors from P to segment start, and segment start to end
        v1 = P - self.A     # (M, 2)
        v2 = self.B - self.A # (M, 2)
        
        # 2D Cross Products to solve intersection algebraically
        D_x = D[:, 0].reshape(-1, 1) # (N, 1)
        D_y = D[:, 1].reshape(-1, 1) # (N, 1)
        
        v2_x = v2[:, 0].reshape(1, -1) # (1, M)
        v2_y = v2[:, 1].reshape(1, -1) # (1, M)
        
        cross_D_v2 = D_x * v2_y - D_y * v2_x # (N, M)
        
        # Prevent division by zero for parallel lines
        valid = np.abs(cross_D_v2) > 1e-6
        
        v1_x = v1[:, 0].reshape(1, -1) # (1, M)
        v1_y = v1[:, 1].reshape(1, -1) # (1, M)
        
        cross_v1_v2 = v1_x * v2_y - v1_y * v2_x # (1, M) broadcasts -> (N, M)
        cross_v1_D = v1_x * D_y - v1_y * D_x    # (N, M)
        
        # Ray distance `t`, segment fraction `u`
        t = -cross_v1_v2 / (cross_D_v2 + 1e-10)
        u = -cross_v1_D / (cross_D_v2 + 1e-10)
        
        # Valid hits: in front of ray (t>0) and within segment bounds (0 <= u <= 1)
        hit = valid & (t > 0) & (u >= 0) & (u <= 1)
        
        # Overwrite non-hits with max_range
        t[~hit] = self.max_range
        
        # Minimum distance hit per ray
        distances = np.min(t, axis=1)
        distances = np.clip(distances, 0, self.max_range)
        
        return distances / self.max_range
