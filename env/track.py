import numpy as np
from typing import Optional


class Track:
    """
    Track loaded from TUMFTM racetrack-database CSV.
    CSV format: x_m, y_m, w_tr_right_m, w_tr_left_m
    """

    def __init__(self, centerline: np.ndarray, widths_right: np.ndarray,
                 widths_left: np.ndarray, pixels_per_meter: float = 3.0):
        self.centerline = centerline        # (N, 2)
        self.widths_right = widths_right    # (N,)
        self.widths_left = widths_left      # (N,)
        self.ppm = pixels_per_meter
        self.num_points = len(centerline)

        # Compute normals, boundaries, arc lengths
        self._compute_normals()
        self._compute_boundaries()
        self._compute_arc_lengths()

    @classmethod
    def load(cls, csv_path: str, pixels_per_meter: float = 3.0) -> "Track":
        """Load track from TUMFTM CSV file."""
        data = np.loadtxt(csv_path, delimiter=",", comments="#")
        centerline = data[:, :2]
        widths_right = data[:, 2]
        widths_left = data[:, 3]
        return cls(centerline, widths_right, widths_left, pixels_per_meter)

    def _compute_normals(self):
        """Compute outward-facing normal vectors at each centerline point."""
        # Tangent vectors via finite differences (closed loop)
        tangents = np.zeros_like(self.centerline)
        tangents[:-1] = self.centerline[1:] - self.centerline[:-1]
        tangents[-1] = self.centerline[0] - self.centerline[-1]

        # Normalise
        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)
        tangents = tangents / lengths

        # Right-hand normals (rotate tangent 90° clockwise)
        self.normals_right = np.column_stack([tangents[:, 1], -tangents[:, 0]])
        self.normals_left = -self.normals_right

    def _compute_boundaries(self):
        """Compute inner and outer boundary polylines."""
        self.boundary_right = (
            self.centerline + self.normals_right * self.widths_right[:, None]
        )
        self.boundary_left = (
            self.centerline + self.normals_left * self.widths_left[:, None]
        )

    def _compute_arc_lengths(self):
        """Cumulative arc-length along the centerline for progress."""
        diffs = np.diff(self.centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.zeros(self.num_points)
        self.arc_lengths[1:] = np.cumsum(seg_lengths)
        # Add closing segment
        close_len = np.linalg.norm(self.centerline[0] - self.centerline[-1])
        self.total_length = self.arc_lengths[-1] + close_len

    def is_on_track(self, x: float, y: float) -> bool:
        """Check if world position is on the drivable surface dynamically."""
        idx = self.get_nearest_centerline_idx(x, y)
        cx, cy = self.centerline[idx]
        
        # Distance to centerline point
        dist = np.hypot(x - cx, y - cy)
        
        # Average allowed width at this point (conservative approximation)
        allowed_dist = (self.widths_right[idx] + self.widths_left[idx]) / 2.0
        
        return float(dist) <= allowed_dist

    def get_nearest_centerline_idx(self, x: float, y: float) -> int:
        """Find the index of the nearest centerline point."""
        dists = np.linalg.norm(self.centerline - np.array([x, y]), axis=1)
        return int(np.argmin(dists))

    def get_progress(self, x: float, y: float) -> float:
        """Get lap progress as a fraction [0, 1]."""
        idx = self.get_nearest_centerline_idx(x, y)
        return self.arc_lengths[idx] / self.total_length

    def get_start_state(self):
        """Return (x, y, theta) at the start of the track."""
        x, y = self.centerline[0]
        # Heading toward next point
        dx = self.centerline[1, 0] - x
        dy = self.centerline[1, 1] - y
        theta = np.arctan2(dy, dx)
        return x, y, theta
