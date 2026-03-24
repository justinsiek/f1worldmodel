import pygame
import numpy as np


class Renderer:
    """
    Dynamically generates a 3-channel 64x64 ego-centric raster observation.
    Uses an invisible Pygame surface for lightning-fast hardware rendering.
    """

    def __init__(self, raster_size: int = 64, pixels_per_meter: float = 1.5):
        if not pygame.get_init():
            pygame.init()
            
        self.size = raster_size
        self.ppm = pixels_per_meter
        self.surface = pygame.Surface((self.size, self.size))

    def render(self, x: float, y: float, theta: float, track):
        """
        Returns a float32 numpy array of shape (3, 64, 64) with channels:
        [0]: Drivable area
        [1]: Walls / Off-track
        [2]: Ego car mask
        """
        # Channel 1: Background = Walls (Green in RGB)
        self.surface.fill((0, 255, 0))

        # Place ego car near the bottom of the mask so it sees much further ahead
        car_draw_x = self.size / 2
        car_draw_y = self.size * 0.75  # 75% down the frame

        # Rotation matrix to orient the car facing UP (increasing Y natively goes Down in Pygame)
        # F1 standard: East is 0. So facing North (Up) requires rot -theta - pi/2
        angle = -theta - np.pi / 2
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)

        # Vectorized transform for track points
        def transform(pts):
            dx = pts[:, 0] - x
            dy = pts[:, 1] - y
            # Rotate
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            # Scale & translate to car_draw_y
            sx = car_draw_x + rx * self.ppm
            sy = car_draw_y + ry * self.ppm
            return np.column_stack((sx, sy)).astype(int)

        r_pts = transform(track.boundary_right)
        l_pts = transform(track.boundary_left)

        # Channel 0: Drivable road (Red in RGB)
        poly_pts = list(r_pts) + list(l_pts)[::-1]
        pygame.draw.polygon(self.surface, (255, 0, 0), poly_pts)

        # Channel 2: Ego car (Blue in RGB)
        car_len = 5.0 * self.ppm
        car_wid = 2.0 * self.ppm
        car_rect = pygame.Rect(0, 0, car_wid, car_len)
        car_rect.center = (car_draw_x, car_draw_y)
        pygame.draw.rect(self.surface, (0, 0, 255), car_rect)

        # Extract RGB channels
        view = pygame.surfarray.pixels3d(self.surface)

        # The Pygame array is (X, Y, Channels). 
        # Output needs to be (Channels, Y, X) for PyTorch. 
        # But Pygame's X, Y translates to Width, Height, so transposing (2, 1, 0) gives (C, H, W).
        raster = np.transpose(view, (2, 1, 0)).astype(np.float32) / 255.0

        return raster
