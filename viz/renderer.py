import pygame
import numpy as np
import sys


class Visualizer:
    """
    High-fidelity follow-camera visualizer for the F1 environment.
    Zooms in on the car for a realistic sense of speed and scale.
    """

    # Colors (flat palette)
    GRASS_GREEN = (60, 130, 40)
    GRASS_GRID = (50, 110, 30)
    ROAD_GRAY = (90, 90, 90)
    BOUNDARY_WHITE = (255, 255, 255)
    KERB_RED = (200, 50, 50)
    KERB_WHITE = (240, 240, 240)
    CAR_RED = (220, 30, 30)
    CAR_OUTLINE = (40, 40, 40)
    HUD_BG = (30, 30, 30, 200)
    HUD_TEXT = (240, 240, 240)
    CENTERLINE_YELLOW = (220, 220, 60)
    THROTTLE_GREEN = (50, 200, 50)
    BRAKE_RED = (200, 50, 50)

    def __init__(self, track, width: int = 1400, height: int = 900):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("F1 World Model Simulator")
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.track = track
        self.font = pygame.font.SysFont("menlo", 13)
        self.font_large = pygame.font.SysFont("menlo", 16, bold=True)

        # Moderate follow-camera scale: 1 meter = 8.0 pixels
        self.scale = 8.0

        # Viewport properties
        self.hud_width = 280
        self.view_width = self.width - self.hud_width
        self.view_height = self.height

        # Dynamic camera center (world coords)
        self.cam_x = 0.0
        self.cam_y = 0.0

        # Compute minimap transform
        all_pts = np.vstack([self.track.boundary_right, self.track.boundary_left])
        self.track_min = all_pts.min(axis=0) - 30
        self.track_max = all_pts.max(axis=0) + 30
        track_size = self.track_max - self.track_min

        # We want the track to fit inside a 200x200 area for the minimap
        mm_size = 200
        self.mm_scale = min(mm_size / track_size[0], mm_size / track_size[1])
        # Center the minimap inside a 240x240 box at the bottom of the HUD
        self.mm_offset_x = (240 - track_size[0] * self.mm_scale) / 2
        self.mm_offset_y = (240 - track_size[1] * self.mm_scale) / 2

    def get_screen_coords(self, pts):
        """Vectorized conversion of (N, 2) world coordinates to screen pixels."""
        # Translate to camera
        dx = pts[:, 0] - self.cam_x
        dy = pts[:, 1] - self.cam_y

        # Scale and move to screen center (y flipped for standard 2D view)
        sx = self.view_width / 2 + dx * self.scale
        sy = self.view_height / 2 - dy * self.scale

        return np.column_stack((sx, sy)).astype(int)

    def get_minimap_coords(self, pts, hud_x, hud_y):
        """Convert world coordinates to minimap coordinates."""
        dx = pts[:, 0] - self.track_min[0]
        dy = pts[:, 1] - self.track_min[1]
        
        sx = hud_x + 20 + dx * self.mm_scale + self.mm_offset_x
        sy = hud_y + 240 - (dy * self.mm_scale + self.mm_offset_y)  # Flip Y
        
        return np.column_stack((sx, sy)).astype(int)

    def _draw_background_grid(self):
        """Draws a grid on the grass to give a sense of speed."""
        self.screen.fill(self.GRASS_GREEN)
        
        # Grid spacing in world meters
        grid_size = 20.0
        
        # Determine visible world bounds
        w_width_m = self.view_width / self.scale
        w_height_m = self.view_height / self.scale
        
        min_x = int((self.cam_x - w_width_m / 2) / grid_size) * grid_size
        max_x = int((self.cam_x + w_width_m / 2) / grid_size) * grid_size + grid_size
        min_y = int((self.cam_y - w_height_m / 2) / grid_size) * grid_size
        max_y = int((self.cam_y + w_height_m / 2) / grid_size) * grid_size + grid_size

        for x in np.arange(min_x, max_x, grid_size):
            pts = np.array([[x, min_y], [x, max_y]])
            s_pts = self.get_screen_coords(pts)
            pygame.draw.line(self.screen, self.GRASS_GRID, tuple(s_pts[0]), tuple(s_pts[1]), 1)

        for y in np.arange(min_y, max_y, grid_size):
            pts = np.array([[min_x, y], [max_x, y]])
            s_pts = self.get_screen_coords(pts)
            pygame.draw.line(self.screen, self.GRASS_GRID, tuple(s_pts[0]), tuple(s_pts[1]), 1)

    def _draw_track(self):
        """Renders the track dynamically scaled to the camera."""
        # Convert all boundary points to screen coords
        r_screen = self.get_screen_coords(self.track.boundary_right)
        l_screen = self.get_screen_coords(self.track.boundary_left)
        c_screen = self.get_screen_coords(self.track.centerline)

        # Draw road as quads
        n = len(r_screen)
        for i in range(n):
            j = (i + 1) % n
            
            # Simple culling: only draw if rough bounds are inside screen
            min_sx = min(r_screen[i][0], r_screen[j][0], l_screen[i][0])
            max_sx = max(r_screen[i][0], r_screen[j][0], l_screen[i][0])
            min_sy = min(r_screen[i][1], r_screen[j][1], l_screen[i][1])
            max_sy = max(r_screen[i][1], r_screen[j][1], l_screen[i][1])
            
            if max_sx < 0 or min_sx > self.view_width or max_sy < 0 or min_sy > self.view_height:
                continue

            quad = [r_screen[i], r_screen[j], l_screen[j], l_screen[i]]
            pygame.draw.polygon(self.screen, self.ROAD_GRAY, quad)
            
            # Kerbs (alternating red/white on the apexes/exits)
            # A simple rule: color segments alternating every 3 indices
            kerb_color = self.KERB_RED if (i // 3) % 2 == 0 else self.KERB_WHITE
            # Draw kerbs slightly thicker on boundaries
            pygame.draw.line(self.screen, kerb_color, tuple(r_screen[i]), tuple(r_screen[j]), 5)
            pygame.draw.line(self.screen, kerb_color, tuple(l_screen[i]), tuple(l_screen[j]), 5)

        # Draw centerline (dashed)
        step = 4
        for i in range(0, n - 1, step):
            j = min(i + int(step/2), n - 1)
            
            if c_screen[i][0] < 0 or c_screen[i][0] > self.view_width or c_screen[i][1] < 0 or c_screen[i][1] > self.view_height:
                continue
                
            pygame.draw.line(self.screen, self.CENTERLINE_YELLOW,
                             tuple(c_screen[i]), tuple(c_screen[j]), 2)

        # Start/finish line
        pygame.draw.line(self.screen, self.BOUNDARY_WHITE, tuple(r_screen[0]), tuple(l_screen[0]), 6)

    def render_frame(self, car_state, action=None, info=None, obs=None):
        """
        Render one frame.
        Returns True if should continue, False if user closed window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        # Update camera position to follow car
        # Smooth follow (lerp)
        self.cam_x += (car_state["x"] - self.cam_x) * 0.2
        self.cam_y += (car_state["y"] - self.cam_y) * 0.2

        # 1. Background Grid
        self._draw_background_grid()

        # 2. Dynamic Track
        self._draw_track()

        # 3. Car and Lidar
        self._draw_lidar_rays(car_state, obs)
        self._draw_car(car_state)

        # 4. HUD and Minimap
        self._draw_hud(car_state, action, info)
        self._draw_minimap(car_state)

        # 5. Raster Preview
        if obs is not None and "raster" in obs:
            self._draw_raster_preview(obs["raster"])

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def _draw_car(self, car_state):
        """Draw the car as a high-fidelity rectangle with heading."""
        # Convert car world coords to screen
        pts = np.array([[car_state["x"], car_state["y"]]])
        sx, sy = self.get_screen_coords(pts)[0]
        theta = car_state["theta"]

        # Real scale F1 car: Length ~5m, Width ~2m
        # Slightly exaggerated width for visibility
        car_len = 5.0 * self.scale
        car_wid = 2.5 * self.scale

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Forward/right vectors (screen space)
        fx, fy = cos_t * car_len / 2, -sin_t * car_len / 2
        rx, ry = sin_t * car_wid / 2, cos_t * car_wid / 2

        corners = [
            (sx + fx + rx, sy + fy + ry),
            (sx + fx - rx, sy + fy - ry),
            (sx - fx - rx, sy - fy - ry),
            (sx - fx + rx, sy - fy + ry),
        ]
        int_corners = [(int(c[0]), int(c[1])) for c in corners]

        # Draw chassis
        pygame.draw.polygon(self.screen, self.CAR_RED, int_corners)
        pygame.draw.polygon(self.screen, self.CAR_OUTLINE, int_corners, 2)

        # Cockpit/Halo (dark gray rectangle in middle)
        cockpit_len = car_len * 0.3
        cockpit_wid = car_wid * 0.4
        cfx, cfy = cos_t * cockpit_len / 2, -sin_t * cockpit_len / 2
        crx, cry = sin_t * cockpit_wid / 2, cos_t * cockpit_wid / 2
        cockpit = [
            (sx + cfx + crx, sy + cfy + cry),
            (sx + cfx - crx, sy + cfy - cry),
            (sx - cfx - crx, sy - cfy - cry),
            (sx - cfx + crx, sy - cfy + cry),
        ]
        pygame.draw.polygon(self.screen, (20, 20, 20), [(int(c[0]), int(c[1])) for c in cockpit])

        # Wheels (black rectangles at corners)
        wheel_len = car_len * 0.2
        wheel_wid = car_wid * 0.2
        wfx, wfy = cos_t * wheel_len / 2, -sin_t * wheel_len / 2
        wrx, wry = sin_t * wheel_wid / 2, cos_t * wheel_wid / 2

        wheel_centers = [
            # Front right
            (sx + fx * 0.7 + rx * 1.1, sy + fy * 0.7 + ry * 1.1),
            # Front left
            (sx + fx * 0.7 - rx * 1.1, sy + fy * 0.7 - ry * 1.1),
            # Rear right
            (sx - fx * 0.7 + rx * 1.1, sy - fy * 0.7 + ry * 1.1),
            # Rear left
            (sx - fx * 0.7 - rx * 1.1, sy - fy * 0.7 - ry * 1.1),
        ]

        for wc in wheel_centers:
            w_corners = [
                (wc[0] + wfx + wrx, wc[1] + wfy + wry),
                (wc[0] + wfx - wrx, wc[1] + wfy - wry),
                (wc[0] - wfx - wrx, wc[1] - wfy - wry),
                (wc[0] - wfx + wrx, wc[1] - wfy + wry),
            ]
            pygame.draw.polygon(self.screen, (10, 10, 10), [(int(c[0]), int(c[1])) for c in w_corners])


    def _draw_hud(self, car_state, action, info):
        """Draw flat HUD panel on the right side."""
        hud_x = self.width - self.hud_width
        hud_y = 0
        hud_w = self.hud_width
        hud_h = self.height

        # Full right-side semi-transparent panel
        hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surf.fill(self.HUD_BG)
        self.screen.blit(hud_surf, (hud_x, hud_y))
        
        # Border
        pygame.draw.line(self.screen, (70, 70, 70), (hud_x, 0), (hud_x, self.height), 2)

        y = 30
        x = hud_x + 20
        w = hud_w - 40

        # Title
        self.screen.blit(
            self.font_large.render("F1 TELEMETRY", True, self.HUD_TEXT), (x, y))
        y += 35

        # Separator
        pygame.draw.line(self.screen, (70, 70, 70), (x, y), (x + w, y))
        y += 15

        # Speed
        speed = car_state["velocity"]
        self.screen.blit(
            self.font.render(f"Speed:   {speed:5.1f} m/s", True, self.HUD_TEXT), (x, y))
        y += 25
        
        # Speed in km/h
        self.screen.blit(
            self.font.render(f"         {speed*3.6:5.0f} km/h", True, (180, 180, 180)), (x, y))
        y += 30

        # Heading
        heading_deg = np.degrees(car_state["theta"]) % 360
        self.screen.blit(
            self.font.render(f"Heading: {heading_deg:5.0f}°", True, self.HUD_TEXT), (x, y))
        y += 30

        # Progress
        if info and "progress" in info:
            prog = info["progress"] * 100
            self.screen.blit(
                self.font.render(f"Lap Prog:{prog:5.1f}%", True, self.HUD_TEXT), (x, y))
        y += 25

        # Lap Times
        if info and "current_lap_time" in info:
            cur_time = info["current_lap_time"]
            last_time = info["last_lap_time"]
            fast_time = info["fastest_lap_time"]
            lap_count = info["lap_count"]
            
            self.screen.blit(
                self.font.render(f"Lap Temp:{cur_time:6.2f}s", True, self.HUD_TEXT), (x, y))
            y += 20
            
            last_str = f"{last_time:6.2f}s" if last_time else "  --.--s"
            self.screen.blit(
                self.font.render(f"Last Lap:{last_str}", True, (180, 180, 180)), (x, y))
            y += 20
            
            fast_color = (255, 215, 0) if fast_time else (180, 180, 180)
            fast_str = f"{fast_time:6.2f}s" if fast_time else "  --.--s"
            self.screen.blit(
                self.font.render(f"Fastest: {fast_str} (L{lap_count})", True, fast_color), (x, y))
        y += 30

        # Status
        if info:
            on_track = info.get("on_track", True)
            status = "ON TRACK" if on_track else "OFF TRACK"
            color = self.THROTTLE_GREEN if on_track else self.BRAKE_RED
            self.screen.blit(
                self.font.render(f"Status:  {status}", True, color), (x, y))
        y += 40

        # Controls section
        if action is not None:
            pygame.draw.line(self.screen, (70, 70, 70), (x, y), (x + w, y))
            y += 15
            self.screen.blit(
                self.font_large.render("INPUTS", True, self.HUD_TEXT), (x, y))
            y += 35

            bar_w = w
            bar_h = 14

            # Steer
            self.screen.blit(
                self.font.render(f"Steer   {action[0]:+.2f}", True, self.HUD_TEXT), (x, y))
            y += 20
            pygame.draw.rect(self.screen, (40, 40, 40), (x, y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (60, 60, 60), (x, y, bar_w, bar_h), 1)
            
            mid = x + bar_w // 2
            steer_w = int(action[0] * bar_w / 2)
            if steer_w > 0:
                pygame.draw.rect(self.screen, self.BOUNDARY_WHITE,
                                 (mid, y + 2, steer_w, bar_h - 4))
            else:
                pygame.draw.rect(self.screen, self.BOUNDARY_WHITE,
                                 (mid + steer_w, y + 2, -steer_w, bar_h - 4))
            pygame.draw.line(self.screen, (150, 150, 150), (mid, y), (mid, y + bar_h))
            y += 30

            # Throttle
            self.screen.blit(
                self.font.render(f"Throttle {action[1]:.2f}", True, self.HUD_TEXT), (x, y))
            y += 20
            pygame.draw.rect(self.screen, (40, 40, 40), (x, y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (60, 60, 60), (x, y, bar_w, bar_h), 1)
            t_px = int(action[1] * bar_w)
            if t_px > 0:
                pygame.draw.rect(self.screen, self.THROTTLE_GREEN, (x, y + 2, t_px, bar_h - 4))
            y += 30

            # Brake
            self.screen.blit(
                self.font.render(f"Brake    {action[2]:.2f}", True, self.HUD_TEXT), (x, y))
            y += 20
            pygame.draw.rect(self.screen, (40, 40, 40), (x, y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (60, 60, 60), (x, y, bar_w, bar_h), 1)
            b_px = int(action[2] * bar_w)
            if b_px > 0:
                pygame.draw.rect(self.screen, self.BRAKE_RED, (x, y + 2, b_px, bar_h - 4))

    def _draw_minimap(self, car_state):
        """Draw a full-track minimap with a car indicator."""
        mm_size = 240
        mm_x = self.width - self.hud_width + 20
        mm_y = self.height - mm_size - 30

        # Background box
        pygame.draw.rect(self.screen, (25, 25, 25), (mm_x, mm_y, mm_size, mm_size))
        pygame.draw.rect(self.screen, (70, 70, 70), (mm_x, mm_y, mm_size, mm_size), 1)

        # Draw track boundaries
        r_pts = self.get_minimap_coords(self.track.boundary_right, mm_x - 20, mm_y - 240 + mm_size)
        l_pts = self.get_minimap_coords(self.track.boundary_left, mm_x - 20, mm_y - 240 + mm_size)
        
        pygame.draw.polygon(self.screen, self.ROAD_GRAY, list(r_pts) + list(l_pts)[::-1])
        pygame.draw.lines(self.screen, self.BOUNDARY_WHITE, True, r_pts, 1)
        pygame.draw.lines(self.screen, self.BOUNDARY_WHITE, True, l_pts, 1)

        # Draw car dot
        car_pt = np.array([[car_state["x"], car_state["y"]]])
        c_mm = self.get_minimap_coords(car_pt, mm_x - 20, mm_y - 240 + mm_size)[0]
        pygame.draw.circle(self.screen, self.CAR_RED, c_mm, 4)

    def _draw_raster_preview(self, raster):
        """Draw the 64x64 ego-centric observation as a scaled preview in the bottom left."""
        preview_size = 180
        preview_x = 20
        preview_y = self.height - preview_size - 20 

        # Label
        self.screen.blit(
            self.font.render("EGO SENSOR", True, self.HUD_TEXT),
            (preview_x, preview_y - 25))

        # Convert 3-channel (C, H, W) raster to RGB
        h, w = raster.shape[1], raster.shape[2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Background (off-track) = dark green, drivable = gray, walls = white, ego = red
        rgb[:, :, 0] = (raster[0] * 110 + raster[1] * 230 + raster[2] * 220).clip(0, 255).astype(np.uint8)
        rgb[:, :, 1] = (raster[0] * 110 + raster[1] * 230 + raster[2] * 30).clip(0, 255).astype(np.uint8)
        rgb[:, :, 2] = (raster[0] * 110 + raster[1] * 230 + raster[2] * 30).clip(0, 255).astype(np.uint8)

        # Pygame uses (W, H, C) surfarray
        surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (preview_size, preview_size))

        pygame.draw.rect(self.screen, (100, 100, 100),
                         (preview_x - 3, preview_y - 3,
                          preview_size + 6, preview_size + 6), 3)
        self.screen.blit(surf, (preview_x, preview_y))

    def _draw_lidar_rays(self, car_state, obs):
        """Cast mathematical Lidar beams from the car dynamically onto the screen."""
        if obs is None or "aux" not in obs or len(obs["aux"]) < 16:
            return
            
        x, y, theta = car_state["x"], car_state["y"], car_state["theta"]
        
        # Rays are index 1 to 15 in the aux vector (0 is speed)
        ray_dists = obs["aux"][1:16]
        max_range = 150.0
        fov = np.pi
        
        angles = np.linspace(-fov/2, fov/2, 15) + theta
        
        # Screen origin
        s_cx, s_cy = self.get_screen_coords(np.array([[x, y]]))[0]
        
        # Draw translucent laser lines to the hit point
        for i, angle in enumerate(angles):
            dist = ray_dists[i] * max_range
            end_x = x + dist * np.cos(angle)
            end_y = y + dist * np.sin(angle)
            
            e_x, e_y = self.get_screen_coords(np.array([[end_x, end_y]]))[0]
            pygame.draw.line(self.screen, (255, 120, 0), (s_cx, s_cy), (e_x, e_y), 1)

    def close(self):
        pygame.quit()
