import argparse
import sys
import os
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.default import Config
from configs.default import Config
from env.f1_env import F1Env
from data.controllers import RandomPolicy, ScriptedPolicy, NoisyScriptedPolicy, AdvancedScriptedPolicy
from viz.renderer import Visualizer

class RaceVisualizer(Visualizer):
    def render_race(self, car1_state, car2_state, p1_name, p2_name, info1, info2, done1=False, done2=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        # Dynamic Camera Framing Logic
        if done1 and not done2:
            target_x, target_y = car2_state["x"], car2_state["y"]
            target_scale = 8.0
        elif done2 and not done1:
            target_x, target_y = car1_state["x"], car1_state["y"]
            target_scale = 8.0
        else:
            # Both alive (or both dead): Track midpoint
            target_x = (car1_state["x"] + car2_state["x"]) / 2.0
            target_y = (car1_state["y"] + car2_state["y"]) / 2.0
            
            # Calculate distance spread
            dx = max(abs(car1_state["x"] - car2_state["x"]), 5.0)
            dy = max(abs(car1_state["y"] - car2_state["y"]), 5.0)
            
            # We want the cars to occupy at most 70% of the screen width/height to leave padding
            scale_x = (self.view_width * 0.7) / dx
            scale_y = (self.view_height * 0.7) / dy
            
            # Pick the most restrictive scale, bound it between 1.0 (super far) and 8.0 (default close)
            target_scale = min(max(min(scale_x, scale_y), 1.0), 8.0)
        
        # Smooth follow interpolation
        self.cam_x += (target_x - self.cam_x) * 0.15
        self.cam_y += (target_y - self.cam_y) * 0.15
        self.scale += (target_scale - self.scale) * 0.10

        self._draw_background_grid()
        self._draw_track()

        # Draw Player 2 (Blue)
        self.CAR_RED = (30, 100, 220)
        self._draw_car(car2_state)
        
        # Draw Player 1 (Red)
        self.CAR_RED = (220, 30, 30)
        self._draw_car(car1_state)

        # Draw Minimap with both cars
        mm_size = 240
        mm_x = self.width - self.hud_width + 20
        mm_y = self.height - mm_size - 30
        pygame.draw.rect(self.screen, (25, 25, 25), (mm_x, mm_y, mm_size, mm_size))
        pygame.draw.rect(self.screen, (70, 70, 70), (mm_x, mm_y, mm_size, mm_size), 1)
        r_pts = self.get_minimap_coords(self.track.boundary_right, mm_x - 20, mm_y - 240 + mm_size)
        l_pts = self.get_minimap_coords(self.track.boundary_left, mm_x - 20, mm_y - 240 + mm_size)
        pygame.draw.polygon(self.screen, self.ROAD_GRAY, list(r_pts) + list(l_pts)[::-1])
        pygame.draw.lines(self.screen, self.BOUNDARY_WHITE, True, r_pts, 1)
        pygame.draw.lines(self.screen, self.BOUNDARY_WHITE, True, l_pts, 1)

        c1_mm = self.get_minimap_coords(np.array([[car1_state["x"], car1_state["y"]]]), mm_x - 20, mm_y - 240 + mm_size)[0]
        c2_mm = self.get_minimap_coords(np.array([[car2_state["x"], car2_state["y"]]]), mm_x - 20, mm_y - 240 + mm_size)[0]
        pygame.draw.circle(self.screen, (30, 100, 220), c2_mm, 4)
        pygame.draw.circle(self.screen, (220, 30, 30), c1_mm, 4)

        # Draw HUD for Race
        self.draw_race_hud(car1_state, car2_state, p1_name, p2_name, info1, info2)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def draw_race_hud(self, c1, c2, p1_name, p2_name, i1, i2):
        hud_x = self.width - self.hud_width
        hud_surf = pygame.Surface((self.hud_width, self.height), pygame.SRCALPHA)
        hud_surf.fill(self.HUD_BG)
        self.screen.blit(hud_surf, (hud_x, 0))
        pygame.draw.line(self.screen, (70, 70, 70), (hud_x, 0), (hud_x, self.height), 2)

        x, y = hud_x + 20, 30
        self.screen.blit(self.font_large.render("1v1 GHOST RACE", True, self.HUD_TEXT), (x, y))
        y += 35
        
        # P1 (Red)
        self.screen.blit(self.font_large.render(f"P1: {p1_name.upper()}", True, (220, 30, 30)), (x, y))
        y += 20
        self.screen.blit(self.font.render(f"Speed: {c1['velocity']:5.1f} m/s", True, self.HUD_TEXT), (x, y))
        y += 20
        prog1 = i1.get('progress', 0) * 100 if i1 else 0
        self.screen.blit(self.font.render(f"Prog:  {prog1:5.1f}%", True, self.HUD_TEXT), (x, y))
        y += 40

        # P2 (Blue)
        self.screen.blit(self.font_large.render(f"P2: {p2_name.upper()}", True, (50, 150, 255)), (x, y))
        y += 20
        self.screen.blit(self.font.render(f"Speed: {c2['velocity']:5.1f} m/s", True, self.HUD_TEXT), (x, y))
        y += 20
        prog2 = i2.get('progress', 0) * 100 if i2 else 0
        self.screen.blit(self.font.render(f"Prog:  {prog2:5.1f}%", True, self.HUD_TEXT), (x, y))

def load_policy(name, env):
    if name == "planner":
        import torch
        from models.world_model import WorldModel
        from planner.mpc import CEMPlanner
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = WorldModel().to(device)
        model.load_state_dict(torch.load("checkpoints/world_model_v1.pth", map_location=device))
        
        # Setting horizon down slightly strictly for visualizer frame-rate since it's running 2 entire 400-sequence mental rollouts simultaneously!
        return CEMPlanner(model, num_candidates=400, horizon=25, iterations=4, device=device), False
    elif name == "advanced":
        return AdvancedScriptedPolicy(env.track, lookahead=15), True
    elif name == "scripted":
        return ScriptedPolicy(env.track, lookahead=5), True
    elif name == "noisy":
        return NoisyScriptedPolicy(env.track, lookahead=5), True
    elif name == "random":
        return RandomPolicy(), False
    else:
        raise ValueError("Unknown policy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=str, default="planner")
    parser.add_argument("--p2", type=str, default="scripted")
    parser.add_argument("--track", type=str, default=None)
    args = parser.parse_args()

    import glob
    config = Config()
    if args.track:
        config.track_csv = args.track
    else:
        # Track Picker
        tracks = sorted(glob.glob("tracks/*.csv"))
        if not tracks:
            print("No tracks found in tracks/ directory!")
            sys.exit(1)
            
        print("\n--- SELECT TRACK ---")
        for i, t in enumerate(tracks):
            name = os.path.basename(t).replace('.csv', '')
            print(f"[{i+1}] {name}")
            
        while True:
            try:
                choice = input(f"Enter track number (1-{len(tracks)}) [1]: ").strip()
                if choice == "":
                    idx = 0
                    break
                idx = int(choice) - 1
                if 0 <= idx < len(tracks):
                    break
                print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")
                
        config.track_csv = tracks[idx]
    
    # We instantiate TWO physically isolated universes so they can Ghost Race without collision physics bugs!
    env1 = F1Env.from_config(config)
    env2 = F1Env.from_config(config)
    
    pol1, c1_needs_state = load_policy(args.p1, env1)
    pol2, c2_needs_state = load_policy(args.p2, env2)
    
    viz = RaceVisualizer(env1.track)
    
    obs1 = env1.reset()
    obs2 = env2.reset()
    
    done1, done2 = False, False
    i1, i2 = {}, {}
    
    print(f"Starting 1v1 Ghost Race on {args.track}...")
    print(f"Player 1 (RED): {args.p1.upper()}")
    print(f"Player 2 (BLUE): {args.p2.upper()}")

    while True:
        if not done1:
            a1 = pol1(obs1, car_state=env1.get_car_state()) if c1_needs_state else pol1(obs1)
            obs1, r1, done1, i1 = env1.step(a1)
        if not done2:
            a2 = pol2(obs2, car_state=env2.get_car_state()) if c2_needs_state else pol2(obs2)
            obs2, r2, done2, i2 = env2.step(a2)
            
        if not viz.render_race(env1.get_car_state(), env2.get_car_state(), args.p1, args.p2, i1, i2, done1, done2):
            break
            
        if done1 and done2:
            print("Race Finished!")
            break

if __name__ == "__main__":
    main()
