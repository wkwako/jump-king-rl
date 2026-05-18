import pydirectinput
pydirectinput.PAUSE=0.01
import time
import json
import numpy as np
from scipy.optimize import curve_fit
import os

from PlatformParser import PlatformParser

class Planning:
    def __init__(self):
        self.graph = {}
        self.platforms = {}
        self.slopes = {}

        self.MAX_JUMP_SECONDS = 0.5833
        self.MAX_JUMP_FRAMES = 35
        self.WALL_REBOUND_VX = 1.75
        self.SCREEN_LEFT = 0
        self.SCREEN_RIGHT = 464
        self.PLATFORM_LAND_TOLERANCE = 8
        self.jump_offsets = self.load_jump_offsets()

        self.jump_models = self.load_jump_models()
        self.parser = PlatformParser()

        jump_curves_path = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/jump_curves.json"
        if os.path.exists(jump_curves_path):
            self.load_jump_curves(jump_curves_path)
        else:
            self.build_jump_curves_analytical()

        with open("C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/full_registry_clean.txt") as f:
            self.platform_data = json.load(f)

        # build slope segments for screens that have beneficial slopes
        slope_data = self.parser.load_slope_data(
            "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/slopedata.txt"
        )
        self.slope_segments_by_screen = {}
        for screen in [37, 38]:
            tiles = slope_data.get(screen, [])
            platform_list = self.platform_data.get(str(screen), [])
            segments = self.parser.build_slope_segments(tiles, platform_list)
            self.slope_segments_by_screen[screen] = [
                s for s in segments if s["landing_platform"] is not None
            ]

    def load_jump_models(self, path="C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/jump_models.json"):
        with open(path, 'r') as f:
            return json.load(f)
        
    def load_jump_offsets(self, path="C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/jump_offsets.json"):
        with open(path, 'r') as f:
            return json.load(f)
        
    def get_model(self, duration_seconds):
        """Returns jump model for duration, capped at max jump."""
        duration_seconds = min(duration_seconds, self.MAX_JUMP_SECONDS)
        frames = round(duration_seconds * 60)
        frames = max(1, min(frames, self.MAX_JUMP_FRAMES))
        available = [int(k) for k in self.jump_models.keys()]
        closest = min(available, key=lambda k: abs(k - frames))
        return self.jump_models[str(closest)]

    def jump(self, units):
        
        #grace period to alt tab into game
        time.sleep(1)

        if units >= 1:
            s = units/60

        else:
            s = units
        #convert frames to seconds
        
        
        #press buttons
        pydirectinput.keyDown("space")
        pydirectinput.keyDown("right")

        #duration buttons are held
        time.sleep(s)

        #release buttons
        pydirectinput.keyUp("space")
        time.sleep(0.03)
        pydirectinput.keyUp("right")

    def parabola(self, x, a, b, c):
        return a * x**2 + b * x + c

    def fit_parabola_relative(self, dx, full_dy, arc_frames):
        """Fits parabola in relative coordinates from start position.
        Returns (a, b, c) coefficients, or (None, None, None) on failure.
        """
        cum_x = 0.0
        cum_y = 0.0
        rel_xs = []
        rel_ys = []
    
        for i in range(arc_frames):
            cum_x += dx
            cum_y += full_dy[i]
            rel_xs.append(cum_x)
            rel_ys.append(cum_y)
    
        rel_xs = np.array(rel_xs)
        rel_ys = np.array(rel_ys)

        try:
            popt, _ = curve_fit(self.parabola, rel_xs, rel_ys)
            a, b, c = popt
            return float(a), float(b), float(c)
        except:
            return None, None, None
        
    def load_jump_curves(self, path="C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/jump_curves.json"):
        with open(path) as f:
            data = json.load(f)
        self.jump_curves = {int(k): v for k, v in data.items()}
        print(f"Loaded {len(self.jump_curves)} jump curves")

    def build_jump_curves_analytical(self):
        """Builds jump curves analytically using known physics constants.
        g = 0.25 px/frame², dx = 3.5 px/frame
        vy_initial varies by jump strength.
        """
        G = 0.25  # px/frame²
        DX = 3.5  # px/frame

        # vy_initial per jump strength — derived from apex height
        # apex_height = vy_initial² / (2*g)
        # vy_initial = sqrt(2*g*apex_height)
        # we get apex_height from trajectories data

        with open("C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/jump_offsets.json") as f:
            jump_offsets = json.load(f)

        curves = {}
        for sf, model in jump_offsets.items():
            sf = int(sf)
            if sf > 36:
                continue

            # get apex height from first half dy sequence
            full_dy = model["full_dy"]
            arc_frames = model["arc_frames"]
            half_frames = arc_frames // 2

            # apex height = sum of ascending dy values
            apex_height = sum(full_dy[:half_frames])

            # derive vy_initial analytically
            vy_initial = np.sqrt(2 * G * apex_height) if apex_height > 0 else 0

            if sf == 6:
                print (f"sf 6 vy_initial: {vy_initial}")

            # half arc frames = vy_initial / g
            half_arc_frames = vy_initial / G

            # apex x = dx * half_arc_frames
            apex_x = DX * half_arc_frames

            # build parabola in relative coords
            # y(x) = vy_initial * (x/dx) - 0.5 * g * (x/dx)²
            # converting x to frame: t = x/dx
            # y = vy_initial * t - 0.5 * g * t²
            # substituting t = x/dx:
            # y = (vy_initial/dx) * x - (g/(2*dx²)) * x²
            # so: a = -g/(2*dx²), b = vy_initial/dx, c = 0
            a = -G / (2 * DX**2)
            b = vy_initial / DX
            c = 0.0

            max_x = DX * arc_frames

            curves[sf] = {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "apex_x": float(apex_x),
                "max_x": float(max_x),
                "dx": DX,
                "vy_initial": float(vy_initial),
                "space_seconds": model["space_seconds"]
            }

            print(f"sf={sf} | vy_initial={vy_initial:.4f} | apex_x={apex_x:.2f} | "
                f"a={a:.6f} b={b:.4f} c={c:.4f}")

        self.jump_curves = curves

        output_path = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/jump_curves.json"
        with open(output_path, "w") as f:
            json.dump({str(k): v for k, v in curves.items()}, f, indent=2)
        print(f"Jump curves saved to {output_path}")

        return curves

    def find_jump(self, start_x, start_y, end_x, end_y, threshold=5.0):
        rel_x = abs(end_x - start_x)
        rel_y = end_y - start_y
        direction = "right" if end_x > start_x else "left"
        print(f"find_jump: rel_x={rel_x:.2f} rel_y={rel_y:.2f} direction={direction}")

        G = 0.26
        TERMINAL_VY = -10.0
        best_match = None
        best_error = float('inf')

        for sf, curve in self.jump_curves.items():
            if int(sf) != 6:
                continue
            
            dx = curve["dx"]
            vy_initial = curve["vy_initial"]
            
            cum_x = 0.0
            cum_y = 0.0
            vy = vy_initial
            
            print(f"sf=6 vy_initial={vy_initial:.4f} dx={dx:.4f}")
            for frame in range(50):
                cum_x += dx
                vy = max(vy - 0.26, -10.0)
                cum_y += vy
                if frame < 20 or abs(cum_x - 154) < 5:
                    print(f"  frame {frame+1}: cum_x={cum_x:.2f} cum_y={cum_y:.2f} vy={vy:.4f}")

                # only check when descending
                if vy >= 0:
                    continue

                # check if we've reached target x
                if abs(cum_x - rel_x) <= dx:
                    error = abs(cum_y - rel_y)
                    if error < best_error:
                        best_error = error
                        best_match = (direction, curve["space_seconds"], sf, error)
                    break

                # stop if we've gone too far
                if cum_x > rel_x + dx:
                    break

        if best_match and best_match[3] <= threshold:
            direction, seconds, sf, error = best_match
            print(f"Found jump: direction={direction} duration={seconds:.4f}s "
                f"space_frames={sf} error={error:.2f}px")
            return (direction, seconds)

        print(f"No jump found. Best error: {best_error:.2f}px")
        return None

    def slope_to_platform(self):
        #returns a platform if character would land on slope that carries them to a platform; returns False otherwise
        pass

    def BFS(self):
        #outputs platform id pairs, location to perform jump, and actions
        #(1, 2): (x, y, action1)
        #(2, 4): (x, y, action4)
        #(4, 5): (x, y, action2)
        pass

    def create_graph(self):
        #first pass: find all platform id pairs and actions between them
        #1, 3 --> (x, y, action1)
        #1, 3 --> (x, y, action2)
        #1, 3 --> (x, y, action3)
        
        #second pass: clean graph, select best key value pairs. platform id pairs are unique
        pass

    def execute_plan(self):
        pass

    def move_to_location(self, location):
        #walks to location
        x,y = location
        pass

    def find_jump(self, start_x, start_y, end_x, end_y, threshold=5.0):
        rel_x = abs(end_x - start_x)
        rel_y = end_y - start_y
        direction = "right" if end_x > start_x else "left"
        print(f"find_jump: rel_x={rel_x:.2f} rel_y={rel_y:.2f} direction={direction}")

        best_match = None
        best_error = float('inf')

        for sf, curve in self.jump_curves.items():
            a = curve["a"]
            b = curve["b"]
            c = curve["c"]
            apex_x = curve["apex_x"]

            # slope at target x must be negative (descending)
            slope = 2 * a * rel_x + b
            if slope >= 0:
                continue

            # evaluate parabola at target x — no bounds on how far we extend
            predicted_y = a * rel_x**2 + b * rel_x + c
            error = abs(predicted_y - rel_y)

            if sf == 6:
                print(f"sf=6 debug: a={a:.6f} b={b:.4f} apex_x={apex_x:.2f}")
                print(f"  rel_x={rel_x:.2f} slope={slope:.4f} predicted_y={predicted_y:.2f}")

            #print(f"  sf={sf} predicted_y={predicted_y:.2f} rel_y={rel_y:.2f} error={error:.2f}")

            if error < best_error:
                best_error = error
                best_match = (direction, curve["space_seconds"], sf, error)

        if best_match and best_match[3] <= threshold:
            direction, seconds, sf, error = best_match
            print(f"Found jump: direction={direction} duration={seconds:.4f}s "
                f"space_frames={sf} error={error:.2f}px")
            return (direction, seconds)

        print(f"No jump found. Best error: {best_error:.2f}px "
            f"(threshold={threshold}px)")
        return None
    
#plan = Planning()
#plan.jump(0.1)

#result = plan.find_jump(start_x=270.5, start_y=-14, end_x=421, end_y=-158)
#print(f"Result: {result}")

if __name__ == "__main__":
    plan = Planning()
    
    # debug sf=6 frame by frame
    curve = plan.jump_curves[6]
    dx = curve["dx"]
    vy_initial = curve["vy_initial"]
    
    cum_x = 0.0
    cum_y = 0.0
    vy = vy_initial
    
    print(f"sf=6 vy_initial={vy_initial:.4f} dx={dx:.4f}")
    for frame in range(50):
        cum_x += dx
        vy = max(vy - 0.26, -10.0)
        cum_y += vy
        print(f"frame {frame+1}: cum_x={cum_x:.2f} cum_y={cum_y:.2f} vy={vy:.4f}")
    
    result = plan.find_jump(
        start_x=424.5, start_y=-158,
        end_x=270.5, end_y=-14
    )
    print(f"Result: {result}")