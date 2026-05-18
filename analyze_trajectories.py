import numpy as np
from scipy.optimize import curve_fit
import json

TRAJECTORY_PATH = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/trajectories.txt"
MAX_JUMP_SECONDS = 0.5833
MAX_JUMP_FRAMES = round(MAX_JUMP_SECONDS * 60)  # 35

def parse_trajectories(filepath):
    flat_jumps = []
    wall_bounces = []
    in_wall_section = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Start session"):
                continue
            if line == "WALL BOUNCES":
                in_wall_section = True
                continue
            try:
                header, frames_str = line.split(";")
                parts = header.split(",")
                start_x = float(parts[0])
                start_y = float(parts[1])
                end_x = float(parts[2])
                end_y = float(parts[3])
                space_frames = int(parts[4])
                space_seconds = float(parts[5])

                positions = []
                for f in frames_str.split("|"):
                    fx, fy = map(float, f.split(","))
                    positions.append((fx, fy))

                positions.append((end_x, end_y))

                record = {
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "space_frames": space_frames,
                    "space_seconds": space_seconds,
                    "positions": positions
                }

                if in_wall_section:
                    wall_bounces.append(record)
                else:
                    flat_jumps.append(record)
            except Exception as e:
                print(f"Skipping line: {e}")
                continue

    return flat_jumps, wall_bounces


def get_first_half(positions, start_y):
    apex_idx = max(range(len(positions)), key=lambda i: positions[i][1])
    return positions[:apex_idx + 1]


def parabola(x, a, b, c):
    return a * x**2 + b * x + c


def fit_parabola_from_half(start_x, start_y, half_positions):
    xs = [start_x] + [p[0] for p in half_positions]
    ys = [start_y] + [p[1] for p in half_positions]
    xs = np.array(xs)
    ys = np.array(ys)
    try:
        popt, _ = curve_fit(parabola, xs, ys)
        a, b, c = popt
        apex_x = -b / (2 * a)
        apex_y = parabola(apex_x, a, b, c)
        half_width = apex_x - start_x
        end_x_predicted = apex_x + half_width
        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "apex_x": float(apex_x),
            "apex_y": float(apex_y),
            "end_x_predicted": float(end_x_predicted),
        }
    except Exception as e:
        return None


def analyze_flat_jumps(flat_jumps):
    print("=" * 60)
    print("FLAT GROUND JUMP ANALYSIS")
    print("=" * 60)

    by_frames = {}
    for j in flat_jumps:
        sf = j["space_frames"]
        if sf not in by_frames:
            by_frames[sf] = []
        by_frames[sf].append(j)

    results = {}
    for sf in sorted(by_frames.keys()):
        jumps = by_frames[sf]
        fits = []
        horizontal_distances = []
        apex_heights = []
        air_frames_list = []

        for j in jumps:
            half = get_first_half(j["positions"], j["start_y"])
            fit = fit_parabola_from_half(j["start_x"], j["start_y"], half)
            if fit:
                fits.append(fit)
                dist = 2 * (fit["apex_x"] - j["start_x"])
                horizontal_distances.append(dist)
                apex_heights.append(fit["apex_y"] - j["start_y"])
                air_frames_list.append(len(j["positions"]))

        if not fits:
            continue

        avg_a = np.mean([f["a"] for f in fits])
        avg_b = np.mean([f["b"] for f in fits])
        avg_c = np.mean([f["c"] for f in fits])
        avg_dist = np.mean(horizontal_distances)
        avg_apex = np.mean(apex_heights)
        avg_air_frames = np.mean(air_frames_list)
        seconds = jumps[0]["space_seconds"]

        # derive physics constants
        half_air = avg_air_frames / 2
        g = 2 * avg_apex / (half_air ** 2)  # px/frame^2
        vy_initial = g * half_air             # px/frame
        vx = avg_dist / avg_air_frames        # px/frame

        print(f"\nspace_frames={sf} ({seconds:.4f}s) | n={len(jumps)}")
        print(f"  avg horizontal distance: {avg_dist:.2f}px")
        print(f"  avg apex height gain:    {avg_apex:.2f}px")
        print(f"  avg air frames:          {avg_air_frames:.1f}")
        print(f"  derived g:               {g:.4f} px/frame^2")
        print(f"  derived vy_initial:      {vy_initial:.4f} px/frame")
        print(f"  derived vx:              {vx:.4f} px/frame")

        results[sf] = {
            "space_seconds": seconds,
            "avg_horizontal_distance": avg_dist,
            "avg_apex_height_gain": avg_apex,
            "avg_air_frames": avg_air_frames,
            "parabola_a": avg_a,
            "parabola_b": avg_b,
            "parabola_c": avg_c,
            "g": g,
            "vy_initial": vy_initial,
            "vx": vx,
            "n_samples": len(jumps)
        }

    return results


def analyze_wall_bounces(wall_bounces):
    print("\n" + "=" * 60)
    print("WALL BOUNCE ANALYSIS")
    print("=" * 60)

    for i, j in enumerate(wall_bounces):
        positions = j["positions"]
        if len(positions) < 4:
            continue

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        wall_hit_idx = None
        for k in range(1, len(xs) - 1):
            dx_before = xs[k] - xs[k-1]
            dx_after = xs[k+1] - xs[k]
            if dx_before * dx_after < 0:
                wall_hit_idx = k
                break
            if xs[k] == xs[k-1] and k > 2:
                wall_hit_idx = k
                break

        ceiling_hit_idx = None
        for k in range(1, len(ys) - 1):
            if ys[k] == ys[k-1] and k > 2:
                ceiling_hit_idx = k
                break

        sf = j["space_frames"]
        seconds = j["space_seconds"]
        print(f"\nWall bounce {i+1}: space_frames={sf} ({seconds:.4f}s)")
        print(f"  start: ({j['start_x']:.2f}, {j['start_y']:.2f})")
        print(f"  end:   ({j['end_x']:.2f}, {j['end_y']:.2f})")
        print(f"  total frames: {len(positions)}")

        if wall_hit_idx is not None:
            wx, wy = positions[wall_hit_idx]
            print(f"  wall hit at frame {wall_hit_idx}: ({wx:.2f}, {wy:.2f})")
            post_bounce = positions[wall_hit_idx:]
            if len(post_bounce) > 3:
                post_xs = np.array([p[0] for p in post_bounce])
                frames = np.arange(len(post_xs))
                if len(frames) > 1:
                    x_vel = np.polyfit(frames, post_xs, 1)[0]
                    print(f"  post-bounce x velocity: {x_vel:.2f}px/frame")

        if ceiling_hit_idx is not None:
            cx, cy = positions[ceiling_hit_idx]
            print(f"  ceiling hit at frame {ceiling_hit_idx}: ({cx:.2f}, {cy:.2f})")

def extract_jump_offsets(flat_jumps, freefall_frames=60):
    """Extracts per-frame offsets for each jump duration.
    Always uses ascending portion only (up to apex), mirrors for descent.
    
    freefall_frames: additional frames appended at terminal velocity after arc
    """
    by_frames = {}
    for j in flat_jumps:
        sf = j["space_frames"]
        if sf not in by_frames:
            by_frames[sf] = []
        by_frames[sf].append(j)

    results = {}
    for sf in sorted(by_frames.keys()):
        if sf > 36:
            continue

        jumps = by_frames[sf]
        all_dx = []
        all_dy_sequences = []

        for j in jumps:
            positions = j["positions"]
            if len(positions) < 3:
                continue

            prev_x, prev_y = j["start_x"], j["start_y"]
            dx_vals = []
            dy_vals = []

            for px, py in positions:
                dx_vals.append(px - prev_x)
                dy_vals.append(py - prev_y)
                prev_x, prev_y = px, py

            # use frame 2->3 delta for dx (avoids zero first frame)
            empirical_dx = abs(dx_vals[2])
            all_dx.append(empirical_dx)

            # skip first frame (always 0.0), find apex from frame 1 onwards
            apex_idx = len(dy_vals)
            for i, dy in enumerate(dy_vals[1:], start=1):
                if dy <= 0:
                    apex_idx = i
                    break

            # only keep ascending portion, skip first zero frame
            first_half = dy_vals[1:apex_idx]
            if not first_half:
                continue

            all_dy_sequences.append(first_half)

        if not all_dx or not all_dy_sequences:
            continue

        avg_dx = float(np.mean(all_dx))

        # pad sequences to same length and average
        max_len = max(len(s) for s in all_dy_sequences)
        padded = []
        for seq in all_dy_sequences:
            if len(seq) < max_len:
                seq = seq + [seq[-1]] * (max_len - len(seq))
            padded.append(seq)

        first_half_dy = [float(np.mean([s[i] for s in padded])) for i in range(max_len)]

        # mirror for second half
        second_half_dy = [-v for v in reversed(first_half_dy)]

        # terminal velocity for freefall
        terminal_dy = second_half_dy[-1]
        freefall_dy = [terminal_dy] * freefall_frames

        full_dy = first_half_dy + second_half_dy + freefall_dy
        arc_frames = len(first_half_dy) + len(second_half_dy)
        full_dx = [avg_dx] * arc_frames + [0.0] * freefall_frames

        seconds = jumps[0]["space_seconds"]
        print(f"space_frames={sf} ({seconds:.4f}s) | n={len(jumps)} | "
              f"dx={avg_dx:.3f} | arc_frames={arc_frames} | "
              f"first_half_len={len(first_half_dy)}")

        results[sf] = {
            "space_seconds": seconds,
            "dx": avg_dx,
            "full_dx": full_dx,
            "full_dy": full_dy,
            "arc_frames": arc_frames,
            "terminal_dy": terminal_dy,
            "n_samples": len(jumps)
        }

    return results

def main():
    flat_jumps, wall_bounces = parse_trajectories(TRAJECTORY_PATH)
    print(f"Loaded {len(flat_jumps)} flat jumps, {len(wall_bounces)} wall bounce jumps")

    flat_results = analyze_flat_jumps(flat_jumps)
    analyze_wall_bounces(wall_bounces)

    print("\n" + "=" * 60)
    print("JUMP OFFSET EXTRACTION")
    print("=" * 60)
    offset_results = extract_jump_offsets(flat_jumps)
    offset_output = {str(k): v for k, v in offset_results.items()}
    with open("jump_offsets.json", "w") as f:
        json.dump(offset_output, f, indent=2)
    print(f"Jump offsets saved to jump_offsets.json")

    output = {str(k): v for k, v in flat_results.items()}
    with open("jump_models.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Jump models saved to jump_models.json")

if __name__ == "__main__":
    main()

# def main():
#     flat_jumps, wall_bounces = parse_trajectories(TRAJECTORY_PATH)
#     print(f"Loaded {len(flat_jumps)} flat jumps, {len(wall_bounces)} wall bounce jumps")

#     flat_results = analyze_flat_jumps(flat_jumps)
#     analyze_wall_bounces(wall_bounces)

#     output = {str(k): v for k, v in flat_results.items()}
#     with open("jump_models.json", "w") as f:
#         json.dump(output, f, indent=2)
#     print(f"\nJump models saved to jump_models.json")

# if __name__ == "__main__":
#     main()