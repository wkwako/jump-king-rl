import numpy as np
from scipy.optimize import curve_fit
import json

TRAJECTORY_PATH = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/trajectories.txt"

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
    """Returns only frames up to and including the apex."""
    apex_idx = max(range(len(positions)), key=lambda i: positions[i][1])
    return positions[:apex_idx + 1]


def parabola(x, a, b, c):
    return a * x**2 + b * x + c


def fit_parabola_from_half(start_x, start_y, half_positions):
    """Fits a parabola to first half of trajectory, mirrors for second half."""
    xs = [start_x] + [p[0] for p in half_positions]
    ys = [start_y] + [p[1] for p in half_positions]

    xs = np.array(xs)
    ys = np.array(ys)

    try:
        popt, _ = curve_fit(parabola, xs, ys)
        a, b, c = popt

        # find apex x from derivative: 2ax + b = 0 -> x = -b/(2a)
        apex_x = -b / (2 * a)
        apex_y = parabola(apex_x, a, b, c)

        # mirror: second half is symmetric about apex
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

    # group by space_frames
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

        for j in jumps:
            half = get_first_half(j["positions"], j["start_y"])
            fit = fit_parabola_from_half(j["start_x"], j["start_y"], half)
            if fit:
                fits.append(fit)
                # horizontal distance is symmetric: 2 * (apex_x - start_x)
                dist = 2 * (fit["apex_x"] - j["start_x"])
                horizontal_distances.append(dist)
                # apex height relative to start
                apex_heights.append(fit["apex_y"] - j["start_y"])

        if not fits:
            continue

        avg_a = np.mean([f["a"] for f in fits])
        avg_b = np.mean([f["b"] for f in fits])
        avg_c = np.mean([f["c"] for f in fits])
        avg_dist = np.mean(horizontal_distances)
        avg_apex = np.mean(apex_heights)
        seconds = jumps[0]["space_seconds"]

        print(f"\nspace_frames={sf} ({seconds:.4f}s) | n={len(jumps)}")
        print(f"  avg horizontal distance: {avg_dist:.2f}px")
        print(f"  avg apex height gain:    {avg_apex:.2f}px")
        print(f"  parabola coefficients:   a={avg_a:.6f}, b={avg_b:.4f}, c={avg_c:.4f}")

        results[sf] = {
            "space_seconds": seconds,
            "avg_horizontal_distance": avg_dist,
            "avg_apex_height_gain": avg_apex,
            "parabola_a": avg_a,
            "parabola_b": avg_b,
            "parabola_c": avg_c,
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

        # detect wall hit: x stops changing or reverses direction
        # look for the frame where x velocity flips sign
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # find x reversal point
        wall_hit_idx = None
        for k in range(1, len(xs) - 1):
            dx_before = xs[k] - xs[k-1]
            dx_after = xs[k+1] - xs[k]
            if dx_before * dx_after < 0:  # sign flip
                wall_hit_idx = k
                break
            # also check for x clamping (hitting wall = x stuck at same value)
            if xs[k] == xs[k-1] and k > 2:
                wall_hit_idx = k
                break

        # detect ceiling hit: y stops increasing and x reversal isn't the cause
        ceiling_hit_idx = None
        for k in range(1, len(ys) - 1):
            dy_before = ys[k] - ys[k-1]
            dy_after = ys[k+1] - ys[k]
            if dy_before > 0 and dy_after < 0:
                # this is just the apex, not a ceiling hit
                pass
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

            # analyze post-bounce trajectory
            post_bounce = positions[wall_hit_idx:]
            if len(post_bounce) > 3:
                post_xs = np.array([p[0] for p in post_bounce])
                post_ys = np.array([p[1] for p in post_bounce])
                # fit line to post-bounce x motion to get horizontal velocity
                frames = np.arange(len(post_xs))
                if len(frames) > 1:
                    x_vel = np.polyfit(frames, post_xs, 1)[0]
                    print(f"  post-bounce x velocity: {x_vel:.2f}px/frame")

        if ceiling_hit_idx is not None:
            cx, cy = positions[ceiling_hit_idx]
            print(f"  ceiling hit at frame {ceiling_hit_idx}: ({cx:.2f}, {cy:.2f})")


def main():
    flat_jumps, wall_bounces = parse_trajectories(TRAJECTORY_PATH)

    print(f"Loaded {len(flat_jumps)} flat jumps, {len(wall_bounces)} wall bounce jumps")

    flat_results = analyze_flat_jumps(flat_jumps)
    analyze_wall_bounces(wall_bounces)

    # save flat jump results as JSON for BFS use
    output = {str(k): v for k, v in flat_results.items()}
    with open("jump_models.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJump models saved to jump_models.json")


if __name__ == "__main__":
    main()