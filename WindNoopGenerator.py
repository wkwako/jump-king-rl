import json
import time
from datetime import datetime

WIND_CYCLE_FRAMES = 780
FRAMES_PER_SECOND = 60
NOOP_THRESHOLD_SECONDS = 0.4
NOOP_THRESHOLD_FRAMES = int(NOOP_THRESHOLD_SECONDS * FRAMES_PER_SECOND)  # 24 frames
MAX_NOOP_FRAMES = WIND_CYCLE_FRAMES  # cap at 1 cycle
frame_skips = 30

def parse_timestamp(ts_str):
    """Parse timestamp string to datetime object."""
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

def load_wind_recording(filepath):
    """Loads wind-only recording with timestamps.
    Returns list of (timestamp, state_dict, action) tuples.
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Start session"):
                continue
            try:
                parts = line.split("|")
                if len(parts) == 3:
                    ts_str, state_str, durations_str = parts
                elif len(parts) == 2:
                    ts_str = None
                    state_str, durations_str = parts
                else:
                    continue

                state = json.loads(state_str)
                left, right, space = map(float, durations_str.split(","))
                ts = parse_timestamp(ts_str) if ts_str else None
                records.append((ts, state, (left, right, space)))
            except Exception as e:
                continue

    print(f"Loaded {len(records)} wind records")
    return records

def fill_wind_noops(records, screen, verbose=True):
    """Inserts no-op actions between recorded wind screen actions.
    
    Rules:
    1. If time gap between consecutive actions > 0.4s, fill with no-ops
    2. Cap at 1 wind cycle (780 frames) regardless of actual gap
    
    Returns list of (state_dict, action) tuples with no-ops inserted.
    """
    filled = []
    noop_action = (0, 0, 0)
    total_noops = 0

    for i, (ts, state_dict, action) in enumerate(records):
        filled.append((state_dict, action))

        if i >= len(records) - 1:
            continue

        next_ts, next_state_dict, next_action = records[i + 1]

        # only fill if both records have timestamps
        if ts is None or next_ts is None:
            continue

        # compute time gap
        gap_seconds = (next_ts - ts).total_seconds()

        # ignore negative gaps (session boundary) or huge gaps (AFK)
        if gap_seconds <= 0:
            continue

        # cap at 1 wind cycle worth of time
        gap_seconds = min(gap_seconds, MAX_NOOP_FRAMES / FRAMES_PER_SECOND)

        # convert to frames
        gap_frames = int(gap_seconds * FRAMES_PER_SECOND)

        # only insert if gap is meaningful
        if gap_frames <= NOOP_THRESHOLD_FRAMES:
            continue

        # insert no-ops — use current state_dict for all no-op states
        noops_to_insert = gap_frames // frame_skips
        for _ in range(noops_to_insert):
            filled.append((state_dict, noop_action))
        total_noops += noops_to_insert

        if verbose:
            print(f"  Record {i}: gap={gap_seconds:.2f}s → inserted {noops_to_insert} no-ops "
                  f"(wind_vel={state_dict.get('wind_velocity', 0):.4f})")

    print(f"\nScreen {screen}: {len(records)} records → {len(filled)} after no-op fill "
          f"({total_noops} no-ops inserted)")
    return [(s, a) for s, a in filled]


def analyze_wind_recording(filepath, screen):
    """Loads, analyzes and fills no-ops for a wind recording."""
    records = load_wind_recording(filepath)
    
    # filter to just this screen
    screen_records = [(ts, s, a) for ts, s, a in records 
                      if int(s.get("current_screen", -1)) == screen]
    print(f"Screen {screen}: {len(screen_records)} records before fill")
    
    if not screen_records:
        print(f"No records found for screen {screen}")
        return []
    
    # analyze gaps
    print("\nGap analysis:")
    gaps = []
    for i in range(len(screen_records) - 1):
        ts1 = screen_records[i][0]
        ts2 = screen_records[i+1][0]
        if ts1 and ts2:
            gap = (ts2 - ts1).total_seconds()
            if gap > 0:
                gaps.append(gap)
    
    if gaps:
        import statistics
        print(f"  Total gaps: {len(gaps)}")
        print(f"  Mean gap: {statistics.mean(gaps):.2f}s")
        print(f"  Max gap: {max(gaps):.2f}s")
        print(f"  Gaps > 0.4s: {sum(1 for g in gaps if g > 0.4)}")
        print(f"  Gaps > 5s (likely AFK): {sum(1 for g in gaps if g > 5)}")
    
    print("\nFilling no-ops:")
    filled = fill_wind_noops(screen_records, screen)
    return filled


if __name__ == "__main__":
    RECORDING_PATH = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/recording_wind_only.txt"
    
    import static_variables
    for screen in sorted(static_variables.WIND_SCREENS):
        print(f"\n{'='*50}")
        filled = analyze_wind_recording(RECORDING_PATH, screen)