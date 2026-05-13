import numpy as np
import json

import static_variables

from PlatformParser import PlatformParser

class RecordingParser:
    def __init__(self):
        self.filepath = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/recording.txt"
        #self.filepath = "C:/Users/wkwak/Documents/CodingWork/PythonStuff/jump-king-rl/recording.txt"
        self.platform_parser = PlatformParser()

    def get_state_size(self, screen):
        if screen in static_variables.WIND_SCREENS:
            return 4  # x, y, wind_velocity, ceiling
        elif screen in static_variables.ICE_SCREENS:
            return 4  # x, y, vel_x, ceiling
        else:
            return 3  # x, y, ceiling

    def equalize_actions(self, actions):
        """For jump actions, sets arrow key duration equal to spacebar duration."""
        equalized = []
        for left, right, space in actions:
            if space > 0:
                # it's a jump — equalize non-zero arrow key to space duration
                new_left = space if left > 0 else 0
                new_right = space if right > 0 else 0
                equalized.append((new_left, new_right, space))
            else:
                # it's a walk — leave as is
                equalized.append((left, right, space))
        return equalized
    
    def cap_actions(self, actions, max_jump=0.6, max_walk=0.2):
        """Caps jump durations to max_jump and walk durations to max_walk."""
        capped = []
        for left, right, space in actions:
            if space > 0:
                # jump action
                space = min(space, max_jump)
                left = min(left, max_jump) if left > 0 else 0
                right = min(right, max_jump) if right > 0 else 0
            else:
                # walk action
                left = min(left, max_walk) if left > 0 else 0
                right = min(right, max_walk) if right > 0 else 0
            capped.append((left, right, space))
        return capped
    
    def snap_to_increment(self, actions, increment=0.05):
        """Snaps all action durations to the nearest increment."""
        snapped = []
        for left, right, space in actions:
            left = round(round(left / increment) * increment, 2)
            right = round(round(right / increment) * increment, 2)
            space = round(round(space / increment) * increment, 2)
            snapped.append((left, right, space))
        return snapped
    
    def convert_to_discretized_actions(self, actions, action_map):
        """Rounds each action tuple to the nearest action in the action map.
        Returns list of action indices."""
        def distance(a1, a2):
            return sum((x - y) ** 2 for x, y in zip(a1, a2)) ** 0.5

        indices = []
        for action in actions:
            best_idx = min(range(len(action_map)), 
                          key=lambda i: distance(action, action_map[i]))
            indices.append(best_idx)
        return indices

    def generate_state(self, state_dict):
        """Generates full 25-value state vector from a recording state dict."""
        x = state_dict["x"]
        y = state_dict["y"]
        current_screen = int(state_dict["current_screen"])
        is_on_ice = float(state_dict["is_on_ice"])
        is_in_snow = float(state_dict["is_in_snow"])
        wind_velocity = float(state_dict["wind_velocity"])

        # get platform data
        self.platform_parser.parse_result = self.platform_parser.read_platform_data(
            (x, y), current_screen
        )

        if self.platform_parser.parse_result is not None:
            pos_state_data = list(self.platform_parser.parse_result[0])
        else:
            pos_state_data = [-9999, 9999, 9999, -9999, 9999]

        # get sector data
        sector_state_data = self.platform_parser.process_registry(current_screen, (x, y))

        # get rebound state
        can_bounce_right, can_bounce_left = self.platform_parser.set_rebound_state(
            (x, y), current_screen
        )

        pos_state = [x, y, current_screen, is_on_ice, is_in_snow, wind_velocity, 
                    can_bounce_right, can_bounce_left]

        return np.array(pos_state + pos_state_data + sector_state_data, dtype=np.float32)
    
    def generate_dataset(self, records, action_indices):
        """Generates full state vectors for all records."""
        states = []
        for i, (state_dict, _) in enumerate(records):
            if i % 100 == 0:
                print(f"Generating state {i}/{len(records)}...")
            state = self.generate_state(state_dict)
            states.append(state)
        return np.array(states), np.array(action_indices)
    
    def generate_state_per_screen(self, state_dict, screen):
        """Generates minimal state vector for per-screen agent."""
        x = float(state_dict["x"])
        y = float(state_dict["y"])
        
        # get ceiling distance from platform parser
        self.platform_parser.parse_result = self.platform_parser.read_platform_data(
            (x, y), screen
        )
        
        if self.platform_parser.parse_result is not None:
            ceiling = float(self.platform_parser.parse_result[0][2])
        else:
            ceiling = 9999.0
        
        if screen in static_variables.WIND_SCREENS:
            wind_velocity = float(state_dict["wind_velocity"])
            return np.array([x, y, wind_velocity, ceiling], dtype=np.float32)
        elif screen in static_variables.ICE_SCREENS:
            vel_x = float(state_dict["vel_x"])
            return np.array([x, y, vel_x, ceiling], dtype=np.float32)
        else:
            return np.array([x, y, ceiling], dtype=np.float32)

    def generate_dataset_per_screen(self, records, action_indices, screen):
        """Generates minimal state vectors for all records on a given screen."""
        states = []
        for i, (state_dict, _) in enumerate(records):
            if i % 100 == 0:
                print(f"Generating state {i}/{len(records)}...")
            state = self.generate_state_per_screen(state_dict, screen)
            states.append(state)
        return np.array(states), np.array(action_indices)

    def tally_actions(self, actions, threshold=0):
        """Tallies action durations by category. Accepts plain list of (left, right, space) tuples."""
        left_counts = {}
        right_counts = {}
        space_counts = {}

        for left, right, space in actions:
            if threshold > 0:
                left = round(round(left / threshold) * threshold, 3)
                right = round(round(right / threshold) * threshold, 3)
                space = round(round(space / threshold) * threshold, 3)

            if left > 0:
                left_counts[left] = left_counts.get(left, 0) + 1
            if right > 0:
                right_counts[right] = right_counts.get(right, 0) + 1
            if space > 0:
                space_counts[space] = space_counts.get(space, 0) + 1

        left_counts = dict(sorted(left_counts.items()))
        right_counts = dict(sorted(right_counts.items()))
        space_counts = dict(sorted(space_counts.items()))

        return left_counts, right_counts, space_counts
    
    def separate_actions_and_state(self, records):
        """Separates records into state dicts and action tuples."""
        states = [r[0] for r in records]
        actions = [r[1] for r in records]
        return states, actions
    
    def get_screen_action_map(self, screen):
        """Generates action map tuples for a given screen from SCREEN_ACTION_MAPS config."""
        config = static_variables.SCREEN_ACTION_MAPS[screen]
        action_map = []

        for duration in config.get("walks", []):
            action_map.append((duration, 0, 0))
            action_map.append((0, duration, 0))

        for duration in config.get("jumps", []):
            action_map.append((duration, 0, duration))
            action_map.append((0, duration, duration))

        for duration in config.get("only_jump", []):
            action_map.append((0, 0, duration))

        # add no-op for wind screens
        if screen in static_variables.WIND_SCREENS:
            action_map.append((0, 0, 0))

        return action_map
    
    def load_recording(self):
        """Reads recording.txt and returns list of (state_dict, (left, right, space)) tuples."""
        records = []
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Start session"):
                    continue
                try:
                    parts = line.split("|")
                    
                    # handle both formats: with or without timestamp
                    if len(parts) == 3:
                        # timestamp|state|actions
                        _, state_str, durations_str = parts
                    elif len(parts) == 2:
                        # state|actions (old format)
                        state_str, durations_str = parts
                    else:
                        continue
                        
                    state = json.loads(state_str)
                    left, right, space = map(float, durations_str.split(","))
                    records.append((state, (left, right, space)))
                    if len(records) % 500 == 0:
                        print(f"Loaded {len(records)} records...")
                except:
                    continue
        return records
    
    def clean_actions(self, records, max_raw=2.0, max_jump=0.6, max_walk=0.2, increment=0.05):
        """Filters malformed records, then equalizes, caps, and snaps actions in one pass."""
        cleaned = []
        filtered_count = 0
        
        for state_dict, action in records:
            left, right, space = action
            # drop implausible values
            if left > max_raw or right > max_raw or space > max_raw:
                filtered_count += 1
                continue
            # drop walks with both directions held
            if space == 0 and left > 0 and right > 0:
                filtered_count += 1
                continue

            # equalize
            if space > 0:
                left = space if left > 0 else 0
                right = space if right > 0 else 0

            # cap
            if space > 0:
                space = min(space, max_jump)
                left = min(left, max_jump) if left > 0 else 0
                right = min(right, max_jump) if right > 0 else 0
            else:
                left = min(left, max_walk) if left > 0 else 0
                right = min(right, max_walk) if right > 0 else 0

            # snap
            left = round(round(left / increment) * increment, 2)
            right = round(round(right / increment) * increment, 2)
            space = round(round(space / increment) * increment, 2)

            cleaned.append((state_dict, (left, right, space)))

        print(f"Filtered {filtered_count} malformed records, {len(cleaned)} remaining.")
        return cleaned
    
    def split_recording_by_screen(self, records):
        """Groups records by current_screen."""
        records = list(records)
        by_screen = {}
        for i, (state_dict, action) in enumerate(records):
            if i % 500 == 0:
                print(f"Processing record {i}/{len(records)}...")
            screen = int(state_dict["current_screen"])
            if screen not in by_screen:
                by_screen[screen] = []
            by_screen[screen].append((state_dict, action))
        
        print(f"Done. Found {len(by_screen)} screens, {len(records)} total records.")
        return by_screen