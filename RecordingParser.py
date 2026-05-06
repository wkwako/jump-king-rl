import numpy as np
import json

class RecordingParser:
    def __init__(self):
        self.filepath = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/recording.txt"

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

    def tally_actions(self, records, threshold=0):
        """Tallies action durations by category. threshold bins values into groups of that width."""
        left_counts = {}
        right_counts = {}
        space_counts = {}

        for _, (left, right, space) in records:
            # bin the value
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

        # sort by duration
        left_counts = dict(sorted(left_counts.items()))
        right_counts = dict(sorted(right_counts.items()))
        space_counts = dict(sorted(space_counts.items()))

        return left_counts, right_counts, space_counts
    
    def separate_actions_and_state(self, records):
        """Separates records into state dicts and action tuples."""
        states = [r[0] for r in records]
        actions = [r[1] for r in records]
        return states, actions
    
    def get_screen_action_map(self, screen_records, env, top_n=6):
        """Returns a reduced action map based on most used actions on this screen."""
        _, actions = self.separate_actions_and_state(screen_records)
        actions = self.equalize_actions(actions)
        actions = self.cap_actions(actions)
        actions = self.snap_to_increment(actions)
        
        # get full action indices against complete action map
        indices = self.convert_to_discretized_actions(actions, env.action_map)
        
        # count and pick top N
        counts = np.bincount(indices, minlength=len(env.action_map))
        top_indices = np.argsort(counts)[::-1][:top_n]
        
        return [env.action_map[i] for i in sorted(top_indices)]
    
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
                except:
                    continue
        return records
    
    def split_recording_by_screen(self, records):
        """Groups records by current_screen."""
        by_screen = {}
        for state_dict, action in records:
            screen = int(state_dict["current_screen"])
            if screen not in by_screen:
                by_screen[screen] = []
            by_screen[screen].append((state_dict, action))
        return by_screen