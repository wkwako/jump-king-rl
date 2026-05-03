import json

class BehavioralCloning:
    def __init__(self):
        self.filepath = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/recording.txt"

    def load_recording(self):
        """Reads recording.txt and returns list of (state_dict, (left, right, space)) tuples."""
        records = []
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Start session"):
                    continue
                try:
                    state_str, durations_str = line.split("|")
                    state = json.loads(state_str)
                    left, right, space = map(float, durations_str.split(","))
                    records.append((state, (left, right, space)))
                except:
                    continue
        return records

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