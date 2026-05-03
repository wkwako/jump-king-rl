import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from PlatformParser import PlatformParser
from Ray import Ray

class JumpKingDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class BehavioralCloning:
    def __init__(self):
        self.filepath = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/recording.txt"
        self.action_map = []
        self.records = []
        self.platform_parser = PlatformParser()
        self.ray_caster = Ray(max_distance=600, step_size=8)

    def load_model(self, model_path, input_dim=42, output_dim=28, hidden_dim=256):
        """Loads a saved BC policy."""
        self.model = BCPolicy(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Loaded BC model from {model_path}")
        return self.model

    def predict(self, state, temperature=1.0):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = self.model(state_tensor) / temperature
            probs = torch.softmax(logits, dim=1)
            action_idx = torch.multinomial(probs, 1).item()
        return action_idx
        
    def train(self, X, y, action_dim, model_path, 
          epochs=100, batch_size=64, lr=1e-3, hidden_dim=256):
        """Trains BC policy on dataset and saves weights."""

        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_dataset = JumpKingDataset(X_train, y_train)
        val_dataset = JumpKingDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # model, loss, optimizer
        model = BCPolicy(input_dim=X.shape[1], output_dim=action_dim, hidden_dim=hidden_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # training
            model.train()
            train_loss = 0
            train_correct = 0
            for states_batch, actions_batch in train_loader:
                optimizer.zero_grad()
                logits = model(states_batch)
                loss = criterion(logits, actions_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (logits.argmax(dim=1) == actions_batch).sum().item()

            # validation
            model.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for states_batch, actions_batch in val_loader:
                    logits = model(states_batch)
                    loss = criterion(logits, actions_batch)
                    val_loss += loss.item()
                    val_correct += (logits.argmax(dim=1) == actions_batch).sum().item()

            train_acc = train_correct / len(X_train)
            val_acc = val_correct / len(X_val)
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return model

    def generate_state(self, state_dict):
        """Generates full state vector from a recording state dict."""
        x = state_dict["x"]
        y = state_dict["y"]
        current_screen = state_dict["current_screen"]
        is_on_ice = float(state_dict["is_on_ice"])
        is_in_snow = float(state_dict["is_in_snow"])
        wind_velocity = state_dict["wind_velocity"]

        # get tile data for current and next screen
        self.platform_parser.parse_result = self.platform_parser.read_platform_data(
            (x, y), current_screen
        )

        # build ray state
        self.ray_caster.build_ray_collision_index(
            self.platform_parser.current_tiles,
            self.platform_parser.next_tiles
        )
        ray_state = self.ray_caster.build_ray_states(num_angles=36)

        pos_state = [x, y, current_screen, is_on_ice, is_in_snow, wind_velocity]
        return np.array(pos_state + ray_state, dtype=np.float32)
    
    def generate_dataset(self, records, action_indices):
        """Generates full state vectors for all records."""
        states = []
        for i, (state_dict, _) in enumerate(records):
            if i % 100 == 0:
                print(f"Generating state {i}/{len(records)}...")
            state = self.generate_state(state_dict)
            states.append(state)
        return np.array(states), np.array(action_indices)


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
        self.records = records
        return records
    
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