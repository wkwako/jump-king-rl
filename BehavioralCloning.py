import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from stable_baselines3.common.callbacks import CallbackList

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
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
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
    
    def transfer_weights_to_ppo(self, ppo_model, model_path):
        """Transfers BC policy weights into PPO's policy network."""
        # load BC model
        bc_state = torch.load(model_path)
        ppo_policy = ppo_model.policy

        # transfer hidden layers to policy_net
        ppo_policy.mlp_extractor.policy_net[0].weight.data = bc_state["net.0.weight"]
        ppo_policy.mlp_extractor.policy_net[0].bias.data = bc_state["net.0.bias"]
        ppo_policy.mlp_extractor.policy_net[2].weight.data = bc_state["net.2.weight"]
        ppo_policy.mlp_extractor.policy_net[2].bias.data = bc_state["net.2.bias"]

        # transfer output layer
        ppo_policy.action_net.weight.data = bc_state["net.4.weight"]
        ppo_policy.action_net.bias.data = bc_state["net.4.bias"]

        print("BC weights transferred to PPO policy network.")
        return ppo_model
        
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