import math
from typing import Optional, Union
import time
import keyboard
import pydirectinput
import numpy as np
import signal
import os
import json
from datetime import datetime
from PlatformParser import PlatformParser
from JumpKingEnv import JumpKingEnv
from BehavioralCloning import BehavioralCloning
from RecordingParser import RecordingParser
import torch
import torch.nn as nn

import sys
sys.path.append("C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL")

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure

from Ray import Ray
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

class FreezePolicyCallback(BaseCallback):
    def __init__(self, freeze_updates=20, verbose=0):
        super().__init__(verbose)
        self.freeze_updates = freeze_updates
        self.frozen = True

    def _on_training_start(self):
        """Freeze policy layers at the start of training."""
        self._freeze_policy()

    def _on_rollout_end(self):
        """Unfreeze policy after n updates."""
        if self.frozen and self.model._n_updates >= self.freeze_updates:
            self._unfreeze_policy()
            print(f"Policy unfrozen after {self.freeze_updates} updates")
            self.frozen = False

    def _freeze_policy(self):
        """Freeze all policy network parameters except value function."""
        for name, param in self.model.policy.named_parameters():
            if "value_net" not in name:
                param.requires_grad = False
        print("Policy frozen — training value function only")

    def _unfreeze_policy(self):
        """Unfreeze all parameters."""
        for name, param in self.model.policy.named_parameters():
            param.requires_grad = True

    def _on_step(self):
        return True

class JumpKingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].env
        if env.gamedata is not None:
            self.logger.record("custom/current_screen", env.gamedata["current_screen"])
            self.logger.record("custom/max_height", env.gamedata["y"])
        return True

class EpisodeMode:
    ACTION = "action"
    SCREEN = "screen"
    HEIGHT = "height"
    ACTION_HEIGHT = "action_height"
    CURRICULUM = "curriculum"

class JumpKingRL:

    def __init__(self):
        self.model_direc = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/"
        #self.model_direc = "C:/Users/wkwak/Documents/CodingWork/PythonStuff/jump-king-rl/models/"

        self.MODEL_CONFIGS = {
            "PPO": {
                "class": PPO,
                "defaults": {"n_steps": 64}
            },
            "DQN": {
                "class": DQN,
                "defaults": {"learning_starts": 1000, "batch_size": 64}
            }
        }

        PPO_PARAMS = ["n_steps", "learning_rate", "batch_size", "n_epochs", "gamma", 
              "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"]

        DQN_PARAMS = ["learning_rate", "batch_size", "learning_starts", "gamma", 
                    "target_update_interval", "exploration_fraction", 
                    "exploration_initial_eps", "exploration_final_eps"]

        self.MODEL_PARAMS = {
            "PPO": PPO_PARAMS,
            "DQN": DQN_PARAMS
        }

    def train_per_screen_agents(self):
        
        pass

    def pretrain_value_function(self, ppo_model, X, epochs=50):
        # freeze policy stream to protect BC weights
        for name, param in ppo_model.policy.named_parameters():
            if "action_net" in name or "policy_net" in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, ppo_model.policy.parameters()), lr=1e-3
        )

        states = torch.FloatTensor(X)

        y_values = states[:, 1]
        upper_left_dist = states[:, 9]
        upper_right_dist = states[:, 11]
        next_upper_left_dist = states[:, 21]
        next_upper_right_dist = states[:, 23]

        y_norm = (y_values - y_values.mean()) / y_values.std()

        def platform_bonus(dist, max_dist=400):
            visible = (dist != -9999).float()
            proximity = visible * (1 - dist.clamp(0, max_dist) / max_dist)
            return proximity

        heuristic_values = (
            y_norm
            + platform_bonus(upper_left_dist) * 0.5
            + platform_bonus(upper_right_dist) * 0.5
            + platform_bonus(next_upper_left_dist) * 0.75
            + platform_bonus(next_upper_right_dist) * 0.75
        )

        for epoch in range(epochs):
            optimizer.zero_grad()
            features = ppo_model.policy.mlp_extractor(states)[1]
            predicted_values = ppo_model.policy.value_net(features).squeeze()
            loss = nn.MSELoss()(predicted_values, heuristic_values)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Value pretrain epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        # restore all parameters before RL starts
        for name, param in ppo_model.policy.named_parameters():
            param.requires_grad = True

        print("Value function pretraining complete.")

    def to_json_safe(self, value):
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        # handles SB3 schedule objects and other callables
        if callable(value):
            return str(value)
        return str(value)  # fallback for anything else unexpected

    def init_metadata(self, model):
        env = model.env.envs[0].env      
        model_type = type(model).__name__

        model_hyperparameters = {
            param: self.to_json_safe(getattr(model, param, None))
            for param in self.MODEL_PARAMS.get(model_type, [])
            if getattr(model, param, None) is not None
        }

        metadata = {
            "total_jumps": 0,
            "total_timesteps": 0,
            "episode_mode": env.episode_mode,
            "hyperparameters": {
                "max_episode_actions": env.max_episode_actions,
                "new_screen_reward": env.new_screen_reward,
                "jump_penalty": env.jump_penalty,
                "max_jump_bonus": env.max_jump_bonus,
                "grid_size": env.grid_size,
                "exploration_reward": env.exploration_reward,
            },
            "model_type": model_type,
            "model_hyperparameters" : model_hyperparameters,

            "architectural": {
                "observation_space": int(env.observation_space.shape[0]),
                "action_space": int(env.action_space.n)
            }
        }
        return metadata

    def load_model(self, name):
        # load metadata
        with open(self.model_direc + name + "_metadata.json") as f:
            self.metadata = json.load(f)
        
        # recreate env from metadata
        env = JumpKingEnv(
            episode_mode=self.metadata["episode_mode"],
            max_episode_actions=self.metadata["hyperparameters"]["max_episode_actions"]
        )
        
        model_type = self.metadata["model_type"]
        model_class = self.MODEL_CONFIGS[model_type]["class"]

        # load model
        print ("Loading existing model...")
        model = model_class.load(self.model_direc + name, env=env)
        
        return model

    def save_metadata(self, name, model, metadata, new=False):
        #if no metadata exists, create it

        env = model.env.envs[0].env

        if new:
            metadata = self.init_metadata(model)
        
        #if it does exist, update it
        else:
            metadata["total_jumps"] += env.jump_counter_metadata
            metadata["total_timesteps"] += model.num_timesteps

        with open(self.model_direc + name + "_metadata.json", "w") as f:
            json.dump(metadata, f)

    def load_metadata(self, name):
        #loads and returns metadata
        metadata_path = self.model_direc + name + "_metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file \"{metadata_path}\" does not exist.")
        
        with open(self.model_direc + name + "_metadata.json", "r") as f:
            metadata = json.load(f)

        return metadata

    def create_model(self, name, env, model_type, verbose, **kwargs):
        #creates a new model. will throw an error if model_name already exists
        model_path = self.model_direc + name
        if os.path.exists(model_path + ".zip"):
            raise FileExistsError("This model already exists. Please use a different name, delete it, or use the overwrite_model() function.")
        
        config = self.MODEL_CONFIGS[model_type]
        params = {**config["defaults"], **kwargs}

        print ("Creating new model...")
        model = config["class"]("MlpPolicy", env, verbose=verbose, **params)

        logger = configure(model_path + "_log/", ["stdout", "csv"])
        #logger = configure(model_path + "_log/", ["csv"])
        model.set_logger(logger)

        model.save(model_path)

        print ("Creating new metadata file...")
        self.save_metadata(name, model, None, True)

        return model

    def overwrite_model(self, name, model):
        metadata = self.load_metadata(name)
        self.delete_model(name)
        model.save(self.model_direc + name)
        self.save_metadata(name, model, metadata)
        return model

    def delete_model(self, name):
        #deletes a model with name

        if not os.path.exists(self.model_direc + name + ".zip"):
            raise FileNotFoundError(f"Model \"{name}\" does not exist.")

        print (f"Deleting model \"{name}\"...")
        os.remove(self.model_direc + name + ".zip")
        print ("Deleting metadata...")
        os.remove(self.model_direc + name + "_metadata.json")

    def train_model(self, name, model, total_timesteps, callback):
        print("Starting training...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.model_direc + name + "_log/" + timestamp + "/"
        
        logger = configure(log_path, ["stdout", "csv"])
        #logger = configure(log_path, ["csv"])
        model.set_logger(logger)
        
        try:
            model.learn(total_timesteps=total_timesteps, callback=callback)
            print("Training complete. Saving...")
        except KeyboardInterrupt:
            print("Interrupted. Saving model and metadata...")
        finally:
            self.overwrite_model(name, model)
            env = model.env.envs[0].env
            env.jump_counter_metadata = 0
            env.reset_keys()

env = JumpKingEnv(episode_mode="action", max_episode_actions=8, spacing=0.05)
bc = BehavioralCloning()
parser = RecordingParser()
records = parser.load_recording()
records = parser.clean_actions(records)
by_screen = parser.split_recording_by_screen(records)

_, actions = parser.separate_actions_and_state(by_screen[1])
left_counts, right_counts, space_counts = parser.tally_actions(actions)
print (f"left counts: {left_counts}")
print (f"right counts: {right_counts}")
print (f"space counts: {space_counts}")

#behavioral cloning test
# env = JumpKingEnv(episode_mode="action", max_episode_actions=8, spacing=0.05)
# bc = BehavioralCloning()

# records = bc.load_recording()
# states, actions = bc.separate_actions_and_state(records)
# actions = bc.equalize_actions(actions)
# actions = bc.cap_actions(actions)
# actions = bc.snap_to_increment(actions, increment=0.05)
# action_indices = bc.convert_to_discretized_actions(actions, env.action_map)

# X, y_labels = bc.generate_dataset(records, action_indices)

# model = bc.train(
#     X, y_labels,
#     action_dim=28,
#     model_path="models/bc_policy_sectors_tanh.pth",
#     epochs=100,
#     batch_size=64,
#     lr=1e-3,
#     hidden_dim=256
# )

# bc.load_model("models/bc_policy_sectors_tanh.pth", input_dim=25)

# for episode in range(3):
#     obs, _ = env.reset()
#     done = False
#     while not done:
#         state = bc.generate_state(env.gamedata)
#         action_idx = bc.predict(state)
#         obs, reward, terminated, truncated, info = env.step(action_idx)
#         done = terminated or truncated
#     print(f"Episode {episode+1}: screen={env.current_screen}")

# bc set up
# bc = BehavioralCloning()
# env = JumpKingEnv(episode_mode=EpisodeMode.ACTION_HEIGHT, max_episode_actions=8)

# records = bc.load_recording()
# states, actions = bc.separate_actions_and_state(records)
# actions = bc.equalize_actions(actions)
# actions = bc.cap_actions(actions)
# actions = bc.snap_to_increment(actions, increment=0.05)
# action_indices = bc.convert_to_discretized_actions(actions, env.action_map)

# # check action distribution
# print(f"Total records: {len(records)}")
# action_counts = np.bincount(action_indices, minlength=len(env.action_map))
# for i, count in enumerate(action_counts):
#     print(f"Action {i} {env.action_map[i]}: {count}")

# # generate dataset
# X, y_labels = bc.generate_dataset(records, action_indices)
# print(f"Dataset shape: {X.shape}")

# training BC model - only need to run this again if new env or more data
# model = bc.train(
#     X, y_labels,
#     action_dim=28,
#     model_path="models/bc_policy_sectors_tanh.pth",
#     epochs=100,
#     batch_size=64,
#     lr=1e-3,
#     hidden_dim=256
# )

#transferring BC model to PPO weights
# JK = JumpKingRL()
# callback = JumpKingCallback()
# env = JumpKingEnv(episode_mode=EpisodeMode.ACTION_HEIGHT, max_episode_actions=8)

# model = JK.create_model("jk_bc_ppo_valuepretraining", env, "PPO", verbose=1,
#                          n_steps=2048, ent_coef=0.005, learning_rate=0.00003,
#                          policy_kwargs={"net_arch": [256, 256]})

# bc.transfer_weights_to_ppo(model, "models/bc_policy_sectors_tanh.pth")
# JK.pretrain_value_function(model, X)
# #model = JK.load_model("jk_bc_ppo_valuepretraining")
# #freezing callback
# freeze_callback = FreezePolicyCallback(freeze_updates=20)
# jk_callback = JumpKingCallback()
# callbacks = CallbackList([freeze_callback, jk_callback])
# JK.train_model("jk_bc_ppo_valuepretraining", model, total_timesteps=100000, callback=callbacks)

# BC model testing, no RL
#bc.load_model("models/bc_policy.pth")
#print (bc.model)
# env = JumpKingEnv(episode_mode=EpisodeMode.ACTION, max_episode_actions=100)
# obs, _ = env.reset()
# print("Running BC policy...")
# total_reward = 0
# num_episodes = 5
# for episode in range(num_episodes):
#     obs, _ = env.reset()
#     episode_reward = 0
#     done = False
#     while not done:
#         # generate fresh state from current game data
#         state = bc.generate_state(env.gamedata)
#         action_idx = bc.predict(state, temperature=1.5)
#         obs, reward, terminated, truncated, info = env.step(action_idx)
#         episode_reward += reward
#         done = terminated or truncated
#     print(f"Episode {episode+1}: reward={episode_reward:.2f}, screen={env.current_screen}")
#     total_reward += episode_reward
# print(f"Average reward: {total_reward/num_episodes:.2f}")

#regular PPO
# JK = JumpKingRL()
# max_episode_actions = 8
# env = JumpKingEnv(episode_mode=EpisodeMode.ACTION_HEIGHT, max_episode_actions=max_episode_actions)
# n_steps=64
# callback = JumpKingCallback()
# platform_parser = PlatformParser()
# # #create, load, train model
# #model = JK.create_model("jk_ppo_dummy", env, "PPO", verbose=1, n_steps=n_steps)
# model = JK.load_model("jk_ppo_dummy")
# JK.train_model("jk_ppo_dummy", model, total_timesteps=10000, callback=callback) #default is 2k

# #sector information debugging
# gamedata = env.get_gamedata_old()
# platform_parser.parse_result = platform_parser.read_platform_data((gamedata["x"], gamedata["y"]), gamedata["current_screen"])
# pos_state_data = list(platform_parser.parse_result[0])
# sector_state_data = platform_parser.process_registry(gamedata["current_screen"], (gamedata["x"], gamedata["y"]))
# can_bounce_right, can_bounce_left = platform_parser.set_rebound_state((gamedata["x"], gamedata["y"]), gamedata["current_screen"])
# pos_state = [gamedata["x"], gamedata["y"], gamedata["current_screen"], env.is_on_ice, env.is_in_snow, env.wind_velocity, can_bounce_left, can_bounce_right]
# state = np.array(pos_state + pos_state_data + sector_state_data, dtype=np.float32)

# print(f"x={gamedata['x']:.1f}, y={gamedata['y']:.1f}, screen={gamedata['current_screen']}")
# print(f"left_wall={pos_state_data[0]}, right_wall={pos_state_data[1]}, ceiling={pos_state_data[2]}")
# print(f"is_on_ice={pos_state[3]}, is_in_snow={pos_state[4]}, wind_velocity={pos_state[5]}")
# print(f"can_bounce_left={pos_state[-2]}, can_bounce_right={pos_state[-1]}")
# print(f"sectors: {sector_state_data}")


# #ray info debugging
# env.gamedata = env.read_gamedata()
# env.load_game_attributes()
# platform_parser = PlatformParser()
# ray_caster = Ray()
# platform_parser.parse_result = platform_parser.read_platform_data((env.x, env.y), env.current_screen)
# ray_caster.build_ray_collision_index(platform_parser.current_tiles, platform_parser.next_tiles)
# ray_data = ray_caster.build_ray_states(num_angles=36)
# print(f"x={env.x:.1f}, y={env.y:.1f}, screen={env.current_screen}")
# print(f"ray data ({len(ray_data)} values):")
# for i, dist in enumerate(ray_data):
#     angle = i * (360 / 36)
#     print(f"  {angle:6.1f}°: {dist:.1f}px")


