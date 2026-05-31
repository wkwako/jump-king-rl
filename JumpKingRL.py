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
from JumpKingEnv import JumpKingEnv, ScreenTransitionException
from BehavioralCloning import BehavioralCloning
from RecordingParser import RecordingParser
import torch
import torch.nn as nn
from stable_baselines3.common.utils import obs_as_tensor
import static_variables
pydirectinput.PAUSE=0

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
        if self.frozen and self.model.num_timesteps >= self.freeze_updates * self.model.n_steps:
            self._unfreeze_policy()
            self.frozen = False
            print(f"Policy unfrozen at timestep {self.model.num_timesteps}")

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
        self.best_reward = -np.inf
        JK = JumpKingRL()
        self.save_path = JK.model_direc

    # def _on_step(self) -> bool:
    #     env = self.training_env.envs[0].env
    #     if env.gamedata is not None:
    #         self.logger.record("custom/current_screen", env.gamedata["current_screen"])
    #         self.logger.record("custom/max_height", env.gamedata["y"])
    #     return True

    def _on_rollout_end(self):
        ep_rew_mean = self.model.ep_info_buffer
        if ep_rew_mean:
            mean_reward = np.mean([ep['r'] for ep in ep_rew_mean])
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(f"{self.save_path}_best")
                print(f"New best reward {mean_reward:.2f} — model saved")

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].env
        
        if env.gamedata is not None:
            self.logger.record("custom/current_screen", env.gamedata["current_screen"])
            self.logger.record("custom/max_height", env.gamedata["y"])
        
        #print action probabilities
        # obs = self.locals.get("obs_tensor")
        # if obs is not None:
        #     with torch.no_grad():
        #         dist = self.model.policy.get_distribution(obs)
        #         probs = dist.distribution.probs[0].cpu().numpy()
        #         action_map = env.action_map
        #         print("Action probs:")
        #         for i, (prob, action) in enumerate(zip(probs, action_map)):
        #             print(f"  {i} {action}: {prob:.4f}")
        
        return True

class EpisodeMode:
    ACTION = "action"
    SCREEN = "screen"
    HEIGHT = "height"
    ACTION_HEIGHT = "action_height"
    CURRICULUM = "curriculum"
    PER_SCREEN = "per_screen"
    JUMPED = "jumped"

class JumpKingRL:

    def __init__(self):
        self.X_by_screen = {}
        self.model_direc = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/"
        self.wind_path = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/recording_wind_only.txt"
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

    def reset_keys(self):
        #print ("RESETTING KEYS")
        pydirectinput.keyUp("space")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("left")

    def pretrain_value_function(self, ppo_model, X, epochs=50, per_screen=False):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, ppo_model.policy.parameters()), lr=1e-3
        )

        # freeze policy stream
        for name, param in ppo_model.policy.named_parameters():
            if "action_net" in name or "policy_net" in name:
                param.requires_grad = False

        states = torch.FloatTensor(X)
        y_values = states[:, 1]
        y_norm = (y_values - y_values.mean()) / (y_values.std() + 1e-8)

        if not per_screen:
            # full heuristic with platform bonuses
            upper_left_dist = states[:, 9]
            upper_right_dist = states[:, 11]
            next_upper_left_dist = states[:, 21]
            next_upper_right_dist = states[:, 23]

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
        else:
            # per-screen: just normalized y
            heuristic_values = y_norm

        for epoch in range(epochs):
            optimizer.zero_grad()
            features = ppo_model.policy.mlp_extractor(states)[1]
            predicted_values = ppo_model.policy.value_net(features).squeeze()
            loss = nn.MSELoss()(predicted_values, heuristic_values)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Value pretrain epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        # restore all parameters
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
                "jump_penalty": env.jump_penalty,
                "max_jump_bonus": env.max_jump_bonus,
                "grid_size": env.grid_size,
                "exploration_reward": env.exploration_reward,
            },
            "model_type": model_type,
            "model_hyperparameters": model_hyperparameters,
            "architectural": {
                "observation_space": int(env.observation_space.shape[0]),
                "action_space": int(env.action_space.n)
            },
            "env_config": {
                "episode_mode": env.episode_mode,
                "max_episode_actions": env.max_episode_actions,
                "action_cutoff": env.action_cutoff,
                "per_screen": env.per_screen,
                "current_screen": env.current_screen,
                "n_steps": model.n_steps,
                "n_epochs": model.n_epochs,
                "ent_coef": model.ent_coef,
                "learning_rate": model.learning_rate if isinstance(model.learning_rate, float) else 0.0001,
                "vf_coef": model.vf_coef,
                "clip_range": model.clip_range if isinstance(model.clip_range, float) else 0.2,
                "target_kl": model.target_kl,
            }
        }
        return metadata

    def load_model(self, name, screen=None, model_prefix="ppo", env=None, only_agent=False):
        if screen is not None:
            model_path = f"{self.model_direc}{name}/{model_prefix}_screen_{screen}"
            metadata_path = f"{self.model_direc}{name}/{model_prefix}_screen_{screen}_metadata.json"
        else:
            model_path = f"{self.model_direc}{name}"
            metadata_path = f"{self.model_direc}{name}_metadata.json"

        if not only_agent:
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        parser = RecordingParser()
        action_map = parser.get_screen_action_map(screen) if screen is not None else None

        if env is None:
            if only_agent:
                # load env config from metadata
                with open(metadata_path) as f:
                    saved_metadata = json.load(f)
                
                env_config = saved_metadata.get("env_config", {})
                env = JumpKingEnv(
                    episode_mode=env_config.get("episode_mode", EpisodeMode.SCREEN),
                    max_episode_actions=env_config.get("max_episode_actions", 22),
                    per_screen=env_config.get("per_screen", True),
                    action_map=action_map,
                    current_screen=env_config.get("current_screen", screen),
                    action_cutoff=env_config.get("action_cutoff", 22),
                    dummyenv=False
                )
            else:
                env = JumpKingEnv(
                    episode_mode=self.metadata["episode_mode"],
                    max_episode_actions=self.metadata["hyperparameters"]["max_episode_actions"],
                    per_screen=screen is not None,
                    action_map=action_map,
                    current_screen=screen if screen is not None else 0,
                    dummyenv=False
                )

        # use PPO as default model class when only_agent=True
        model_class = PPO if only_agent else self.MODEL_CONFIGS[self.metadata["model_type"]]["class"]
        
        model = model_class.load(
            model_path,
            env=env,
            custom_objects={
                "action_space": env.action_space,
                "observation_space": env.observation_space
            }
        )

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

    # def create_model(self, name, env, model_type, verbose, **kwargs):
    #     model_path = self.model_direc + name
    #     if os.path.exists(model_path + ".zip"):
    #         raise FileExistsError("This model already exists. Please use a different name, delete it, or use the overwrite_model() function.")
        
    #     config = self.MODEL_CONFIGS[model_type]
    #     params = {**config["defaults"], **kwargs}

    #     print("Creating new model...")
    #     model = config["class"]("MlpPolicy", env, verbose=verbose, **params)

    #     os.makedirs(model_path + "_log/", exist_ok=True)
    #     logger = configure(model_path + "_log/", ["stdout", "csv"])
    #     model.set_logger(logger)

    #     model.save(model_path)
    #     print(f"Model saved to {model_path}.zip")

    #     print("Creating new metadata file...")
    #     self.save_metadata(name, model, None, True)
    #     print(f"Metadata saved.")

    #     return model

    def create_model(self, name, env, model_type, verbose, **kwargs):
        model_path = self.model_direc + name
        if os.path.exists(model_path + ".zip"):
            raise FileExistsError("This model already exists. Please use a different name, delete it, or use the overwrite_model() function.")
        
        config = self.MODEL_CONFIGS[model_type]
        params = {**config["defaults"], **kwargs}

        print("Creating new model...")
        model = config["class"]("MlpPolicy", env, verbose=verbose, **params)

        model.save(model_path)
        print(f"Model saved to {model_path}.zip")

        print("Creating new metadata file...")
        self.save_metadata(name, model, None, True)
        print(f"Metadata saved.")

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

    def gen_BC_bulk(self, folder_name, records):
        """Trains BC models for all screens and saves them to folder."""
        folder_path = self.model_direc + folder_name
        if os.path.exists(folder_path):
            raise FileExistsError(f"Folder '{folder_name}' already exists. Please use a different name or delete it first.")
        os.makedirs(folder_path)
        #os.makedirs(self.model_direc + folder_name, exist_ok=True)
        
        parser = RecordingParser()
        records = parser.clean_actions(records)
        by_screen = parser.split_recording_by_screen(records)
        
        print(f"\n{'='*50}")
        print(f"Total screens with data: {len(by_screen)}")
        print(f"Total records: {sum(len(v) for v in by_screen.values())}")
        print(f"{'='*50}")
        
        for screen, screen_records in sorted(by_screen.items()):
            print(f"\n--- Screen {screen} ({len(screen_records)} records) ---")
            
            if len(screen_records) < 10:
                print(f"Skipping — insufficient data")
                continue
            
            action_map = parser.get_screen_action_map(screen)
            state_size = parser.get_state_size(screen)
            
            print(f"State size: {state_size}")
            print(f"Action map ({len(action_map)} actions):")
            for i, action in enumerate(action_map):
                print(f"  {i}: left={action[0]}, right={action[1]}, space={action[2]}")
            
            _, actions = parser.separate_actions_and_state(screen_records)
            action_indices = parser.convert_to_discretized_actions(actions, action_map)
            
            counts = np.bincount(action_indices, minlength=len(action_map))
            print(f"Action distribution:")
            for i, count in enumerate(counts):
                pct = count / len(action_indices) * 100
                print(f"  action {i}: {count} ({pct:.1f}%)")
            
            X, y_labels = parser.generate_dataset_per_screen(screen_records, action_indices, screen)
            print(f"Dataset shape: {X.shape}")
            
            self.X_by_screen[screen] = X
            
            model_path = f"{self.model_direc}{folder_name}/bc_screen_{screen}.pth"
            bc = BehavioralCloning()
            bc.train(
                X, y_labels,
                action_dim=len(action_map),
                model_path=model_path,
                hidden_dim=256,
                epochs=100,
                batch_size=32,
                lr=1e-3
            )
            print(f"Saved to {model_path}")
        
        print(f"\n{'='*50}")
        print("BC bulk training complete.")
        print(f"{'='*50}")

    def gen_RL_bulk(self, folder_name, n_steps=128, episode_mode=EpisodeMode.ACTION_HEIGHT):
        """Creates PPO models for all screens with BC weight transfer and value pretraining."""
        os.makedirs(self.model_direc + folder_name, exist_ok=True)
        
        parser = RecordingParser()
        bc = BehavioralCloning()
        
        for screen in sorted(self.X_by_screen.keys()):
            print(f"\n--- Screen {screen} ---")
            
            bc_model_path = f"{self.model_direc}{folder_name}/bc_screen_{screen}.pth"
            if not os.path.exists(bc_model_path):
                print(f"Skipping screen {screen} — no BC model found")
                continue
            
            action_map = parser.get_screen_action_map(screen)
            state_size = parser.get_state_size(screen)
            
            env = JumpKingEnv(
                episode_mode=episode_mode,
                max_episode_actions=12,
                per_screen=True,
                action_map=action_map,
                current_screen=screen
            )
            
            model_name = f"{folder_name}/ppo_screen_{screen}"

            model = self.create_model(
                model_name, env, "PPO",
                verbose=1,
                n_steps=n_steps,
                batch_size=64,
                n_epochs=10,
                ent_coef=0.02, #was 0.005 for the good screen1 run
                learning_rate=0.0001, #was 0.00003 for the good screen1 run
                policy_kwargs={"net_arch": [256, 256]}
            )


            print(f"X shape for screen {screen}: {self.X_by_screen[screen].shape}")
            bc_state = torch.load(bc_model_path)
            print(f"BC input layer shape: {bc_state['net.0.weight'].shape}")
            bc.transfer_weights_to_ppo(model, bc_model_path)
            self.pretrain_value_function(model, self.X_by_screen[screen], per_screen=True)
            
            self.overwrite_model(model_name, model)
            print(f"Screen {screen} PPO model saved.")
        
        print("\nRL bulk generation complete.")

    # def gen_DQN_bulk(self, folder_name):
    #     """Creates DQN models for all screens with BC weight transfer."""
    #     os.makedirs(self.model_direc + folder_name, exist_ok=True)
        
    #     parser = RecordingParser()
    #     bc = BehavioralCloning()
        
    #     for screen in sorted(self.X_by_screen.keys()):
    #         print(f"\n--- Screen {screen} ---")
            
    #         bc_model_path = f"{self.model_direc}{folder_name}/bc_screen_{screen}.pth"
    #         if not os.path.exists(bc_model_path):
    #             print(f"Skipping screen {screen} — no BC model found")
    #             continue
            
    #         action_map = parser.get_screen_action_map(screen)
            
    #         env = JumpKingEnv(
    #             episode_mode=EpisodeMode.ACTION_HEIGHT,
    #             max_episode_actions=12,
    #             per_screen=True,
    #             action_map=action_map,
    #             current_screen=screen
    #         )
            
    #         model_name = f"{folder_name}/dqn_screen_{screen}"
    #         model = self.create_model(
    #             model_name, env, "DQN",
    #             verbose=1,
    #             learning_rate=0.00003,
    #             batch_size=32,
    #             learning_starts=50,
    #             exploration_fraction=0.03,
    #             exploration_final_eps=0.1,
    #             policy_kwargs={"net_arch": [256, 256]}
    #         )
            
    #         bc.transfer_weights_to_dqn(model, bc_model_path)
            
    #         self.overwrite_model(model_name, model)
    #         print(f"Screen {screen} DQN model saved.")
        
    #     print("\nDQN bulk generation complete.")

    # def train_model_per_screen(self, folder_name, start_screen=0, total_timesteps=999999):
    #     """Kicks off per-screen training. Handles screen transitions and keyboard interrupts."""
    #     current_screen = start_screen
        
    #     while True:
    #         print(f"\n--- Loading model for screen {current_screen} ---")
            
    #         model_path = f"{self.model_direc}{folder_name}/ppo_screen_{current_screen}"
    #         if not os.path.exists(model_path + ".zip"):
    #             print(f"No model found for screen {current_screen}, stopping.")
    #             break

    #         # get stable screen reading
    #         time.sleep(0.3)
    #         model = self.load_model(folder_name, screen=current_screen)
    #         model.env.envs[0].env.expected_screen = current_screen
    #         model.env.envs[0].env.total_screen_actions = 0

    #         actual_screen = model.env.envs[0].env.read_gamedata()["current_screen"]
    #         print(f"Loaded model for screen: {current_screen}, actual screen: {actual_screen}")

    #         if actual_screen != current_screen:
    #             print(f"Screen mismatch — switching to screen {actual_screen}")
    #             current_screen = actual_screen
    #             continue

    #         try:
    #             jk_callback = JumpKingCallback()
    #             callbacks = CallbackList([jk_callback])
                
    #             log_path = f"{self.model_direc}{folder_name}/ppo_screen_{current_screen}_log/"
    #             os.makedirs(log_path, exist_ok=True)
    #             logger = configure(log_path, ["csv"])  # remove "stdout" if too noisy
    #             model.set_logger(logger)
                
    #             model.learn(
    #                 total_timesteps=total_timesteps,
    #                 reset_num_timesteps=False,
    #                 callback=callbacks
    #             )
    #             print(f"Screen {current_screen} training complete.")
    #             self.overwrite_model(f"{folder_name}/ppo_screen_{current_screen}", model)

    #         except ScreenTransitionException as e:
    #             #print(f"ScreenTransitionException caught")
    #             self.reset_keys()
    #             self.overwrite_model(f"{folder_name}/ppo_screen_{current_screen}", model)
    #             model.env.envs[0].env.reset_keys()
                
    #             time.sleep(0.75)
    #             model.env.envs[0].env.gamedata = model.env.envs[0].env.read_gamedata()
    #             model.env.envs[0].env.load_game_attributes()
    #             current_screen = model.env.envs[0].env.current_screen
    #             print(f"Transitioning to screen {current_screen}")

    #         except KeyboardInterrupt:
    #             self.reset_keys()
    #             print(f"Interrupted on screen {current_screen}, saving...")
    #             self.overwrite_model(f"{folder_name}/ppo_screen_{current_screen}", model)
    #             model.env.envs[0].env.reset_keys()
    #             break

    #         finally:
    #             model.env.envs[0].env.reset_keys()

    # def train_DQN_per_screen(self, folder_name, start_screen=0, total_timesteps=999999):
    #     """Kicks off per-screen DQN training. Handles screen transitions and keyboard interrupts."""
    #     current_screen = start_screen
        
    #     while True:
    #         print(f"\n--- Loading DQN model for screen {current_screen} ---")
            
    #         model_path = f"{self.model_direc}{folder_name}/dqn_screen_{current_screen}"
    #         if not os.path.exists(model_path + ".zip"):
    #             print(f"No model found for screen {current_screen}, stopping.")
    #             break

    #         time.sleep(0.3)
    #         model = self.load_model(folder_name, screen=current_screen, model_prefix="dqn")
    #         model.env.envs[0].env.expected_screen = current_screen
    #         model.env.envs[0].env.total_screen_actions = 0

    #         actual_screen = model.env.envs[0].env.read_gamedata()["current_screen"]
    #         print(f"Loaded model for screen: {current_screen}, actual screen: {actual_screen}")

    #         if actual_screen != current_screen:
    #             print(f"Screen mismatch — switching to screen {actual_screen}")
    #             current_screen = actual_screen
    #             continue

    #         try:
    #             jk_callback = JumpKingCallback()
    #             callbacks = CallbackList([jk_callback])
                
    #             log_path = f"{self.model_direc}{folder_name}/dqn_screen_{current_screen}_log/"
    #             logger = configure(log_path, ["stdout", "csv"])
    #             model.set_logger(logger)
                
    #             model.learn(
    #                 total_timesteps=total_timesteps,
    #                 reset_num_timesteps=False,
    #                 callback=callbacks
    #             )
    #             print(f"Screen {current_screen} training complete.")
    #             self.overwrite_model(f"{folder_name}/dqn_screen_{current_screen}", model)

    #         except ScreenTransitionException as e:
    #             self.reset_keys()
    #             self.overwrite_model(f"{folder_name}/dqn_screen_{current_screen}", model)
    #             model.env.envs[0].env.reset_keys()
                
    #             time.sleep(0.75)
    #             model.env.envs[0].env.gamedata = model.env.envs[0].env.read_gamedata()
    #             model.env.envs[0].env.load_game_attributes()
    #             current_screen = model.env.envs[0].env.current_screen
    #             print(f"Transitioning to screen {current_screen}")

    #         except KeyboardInterrupt:
    #             self.reset_keys()
    #             print(f"Interrupted on screen {current_screen}, saving...")
    #             self.overwrite_model(f"{folder_name}/dqn_screen_{current_screen}", model)
    #             model.env.envs[0].env.reset_keys()
    #             break

    #         finally:
    #             model.env.envs[0].env.reset_keys()

    def create_BC_screen(self, name, screen, records, epochs=100, batch_size=32, lr=1e-3):
        """Trains a single BC model for one screen and saves to models/<name>/."""
        folder_path = self.model_direc + name
        os.makedirs(folder_path, exist_ok=True)
 
        parser = RecordingParser()
        records = parser.clean_actions(records)
        by_screen = parser.split_recording_by_screen(records)
 
        if screen not in by_screen:
            print(f"No records found for screen {screen}.")
            return
 
        if screen in static_variables.WIND_SCREENS:
            wind_records = parser.load_wind_recording(self.wind_path)
            wind_screen_records = [(ts, s, a) for ts, s, a in wind_records
                                if int(s.get("current_screen", -1)) == screen]
            screen_records = parser.fill_wind_noops(wind_screen_records, screen, noop_divisor=30)
        else:
            screen_records = by_screen[screen]


        print(f"\n--- Screen {screen} ({len(screen_records)} records) ---")
 
        if len(screen_records) < 10:
            print(f"Insufficient data for screen {screen}, aborting.")
            return
 
        action_map = parser.get_screen_action_map(screen)
        state_size = parser.get_state_size(screen)
 
        print(f"State size: {state_size}")
        print(f"Action map ({len(action_map)} actions):")
        for i, action in enumerate(action_map):
            print(f"  {i}: left={action[0]}, right={action[1]}, space={action[2]}")
 
        _, actions = parser.separate_actions_and_state(screen_records)
        action_indices = parser.convert_to_discretized_actions(actions, action_map)
 
        counts = np.bincount(action_indices, minlength=len(action_map))
        print(f"Action distribution:")
        for i, count in enumerate(counts):
            pct = count / len(action_indices) * 100
            print(f"  action {i}: {count} ({pct:.1f}%)")
 
        X, y_labels = parser.generate_dataset_per_screen(screen_records, action_indices, screen)
        print(f"Dataset shape: {X.shape}")
 
        self.X_by_screen[screen] = X
 
        model_path = f"{self.model_direc}{name}/bc_screen_{screen}.pth"
        bc = BehavioralCloning()
        bc.train(
            X, y_labels,
            action_dim=len(action_map),
            model_path=model_path,
            hidden_dim=256,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
        print(f"BC model saved to {model_path}")
 
    def create_RL_screen(self, name, screen, n_steps=2048, episode_mode=EpisodeMode.SCREEN,
                         freeze_updates=5, ent_coef=0.02, learning_rate=0.0001, vf_coef=0.5,
                        n_epochs=10, clip_range=0.2, target_kl=0.02, action_cutoff=22):
        """Creates a PPO model for one screen with BC weight transfer and value pretraining.
        Saves to models/<name>/."""
        folder_path = self.model_direc + name
        os.makedirs(folder_path, exist_ok=True)
 
        bc_model_path = f"{self.model_direc}{name}/bc_screen_{screen}.pth"
        if not os.path.exists(bc_model_path):
            print(f"No BC model found at {bc_model_path}. Run create_BC_screen first.")
            return
 
        if screen not in self.X_by_screen:
            print(f"No X data for screen {screen} in memory. Run create_BC_screen first.")
            return
 
        parser = RecordingParser()
        bc = BehavioralCloning()
 
        action_map = parser.get_screen_action_map(screen)
 
        env = JumpKingEnv(
            episode_mode=episode_mode,
            max_episode_actions=12,
            per_screen=True,
            action_map=action_map,
            current_screen=screen,
            action_cutoff=action_cutoff,
        )
 
        model_name = f"{name}/ppo_screen_{screen}"
        model = self.create_model(
            model_name, env, "PPO",
            verbose=1,
            n_steps=n_steps,
            batch_size=64,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            vf_coef=vf_coef,
            clip_range=clip_range,
            target_kl=target_kl, #added this. default is none
            policy_kwargs={"net_arch": [256, 256]}
        )
 
        bc_state = torch.load(bc_model_path)
        print(f"BC input layer shape: {bc_state['net.0.weight'].shape}")
        bc.transfer_weights_to_ppo(model, bc_model_path)
        #self.pretrain_value_function(model, self.X_by_screen[screen], per_screen=True)
 
        self.overwrite_model(model_name, model)
        print(f"Screen {screen} PPO model saved to {self.model_direc}{model_name}")
        return env

    def train_model_one_screen(self, folder_name, screen, total_timesteps=500000, freeze_updates=0):
        model_path = f"{self.model_direc}{folder_name}/ppo_screen_{screen}"
        if not os.path.exists(model_path + ".zip"):
            print(f"No model found for screen {screen}, stopping.")
            return

        model = self.load_model(folder_name, screen=screen, only_agent=True)
        model.env.envs[0].env.expected_screen = screen
        model.env.envs[0].env.total_screen_actions = 0

        actual_screen = model.env.envs[0].env.read_gamedata()["current_screen"]
        if actual_screen != screen:
            print(f"Warning: expected screen {screen}, got {actual_screen}. Teleporting...")
            model.env.envs[0].env.teleport(screen)

        try:
            freeze_callback = FreezePolicyCallback(freeze_updates=freeze_updates)
            jk_callback = JumpKingCallback()
            callbacks = CallbackList([freeze_callback, jk_callback])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"{self.model_direc}{folder_name}/ppo_screen_{screen}_log/{timestamp}/"
            logger = configure(log_path, ["stdout", "csv"])
            model.set_logger(logger)

            model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=True,
                callback=callbacks
            )
            print(f"Screen {screen} training complete.")

        except KeyboardInterrupt:
            print(f"Interrupted on screen {screen}, saving...")

        finally:
            self.reset_keys()
            model.env.envs[0].env.reset_keys()
            self.overwrite_model(f"{folder_name}/ppo_screen_{screen}", model)
            print(f"Screen {screen} model saved.")

    def play_game_per_screen(self, start_screen=0):
        current_screen = start_screen
        
        while True:
            print(f"\n--- Loading model for screen {current_screen} ---")
            
            model_path = f"{self.model_direc}/screen{current_screen}/ppo_screen_{current_screen}"
            if not os.path.exists(model_path + ".zip"):
                print(f"No model found for screen {current_screen}, stopping.")
                break

            time.sleep(0.3)
            model = self.load_model(f"screen{current_screen}", screen=current_screen, only_agent=True)
            
            model.env.envs[0].env.play = True
            model.policy.set_training_mode(False)
            model.env.envs[0].env.expected_screen = current_screen

            actual_screen = model.env.envs[0].env.read_gamedata()["current_screen"]
            print(f"Loaded model for screen: {current_screen}, actual screen: {actual_screen}")

            if actual_screen != current_screen:
                print(f"Screen mismatch — switching to screen {actual_screen}")
                current_screen = actual_screen
                continue

            try:
                obs = model.env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True) #deterministic=True
                    obs, reward, done, _ = model.env.step(action)

                new_screen = model.env.envs[0].env.current_screen
                if new_screen > current_screen:
                    print(f"Transitioning from screen {current_screen} to {new_screen}")
                    current_screen = new_screen
                else:
                    print(f"Fell back to screen {new_screen}, retrying screen {current_screen}")

            except KeyboardInterrupt:
                self.reset_keys()
                model.env.envs[0].env.reset_keys()
                print(f"Interrupted on screen {current_screen}")
                break      
  
JK = JumpKingRL()
parser = RecordingParser()
records = parser.load_recording()
screen = 21
JK.create_BC_screen(f"screen{screen}_dummy", screen=screen, records=records)
env = JK.create_RL_screen(f"screen{screen}_dummy", screen=screen, action_cutoff=200, n_steps=64, n_epochs=5, ent_coef=0.30, target_kl=0.04, learning_rate=0.0001, episode_mode=EpisodeMode.SCREEN)
JK.train_model_one_screen(f"screen{screen}_dummy", screen=screen, freeze_updates=0)


#JK.play_game_per_screen(start_screen=0)


# parser = RecordingParser()
# action_map = parser.get_screen_action_map(5)
# print (action_map)

# env = JumpKingEnv(episode_mode="action", max_episode_actions=8, spacing=0.05)
# bc = BehavioralCloning()
# parser = RecordingParser()
# records = parser.load_recording()
# records = parser.clean_actions(records, increment=0.025)
# by_screen = parser.split_recording_by_screen(records)

# _, actions = parser.separate_actions_and_state(by_screen[31])
# left_counts, right_counts, space_counts = parser.tally_actions(actions)
# print (f"left counts: {left_counts}")
# print (f"right counts: {right_counts}")
# print (f"space counts: {space_counts}")


#UNCOMMENT TO ADD PLATFORM DATA
# JK = JumpKingRL()
# max_episode_actions = 8
# env = JumpKingEnv(episode_mode=EpisodeMode.ACTION_HEIGHT, max_episode_actions=max_episode_actions, dummyenv=True)
# n_steps=64
# callback = JumpKingCallback()
# platform_parser = PlatformParser()
# #create, load, train model
# #model = JK.create_model("dummy", env, "PPO", verbose=1, n_steps=n_steps)
# model = JK.load_model("dummy")
# JK.train_model("dummy", model, total_timesteps=10000, callback=callback) #default is 2k

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


