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

import sys
sys.path.append("C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL")

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.logger import configure

from JumpKingEnv import JumpKingEnv
from stable_baselines3.common.callbacks import BaseCallback

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
            
def human_readable_platforms(platforms):
    print (f"left wall: {platforms[0][0]}")
    print (f"right wall: {platforms[0][1]}")
    print (f"ceiling: {platforms[0][2]}")
    print (f"left edge of current platform: {platforms[0][3]}")
    print (f"right edge of current platform: {platforms[0][4]}")
    print ("Sector info (relative y, length)")
    print (f"up left: {platforms[1][0]}")
    print (f"up right: {platforms[1][1]}")
    print (f"left: {platforms[1][2]}")
    print (f"right: {platforms[1][3]}")
    print (f"next screen, up left: {platforms[1][4]}")
    print (f"next screen, up right: {platforms[1][5]}")

#create model first
JK = JumpKingRL()
max_episode_actions = 4
env = JumpKingEnv(episode_mode=EpisodeMode.ACTION_HEIGHT, max_episode_actions=max_episode_actions)
n_steps=64
callback = JumpKingCallback()
platform_parser = PlatformParser()

#create, load, train model. create not needed if already created
#model = JK.create_model("jk_ppo_temp1", env, "PPO", verbose=1, n_steps=n_steps)
model = JK.load_model("jk_ppo_temp1")
JK.train_model("jk_ppo_temp1", model, total_timesteps=10000, callback=callback) #default is 2k

#debug information
x, y, vel_x, vel_y, is_on_ground, current_screen = env.get_gamedata_old()
platform_parser.update_registry(current_screen, (x, y))
pos_state_data = list(platform_parser.parse_result[0])
sector_state_data = platform_parser.process_registry(current_screen, (x, y))
pos_state = [x, y, current_screen]
state = np.array(pos_state + pos_state_data + sector_state_data, dtype=np.float32)
print(f"x={x:.1f}, y={y:.1f}, screen={current_screen}")
print(f"left_wall={pos_state_data[0]}, right_wall={pos_state_data[1]}, ceiling={pos_state_data[2]}")
print(f"platform_x_start={pos_state_data[3]}, platform_x_end={pos_state_data[4]}")
print(f"sectors: {sector_state_data}")
#print(f"full state ({len(state)} values): {state}")

# model = JK.create_model("jk_dqn_test2", env, "DQN", learning_starts=1000, 
#                         exploration_fraction=0.8,
#                         exploration_initial_eps=1.0,
#                         exploration_final_eps=0.05,
#                         batch_size=64)

#env.close()


