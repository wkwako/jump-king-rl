import math
from typing import Optional, Union
import time
import keyboard
import pydirectinput
import numpy as np
import signal
import os
import json

import sys
sys.path.append("C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL")

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from JumpKingEnv import JumpKingEnv

class EpisodeMode:
    ACTION = "action"
    SCREEN = "screen"
    HEIGHT = "height"
    ACTION_HEIGHT = "action_height"
    CURRICULUM = "curriculum"

class JumpKingRL:

    def __init__(self):
        self.model_direc = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/"

    def init_metadata(self, model):
        env = model.env.envs[0].env        
        metadata = {
            "total_jumps": 0,
            "total_timesteps": 0,
            "episode_mode": env.episode_mode,
            "hyperparameters": {
                "max_episode_actions": env.max_episode_actions,
                "n_steps": model.n_steps,
                "new_screen_reward": env.new_screen_reward,
                "jump_penalty": env.jump_penalty,
                "max_jump_bonus": env.max_jump_bonus,
                "grid_size": env.grid_size,
                "exploration_reward": env.exploration_reward,
            },
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
        
        # load model
        print ("Loading existing model...")
        model = PPO.load(self.model_direc + name, env=env)

        logger = configure(self.model_direc + name + "_log/", ["stdout", "csv"])
        model.set_logger(logger)
        
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

    def create_model(self, name, env, n_steps, verbose=1, model_name="MlpPolicy"):
        #creates a new model. will throw an error if model_name already exists
        model_path = self.model_direc + name
        if os.path.exists(model_path + ".zip"):
            raise FileExistsError("This model already exists. Please use a different name, delete it, or use the overwrite_model() function.")
        
        print ("Creating new model...")
        model = PPO(model_name, env, verbose=verbose, n_steps=n_steps)

        logger = configure(model_path + "_log/", ["stdout", "csv"])
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

    def train_model(self, name, model, total_timesteps):
        print ("Starting training...")
        try:
            #train the model until complete or interrupted
            model.learn(total_timesteps=total_timesteps)
            print ("Training complete. Saving...")
        
        except KeyboardInterrupt:
            print ("Interrupted. Saving model and metadata...")

        finally:
            #overwrite what we have and reset variables
            self.overwrite_model(name, model)
            env = model.env.envs[0].env
            env.jump_counter_metadata = 0
            
        
#create model first
JK = JumpKingRL()
max_episode_actions = 5
env = JumpKingEnv(episode_mode=EpisodeMode.ACTION_HEIGHT, max_episode_actions=max_episode_actions)
n_steps=128
 
 #JK.delete_model("jk_ppo_action_height1")

model = JK.create_model("jk_ppo_action4", env, n_steps)
model = JK.load_model("jk_ppo_action4")

JK.train_model("jk_ppo_action4", model, total_timesteps=30000)

env.close()


#then train it

#MODEL_PATH = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/jumpking_ppo"

# max_episode_actions = 10
# env = JumpKingEnv(episode_mode=EpisodeMode.ACTION, max_episode_actions=max_episode_actions)
# OVERWRITE_MODEL = True
# n_steps = 512

# #if model exists, load it
# if os.path.exists(MODEL_PATH + ".zip") and OVERWRITE_MODEL is False:
#     print ("Loading existing model...")
#     model = PPO.load(MODEL_PATH, env=env)

# #if it doesn't, create a new model
# else:
#     print ("Creating new model...")
#     model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps)

# #save and quit if q is pressed
# #keyboard.add_hotkey("ctrl+q", save_and_quit)

# try:
#     model.learn(total_timesteps=1000)
#     print ("Training complete. Saving...")
#     model.save(MODEL_PATH)

# except KeyboardInterrupt:
#     print ("Interrupted. Saving...")
#     model.save(MODEL_PATH)



#env.close()


