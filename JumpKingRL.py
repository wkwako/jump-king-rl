import math
from typing import Optional, Union
import time
import keyboard
import pydirectinput
import numpy as np
import signal
import os
import json

import gymnasium as gym
from stable_baselines3 import PPO

from JumpKingEnv import JumpKingEnv

#TODO: turn this into a class

MODEL_PATH = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/jumpking_ppo"

class EpisodeMode:
    ACTION = "action"
    SCREEN = "screen"
    HEIGHT = "height"
    CURRICULUM = "curriculum"

class JumpKingRL:

    def __init__(self):
        self.init_metadata()
        self.model_direc = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/"

    def init_metadata(self):
    #turn this into a class variable
        self.metadata = {
            "total_jumps": 0,
            "total_timesteps": model.num_timesteps,
            "episode_mode": EpisodeMode.JUMP,
            "max_jumps": max_jumps,
            "hyperparameters": {
                "n_steps": n_steps,
                "new_screen_reward": env.new_screen_reward,
                "jump_penalty": env.jump_penalty,
                "max_jump_bonus": env.max_jump_bonus,
            }
    }

    def load_metadata(self):
        pass

    def save_metadata(self):
        with open(MODEL_PATH + "_metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def create_model(self, model_name):
        #creates a new model. will throw an error if model_name already exists
        pass

    def overwrite_model(self, model_name):
        #deletes a model with model_name, then creates a new one in its place
        pass

    def delete_model(self, model_name):
        #deletes a model with model_name
        pass

max_jumps = 10
env = JumpKingEnv(episode_mode=EpisodeMode.ACTION, max_jumps=max_jumps)
OVERWRITE_MODEL = True
n_steps = 512

#if model exists, load it
if os.path.exists(MODEL_PATH + ".zip") and OVERWRITE_MODEL is False:
    print ("Loading existing model...")
    model = PPO.load(MODEL_PATH, env=env)

#if it doesn't, create a new model
else:
    print ("Creating new model...")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=n_steps)

#save and quit if q is pressed
#keyboard.add_hotkey("ctrl+q", save_and_quit)

try:
    model.learn(total_timesteps=10000)
    print ("Training complete. Saving...")
    model.save(MODEL_PATH)

except KeyboardInterrupt:
    print ("Interrupted. Saving...")
    model.save(MODEL_PATH)



env.close()


