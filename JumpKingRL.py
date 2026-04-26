import math
from typing import Optional, Union
import time
import keyboard
import pydirectinput
import numpy as np
import signal
import os

import gymnasium as gym
from stable_baselines3 import PPO

from JumpKingEnv import JumpKingEnv

MODEL_PATH = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/models/jumpking_ppo"


# def save_and_quit():
#     print("Saving model and quitting...")
#     model.save(MODEL_PATH)
#     exit()


env = JumpKingEnv()
OVERWRITE_MODEL = True

#if model exists, load it
if os.path.exists(MODEL_PATH + ".zip") and OVERWRITE_MODEL is False:
    print ("Loading existing model...")
    model = PPO.load(MODEL_PATH, env=env)

#if it doesn't, create a new model
else:
    print ("Creating new model...")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512)

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