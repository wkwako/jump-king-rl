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

import JumpKingEnv

def save_and_quit():
    print("Saving model and quitting...")
    model.save(MODEL_NAME)
    exit()

MODEL_NAME = "jumpking_ppo"

env = JumpKingEnv()

#if model exists, load it
if os.path.exists(MODEL_NAME + ".zip"):
    print ("Loading existing model...")
    model = PPO.load(MODEL_NAME, env=env)

#if it doesn't, create a new model
else:
    print ("Creating new model...")
    model = PPO("MlpPolicy", env, verbose=1)

#save and quit if q is pressed
keyboard.add_hotkey("ctrl+q", save_and_quit)

try:
    model.learn(total_timesteps=20)
    print ("Training complete. Saving...")
    model.save(MODEL_NAME)

except KeyboardInterrupt:
    print ("Interrupted. Saving...")
    model.save(MODEL_NAME)

env.close()