import math
from typing import Optional, Union
import time
import keyboard
import pydirectinput
import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class JumpKingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self):
        #define action space here, tuples of buttons with presses for each
        #pass into function that activates it via pydirectinput
        self.action_space = self.init_action_space()
        self.state = None #TODO: set this
        pass

    def step(self, action):
        reward = None #TODO: remove this and set it elsewhere
        terminated = None #TODO: remove this and set it elsewhere

        #map action to pydirectinput here        

        #define rewards
        #reward screen transitions the most
        #reward full jumps that land higher more
        #reward jumps that land higher more
        #reward jumps that land at the same height but in a different location
        #punish falling
        #reward moving right after a big fall so the player stands up?

        #define a function that sets termination bool

        #define state with location, etc.

        #state, reward, if the episode is terminated, truncation, and info dict
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    def read_from_file():
        pass

    def init_action_space():
        #left, right, spacebar
        action_space = []

        #walk left
        action_space.append([0.2, 0, 0])

        #walk small left
        action_space.append([0.1, 0, 0])

        #walk right
        action_space.append([0, 0.2, 0])

        #walk small right
        action_space.append([0, 0.1, 0])

        #jump up
        action_space.append([0, 0, 0.6])

        #jump right, 0.1s
        action_space.append([0, 0.1, 0.1])

        #jump right, 0.2s
        action_space.append([0, 0.2, 0.2])

        #jump right, 0.3s
        action_space.append([0, 0.3, 0.3])

        #jump right, 0.4s
        action_space.append([0, 0.4, 0.4])

        #jump right, 0.5s
        action_space.append([0, 0.5, 0.5])

        #jump right, 0.6s
        action_space.append([0, 0.6, 0.6])

        #jump left, 0.1s
        action_space.append([0.1, 0, 0.1])

        #jump left, 0.2s
        action_space.append([0.2, 0, 0.2])

        #jump left, 0.3s
        action_space.append([0.3, 0, 0.3])

        #jump left, 0.4s
        action_space.append([0.4, 0, 0.4])

        #jump left, 0.5s
        action_space.append([0.5, 0, 0.5])

        #jump left, 0.6s
        action_space.append([0.6, 0, 0.6])

        return action_space