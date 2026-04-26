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
        self.action_map = {}
        self.gamedata = None
        self.gamadata_prev
        self.new_screen_reward = 300
        self.jumped = False

    def step(self, action):
        reward = 0

        #duplicate gamedata so we have previous info after action
        self.gamedata_prev = list(self.gamedata)

        #executes action
        self.execute_action(action)

        #reads gamedata
        self.gamedata = None
        self.gamedata = self.read_gamedata()
        x, y, vel_x, vel_y, is_on_ground, current_screen, total_screens, jump_frames, jump_percentage, max_height_this_jump = self.gamedata   
        x_prev, y_prev, vel_x_prev, vel_y_prev, is_on_ground_prev, current_screen_prev, total_screens_prev, jump_frames_prev, jump_percentage_prev, max_height_this_jump_prev = self.gamedata_prev

        #define rewards
        #if we go up to a new screen, very large reward
        if current_screen > current_screen_prev:
            reward += self.new_screen_reward

        #if we landed higher, moderate reward. if it was a max jump, bonus reward
        if y > y_prev:
            reward += self.new_height_reward(y, y_prev, jump_percentage)

        if y == y_prev:
            pass
        
        #reward screen transitions the most
        #reward full jumps that land higher more
        #reward jumps that land higher more
        #reward jumps that land at the same height but in a different location
        #punish falling
        #reward moving right after a big fall so the player stands up?

        #define a function that sets termination bool

        #define state with location, etc.

        #terminated is always True since an episode is one jump?
        #not necessarily. action could just be walking

        if self.jumped:
            terminated = True
        
        else:
            terminated = False

        #reset jumped boolean
        self.jumped = False

        #state, reward, if the episode is terminated, truncation, and info dict
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    def new_height_reward(self, y, y_prev, jump_percentage, max_jump_bonus=1.20):
        #if jump was max height, increase reward by 20%

        reward = y - y_prev

        if jump_percentage == 1:
            reward *= max_jump_bonus

        return reward

    def read_gamedata(self):
        """Reads the game state and outputs a list with 9 parts.
           [x, y, velX, velY, isOnGround, currentScreen, totalScreens, chargeTimer, maxHeightThisJump]"""
        while True:
            try:
                with open("C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/gamestate.txt") as f:
                    content = f.read()
                parts = content.split(",")
                if len(parts) == 10:
                    return parts
            except:
                pass
                
    def execute_action(self, action):
        #map action index to keypresses
        left, right, jump = self.action_space[action]

        #walking
        if not jump:
            if left:
                self.key_press("left", left)
            else:
                self.key_press("right", right)

        #jumping
        else:
            self.jumped = True
            #jumping straight up
            if not left and not right:
                self.key_press("space", jump)

            #jumping left
            elif left:
                self.key_press("space", jump, "left")

            #jumping right
            else:
                self.key_press("space", jump, "right")

    
    def key_press(self, key, duration, key2=None):
        #key2 is released first so directional jumps occur
        pydirectinput.keyDown(key)

        if key2:
            pydirectinput.keyDown(key2)

        time.sleep(duration)

        if key2:
            pydirectinput.keyUp(key2)

        pydirectinput.keyUp(key)

    def init_action_space(self):
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
    
    def init_states(self):
        #self.x, self.y, self.
        pass