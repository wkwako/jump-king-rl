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

    def __init__(self, episode_mode, max_jumps=10):
        self.action_map = self.init_action_map()
        self.action_space = spaces.Discrete(len(self.action_map))
        self.state = None
        self.gamedata = None
        self.gamedata_prev = None
        self.new_screen_reward = 300
        self.same_level_reward = 5
        self.jumped = False
        self.sleep_time = 0.1
        self.jump_counter_metadata = 0
        self.jump_penalty = -1
        self.max_jump_bonus = 1.40

        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf]), dtype=np.float32
)

    def step(self, action):
        reward = 0

        #duplicate gamedata so we have previous info after action
        self.gamedata_prev = list(self.gamedata)

        #executes action
        self.execute_action(action)

        #reads gamedata
        self.gamedata = self.read_gamedata()
        x, y, vel_x, vel_y, is_on_ground, current_screen, total_screens, jump_frames, jump_percentage, max_height_this_jump = self.gamedata   
        x_prev, y_prev, vel_x_prev, vel_y_prev, is_on_ground_prev, current_screen_prev, total_screens_prev, jump_frames_prev, jump_percentage_prev, max_height_this_jump_prev = self.gamedata_prev

        #define rewards
        #if we go up to a new screen, very large reward
        if current_screen > current_screen_prev:
            reward += self.new_screen_reward

        #if we landed higher, moderate reward. if it was a max jump, bonus reward. or punishes falling
        if y > y_prev or y < y_prev:
            reward += self.new_height_reward(y, y_prev, jump_percentage)

        #if we jumped and stayed at the same y level, small reward
        # if self.jumped and y == y_prev:
        #     reward += 5
        
        #reward jumps that land at the same height but in a different location?
        #reward moving right after a big fall so the player stands up?

        

        #define state
        self.state = (x, y, vel_x, vel_y)

        #if we jumped, terminate episode
        if self.jumped:
            self.jump_counter_metadata += 1
            #print ("jumped")
            #small negative reward to encourage walking sometimes
            reward += self.jump_penalty
            terminated = True
        
        #if we walked, continue episode
        else:
            #print ("walked")
            terminated = False

        #reset jumped boolean
        self.jumped = False

        #state, reward, if the episode is terminated, truncation, and info dict
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        self.gamedata = self.read_gamedata()
        self.gamedata_prev = list(self.gamedata)

        x, y, vel_x, vel_y = self.gamedata[:4]
        self.state = (x, y, vel_x, vel_y)

        return np.array(self.state, dtype=np.float32), {}

    def new_height_reward(self, y, y_prev, jump_percentage):
        #if jump was max height, increase reward by 20%

        reward = y - y_prev

        if jump_percentage == 1 and y > y_prev:
            reward *= self.max_jump_bonus

        return reward

    def read_gamedata(self):
        """Reads the game state and outputs a list with 9 parts.
           [x, y, velX, velY, isOnGround, currentScreen, totalScreens, jumpFrames, jumpPercentage, maxHeightThisJump]"""
        while True:
            try:
                with open("C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/gamestate.txt") as f:
                    content = f.read()
                parts = content.split(",")
                if len(parts) == 10:
                    return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), parts[4].strip().lower() == "true", int(parts[5]), int(parts[6]), int(parts[7]), float(parts[8]), float(parts[9])]
            except:
                pass
            time.sleep(self.sleep_time)
                
    def execute_action(self, action):
        #map action index to keypresses
        left, right, jump = self.action_map[action]

        #walking
        if not jump:
            if left:
                self.key_press("left", left)
            else:
                self.key_press("right", right)

        #jumping
        else:
            self.jumped = True
            #jumping straight up - removed for now
            #if not left and not right:
                #self.key_press("space", jump)

            #jumping left
            if left:
                self.key_press("space", jump, "left")

            #jumping right
            else:
                #very small sleep timer to ensure space is released first
                time.sleep(0.05)
                self.key_press("space", jump, "right")

    
    def key_press(self, key, duration, key2=None):
        #key is released first when key2 is present so directional jumps occur
        pydirectinput.keyDown(key)

        if key2:
            pydirectinput.keyDown(key2)

        time.sleep(duration)

        pydirectinput.keyUp(key)

        if key2:
            pydirectinput.keyUp(key2)

    def init_action_map(self):
        #left, right, spacebar
        action_map = []

        #walk left
        action_map.append([0.2, 0, 0])

        #walk small left
        action_map.append([0.1, 0, 0])

        #walk right
        action_map.append([0, 0.2, 0])

        #walk small right
        action_map.append([0, 0.1, 0])

        #jump right, 0.1s
        action_map.append([0, 0.1, 0.1])

        #jump right, 0.2s
        action_map.append([0, 0.2, 0.2])

        #jump right, 0.3s
        action_map.append([0, 0.3, 0.3])

        #jump right, 0.4s
        action_map.append([0, 0.4, 0.4])

        #jump right, 0.5s
        action_map.append([0, 0.5, 0.5])

        #jump right, 0.6s
        action_map.append([0, 0.6, 0.6])

        #jump left, 0.1s
        action_map.append([0.1, 0, 0.1])

        #jump left, 0.2s
        action_map.append([0.2, 0, 0.2])

        #jump left, 0.3s
        action_map.append([0.3, 0, 0.3])

        #jump left, 0.4s
        action_map.append([0.4, 0, 0.4])

        #jump left, 0.5s
        action_map.append([0.5, 0, 0.5])

        #jump left, 0.6s
        action_map.append([0.6, 0, 0.6])

        return action_map