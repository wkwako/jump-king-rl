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

    def __init__(self, episode_mode, max_episode_actions=10, curriculum_screens=5):
        self.action_map = self.init_action_map()
        self.action_space = spaces.Discrete(len(self.action_map))
        self.state = None
        self.gamedata = None
        self.gamedata_prev = None
        self.new_screen_reward = 10
        self.jumped = False
        self.sleep_time = 0.1
        self.jump_counter_metadata = 0
        self.jump_penalty = -0.25
        self.max_jump_bonus = 1.50
        self.episode_mode = episode_mode
        self.max_episode_actions = max_episode_actions
        self.action_counter = 0
        self.curriculum_screens = curriculum_screens
        self.visited_cells = set()
        self.grid_size = 8
        self.exploration_reward = 0.5

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action):
        #set reward to 0 for this step
        reward = 0

        #duplicate gamedata so we have previous info after action
        self.gamedata_prev = list(self.gamedata)

        #executes action. pauses here until the action is complete (we finish walking or release the jump button)
        self.execute_action(action)

        #reads gamedata. pauses here until the character lands
        self.gamedata = self.read_gamedata()

        #release spacebar if it's beind held before we choose another action
        self.reset_keys()

        #set game data into individual variables
        x, y, vel_x, vel_y, is_on_ground, current_screen, total_screens, jump_frames, jump_percentage, max_height_this_jump = self.gamedata   
        x_prev, y_prev, vel_x_prev, vel_y_prev, is_on_ground_prev, current_screen_prev, total_screens_prev, jump_frames_prev, jump_percentage_prev, max_height_this_jump_prev = self.gamedata_prev

        #reward calculation
        #must land for current_screen to be registered as above previous screen
        if current_screen > current_screen_prev:
            reward += self.new_screen_reward

        #if we landed higher, moderate reward. if it was a max jump, bonus reward. or punishes falling
        if y > y_prev or y < y_prev:
            reward += self.new_height_reward(y, y_prev, jump_percentage)

        #reward moving right after a big fall so the player stands up?

        #reward exploring unexplored grid cells this episode
        cell = self.get_grid_cell(x, y)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            reward += self.exploration_reward

        #create state tuple. this is what the agent uses to determine actions
        self.state = (x, y, vel_x, vel_y, current_screen)

        #increment jump count metadata
        if self.jumped:
            reward += self.jump_penalty
            self.jump_counter_metadata += 1

            #reset jump boolean after every action if it was true
            self.jumped = False

        #set terminated bool based on episode_mode
        terminated = self.set_terminated(current_screen, current_screen_prev, y, y_prev)

        #state, reward, if the episode is terminated, truncation, and info dict
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    def set_terminated(self, current_screen, current_screen_prev, y, y_prev):
        #check for termination based on episode type

        #terminate after a certain number of actions
        if self.episode_mode == "action":
            result = self.terminate_action_episode()
        
        #terminate after reaching a new screen and landing there
        elif self.episode_mode == "screen":
            result = self.terminate_screen_episode(current_screen, current_screen_prev)
        
        #terminate after landing with a positive height gain
        elif self.episode_mode == "height":
            result = self.terminate_height_episode(y, y_prev)

        elif self.episode_mode == "action_height":
            result = self.terminate_action_episode() or self.terminate_height_episode(y, y_prev)

        #combines multiple episode types. type1 for first n/2 screens, type2 for second n/2 screens
        elif self.episode_mode == "curriculum":
            #uses jump episodes beneath n screens, and screen episodes above n screens
            if current_screen < self.curriculum_screens:
                result = self.terminate_height_episode(y, y_prev)
            else:
                result = self.terminate_screen_episode(current_screen, current_screen_prev)

        return result
    

    def reset_keys(self):
        #stops pressing all keys
        pydirectinput.keyUp("space")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("left")

    def terminate_height_episode(self, y, y_prev):
        if y > y_prev:
            return True
        return False

    def terminate_action_episode(self):
        #returns True if we should terminate the episode (based on jumps). False otherwise
        self.action_counter += 1
        if self.action_counter >= self.max_episode_actions:
            self.action_counter = 0
            return True
    
        return False

    def terminate_screen_episode(self, current_screen, current_screen_prev):
        #returns True if we should terminate the episode (based on screens). False otherwise
        if current_screen > current_screen_prev:
            return True

        return False
    
    def get_grid_cell(self, x, y):
        return (int(x // self.grid_size), int(y // self.grid_size))

    def reset(self, seed=None, options=None):
        self.gamedata = self.read_gamedata()
        self.gamedata_prev = list(self.gamedata)
        self.visited_cells.clear()
        self.action_counter = 0

        x, y, vel_x, vel_y, is_on_ground, current_screen = self.gamedata[:6]
        self.state = (x, y, vel_x, vel_y, current_screen)

        return np.array(self.state, dtype=np.float32), {}

    def new_height_reward(self, y, y_prev, jump_percentage):
        #if jump was max height, increase reward by 20%

        reward = (y - y_prev) / 5

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
        action_map.append([0.05, 0, 0])

        #walk right
        action_map.append([0, 0.2, 0])

        #walk small right
        action_map.append([0, 0.05, 0])

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