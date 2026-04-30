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

from PlatformParser import PlatformParser

class JumpKingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self, episode_mode, max_episode_actions=10, curriculum_screens=5):
        self.teleport_path = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/teleport.txt"
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
        self.max_jump_bonus = 1.0
        self.episode_mode = episode_mode
        self.max_episode_actions = max_episode_actions
        self.action_counter = 0
        self.curriculum_screens = curriculum_screens
        self.visited_cells = set()
        self.grid_size = 20
        self.exploration_reward = 0.1
        self.gamedata_start_of_episode = None
        self.platform_parser = PlatformParser()

        self.recent_landings = []
        self.landing_memory = 5  # how many recent landings to remember
        self.stuck_penalty = -3
        self.stuck_threshold = 10  # pixels — how close counts as "same spot"

        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 20, dtype=np.float32),
            high=np.array([np.inf] * 20, dtype=np.float32),
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
        #self.reset_keys()

        #set game data into individual variables
        x, y, vel_x, vel_y, is_on_ground, current_screen, total_screens, jump_frames, jump_percentage, max_height_this_jump = self.gamedata   
        x_prev, y_prev, vel_x_prev, vel_y_prev, is_on_ground_prev, current_screen_prev, total_screens_prev, jump_frames_prev, jump_percentage_prev, max_height_this_jump_prev = self.gamedata_prev

        # update registry and get state
        self.platform_parser.update_registry(current_screen, (x, y))

        if self.platform_parser.parse_result is not None:
            pos_state_data = list(self.platform_parser.parse_result[0])
        else:
            pos_state_data = [-9999, 9999, 9999, -9999, 9999]

        pos_state_data = list(self.platform_parser.parse_result[0])
        pos_state_data[2] += -50
        sector_state_data = self.platform_parser.process_registry(current_screen, (x, y))
        #ceiling_data = sector_state_data[0][2] - 50

        pos_state = [x, y, current_screen]

        #time.sleep(0.5)

        self.state = np.array(pos_state + pos_state_data + sector_state_data, dtype=np.float32)

        #reward calculation
        #must land for current_screen to be registered as above previous screen
        if current_screen > current_screen_prev:
            #print (f"Reward for new screen: {self.new_screen_reward}")
            reward += self.new_screen_reward

        #if we landed higher, reward. if we landed lower, punish
        if y > y_prev or y < y_prev:
            #print (f"Reward for landing at new altitude: {self.new_height_reward(y, y_prev, jump_percentage)}")
            reward += self.new_height_reward(y, y_prev, jump_percentage)

        #punish repeated jumps that land in the same spot
        self.add_landing(x, y)
        if self.check_landing_cluster():
            reward += self.stuck_penalty

        #reward moving right after a big fall so the player stands up?

        #reward exploring unexplored grid cells this episode
        # cell = self.get_grid_cell(x, y)
        # if cell not in self.visited_cells:
        #     self.visited_cells.add(cell)
        #     reward += self.exploration_reward

        #create state tuple. this is what the agent uses to determine actions
        #self.state = (x, y, vel_x, vel_y, current_screen)

        #increment jump count metadata
        if self.jumped:
            reward += self.jump_penalty
            self.jump_counter_metadata += 1

            #reset jump boolean after every action if it was true
            self.jumped = False

        #set terminated bool based on episode_mode
        terminated = self.set_terminated(current_screen, current_screen_prev, y, y_prev)

        # if terminated:
        #     y_start = self.gamedata_start_of_episode[1]
        #     reward += (y - y_start) / 5

        #print (f"State: {self.state}")

        #state, reward, if the episode is terminated, truncation, and info dict
        return self.state, reward, terminated, False, {}
    
    def add_landing(self, x, y):
        if not self.jumped:
            return
        self.recent_landings.append((x, y))
        if len(self.recent_landings) > self.landing_memory:
            self.recent_landings.pop(0)

    def check_landing_cluster(self):
        if len(self.recent_landings) < self.landing_memory:
            return False
        
        xs = [l[0] for l in self.recent_landings]
        ys = [l[1] for l in self.recent_landings]
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)
        
        return x_spread < self.stuck_threshold and y_spread < self.stuck_threshold

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
        #time.sleep(2)
        self.gamedata = self.read_gamedata()
        self.gamedata_prev = list(self.gamedata)
        self.visited_cells.clear()
        self.action_counter = 0
        self.gamedata_start_of_episode = list(self.gamedata)

        x, y, vel_x, vel_y, is_on_ground, current_screen = self.gamedata[:6]

        # reuse last state if available, otherwise use sentinels
        if self.state is not None:
            # update just x, y, current_screen in the existing state
            self.state[0] = x
            self.state[1] = y
            self.state[2] = current_screen
        else:
            # first ever reset - use sentinels
            pos_state_data = [-9999, 9999, 9999, -9999, 9999]
            sector_state_data = [-9999] * 12
            self.state = np.array([x, y, current_screen] + pos_state_data + sector_state_data, dtype=np.float32)

        return self.state, {}

    def new_height_reward(self, y, y_prev, jump_percentage):
        #if jump was max height, increase reward by 20%

        reward = (y - y_prev) / 5

        if jump_percentage == 1 and y > y_prev:
            reward *= self.max_jump_bonus

        return reward

    def read_gamedata(self):
        """Blocks until the player is on the ground (action fully resolved), then returns state."""
        while True:
            try:
                with open("C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/gamestate.txt") as f:
                    content = f.read()
                parts = content.split(",")
                if len(parts) == 10:
                    is_on_ground = parts[4].strip().lower() == "true"
                    if is_on_ground:
                        return [
                            float(parts[0]), float(parts[1]),
                            float(parts[2]), float(parts[3]),
                            True,
                            int(parts[5]), int(parts[6]),
                            int(parts[7]), float(parts[8]), float(parts[9])
                        ]
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
            elif right:
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

        #perform no action - only used for testing
        #action_map.append([0, 0, 0])

        #walk left
        action_map.append([0.2, 0, 0])

        #walk small left
        action_map.append([0.05, 0, 0])

        #walk right
        action_map.append([0, 0.2, 0])

        #walk small right
        action_map.append([0, 0.05, 0])

        #jump right, 0.05s
        action_map.append([0, 0.05, 0.05])

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

        #jump left, 0.05s
        action_map.append([0.05, 0, 0.05])

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