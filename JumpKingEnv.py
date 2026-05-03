import math
from typing import Optional, Union
import time
import keyboard
import pydirectinput
import numpy as np
import json

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from PlatformParser import PlatformParser
from Ray import Ray

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
        self.jump_penalty = 0
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
        self.ray_caster = Ray(max_distance=400, step_size=8)

        self.x = self.y = self.vel_x = self.vel_y = None
        self.is_on_ground = self.current_screen = self.total_screens = None
        self.jump_frames = self.jump_percentage = self.max_height_this_jump = None
        self.is_on_ice = self.is_in_snow = self.is_in_water = self.wind_velocity = None
        self.x_prev = self.y_prev = self.vel_x_prev = self.vel_y_prev = None
        self.is_on_ground_prev = self.current_screen_prev = self.total_screens_prev = None
        self.jump_frames_prev = self.jump_percentage_prev = self.max_height_this_jump_prev = None
        self.is_on_ice_prev = self.is_in_snow_prev = self.is_in_water_prev = self.wind_velocity_prev = None

        self.recent_landings = []
        self.landing_memory = 5  # how many recent landings to remember
        self.stuck_penalty = -3
        self.stuck_threshold = 10  # pixels — how close counts as "same spot"

        #sector observation space
        # self.observation_space = spaces.Box(
        #     low=np.array([-np.inf] * 20, dtype=np.float32),
        #     high=np.array([np.inf] * 20, dtype=np.float32),
        #     dtype=np.float32
        # )

        #ray observation space
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 42, dtype=np.float32),
            high=np.array([np.inf] * 42, dtype=np.float32),
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

        #release spacebar if it's being held before we choose another action
        self.reset_keys()

        #set game data into individual variables
        self.load_game_attributes()
        self.load_game_attributes_prev()

        #ray state building
        self.platform_parser.parse_result = self.platform_parser.read_platform_data((self.x, self.y), self.current_screen)
        self.ray_caster.build_ray_collision_index(self.platform_parser.current_tiles, self.platform_parser.next_tiles)
        ray_state_data = self.ray_caster.build_ray_states(num_angles=36)
        pos_state = [self.x, self.y, self.current_screen, self.is_on_ice, self.is_in_snow, self.wind_velocity]
        self.state = np.array(pos_state + ray_state_data, dtype=np.float32)

        #sector state building
        # self.platform_parser.parse_result = self.platform_parser.read_platform_data((self.x, self.y), self.current_screen)
        # # build state
        # if self.platform_parser.parse_result is not None:
        #     pos_state_data = list(self.platform_parser.parse_result[0])
        #     #pos_state_data[2] += -50  # ceiling offset
        # else:
        #     pos_state_data = [-9999, 9999, 9999, -9999, 9999]
        # sector_state_data = self.platform_parser.process_registry(self.current_screen, (self.x, self.y))
        # pos_state = [self.x, self.y, self.current_screen]
        # self.state = np.array(pos_state + pos_state_data + sector_state_data, dtype=np.float32)

        #reward calculation
        if self.current_screen > self.current_screen_prev:
            reward += self.new_screen_reward

        #if we landed higher, reward. if we landed lower, penalty
        if self.y != self.y_prev:
            #print (f"Reward for landing at new altitude: {self.new_height_reward(y, y_prev, jump_percentage)}")
            reward += self.new_height_reward()

        #penalty for repeated jumps that land in the same spot
        self.add_landing()
        if self.check_landing_cluster():
            reward += self.stuck_penalty

        #reward exploring unexplored grid cells this episode
        # cell = self.get_grid_cell(x, y)
        # if cell not in self.visited_cells:
        #     self.visited_cells.add(cell)
        #     reward += self.exploration_reward

        #increment jump count metadata
        if self.jumped:
            reward += self.jump_penalty
            self.jump_counter_metadata += 1

            #reset jump boolean after every action if it was true
            self.jumped = False

        #set terminated bool based on episode_mode
        terminated = self.set_terminated()

        # if terminated:
        #     y_start = self.gamedata_start_of_episode[1]
        #     reward += (y - y_start) / 5

        return self.state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        #time.sleep(2)
        self.gamedata = self.read_gamedata()
        self.load_game_attributes()
        self.load_game_attributes_prev()
        #self.gamedata_prev = list(self.gamedata)
        #self.visited_cells.clear()
        self.action_counter = 0
        #self.gamedata_start_of_episode = list(self.gamedata)

        #x, y, vel_x, vel_y, is_on_ground, current_screen = self.gamedata[:6]
        self.load_game_attributes()

        # reuse last state if available, otherwise use sentinels
        if self.state is not None:
            # update just x, y, current_screen in the existing state
            self.state[0] = self.x
            self.state[1] = self.y
            self.state[2] = self.current_screen
        else:
            # first ever reset - use sentinels
            #sector sentinels
            #pos_state_data = [-9999, 9999, 9999, -9999, 9999]
            #sector_state_data = [-9999] * 12
            #self.state = np.array([self.x, self.y, self.current_screen] + pos_state_data + sector_state_data, dtype=np.float32)

            #ray sentinels
            ray_state_data = [400] * 36
            self.state = np.array([self.x, self.y, self.current_screen, 0, 0, 0] + ray_state_data, dtype=np.float32)

        return self.state, {}
    
    def add_landing(self):
        if not self.jumped:
            return
        self.recent_landings.append((self.x, self.y))
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

    def set_terminated(self):
        #check for termination based on episode type

        #terminate after a certain number of actions
        if self.episode_mode == "action":
            result = self.terminate_action_episode()
        
        #terminate after reaching a new screen and landing there
        elif self.episode_mode == "screen":
            result = self.terminate_screen_episode()
        
        #terminate after landing with a positive height gain
        elif self.episode_mode == "height":
            result = self.terminate_height_episode()

        elif self.episode_mode == "action_height":
            result = self.terminate_action_episode() or self.terminate_height_episode()

        #combines multiple episode types. type1 for first n/2 screens, type2 for second n/2 screens
        elif self.episode_mode == "curriculum":
            #uses jump episodes beneath n screens, and screen episodes above n screens
            if self.current_screen < self.curriculum_screens:
                result = self.terminate_height_episode()
            else:
                result = self.terminate_screen_episode()

        return result
    
    def reset_keys(self):
        #stops pressing all keys
        pydirectinput.keyUp("space")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("left")

    def terminate_height_episode(self):
        if self.y > self.y_prev:
            return True
        return False

    def terminate_action_episode(self):
        #returns True if we should terminate the episode (based on jumps). False otherwise
        self.action_counter += 1
        if self.action_counter >= self.max_episode_actions:
            self.action_counter = 0
            return True
    
        return False

    def terminate_screen_episode(self):
        #returns True if we should terminate the episode (based on screens). False otherwise
        if self.current_screen > self.current_screen_prev:
            return True

        return False
    
    def get_grid_cell(self, x, y):
        return (int(x // self.grid_size), int(y // self.grid_size))

    def new_height_reward(self):
        #if jump was max height, increase reward by 20%

        reward = (self.y - self.y_prev) / 5

        if self.jump_percentage == 1 and self.y > self.y_prev:
            reward *= self.max_jump_bonus

        return reward

    def read_gamedata(self):
        while True:
            try:
                with open("C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/gamestate.txt") as f:
                    content = f.read()
                data = json.loads(content)
                if data.get("is_on_ground"):
                    return data
            except Exception as e:
                continue
                #print(f"error: {e}")
            time.sleep(self.sleep_time)

    def get_gamedata_old(self):
        self.gamedata = self.read_gamedata()
        self.load_game_attributes()
        return self.x, self.y, self.vel_x, self.vel_y, self.is_on_ground, self.current_screen
                
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

    def init_action_map(self, spacing=0.1):
        #left, right, spacebar
        action_map = []

        #perform no action - only used for testing
        #action_map.append([0, 0, 0])

        for t in [0.1, 0.2]:
            action_map.append([t, 0, 0])
            action_map.append([0, t, 0])

        for t in np.arange(spacing, 0.65, spacing):  # 0.05 to 0.60 inclusive
            action_map.append([0, round(t, 2), round(t, 2)])
            action_map.append([round(t, 2), 0, round(t, 2)])

        return action_map
    
    def load_game_attributes(self):
        """Unpacks current gamedata dict into instance variables."""
        self.x = self.gamedata["x"]
        self.y = self.gamedata["y"]
        self.vel_x = self.gamedata["vel_x"]
        self.vel_y = self.gamedata["vel_y"]
        self.is_on_ground = self.gamedata["is_on_ground"]
        self.current_screen = self.gamedata["current_screen"]
        self.total_screens = self.gamedata["total_screens"]
        self.jump_frames = self.gamedata["jump_frames"]
        self.jump_percentage = self.gamedata["jump_percentage"]
        self.max_height_this_jump = self.gamedata["max_height"]
        self.is_on_ice = self.gamedata["is_on_ice"]
        self.is_in_snow = self.gamedata["is_in_snow"]
        self.is_in_water = self.gamedata["is_in_water"]
        self.wind_velocity = self.gamedata["wind_velocity"]

    def load_game_attributes_prev(self):
        """Unpacks current gamedata dict into instance variables."""
        self.x_prev = self.gamedata["x"]
        self.y_prev = self.gamedata["y"]
        self.vel_x_prev = self.gamedata["vel_x"]
        self.vel_y_prev = self.gamedata["vel_y"]
        self.is_on_ground_prev = self.gamedata["is_on_ground"]
        self.current_screen_prev = self.gamedata["current_screen"]
        self.total_screens_prev = self.gamedata["total_screens"]
        self.jump_frames_prev = self.gamedata["jump_frames"]
        self.jump_percentage_prev = self.gamedata["jump_percentage"]
        self.max_height_this_jump_prev = self.gamedata["max_height"]
        self.is_on_ice_prev = self.gamedata["is_on_ice"]
        self.is_in_snow_prev = self.gamedata["is_in_snow"]
        self.is_in_water_prev = self.gamedata["is_in_water"]
        self.wind_velocity_prev = self.gamedata["wind_velocity"]