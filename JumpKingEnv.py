import math
from typing import Optional, Union
import time
import keyboard
import pydirectinput
pydirectinput.PAUSE = 0.01
import numpy as np
import json
import os
import random

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from PlatformParser import PlatformParser
from RecordingParser import RecordingParser
from Ray import Ray
import static_variables

class ScreenTransitionException(Exception):
    pass

class JumpKingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self, episode_mode, max_episode_actions=10, curriculum_screens=5, spacing=0.05, per_screen=False, action_map=None, current_screen=None):
        self.teleport_path = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/teleport.txt"
        self.gamestate_path = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/gamestate.txt"
        self.spacing = spacing
        #self.action_map = self.init_action_map()
        self.state = None
        self.gamedata = None
        #self.gamedata_prev = None
        self.new_screen_reward_val = 150
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
        self.recording_parser = RecordingParser()
        self.total_screen_actions = 0
        self.expected_screen = None
        self.direction_reward = 0.01
        self.speed_reward = 100
        self.force_teleport = False

        self.recent_walk_actions = []
        self.recent_jump_actions = []
        self.action_repeat_penalty = -10
        self.action_cutoff = 22
        self.action_cutoff_penalty = -50

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
        self.per_screen = per_screen
        self.step_counter = 0

        self.pending_transition = False
        self.pending_transition_screen = None

        self.current_screen = current_screen if current_screen is not None else 0

        #is action_map is passed in, it's a per-screen agent. otherwise, normal agent
        if action_map is not None:
            self.action_map = action_map
        else:
            self.action_map = self.init_action_map()

        self.action_space = spaces.Discrete(len(self.action_map))

        #build dynamic observation space if per-screen agent
        if self.per_screen:
            self.observation_space = self.build_observation_space()
        
        else:
            #sector observation space
            self.observation_space = spaces.Box(
                low=np.array([-np.inf] * 25, dtype=np.float32),
                high=np.array([np.inf] * 25, dtype=np.float32),
                dtype=np.float32
            )

        #ray observation space
        # self.observation_space = spaces.Box(
        #     low=np.array([-np.inf] * 42, dtype=np.float32),
        #     high=np.array([np.inf] * 42, dtype=np.float32),
        #     dtype=np.float32
        # )

    def build_observation_space(self):
        if self.per_screen:
            size = self.recording_parser.get_state_size(self.current_screen)
            return spaces.Box(
                low=np.array([-np.inf] * size, dtype=np.float32),
                high=np.array([np.inf] * size, dtype=np.float32),
                dtype=np.float32
            )
        else:
            return spaces.Box(
                low=np.array([-np.inf] * 25, dtype=np.float32),
                high=np.array([np.inf] * 25, dtype=np.float32),
                dtype=np.float32
            )

    def step(self, action):
        reward = 0

        self.step_counter += 1
        print(f"--- Step {self.step_counter} ---")

        # snapshot prev values BEFORE action
        self.load_game_attributes_prev()
        
        # executes action
        #time.sleep(1.5)

        #print(f"Action selected: {action} — {self.action_map[action]}")
        prev_write_count = self.execute_action(action)

        self.action_counter += 1

        if self.per_screen:
            self.total_screen_actions += 1

        self.wait_for_landing(prev_write_count)

        # reads gamedata — pauses until character lands
        self.gamedata = self.read_gamedata()

        # release spacebar if held
        self.reset_keys()

        # set game data into individual variables
        self.load_game_attributes()

        self.state = self.build_state_per_screen() if self.per_screen else self.build_state()

        #provides a small bonus in direction of progress
        reward += self.get_direction_reward()
        reward += self.get_goal_proximity_reward()

        # height reward for all agents
        height_reward = self.new_height_reward()
        if height_reward != 0:
            print(f"Height reward/penalty: {height_reward:.2f}")
            reward += height_reward

        reward += self.new_screen_reward()

        # penalties
        reward += self.check_tent_penalty()
        reward += self.check_alternating_walk_penalty(action)
        reward += self.check_repeated_jump_penalty(action)

        # stuck penalty
        self.add_landing()
        if self.check_landing_cluster():
            reward += self.stuck_penalty

        # jump penalty/metadata
        if self.jumped:
            reward += self.jump_penalty
            self.jump_counter_metadata += 1
            self.jumped = False

        # termination
        terminated = self.set_terminated()

        if self.action_counter >= self.action_cutoff:
            reward += self.action_cutoff_penalty
            terminated = True
            self.force_teleport = True
            print (f"Action cutoff penalty: {self.action_cutoff_penalty}")

        if terminated:
            print ("--- EPISODE END ---")

        #test this. gives bonus reward for faster screen completion
        if terminated and self.episode_mode == "screen" and self.current_screen > self.current_screen_prev:
            reward += self.speed_reward / self.action_counter
            print(f"Speed bonus: {self.speed_reward / self.action_counter:.2f} ({self.action_counter} actions)")

        return self.state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.gamedata = self.read_gamedata()
        self.load_game_attributes()
        
        # only teleport if we're on the wrong screen
        if self.per_screen and (self.current_screen != self.expected_screen or self.force_teleport):
            self.teleport(self.expected_screen)
            self.gamedata = self.read_gamedata()
            self.load_game_attributes()
            self.force_teleport = False
        
        self.load_game_attributes_prev()
        self.action_counter = 0
        self.recent_walk_actions = []
        self.recent_jump_actions = []
        self.pending_transition = False
        self.pending_transition_screen = None

        if self.per_screen:
            self.state = self.build_state_per_screen()
        else:
            if self.state is not None:
                self.state[0] = self.x
                self.state[1] = self.y
                self.state[2] = self.current_screen
            else:
                pos_state_data = [-9999, 9999, 9999, -9999, 9999]
                sector_state_data = [-9999] * 12
                pos_state = [self.x, self.y, self.current_screen, 0, 0, 0, 0, 0]
                self.state = np.array(pos_state + pos_state_data + sector_state_data, dtype=np.float32)

        return self.state, {}
    
    def new_screen_reward(self):
        if self.current_screen > self.current_screen_prev:
            return self.new_screen_reward_val
        elif self.current_screen < self.current_screen_prev:
            return -50
        return 0
    
    def get_goal_proximity_reward(self):
        next_screen = self.expected_screen + 1
        if next_screen not in static_variables.SCREEN_START_POSITIONS:
            return 0
        
        goal_x, goal_y, _ = static_variables.SCREEN_START_POSITIONS[next_screen]
        
        # distance to goal in current and previous position
        curr_dist = math.sqrt((self.x - goal_x)**2 + (-self.y - goal_y)**2)
        prev_dist = math.sqrt((self.x_prev - goal_x)**2 + (-self.y_prev - goal_y)**2)
        
        # reward reduction in distance, don't punish increase
        improvement = prev_dist - curr_dist
        if improvement > 0:
            goal_proximity_reward = improvement * 0.02
            #print (f"goal proximity reward: {goal_proximity_reward}")
            return goal_proximity_reward
        return 0
    
    def teleport(self, screen):
        x, y, radius = static_variables.SCREEN_START_POSITIONS[screen]
        x += random.randint(-radius, radius)
        
        temp_path = self.teleport_path + ".tmp"
        
        # write to temp file
        with open(temp_path, 'w') as f:
            f.write(f"{x},{y}")
        
        # retry rename if access denied
        for attempt in range(50):
            try:
                os.replace(temp_path, self.teleport_path)
                break
            except PermissionError:
                time.sleep(0.1)
        
        time.sleep(0.2)
        self.gamedata = self.read_gamedata()
        self.load_game_attributes()
        
        if self.current_screen != screen:
            print(f"Teleport warning: expected screen {screen}, got {self.current_screen}")
        
    def get_direction_reward(self):
        direction = static_variables.SCREEN_PROGRESS_DIRECTION.get(self.current_screen, None)
        if direction is None:
            return 0
        
        dx = self.x - self.x_prev
        
        if direction == "right" and dx > 0:
            return dx * self.direction_reward
        elif direction == "left" and dx < 0:
            return abs(dx) * self.direction_reward
        return 0
    
    def check_alternating_walk_penalty(self, action):
        left_walks = {(t, 0, 0) for t in [0.1, 0.2]}
        right_walks = {(0, t, 0) for t in [0.1, 0.2]}
        
        current = tuple(self.action_map[action])
        is_walk = current in left_walks or current in right_walks
        
        if not is_walk:
            self.recent_walk_actions = []
            return 0
        
        self.recent_walk_actions.append(current)
        if len(self.recent_walk_actions) > 6:
            self.recent_walk_actions.pop(0)
        
        if len(self.recent_walk_actions) == 6:
            # check alternating pattern
            if all(self.recent_walk_actions[i] != self.recent_walk_actions[i+1] for i in range(5)):
                print (f"penalty for alternating walks: {self.action_repeat_penalty}")
                return self.action_repeat_penalty
        return 0
    
    def check_repeated_jump_penalty(self, action):
        current = tuple(self.action_map[action])
        is_jump = current[2] > 0  # space > 0 means it's a jump
        
        if not is_jump:
            return 0
        
        self.recent_jump_actions.append(current)
        if len(self.recent_jump_actions) > 4:
            self.recent_jump_actions.pop(0)
        
        if len(self.recent_jump_actions) == 4:
            if len(set(self.recent_jump_actions)) == 1:
                print (f"penalty for repeated jumps in same direction: {self.action_repeat_penalty}")
                return self.action_repeat_penalty
        return 0

    def check_tent_penalty(self):
        if self.current_screen in static_variables.TENT_BOUNDS:
            bounds = static_variables.TENT_BOUNDS[self.current_screen]
            if (bounds["x_min"] <= self.x <= bounds["x_max"] and 
                bounds["y_min"] <= self.y <= bounds["y_max"]):
                print ("tent penalty")
                return -20
        return 0

    def build_state(self):
        """Builds full 25-value state vector for monolithic agent."""
        self.platform_parser.parse_result = self.platform_parser.read_platform_data(
            (self.x, self.y), self.current_screen
        )

        if self.platform_parser.parse_result is not None:
            pos_state_data = list(self.platform_parser.parse_result[0])
        else:
            pos_state_data = [-9999, 9999, 9999, -9999, 9999]

        sector_state_data = self.platform_parser.process_registry(
            self.current_screen, (self.x, self.y)
        )

        can_bounce_right, can_bounce_left = self.platform_parser.set_rebound_state(
            (self.x, self.y), self.current_screen
        )

        pos_state = [self.x, self.y, self.current_screen, self.is_on_ice, 
                    self.is_in_snow, self.wind_velocity, can_bounce_right, can_bounce_left]

        return np.array(pos_state + pos_state_data + sector_state_data, dtype=np.float32)
    
    def build_state_per_screen(self):
        self.platform_parser.parse_result = self.platform_parser.read_platform_data(
            (self.x, self.y), self.current_screen
        )
        
        if self.platform_parser.parse_result is not None:
            ceiling = self.platform_parser.parse_result[0][2]
            platform_x_start = self.platform_parser.parse_result[0][0]
            platform_x_end = self.platform_parser.parse_result[0][1]
            rel_x_start = self.x - platform_x_start
            rel_x_end = platform_x_end - self.x
        else:
            ceiling = 9999
            rel_x_start = 9999
            rel_x_end = 9999
        
        if self.current_screen in static_variables.WIND_SCREENS:
            return np.array([self.x, self.y, self.wind_velocity, ceiling, rel_x_start, rel_x_end], dtype=np.float32)
        elif self.current_screen in static_variables.ICE_SCREENS:
            return np.array([self.x, self.y, self.vel_x, ceiling, rel_x_start, rel_x_end], dtype=np.float32)
        else:
            return np.array([self.x, self.y, ceiling, rel_x_start, rel_x_end], dtype=np.float32)

    def build_state_ray(self):
        #ray state building
        # self.platform_parser.parse_result = self.platform_parser.read_platform_data((self.x, self.y), self.current_screen)
        # self.ray_caster.build_ray_collision_index(self.platform_parser.current_tiles, self.platform_parser.next_tiles)
        # ray_state_data = self.ray_caster.build_ray_states(num_angles=36)
        # pos_state = [self.x, self.y, self.current_screen, self.is_on_ice, self.is_in_snow, self.wind_velocity]
        # self.state = np.array(pos_state + ray_state_data, dtype=np.float32)
        pass

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

        if self.episode_mode == "per_screen":
            return self.terminate_per_screen_episode()
        
        elif self.episode_mode == "action_height":
            return self.terminate_action_episode() or self.terminate_height_episode()

        elif self.episode_mode == "action":
            result = self.terminate_action_episode()
        
        #terminate after reaching a new screen and landing there
        elif self.episode_mode == "screen":
            result = self.terminate_screen_episode()
        
        #terminate after landing with a positive height gain
        elif self.episode_mode == "height":
            result = self.terminate_height_episode()

        #combines multiple episode types. type1 for first n/2 screens, type2 for second n/2 screens
        elif self.episode_mode == "curriculum":
            #uses jump episodes beneath n screens, and screen episodes above n screens
            if self.current_screen < self.curriculum_screens:
                result = self.terminate_height_episode()
            else:
                result = self.terminate_screen_episode()

        return result
    
    def terminate_per_screen_episode(self):
        if self.current_screen != self.expected_screen:
            return True
        
        # skip drop termination for screens that require large drops
        if self.current_screen not in static_variables.NO_DROP_TERMINATION_SCREENS:
            if self.y_prev - self.y > 50:
                return True
        
        return self.terminate_action_episode() or self.terminate_height_episode()
    
    # def terminate_per_screen_episode(self):
    #     # end episode if fell to different screen
    #     if self.current_screen != self.expected_screen:
    #         return True
    #     return self.terminate_action_episode() or self.terminate_height_episode()
    
    # def terminate_per_screen_episode(self):
    #     self.action_counter += 1
        
    #     if self.total_screen_actions < 5:
    #         # early phase: terminate after 2 actions
    #         if self.action_counter >= 2:
    #             self.action_counter = 0
    #             return True
    #     else:
    #         # late phase: terminate on height gain
    #         if self.y > self.y_prev:
    #             return True
    #         if self.action_counter >= 4:
    #             self.action_counter = 0
    #             return True
    
    #     return False

    def wait_for_landing(self, prev_write_count):
        time.sleep(0.2)
        while True:
            try:
                with open(self.gamestate_path) as f:
                    content = f.read()
                if not content or content.isspace():
                    time.sleep(self.sleep_time)
                    continue
                data = json.loads(content)
                if data.get("is_on_ground") and data.get("write_count", 0) > prev_write_count:
                    return
            except (json.JSONDecodeError, KeyError):
                pass
            except Exception as e:
                print(f"wait_for_landing error: {e}")
            time.sleep(self.sleep_time)
    
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
        #self.action_counter += 1
        if self.action_counter >= self.max_episode_actions:
            self.action_counter = 0
            return True
    
        return False

    def terminate_screen_episode(self):
        #returns True if we should terminate the episode (based on screens). False otherwise
        if self.current_screen != self.current_screen_prev:
            return True

        return False
    
    def get_grid_cell(self, x, y):
        return (int(x // self.grid_size), int(y // self.grid_size))

    # def new_height_reward(self):
    #     #if jump was max height, increase reward by 20%

    #     reward = (self.y - self.y_prev) / 5

    #     if self.jump_percentage == 1 and self.y > self.y_prev:
    #         reward *= self.max_jump_bonus

    #     return reward

    def new_height_reward(self):
        if self.y == self.y_prev:
            return 0
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
        return self.gamedata
                
    def execute_action(self, action):
        prev_write_count = self.gamedata.get("write_count", 0) if self.gamedata else 0
        
        left, right, jump = self.action_map[action]
        #print(f"Executing: left={left}, right={right}, jump={jump}")

        if not jump:
            if left:
                self.key_press("left", left)
            elif right:
                self.key_press("right", right)
            time.sleep(0.1) #was 0.3
        else:
            self.jumped = True
            if left:
                self.key_press("space", jump, "left")
            else:
                time.sleep(0.05)
                self.key_press("space", jump, "right")
            time.sleep(0.1) #was 1
        
        return prev_write_count

    
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

        for t in [0.1, 0.2]:
            action_map.append([t, 0, 0])
            action_map.append([0, t, 0])

        for t in np.arange(self.spacing, 0.65, self.spacing):
            action_map.append([0, float(round(t, 2)), float(round(t, 2))])
            action_map.append([float(round(t, 2)), 0, float(round(t, 2))])

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