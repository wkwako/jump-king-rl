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

np.set_printoptions(precision=5, suppress=True)

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from PlatformParser import PlatformParser
from RecordingParser import RecordingParser
from Ray import Ray
import static_variables
from GameStateReceiver import GameStateReceiver
import GeneratePlatformIDs

class ScreenTransitionException(Exception):
    pass

class JumpKingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    def __init__(self, episode_mode, max_episode_actions=20, curriculum_screens=5, spacing=0.05, per_screen=False, action_map=None, current_screen=None, dummyenv=False, play=False, action_cutoff=22):
        print(f"JumpKingEnv created")
        self.teleport_path = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/teleport.txt"
        #self.registry_path = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/registry.txt"
        #self.registry_path = "C:/Users/wkwak/Documents/CodingWork/Environments/PythonStuff/jump-king-rl/registry.txt"
        self.receiver = GameStateReceiver.get_shared()
        self.spacing = spacing
        #self.action_map = self.init_action_map()
        self.state = None
        self.gamedata = None
        #self.gamedata_prev = None
        self.new_screen_reward_val = 150 #was 150
        self.falling_screen_penalty = -150 #was -50
        self.jumped = False
        self.jumped_prev = False
        self.sleep_time = 0.1
        self.jump_counter_metadata = 0
        self.jump_penalty = -5
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
        self.speed_reward = 100
        self.force_teleport = False
        self.dummyenv = dummyenv
        self.play = play
        self.expected_screen = current_screen

        self.actions_since_jump = 0
        self.end_zone_reached = False

        # build height ID map for wind screens before RL starts
        if self.expected_screen in static_variables.WIND_SCREENS:
            raw_records = self.recording_parser.load_wind_recording(self.recording_parser.wind_path)
            screen_records = [
                (state_dict, action) for ts, state_dict, action in raw_records
                if int(state_dict.get("current_screen", -1)) == self.expected_screen
            ]
            if screen_records:
                self.recording_parser.build_height_id_map(self.expected_screen)
            else:
                print(f"Warning: no recordings found for screen {self.expected_screen}, height_id map will be empty")

        self.wind_jump_reward = 50 #was 80
        self.wind_jump_penalty = -100 #was -80
        self.wind_screen_reward = 500 #was 3000
        self.wind_screen_penalty = -300 #was -300
        self.wind_noop_reward = 0.5 #was 20
        self.wind_height_scale = 3 #was 5
        self.last_jump_time = time.time()
        self.noop_cycle_penalty = -80
        self.noop_cycle_limit = 15  # seconds
        self.wind_walk_penalty = -0.5 #not in use. add later.

        self.jump_counter = 0
        self.jump_cutoff = 8
        self.jump_cutoff_penalty = -70

        self.ice_stability_reward = 20
        self.new_platform_reward = 30

        self.recent_walk_actions = []
        self.recent_jump_actions = []
        self.action_repeat_penalty = -10
        self.action_cutoff = action_cutoff #20-30 depending on screen
        self.action_cutoff_penalty = -200 #was -50
        self.wind_action_cutoff_penalty = -1000
        self.ice_action_reward = 15
        self.ice_stuck_penalty = -20

        self.x = self.y = self.vel_x = self.vel_y = None
        self.is_on_ground = self.current_screen = self.total_screens = None
        self.jump_frames = self.jump_percentage = self.max_height_this_jump = None
        self.is_on_ice = self.is_in_snow = self.is_in_water = self.wind_velocity = None
        self.wind_acceleration = None
        self.x_prev = self.y_prev = self.vel_x_prev = self.vel_y_prev = None
        self.is_on_ground_prev = self.current_screen_prev = self.total_screens_prev = None
        self.jump_frames_prev = self.jump_percentage_prev = self.max_height_this_jump_prev = None
        self.is_on_ice_prev = self.is_in_snow_prev = self.is_in_water_prev = self.wind_velocity_prev = None
        self.wind_acceleration_prev = None

        self.recent_landings = []
        self.landing_memory = 7  # how many recent landings to remember
        self.stuck_penalty = -3
        self.stuck_threshold = 10  # pixels — how close counts as "same spot"
        self.per_screen = per_screen
        self.step_counter = 0

        self.frame_count = None
        self.frame_count_prev = None
        self.wind_frame = None
        self.wind_frame_prev = None

        self.pending_transition = False
        self.pending_transition_screen = None

        self.current_screen = current_screen if current_screen is not None else 0

        self.platform_ids = GeneratePlatformIDs.generate_platform_ids(self.platform_parser.registry)
        self.visited_platforms = set()  # reset each episode
        self.prev_platform_id = -1
        self.wait_for_v0 = False
        self.last_jump_was_large = False

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

    def _get_safe_default_state(self):
        """Returns a safe default state when current screen is invalid."""
        if self.state is not None:
            return self.state  # return last known good state
        # fallback zeros matching observation space size
        return np.zeros(self.observation_space.shape, dtype=np.float32)

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

        #ice-specific rewards and penalties. must happen before action occurs
        reward += self.ice_velocity_reward(action)

        # update platform tracking
        current_platform_id = GeneratePlatformIDs.get_platform_id(
            self.x, self.y, self.current_screen,
            self.platform_parser.registry, self.platform_ids
        )

        # ice rewards
        reward += self.ice_short_hop_reward(action)

        # update prev platform
        self.prev_platform_id = current_platform_id
        
        # executes action
        #time.sleep(1)
        prev_write_count = self.execute_action(action)

        self.action_counter += 1

        if self.per_screen and not self.play:
            self.total_screen_actions += 1

        #print ("now waiting for landing")
        self.wait_for_landing(prev_write_count)
        #print ("character has landed")

        if self.end_zone_reached and not self.play:
            print(f"End zone reached on screen {self.expected_screen} — treating as screen complete")
            self.gamedata = self.read_gamedata()
            self.load_game_attributes()
            reward += self.new_screen_reward_val
            self.force_teleport = True
            self.state = self._get_safe_default_state()
            return self.state, reward, True, False, {}

        #ice rewards 2
        #reward += self.ice_v0_reward()  # fires if wait_for_v0 was set. not working bc v is always 0, not updating state enough
        reward += self.ice_new_platform_reward()

        #wind-specific rewards and penalties
        #reward += self.wind_jump()
        #reward += self.wind_noop()
        reward += self.check_noop_cycle_penalty()
        reward += self.check_wind_walk_penalty(action)

        #reads gamedata
        self.gamedata = self.read_gamedata()

        #release spacebar if held
        self.reset_keys()

        #set game data into individual variables
        self.load_game_attributes()
        #print (f"new pos: {self.x, self.y}")

        #self.check_ice_stuck_penalty()

        #if the player drops from too high, they get stunned. wait for that to fade before starting a new action
        self.wait_for_stun()

        if self.dummyenv:
            self.platform_parser.update_registry(self.current_screen, (self.x, self.y))

        #entered alternate map
        if self.current_screen > 42 or self.current_screen < 0:
            print(f"Entered alternate map (screen {self.current_screen}), treating as fall")
            reward += self.new_screen_reward()  # fires fall penalty since current < expected
            self.force_teleport = True
            self.state = self._get_safe_default_state()  # return last known good state
            return self.state, reward, True, False, {}

        self.state = self.build_state_per_screen() if self.per_screen else self.build_state()
        print (f"state: {self.state}")
        print (f"wind timer: {self.wind_timer}")

        #provides a small bonus in direction of progress
        #reward += self.get_goal_proximity_reward()

        # height reward for all agents
        height_reward = self.new_height_reward()
        if height_reward != 0:
            # only apply height reward  if character is grounded
            if self.is_on_ground:
                print(f"Height reward/penalty: {height_reward:.2f}")
                reward += height_reward

        reward += self.new_screen_reward()

        # penalties
        #reward += self.check_tent_penalty()
        #reward += self.check_alternating_walk_penalty(action)
        #reward += self.check_repeated_jump_penalty(action)

        # stuck penalty
        self.add_landing()
        # if self.check_landing_cluster():
        #     reward += self.stuck_penalty

        # termination
        terminated = self.set_terminated()

        #apply jump action cutoff
        # jump_cutoff_reward, jump_cutoff_terminate = self.check_jump_cutoff()
        # reward += jump_cutoff_reward
        # if jump_cutoff_terminate:
        #     terminated = True
        #     self.force_teleport = True

        # jump penalty/metadata
        if self.jumped:
            # if self.expected_screen in static_variables.WIND_SCREENS:
            #     print (f"jump penalty: {self.jump_penalty}")
            #     reward += self.jump_penalty
            self.jump_counter_metadata += 1
            self.jumped_prev = True
            self.jumped = False
            self.actions_since_jump = 0
        
        else:
            self.jumped_prev = False
            self.actions_since_jump += 1

        #apply action cutoff
        if self.action_counter >= self.action_cutoff:
            if self.expected_screen in static_variables.ICE_SCREENS:
                reward += self.action_cutoff_penalty*4
            elif self.expected_screen in static_variables.WIND_SCREENS:
                reward += self.wind_action_cutoff_penalty
            else:
                reward += self.action_cutoff_penalty
            terminated = True
            self.force_teleport = True
            print (f"Action cutoff penalty: {self.action_cutoff_penalty}")

        if terminated:
            print ("--- EPISODE END ---")

        #gives bonus reward for faster screen completion
        # if terminated and self.episode_mode == "screen" and self.current_screen > self.current_screen_prev:
        #     reward += self.speed_reward / self.action_counter
        #     print(f"Speed bonus: {self.speed_reward / self.action_counter:.2f} ({self.action_counter} actions)")

        return self.state, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.platform_parser.parse_result = None
        self.gamedata = self.read_gamedata()
        self.load_game_attributes()
        self.reset_keys()
        self.jumped_prev = False
        self.jump_counter = 0
        self.end_zone_reached = False
        self.actions_since_jump = 0
        
        # only teleport if we're on the wrong screen
        if self.per_screen and (self.current_screen != self.expected_screen or self.force_teleport) and not self.dummyenv and not self.play:
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

        self.visited_platforms = set()
        self.prev_platform_id = -1
        self.wait_for_v0 = False
        self.last_jump_was_large = False

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
    
    def wait_for_stun(self):
        if self.y - self.max_height_this_jump < -170:
            time.sleep(1)

    def check_ice_stuck_penalty(self):
        if self.expected_screen not in static_variables.ICE_SCREENS:
            return 0
        
        if len(self.recent_landings) < self.landing_memory:  # need enough samples
            return 0
        
        xs = [pos[0] for pos in self.recent_landings]
        ys = [pos[1] for pos in self.recent_landings]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        # euclidean diagonal of bounding box
        diagonal = (x_range**2 + y_range**2) ** 0.5
        
        if diagonal < 80:
            print(f"Ice stuck penalty: {self.check_ice_stuck_penalty}")
            return self.ice_stuck_penalty
        
        return 0
    
    def ice_short_hop_reward(self, action):
        """Reward for successful velocity correction after large jump."""
        if self.expected_screen not in static_variables.ICE_SCREENS:
            return 0
        
        left, right, jump = self.action_map[action]
        is_short_hop = (left == 0.05 and jump == 0.05) or \
                    (right == 0.05 and jump == 0.05)
                
        if not is_short_hop:
            # check if this was a large jump to set flag for next action
            if jump >= 0.3:
                self.last_jump_was_large = True
            else:
                self.last_jump_was_large = False
            return 0
        
        # this is a short hop — check if it follows a large jump
        if self.last_jump_was_large:
            self.wait_for_v0 = True
            self.last_jump_was_large = False
            return 0  # reward fires after wait_for_v0 completes, not here
        
        return 0

    def ice_v0_reward(self):
        """Bonus for successfully stopping on a platform after short hop correction."""
        if self.expected_screen not in static_variables.ICE_SCREENS:
            return 0
        
        current_platform_id = GeneratePlatformIDs.get_platform_id(
            self.x, self.y, self.current_screen,
            self.platform_parser.registry,
            self.platform_ids
        )
        
        # still on same platform and velocity is zero
        if current_platform_id == self.prev_platform_id and \
        current_platform_id != -1 and \
        self.vel_x == 0:
            print(f"Ice stability bonus: {self.ice_stability_reward}")
            return self.ice_stability_reward
        
        return 0

    def ice_new_platform_reward(self):
        """Reward for jumping to a new unvisited platform, fires on landing."""
        if self.expected_screen not in static_variables.ICE_SCREENS:
            return 0
        
        if not self.jumped:
            return 0
        
        current_platform_id = GeneratePlatformIDs.get_platform_id(
            self.x, self.y, self.current_screen,
            self.platform_parser.registry,
            self.platform_ids
        )
        
        if current_platform_id == -1:
            return 0
        
        if current_platform_id not in self.visited_platforms:
            self.visited_platforms.add(current_platform_id)
            print(f"New platform reward: {self.new_platform_reward}")
            return self.new_platform_reward
        
        
        return 0

    def ice_velocity_reward(self, action):
        reward = 0
        left, right, jump = self.action_map[action]

        #if we didn't jump the previous action, don't give a reward. or if this isn't an ice screen
        if not self.jumped_prev or self.expected_screen not in static_variables.ICE_SCREENS:
            return 0

        #if velocity is positive, left walks and short left jump give reward
        if self.vel_x > 0:
            #if action in [[0.1, 0, 0], [0.2, 0, 0], [0.1, 0, 0.1]]:
            if left in [0.05, 0.2]:
                reward += self.ice_action_reward
            else:
                reward -= self.ice_action_reward

        #if velocity is negative, right walks and short right jumps give reward
        elif self.vel_x < 0:
            if right in [0.05, 0.2]:
                reward += self.ice_action_reward
            else:
                reward -= self.ice_action_reward

        #otherwise if the action was a jump straight up, give a reward (only applies to screen 37)
        # elif abs(self.vel_x) > 0:
        #     if jump > 0 and left == 0 and right == 0:
        #         reward += self.ice_action_reward
        #     else:
        #         reward -= self.ice_action_reward

        print (f"ice velocity reward: {reward}")

        return reward

    def check_jump_cutoff(self):
        if self.expected_screen not in static_variables.WIND_SCREENS:
            return 0, False
        if self.jumped:
            self.jump_counter += 1
        if self.jump_counter >= self.jump_cutoff:
            self.jump_counter = 0
            print(f"Jump cutoff penalty: {self.jump_cutoff_penalty}")
            return self.jump_cutoff_penalty, True
        return 0, False

    def wind_jump(self):
        reward = 0
        if self.expected_screen in static_variables.WIND_SCREENS and self.jumped:
            if self.get_wind_state() > 0:
                print (f"wind jump reward: {self.wind_jump_reward}")
                reward += self.wind_jump_reward
            else:
                print (f"wind jump penalty: {self.wind_jump_penalty}")
                reward += self.wind_jump_penalty
        return reward
    
    def wind_noop(self):
        reward = 0
        if self.expected_screen not in static_variables.WIND_SCREENS:
            return 0
        #if wind not blowing to the right, give reward for no-op
        if self.get_wind_state() != 100:
            print (f"wind noop reward: {self.wind_noop_reward}")
            reward += self.wind_noop_reward

        #if wind blowing to the right, give penalty for no-op
        else:
            print (f"wind noop penalty: {-self.wind_noop_reward}")
            reward += -self.wind_noop_reward
        return reward
    
    def check_wind_walk_penalty(self, action):
        reward = 0

        if self.expected_screen not in static_variables.WIND_SCREENS:
            return 0

        left, right, jump = self.action_map[action]
        if jump == 0 and (left > 0 or right > 0):
            reward += self.wind_walk_penalty
            print (f"walk penalty: {self.wind_walk_penalty}")

        return reward

    def new_screen_reward(self):
        reward = 0
        if self.current_screen == self.expected_screen + 1:
            if self.expected_screen in static_variables.WIND_SCREENS:
                reward += self.wind_screen_reward
                print (f"New wind screen reward: {self.wind_screen_reward}")
            else:
                reward += self.new_screen_reward_val
                print (f"New screen reward: {self.new_screen_reward_val}")
            
            reward += self.speed_reward / self.action_counter
            print (f"Speed bonus: {self.speed_reward / self.action_counter:.2f} ({self.action_counter} actions)")

        #fell. apply penalty
        elif self.current_screen < self.expected_screen:
            if self.expected_screen in static_variables.WIND_SCREENS:
                reward += self.wind_screen_penalty
                print (f"Wind screen penalty: {self.wind_screen_penalty}")

            else:
                reward += self.falling_screen_penalty
                print (f"Falling screen penalty: {self.falling_screen_penalty}")

        return reward
    
    def check_noop_cycle_penalty(self):
        if self.expected_screen not in static_variables.WIND_SCREENS:
            return 0
        
        now = time.time()
        
        if self.last_jump_time is None:
            self.last_jump_time = now
            return 0
        
        if self.jumped:
            self.last_jump_time = now
            return 0
        
        if now - self.last_jump_time >= self.noop_cycle_limit:
            self.last_jump_time = now  # reset so it doesn't fire every step
            print(f"Noop cycle penalty: {self.noop_cycle_penalty}")
            return self.noop_cycle_penalty
        
        return 0
    
    def get_goal_proximity_reward(self):
        #do not apply reward for non wind screens
        if self.expected_screen not in static_variables.WIND_SCREENS:
            return 0

        next_screen = self.expected_screen + 1
        if next_screen not in static_variables.SCREEN_START_POSITIONS:
            return 0
        
        goal_x, goal_y, _ = static_variables.SCREEN_START_POSITIONS[next_screen]
        
        # distance to goal in current and previous position
        curr_dist = math.sqrt((self.x - goal_x)**2 + (-self.y - goal_y)**2)
        prev_dist = math.sqrt((self.x_prev - goal_x)**2 + (-self.y_prev - goal_y)**2)
        
        improvement = prev_dist - curr_dist
        if improvement < -300:
            improvement = -300
        
        print (f"goal proximity reward: {improvement}")
        return improvement

        # reward reduction in distance, don't punish increase
        # improvement = prev_dist - curr_dist
        # if improvement > 0:
        #     goal_proximity_reward = improvement
        #     if goal_proximity_reward > 20:
        #         print (f"goal proximity reward: {goal_proximity_reward}")
        #         return goal_proximity_reward
        #return 0
    
    def teleport(self, screen):
        x, y, radius = static_variables.SCREEN_START_POSITIONS[screen]
        x += random.randint(-radius, radius)
        
        for attempt in range(10):
            self.receiver.send_teleport(x, y)
            time.sleep(0.75)  # longer wait
            self.gamedata = self.read_gamedata()
            self.load_game_attributes()
            self.load_game_attributes_prev()
            time.sleep(0.25)
            
            if self.current_screen == screen:
                return
            
            print(f"Teleport attempt {attempt+1} failed: expected {screen}, got {self.current_screen}")
            time.sleep(0.5)  # extra wait before retry
        
        print(f"Teleport failed after 10 attempts — manually waiting...")
    
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
        if self.expected_screen in static_variables.WIND_SCREENS:
            return 0
        
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
            left_wall_dist = self.platform_parser.parse_result[0][0]
            right_wall_dist = self.platform_parser.parse_result[0][1]
            ceiling = self.platform_parser.parse_result[0][2]
            rel_x_start = self.platform_parser.parse_result[0][3]
            rel_x_end = self.platform_parser.parse_result[0][4]
        else:
            ceiling = 9999
            rel_x_start = 9999
            rel_x_end = 9999
        
        if self.expected_screen in static_variables.OLD_STATE_SCREENS:
            return np.array([self.x, self.y, ceiling, left_wall_dist, right_wall_dist, rel_x_start, rel_x_end], dtype=np.float32)
        
        elif self.expected_screen in static_variables.XY_STATE_SCREENS:
            return np.array([self.x, self.y % 360])
        
        # elif self.expected_screen in static_variables.WIND_PLATFORM_DETECTION_SCREENS:
        #     height_id = self.get_height_id(self.y, self.expected_screen)
        #     return np.array([self.x/480, height_id, self.wind_timer/13, rel_x_start, rel_x_end])

        elif self.expected_screen in static_variables.WIND_SCREENS:
            height_id = self.recording_parser.get_height_id(self.y, self.expected_screen)
            #return np.array([self.x/480, height_id, self.wind_timer/13, self.actions_since_jump], dtype=np.float32)
            #return np.array([self.x/480, height_id, self.wind_timer/13], dtype=np.float32)
            return np.array([self.x/480, height_id, self.wind_timer/13], dtype=np.float32)
        
            # height_onehot = self.recording_parser.get_height_onehot(self.y, self.expected_screen)
            # x_norm = self.x / 480
            # wind_timer_norm = self.wind_timer / 13
            # base_state = np.array([x_norm, wind_timer_norm, self.actions_since_jump], dtype=np.float32)
            # return np.concatenate([base_state, height_onehot])
        
        elif self.expected_screen in static_variables.ICE_SCREENS:
            return np.array([self.x, self.y % 360, self.vel_x, ceiling, rel_x_start, rel_x_end], dtype=np.float32)
        else:
            return np.array([self.x, self.y % 360, ceiling, left_wall_dist, right_wall_dist, rel_x_start, rel_x_end], dtype=np.float32)

    def get_wind_state(self):
        if self.wind_velocity >= 0.07:
            return 100.0
        elif self.wind_velocity <= -0.07:
            return -100.0
        return 0.0

    def build_state_ray(self):
        #ray state building
        # self.platform_parser.parse_result = self.platform_parser.read_platform_data((self.x, self.y), self.current_screen)
        # self.ray_caster.build_ray_collision_index(self.platform_parser.current_tiles, self.platform_parser.next_tiles)
        # ray_state_data = self.ray_caster.build_ray_states(num_angles=36)
        # pos_state = [self.x, self.y, self.current_screen, self.is_on_ice, self.is_in_snow, self.wind_velocity]
        # self.state = np.array(pos_state + ray_state_data, dtype=np.float32)
        pass

    def add_landing(self):
        # if not self.jumped:
        #     return
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

        # if self.episode_mode == "per_screen":
        #     return self.terminate_per_screen_episode()
        
        if self.episode_mode == "action_height":
            return self.terminate_action_episode() or self.terminate_height_episode() or self.terminate_screen_episode()

        elif self.episode_mode == "action":
            result = self.terminate_action_episode()
        
        #terminate after reaching a new screen and landing there
        elif self.episode_mode == "screen":
            result = self.terminate_screen_episode()# or self.terminate_height_episode() #added self.terminate_height_episode()
        
        #terminate after landing with a positive height gain
        elif self.episode_mode == "height":
            result = self.terminate_height_episode() or self.terminate_screen_episode()

        elif self.episode_mode == "jumped":
            if self.jumped or self.terminate_screen_episode():
                return True
            return False

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

    def wait_for_landing(self, prev_write_count):
        end_zone = static_variables.END_ZONE_POSITIONS.get(self.expected_screen)
        end_zone_radius = static_variables.END_ZONE_RADIUS
        self.end_zone_reached = self.receiver.wait_for_landing(
            self.jumped, prev_write_count,
            end_zone=end_zone, end_zone_radius=end_zone_radius
        )
        time.sleep(0.03)
    
    def reset_keys(self):
        #stops pressing all keys
        #print ("RESETTING KEYS")
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

    def new_height_reward(self, negatives=True):
        if self.y == self.y_prev:
            return 0

        dist = self.y - self.y_prev
        reward = dist / 5

        # if self.jump_percentage == 1 and self.y > self.y_prev:
        #     reward *= self.max_jump_bonus

        #temporary for wind screens - no height penalty:
        # if self.current_screen in static_variables.WIND_SCREENS and reward < 0:
        #     reward = 0

        if self.expected_screen in static_variables.ICE_SCREENS:
            reward *= 3
            if dist > 0 and dist < 20:
                reward = 0

        if self.expected_screen in static_variables.WIND_SCREENS:
            reward *= self.wind_height_scale

        if reward < -100:
            reward = -100

        if not negatives and reward < 0:
            reward = 0

        return reward

    def read_gamedata(self):
        data = self.receiver.read_gamedata()
        if data is None:
            return self.gamedata  # return last known if buffer empty
        return data

    def get_gamedata_old(self):
        self.gamedata = self.read_gamedata()
        self.load_game_attributes()
        return self.gamedata
                
    def execute_action(self, action):
        prev_write_count = self.gamedata.get("write_count", 0) if self.gamedata else 0
        
        left, right, jump = self.action_map[action]
        print(f"Executing action: left={left}, right={right}, jump={jump}")

        if (left, right, jump) == (0, 0, 0):
            #print ("no-op")
            time.sleep(0.05)

        if not jump:
            if left:
                self.key_press("left", left)
            elif right:
                self.key_press("right", right)
            time.sleep(0.1) #was 0.1
        else:
            self.jumped = True
            if left:
                self.key_press("space", jump, "left")
            elif right:
                #time.sleep(0.05)
                self.key_press("space", jump, "right")
            else:
                self.key_press("space", jump)
            time.sleep(0.07) #was 0.1

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

        if self.dummyenv:
            return [[0,0,0]]

        #perform no action - only used for testing
        #action_map.append([0, 0, 0])

        action_map = []

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
        self.wind_acceleration = self.gamedata["wind_acceleration"]
        self.frame_count = self.gamedata.get("frame_count", 0)
        self.wind_frame = self.gamedata.get("wind_frame", 0)
        self.wind_timer = self.gamedata.get("wind_timer", -1)

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
        self.wind_acceleration_prev = self.gamedata["wind_acceleration"]
        self.frame_count_prev = self.gamedata.get("frame_count", 0)
        self.wind_frame = self.gamedata.get("wind_frame", 0)

    def close(self):
        self.receiver.close()
        super().close()