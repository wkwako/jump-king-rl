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
        self.platform_path = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/platformdata.txt"
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

        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 23, dtype=np.float32),
            high=np.array([np.inf] * 23, dtype=np.float32),
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

        platform_data = self.read_platform_data()
        platform_state = self.flatten_platform_state(platform_data)

        #release spacebar if it's beind held before we choose another action
        self.reset_keys()

        #set game data into individual variables
        x, y, vel_x, vel_y, is_on_ground, current_screen, total_screens, jump_frames, jump_percentage, max_height_this_jump = self.gamedata   
        x_prev, y_prev, vel_x_prev, vel_y_prev, is_on_ground_prev, current_screen_prev, total_screens_prev, jump_frames_prev, jump_percentage_prev, max_height_this_jump_prev = self.gamedata_prev

        #reward calculation
        #must land for current_screen to be registered as above previous screen
        if current_screen > current_screen_prev:
            #print (f"Reward for new screen: {self.new_screen_reward}")
            reward += self.new_screen_reward

        #if we landed higher, reward. if we landed lower, punish
        if y > y_prev or y < y_prev:
            #print (f"Reward for landing at new altitude: {self.new_height_reward(y, y_prev, jump_percentage)}")
            reward += self.new_height_reward(y, y_prev, jump_percentage)

        #reward moving right after a big fall so the player stands up?

        #reward exploring unexplored grid cells this episode
        # cell = self.get_grid_cell(x, y)
        # if cell not in self.visited_cells:
        #     self.visited_cells.add(cell)
        #     reward += self.exploration_reward

        #create state tuple. this is what the agent uses to determine actions
        #self.state = (x, y, vel_x, vel_y, current_screen)
        pos_state = (x, y, current_screen)
        self.state = np.concatenate([np.array(pos_state, dtype=np.float32), platform_state])

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

        #state, reward, if the episode is terminated, truncation, and info dict
        return self.state, reward, terminated, False, {}
    
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
        platform_data = self.read_platform_data()
        platform_state = self.flatten_platform_state(platform_data)

        self.gamedata_prev = list(self.gamedata)
        self.visited_cells.clear()
        self.action_counter = 0
        
        self.gamedata_start_of_episode = list(self.gamedata)

        x, y, vel_x, vel_y, is_on_ground, current_screen = self.gamedata[:6]
        pos_state = (x, y, current_screen)
        self.state = np.concatenate([np.array(pos_state, dtype=np.float32), platform_state])


        return self.state, {}

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

    def read_platform_data(self):
        for _ in range(3):
            try:
                with open(self.platform_path) as f:
                    content = f.read()
                if content:
                    result = self.parse_platforms(content)
                    if result is not None:
                        return result
            except:
                pass
            time.sleep(self.sleep_time)
        return None
    
    def flatten_platform_state(self, platform_data):
        if platform_data is None:
            return np.full(20, -9999, dtype=np.float32)
        
        (left_wall, right_wall, ceiling), sectors = platform_data
        
        flat = [left_wall, right_wall, ceiling]
        for sector in sectors:
            flat.extend(sector)
        
        return np.array(flat, dtype=np.float32)

    def parse_platforms(self, platform_str):
        if not platform_str:
            return None

        current_screen_tiles = []
        next_screen_tiles = []
        parsing_next = False

        for line in platform_str.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("player") or line.startswith("DEBUG"):
                continue
            if line.startswith("screen:"):
                if current_screen_tiles:
                    parsing_next = True
                continue
            vals = line.split(",")
            if len(vals) == 4:
                try:
                    tile = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
                    if not parsing_next:
                        current_screen_tiles.append(tile)
                    else:
                        next_screen_tiles.append(tile)
                except ValueError:
                    continue

        # compute wall distances using merge_walls
        walls = self.merge_walls(current_screen_tiles)

        # exclude walls that contain the player's x position (standing platform edges)
        #walls must extend player's vertical position to be considered a wall
        left_walls = [w[0] for w in walls if w[0] < 0 and w[1] < 0 and w[2] > 0]
        right_walls = [w[0] for w in walls if w[0] > 0 and w[1] < 0 and w[2] > 0]

        left_wall_dist = max(left_walls) if left_walls else -9999
        right_wall_dist = min(right_walls) if right_walls else 9999

        #print (f"current walls: {walls}")
        #print (f"left walls: {left_walls}")
        #print (f"right walls: {right_walls}")

        player_width = 8

        ceiling_candidates = [t for t in current_screen_tiles
                      if t[1] < 0 and abs(t[0]) < player_width]

        next_ceiling = [(t[0], t[1] + 360, t[2], t[3]) for t in next_screen_tiles]
        next_ceiling_candidates = [t for t in next_ceiling
                                if t[1] < 0 and abs(t[0]) < player_width]

        all_ceiling = ceiling_candidates + next_ceiling_candidates
        ceiling_dist = -max([t[1] for t in all_ceiling]) if all_ceiling else 9999

        # merge tiles into platforms
        current_platforms = self.merge_tiles(current_screen_tiles)

        # offset next screen tiles by +360 before merging so they fall within
        # the Y filter range and produce valid next_left/next_right sectors
        next_screen_tiles_offset = [(x, y + 360, w, h) for x, y, w, h in next_screen_tiles]
        next_platforms = self.merge_tiles(next_screen_tiles_offset)

        # assign to sectors
        sectors = self.assign_sectors(current_platforms, next_platforms)

        return (left_wall_dist, right_wall_dist, ceiling_dist), sectors

    def merge_tiles(self, tiles):
        tile_w = 8
        tile_spacing = 8
        min_platform_width = 64
        max_jump_height = 180
        max_jump_width = 300

        tiles = [t for t in tiles if abs(t[0]) < max_jump_width and -max_jump_height < t[1] < 100]

        by_y = {}
        for x, y, w, h in tiles:
            key = round(y)
            if key not in by_y:
                by_y[key] = []
            by_y[key].append(x)

        segments_by_y = {}
        for y, x_values in by_y.items():
            x_values.sort()
            start = x_values[0]
            end = x_values[0]
            segments = []

            for x in x_values[1:]:
                if x - end - tile_w <= tile_spacing:  # actual pixel gap between tiles
                    end = x
                else:
                    seg_end = end + tile_w
                    width = seg_end - start
                    if width >= min_platform_width:
                        segments.append((start, seg_end))
                    start = x
                    end = x

            seg_end = end + tile_w
            width = seg_end - start
            if width >= min_platform_width:
                segments.append((start, seg_end))

            if segments:
                segments_by_y[y] = segments

        platforms = []
        for y, segments in segments_by_y.items():
            for x_start, x_end in segments:
                if y - 8 not in segments_by_y:
                    is_top_edge = True
                else:
                    above_segs = segments_by_y[y - 8]
                    overlaps = any(
                        not (ax_end <= x_start or ax_start >= x_end)
                        for ax_start, ax_end in above_segs
                    )
                    is_top_edge = not overlaps

                if is_top_edge:
                    center_x = (x_start + x_end) / 2
                    platforms.append((-y, center_x, x_start, x_end))

        return platforms
    
    def merge_walls(self, tiles):
        tile_h = 8
        tile_spacing = 8
        min_wall_height = 80  # must span at least 10 tiles vertically to count as a wall

        # group by x
        by_x = {}
        for x, y, w, h in tiles:
            by_x.setdefault(round(x), []).append(y)

        walls = []
        for x, y_values in by_x.items():
            y_values.sort()
            # merge vertically, same logic as horizontal merge
            start = y_values[0]; end = y_values[0]
            for y in y_values[1:]:
                if y - end - tile_h <= tile_spacing:
                    end = y
                else:
                    wall_height = (end + tile_h) - start
                    if wall_height >= min_wall_height:
                        walls.append((x, start, end + tile_h, wall_height))
                    start = y; end = y
            wall_height = (end + tile_h) - start
            if wall_height >= min_wall_height:
                walls.append((x, start, end + tile_h, wall_height))

        return walls  # list of (x, y_start, y_end, height)

    def assign_sectors(self, current_platforms, next_platforms):
        sectors = {
            'upper_left': None,
            'upper_right': None,
            'left': None,
            'right': None,
            'next_left': None,
            'next_right': None
        }
        best_dist = {k: float('inf') for k in sectors}

        # filter out standing platform segments (contain x=0 and below player)
        # but keep disconnected same-level platforms
        current_platforms = [
            p for p in current_platforms
            if not (p[2] <= 0 <= p[3] and -50 < p[0] < 0)
        ]

        for rel_y, center_x, x_start, x_end in current_platforms:
            if rel_y < -50:
                continue

            dist = (rel_y**2 + center_x**2) ** 0.5

            # only assign to upper sectors if platform is actually above player
            if rel_y > 0 and abs(rel_y) >= abs(center_x) * 0.3:
                sector = 'upper_left' if center_x <= 0 else 'upper_right'
            else:
                sector = 'left' if center_x <= 0 else 'right'

            if dist < best_dist[sector]:
                best_dist[sector] = dist
                sectors[sector] = (rel_y, x_start, x_end)

        for rel_y, center_x, x_start, x_end in next_platforms:
            sector = 'next_left' if center_x <= 0 else 'next_right'
            dist = (rel_y**2 + center_x**2) ** 0.5
            if dist < best_dist[sector]:
                best_dist[sector] = dist
                sectors[sector] = (rel_y, x_start, x_end)

        sentinel = -9999
        result = []
        for key in ['upper_left', 'upper_right', 'left', 'right', 'next_left', 'next_right']:
            if sectors[key] is not None:
                result.append(sectors[key])
            else:
                result.append((sentinel, sentinel, sentinel))

        return result
                
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