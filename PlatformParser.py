import json
import os
import time
import numpy as np
import math

class PlatformParser:
    def __init__(self):
        self.platform_path = "C:/Program Files (x86)/Steam/steamapps/workshop/content/1061090/3699885336/platformdata.txt"
        self.registry_path = "C:/Users/wkwak/Documents/CodingWork/Environments/workStuffPython/JumpKingRL/registry.txt"
        self.registry = self.load_registry()
        self.parse_result = None
        self.sleep_time = 0.1
        self.current_platforms = None
        self.current_times = []
        self.registry_threshold = 5

    def save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f)

    def load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def update_registry(self, current_screen, player_position):
        max_attempts = 30
        for attempt in range(max_attempts):
            self.parse_result = self.read_platform_data(player_position)
            if self.parse_result is not None:
                break
            time.sleep(self.sleep_time)
        else:
            raise RuntimeError(f"Failed to read platform data after {max_attempts} attempts — game may have crashed or file is corrupted.")

        env_state = self.parse_result[0]
        standing_start, standing_end = env_state[3], env_state[4]

        if standing_start == -9999 or standing_end == 9999:
            return False

        player_x, player_y = player_position

        # standing_start/end are now already absolute since tiles are absolute
        abs_start = round((standing_start + player_x) / 8) * 8
        abs_end = round((standing_end + player_x) / 8) * 8
        length = abs_end - abs_start
        center_x = (abs_start + abs_end) / 2
        new_platform = (abs_start, player_y, abs_end, player_y, center_x, length)

        screen_key = str(current_screen)

        if screen_key not in self.registry:
            self.registry[screen_key] = []

        if not self.is_coord_in_registry(new_platform, screen_key):
            #print (f"Adding new platform on screen {current_screen}: {new_platform}")
            self.registry[screen_key].append(list(new_platform))
            self.save_registry()

        return True
    
    def is_coord_in_registry(self, new_platform, screen_key):
        new_length = new_platform[5]
        new_y = new_platform[1]
        for platform in self.registry[screen_key]:
            if abs(platform[5] - new_length) < self.registry_threshold and abs(platform[1] - new_y) < self.registry_threshold:
                return True
        return False

    def get_angle_and_distance(self, player_x, player_y, platform):
        abs_x_start, abs_y, abs_x_end, _, center_x, length = platform
        #center_x = (abs_x_start + abs_x_end) / 2
        
        # relative coords
        rel_x = center_x - player_x
        rel_y = abs_y - player_y  # positive = above player
        
        # angle: 0 = up, clockwise positive
        angle = math.degrees(math.atan2(rel_x, rel_y)) % 360
        distance = math.sqrt(rel_x**2 + rel_y**2)
        
        return angle, distance, rel_x, rel_y

    def get_sector(self, angle):
        if 5 <= angle <= 85:
            return 'upper_right'
        elif 85 < angle <= 110:
            return 'right'
        elif 250 <= angle <= 275:
            return 'left'
        elif 275 < angle <= 355:
            return 'upper_left'
        else:
            return None

    def process_registry(self, current_screen, player_position):
        player_x, player_y = player_position
        sentinel = (-9999, -9999)
        distance_limit_x = 400
        distance_limit_y = 170

        wide_ceiling_dist = self.detect_wide_ceiling(self.current_tiles) if self.current_tiles else 9999

        sectors = {
            'upper_right': (float('inf'), None),
            'right': (float('inf'), None),
            'upper_left': (float('inf'), None),
            'left': (float('inf'), None),
            'next_upper_right': (float('inf'), None),
            'next_upper_left': (float('inf'), None),
        }

        # current screen platforms
        current_key = str(current_screen)
        for platform in self.registry.get(current_key, []):
            
            #skip the platform the player is standing on
            abs_x_start, abs_y, abs_x_end, _, center_x, length = platform
            if abs_x_start <= player_x <= abs_x_end and abs(abs_y - player_y) < 10:
                continue

            angle, distance, rel_x, rel_y = self.get_angle_and_distance(player_x, player_y, platform)
            
            #don't add to sectors if the platform is too far away
            if abs(rel_x) > distance_limit_x or abs(rel_y) > distance_limit_y:
                continue

            # blocked by ceiling
            if rel_y > wide_ceiling_dist: 
                continue

            sector = self.get_sector(angle)

            if sector and distance < sectors[sector][0]:
                sectors[sector] = (distance, (angle, distance))
                
        # next screen platforms
        next_key = str(current_screen + 1)
        for platform in self.registry.get(next_key, []):
            abs_x_start, abs_y, abs_x_end, _, center_x, length = platform
            angle, distance, rel_x, rel_y = self.get_angle_and_distance(player_x, player_y, platform)

            if abs(rel_y) > 170 or abs(rel_x) > 350:
                continue

            if rel_y > wide_ceiling_dist:
                continue

            sector = self.get_sector(angle)
            if sector == 'upper_right' and distance < sectors['next_upper_right'][0]:
                sectors['next_upper_right'] = (distance, (angle, distance))
            elif sector == 'upper_left' and distance < sectors['next_upper_left'][0]:
                sectors['next_upper_left'] = (distance, (angle, distance))

        # flatten to state
        result = []
        for key in ['upper_left', 'upper_right', 'left', 'right', 'next_upper_left', 'next_upper_right']:
            val = sectors[key][1]
            result.extend(val if val is not None else sentinel)

        return result  # 12 values: 6 sectors x (angle, distance)

    def is_coord_in_registry(self, new_platform, current_screen):
        platforms_in_screen = self.registry[current_screen]
        for platform in platforms_in_screen:
            if self.within_threshold(platform, new_platform):
                return True
        return False

    def _extract_tiles(self, platform_str, player_x, player_y):
        tiles = []
        for line in platform_str.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("screen:") or line.startswith("DEBUG"):
                continue
            vals = line.split(",")
            if len(vals) == 4:
                try:
                    abs_x, abs_y, w, h = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                    rel_x = abs_x - (player_x + 8)
                    rel_y = abs_y + player_y
                    tiles.append((rel_x, rel_y, w, h))
                except ValueError:
                    continue
        return tiles

    def within_threshold(self, list1, list2, thresh=5):
        for i in range(len(list1)):
            if abs(list1[i] - list2[i]) > thresh:
                return False
        return True

    def read_platform_data(self, player_position):
        for _ in range(3):
            try:
                with open(self.platform_path) as f:
                    content = f.read()
                if content:
                    result = self.parse_platforms(content, player_position[0], player_position[1])
                    if result is not None:
                        self.current_tiles = self._extract_tiles(content, player_position[0], player_position[1])
                        return result
            except:
                pass
            time.sleep(self.sleep_time)
        return None
    
    def flatten_platform_state(self, platform_data):
        if platform_data is None:
            return np.full(17, -9999, dtype=np.float32)
        
        (left_wall, right_wall, ceiling, platform_x_start, platform_x_end), sectors = platform_data
        
        flat = [left_wall, right_wall, ceiling, platform_x_start, platform_x_end]
        for sector in sectors:
            flat.extend(sector)  # each sector is now (rel_y, length)
        
        return np.array(flat, dtype=np.float32)

    def parse_platforms(self, platform_str, player_x, player_y):
        if not platform_str:
            return None

        current_screen_tiles = []
        next_screen_tiles = []
        parsing_next = False

        for line in platform_str.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("DEBUG"):
                continue
            if line.startswith("screen:"):
                if current_screen_tiles:
                    parsing_next = True
                continue
            vals = line.split(",")
            if len(vals) == 4:
                try:
                    abs_x, abs_y, w, h = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                    # convert to relative coordinates using player position
                    rel_x = abs_x - (player_x + 8)
                    rel_y = abs_y + player_y
                    tile = (rel_x, rel_y, w, h)
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

        player_width = 8

        ceiling_candidates = [t for t in current_screen_tiles
                      if t[1] < 0 and abs(t[0]) < player_width]

        next_ceiling = [(t[0], t[1] + 360, t[2], t[3]) for t in next_screen_tiles]
        next_ceiling_candidates = [t for t in next_ceiling
                                if t[1] < 0 and abs(t[0]) < player_width]

        all_ceiling = ceiling_candidates + next_ceiling_candidates
        ceiling_dist = -max([t[1] for t in all_ceiling]) if all_ceiling else 9999

        current_platforms = self.merge_tiles(current_screen_tiles)

        # offset next screen tiles by +360 before merging so they fall within
        # the Y filter range and produce valid next_left/next_right sectors
        next_screen_tiles_offset = [(x, y + 360, w, h) for x, y, w, h in next_screen_tiles]
        next_platforms = self.merge_tiles(next_screen_tiles_offset)

        # find standing platform (the one the player is on: contains x=0, just below player)
        standing_platform = None
        below_platforms = [p for p in current_platforms if p[0] < 0 and p[2] <= 20 and p[3] >= -20]
        if below_platforms:
            standing = min(below_platforms, key=lambda p: abs(p[0]))
            standing_platform = (standing[2], standing[3])

        if standing_platform is None:
            platform_x_start, platform_x_end = -9999, 9999
        else:
            platform_x_start, platform_x_end = standing_platform

        wide_ceiling_dist = self.detect_wide_ceiling(current_screen_tiles)

        # assign to sectors
        sectors = self.assign_sectors(current_platforms, next_platforms, wide_ceiling_dist)

        # print(f"current_screen_tiles count: {len(current_screen_tiles)}")
        # print(f"sample tiles: {current_screen_tiles[:5]}")
        # print(f"current_platforms: {current_platforms}")
        # print(f"below_platforms: {below_platforms}")

        return (left_wall_dist, right_wall_dist, ceiling_dist, platform_x_start, platform_x_end), sectors

    def merge_tiles(self, tiles):
        tile_w = 8
        tile_spacing = 16
        min_platform_width = 8
        max_jump_height = 180
        max_jump_width = 300

        tiles = [t for t in tiles if abs(t[0]) < max_jump_width and -max_jump_height < t[1] < 100]

        # find surface tiles - only keep tiles with no tile directly above
        tile_positions = set((round(t[0]), round(t[1])) for t in tiles)
        
        # a tile is a walkable surface if there are no tiles in the N pixels above it
        clearance = 30  # pixels of clear space required above
        tile_positions = set((round(t[0]), round(t[1])) for t in tiles)

        def has_clearance(x, y, clearance, tile_positions, tile_h=8):
            for check_y in range(int(y) - tile_h, int(y) - clearance - tile_h, -tile_h):
                if (round(x), round(check_y)) in tile_positions:
                    return False
            return True

        surface_tiles = [t for t in tiles if has_clearance(t[0], t[1], clearance, tile_positions)]
        
        # group surface tiles by y and merge horizontally
        by_y = {}
        for x, y, w, h in surface_tiles:
            by_y.setdefault(round(y), []).append(x)

        segments_by_y = {}
        for y, x_values in by_y.items():
            x_values.sort()
            start = x_values[0]
            end = x_values[0]
            segments = []

            for x in x_values[1:]:
                if x - end - tile_w <= tile_spacing:
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

    def assign_sectors(self, current_platforms, next_platforms, wide_ceiling_dist=9999):
        # filter out platforms above the wide ceiling
        current_platforms = [p for p in current_platforms if p[0] <= wide_ceiling_dist]
        next_platforms = [p for p in next_platforms if p[0] <= wide_ceiling_dist]
        
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

        # in assign_sectors, filter current platforms that overlap with next platforms
        next_x_ranges = [(p[2], p[3]) for p in next_platforms if p[0] > 0]
        current_platforms = [
            p for p in current_platforms
            if not any(
                not (p[3] <= nx_start or nx_end <= p[2])
                for nx_start, nx_end in next_x_ranges
            )
        ]

        for rel_y, center_x, x_start, x_end in current_platforms:
            if rel_y < -50:
                continue

            dist = (rel_y**2 + center_x**2) ** 0.5

            if rel_y > 0:
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
                rel_y, x_start, x_end = sectors[key]
                length = x_end - x_start
                result.append((rel_y, length))
            else:
                result.append((sentinel, sentinel))

        return result
    
    def detect_wide_ceiling(self, tiles, min_ceiling_width=100):
        # only look at tiles above player
        above_tiles = [t for t in tiles if t[1] < 0]

        if not above_tiles:
            return 9999

        tile_w = 8
        tile_spacing = 8

        # group by y
        by_y = {}
        for x, y, w, h in above_tiles:
            by_y.setdefault(round(y), []).append(x)

        # find the lowest (closest to player) wide segment
        wide_ceiling_dist = 9999
        for y in sorted(by_y.keys(), reverse=True):  # closest y first
            xs = sorted(by_y[y])
            # merge horizontally
            start = xs[0]; end = xs[0]
            for x in xs[1:]:
                if x - end - tile_w <= tile_spacing:
                    end = x
                else:
                    width = (end + tile_w) - start
                    if width >= min_ceiling_width:
                        wide_ceiling_dist = -y  # convert to positive distance
                        return wide_ceiling_dist
                    start = x; end = x
            width = (end + tile_w) - start
            if width >= min_ceiling_width:
                wide_ceiling_dist = -y
                return wide_ceiling_dist

        return wide_ceiling_dist