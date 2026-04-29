import json
import os
import time
import numpy as np


class PlatformParser:
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
            return np.full(17, -9999, dtype=np.float32)
        
        (left_wall, right_wall, ceiling, platform_x_start, platform_x_end), sectors = platform_data
        
        flat = [left_wall, right_wall, ceiling, platform_x_start, platform_x_end]
        for sector in sectors:
            flat.extend(sector)  # each sector is now (rel_y, length)
        
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

        return (left_wall_dist, right_wall_dist, ceiling_dist, platform_x_start, platform_x_end), sectors

    def merge_tiles(self, tiles):
        tile_w = 8
        tile_spacing = 8
        min_platform_width = 10
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
        
        surface_tiles_ur = [t for t in surface_tiles if t[0] > 50 and t[1] < 0]
        print(f"Upper-right surface tiles: {sorted(surface_tiles_ur, key=lambda t: t[1])}")

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

        print (f"current platforms: {current_platforms}")
        
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