import math

class Ray:
    def __init__(self, max_distance=900, step_size=8):
        self.tile_index = set()
        self.max_distance = max_distance
        self.step_size = step_size

    def build_ray_collision_index(self, current_tiles, next_tiles=None):
        self.tile_index = set(
            (int(t[0] // 8) * 8, int(t[1] // 8) * 8)
            for t in current_tiles
        )
        if next_tiles:
            self.tile_index.update(
                (int(t[0] // 8) * 8, int(t[1] // 8) * 8)
                for t in next_tiles
            )

    def ray(self, angle_deg):
        """Casts a single ray using DDA algorithm. Guarantees no tiles are skipped.
        Angle convention: 0 = up, clockwise positive."""
        angle_rad = math.radians(angle_deg)
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)

        # avoid division by zero
        if abs(dx) < 1e-10: dx = 1e-10
        if abs(dy) < 1e-10: dy = 1e-10

        # step direction
        step_x = 8 if dx > 0 else -8
        step_y = 8 if dy > 0 else -8

        # how far along the ray we must travel to cross one grid line in each axis
        delta_dist_x = abs(8 / dx)
        delta_dist_y = abs(8 / dy)

        # initial distances to first grid crossing
        if dx > 0:
            side_dist_x = ((8 - (0 % 8)) % 8) / abs(dx) if abs(dx) > 0 else float('inf')
        else:
            side_dist_x = ((0 % 8)) / abs(dx) if abs(dx) > 0 else float('inf')

        if dy > 0:
            side_dist_y = ((8 - (0 % 8)) % 8) / abs(dy) if abs(dy) > 0 else float('inf')
        else:
            side_dist_y = ((0 % 8)) / abs(dy) if abs(dy) > 0 else float('inf')

        # start at player position (0,0 in relative coords)
        map_x = 0
        map_y = 0
        dist = 0

        while dist < self.max_distance:
            # step to nearest grid boundary
            if side_dist_x < side_dist_y:
                dist = side_dist_x
                side_dist_x += delta_dist_x
                map_x += step_x
            else:
                dist = side_dist_y
                side_dist_y += delta_dist_y
                map_y += step_y

            if dist >= self.max_distance:
                break

            if (map_x, map_y) in self.tile_index:
                return dist

        return self.max_distance

    def build_ray_states(self, num_angles):
        """Casts evenly spaced rays from 0 to 360 degrees.
        Returns a list of distances, one per ray."""
        angle_step = 360 / num_angles
        return [self.ray(i * angle_step) for i in range(num_angles)]