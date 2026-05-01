import math

class Ray:
    def __init__(self, max_distance=400, step_size=8):
        self.tile_index = set()
        self.max_distance = max_distance
        self.step_size = step_size

    def build_ray_collision_index(self, tiles):
        """Builds a hash set of tile positions for fast collision lookup."""
        self.tile_index = set(
            (int(t[0] // 8) * 8, int(t[1] // 8) * 8)
            for t in tiles
        )

    def ray(self, angle_deg):
        """Casts a single ray at the given angle. Returns distance to first hit, or max_distance if no hit.
        Angle convention: 0 = up, clockwise positive."""
        angle_rad = math.radians(angle_deg)
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)  # negative because up = negative y

        steps = int(self.max_distance / self.step_size)

        for i in range(1, steps):
            check_x = dx * i * self.step_size
            check_y = dy * i * self.step_size

            snapped_x = int(check_x // 8) * 8
            snapped_y = int(check_y // 8) * 8

            if (snapped_x, snapped_y) in self.tile_index:
                return math.sqrt(check_x**2 + check_y**2)

        return self.max_distance

    def build_ray_states(self, num_angles):
        """Casts evenly spaced rays from 0 to 360 degrees.
        Returns a list of distances, one per ray."""
        angle_step = 360 / num_angles
        return [self.ray(i * angle_step) for i in range(num_angles)]