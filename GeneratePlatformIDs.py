def generate_platform_ids(registry):
    """Assigns sequential integer IDs to each platform per screen.
    Accepts registry dict directly rather than a path.
    Returns dict: {screen_int: {platform_idx: global_id}}
    """
    platform_ids = {}
    global_id = 0
    
    for screen, platforms in registry.items():
        screen = int(screen)
        platform_ids[screen] = {}
        for idx in range(len(platforms)):
            platform_ids[screen][idx] = global_id
            global_id += 1
    
    return platform_ids

def get_platform_id(x, y, screen, registry, platform_ids, threshold=8):
    """Returns platform ID for given position, or -1 if not on a known platform.
    
    Args:
        x: player x position (raw)
        y: player y position (raw, negated but NOT modulo'd)
        screen: current screen integer
        registry: platform registry dict
        platform_ids: dict from generate_platform_ids()
        threshold: y distance threshold for platform matching
    """
    platforms = registry.get(screen) or registry.get(str(screen))
    if platforms is None:
        return -1

    for idx, platform in enumerate(platforms):
        x_start = platform[0]
        x_end = platform[2]
        y_platform = platform[1]  # just negate to match self.y coordinate system

        if x_start <= x <= x_end and abs(y - y_platform) <= threshold:
            return platform_ids[screen][idx]

    return -1