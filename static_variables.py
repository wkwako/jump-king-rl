WIND_SCREENS = {25, 26, 27, 28, 29, 30, 31}
ICE_SCREENS = {36, 37, 38}

TENT_BOUNDS = {
    0: {"x_min": 72, "x_max": 120, "y_min": -302, "y_max": -200}  # adjust these
}

SCREEN_PROGRESS_DIRECTION = {
    #Redcrown Woods
    0: "up",
    1: "left",
    2: "right",
    3: "right",
    4: "left",
    #Colossal Drain
    5: "right",
    6: "left",
    7: "right", #might need to bisect this one
    8: "right",
    9: "up",

    #False Kings' Keep
    10: "up",
    11: "up",
    12: "up",
    13: "up",

    #Bargainburg
    14: "up",
    15: "right",
    16: "left",
    17: "right",
    18: "up",

    #Great Frontier
    19: "right",
    20: "up",
    21: "up",
    22: "up",
    23: "up",
    24: "right",

    #Windswept Bluff/Stormwall Pass
    25: "up",
    26: "up",
    27: "right",
    28: "up",
    29: "up",
    30: "up",
    31: "right",

    #Chapel Perilous
    32: "left", #switch to right if entering from left
    33: "up",
    34: "up",
    35: "right",

    #Blue Ruin
    36: "left", #kind of need both here
    37: "left",
    38: "up",

    #The Tower
    39: "up",
    40: "right",
    41: "up",
    42: "right"
}

SCREEN_ACTION_MAPS = {
    #Redcrown Woods
    0: {
        "walks": [0.2],
        "jumps": [0.6],
    },
    1: {
        "walks": [0.1, 0.2],
        "jumps": [0.3, 0.45, 0.6],
    },
    2: {
        "walks": [0.1, 0.2],
        "jumps": [0.15, 0.4, 0.6],
    },
    3: {
        "walks": [0.1, 0.2],
        "jumps": [0.15, 0.6], #could add 0.4 here for full path coverage
    },
    4: {
        "walks": [0.1, 0.2],
        "jumps": [0.15, 0.45, 0.6],
    },
    
    #Colossal Drain
    5: {
        "walks": [0.1, 0.2],
        "jumps": [0.35, 0.45, 0.6],
    },
    6: {
        "walks": [0.1, 0.2],
        "jumps": [0.15, 0.45, 0.5, 0.6],
    },
    7: {
        "walks": [0.1, 0.2],
        "jumps": [0.6],
    },
    8: {
        "walks": [0.1, 0.2],
        "jumps": [0.1, 0.15, 0.5, 0.6],
    },
    9: {
        "walks": [0.1, 0.2],
        "jumps": [0.2, 0.45, 0.6],
    },
    
    #False Kings' Keep
    10: {
        "walks": [0.1, 0.2],
        "jumps": [0.15, 0.48, 0.6],
    },
    11: {
        "walks": [0.1, 0.2],
        "jumps": [0.53, 0.6],
    },
    12: {
        "walks": [0.1, 0.2],
        "jumps": [0.45, 0.5, 0.6],
    },
    13: {
        "walks": [0.1, 0.2],
        "jumps": [0.4, 0.6],
    },

    #Bargainburg
    14: {
        "walks": [0.1, 0.2],
        "jumps": [0.4, 0.6],
    },
    15: {
        "walks": [0.1, 0.2],
        "jumps": [0.12, 0.45, 0.5, 0.6],
    },
    16: {
        "walks": [0.1, 0.2],
        "jumps": [0.45, 0.5, 0.6],
    },
    17: {
        "walks": [0.1, 0.2],
        "jumps": [0.1, 0.15, 0.45, 0.5, 0.6],
    },
    18: {
        "walks": [0.1, 0.2],
        "jumps": [0.45, 0.5, 0.6],
    },

    #Great Frontier
    19: {
        "walks": [0.1, 0.2],
        "jumps": [0.15, 0.48, 0.6],
    },
    20: {
        "walks": [0.1, 0.2],
        "jumps": [0.5, 0.6],
    },
    21: {
        "walks": [0.1, 0.2],
        "jumps": [0.23, 0.5, 0.6],
    },
    22: {
        "walks": [0.1, 0.2],
        "jumps": [0.3, 0.48, 0.6],
    },
    23: {
        "walks": [0.1, 0.2],
        "jumps": [0.5, 0.6],
    },
    24: {
        "walks": [0.1, 0.2],
        "jumps": [0.23, 0.43, 0.6],
    },

    #Windswept Bluff/Stormwall Pass
    25: {
        "walks": [0.1, 0.2],
        "jumps": [0.25],
        "only_jump": [0.6],
    },
    26: {
        "walks": [],
        "jumps": [],
        "only_jump": [0.15, 0.6],
    },
    27: {
        "walks": [0.2],
        "jumps": [0.35],
        "only_jump": [0.15, 0.6],
    },
    28: {
        "walks": [],
        "jumps": [0.6],
        "only_jump": [0.15, 0.6],
    },
    29: {
        "walks": [],
        "jumps": [0.6],
        "only_jump": [0.15, 0.6],
    },
    30: {
        "walks": [],
        "jumps": [0.28],
        "only_jump": [0.6],
    },
    31: {
        "walks": [0.1, 0.2],
        "jumps": [],
        "only_jump": [0.6],
    },

    #Chapel Perilous
    32: {
        "walks": [0.1, 0.2],
        "jumps": [0.15, 0.6],
    },
    33: {
        "walks": [0.1, 0.2],
        "jumps": [0.5, 0.6],
    },
    34: {
        "walks": [0.1, 0.2],
        "jumps": [0.45, 0.6],
    },
    35: {
        "walks": [0.1, 0.2],
        "jumps": [0.6],
    },

    #Blue Ruin
    36: {
        "walks": [0.1, 0.2],
        "jumps": [0.1, 0.45, 0.5],
    },
    37: {
        "walks": [0.1, 0.2],
        "jumps": [0.1, 0.6],
        "only_jump": [0.1],
    },
    38: {
        "walks": [0.1, 0.2],
        "jumps": [0.1, 0.28, 0.6],
    },

    #The Tower
    39: {
        "walks": [0.1, 0.2],
        "jumps": [0.33, 0.45, 0.5, 0.6], #could set 0.48 here instead of 0.45 and 0.5
    },
    40: {
        "walks": [0.1, 0.2],
        "jumps": [0.3, 0.48, 0.6],
    },
    41: {
        "walks": [0.1, 0.2],
        "jumps": [0.38, 0.48, 0.6],
    },
    42: {
        "walks": [0.1, 0.2],
        "jumps": [0.23, 0.6],
    },
}