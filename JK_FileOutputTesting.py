import time

path = r"C:\Users\wkwak\Documents\CodingWork\Environments\workStuffPython\JumpKingRL\gamestate.txt"

while True:

    try:
        with open(path) as f:
            data = f.read().split(",")
        x, y, velX, velY = data[0], data[1], data[2], data[3]
        isOnGround = data[4] == "True"
        currentScreen = data[5]
        totalScreens = data[6]
        jumpTime = data[7]
        
        print(f"X:{x} Y:{y} VelX:{velX} VelY:{velY} Ground:{isOnGround} Screen:{currentScreen}/{totalScreens} Jump Time: {jumpTime}")

    except:
        pass

    time.sleep(0.1)