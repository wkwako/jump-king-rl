import pydirectinput
pydirectinput.PAUSE=0
import time


class TrajectoryCalculations:
    def __init__(self):
        pass

    def jump(self, frames):
        
        #grace period to alt tab into game
        time.sleep(1)

        #convert frames to seconds
        s = frames/60
        
        #press buttons
        pydirectinput.keyDown("space")
        pydirectinput.keyDown("right")

        #duration buttons are held
        time.sleep(s)

        #release buttons
        pydirectinput.keyUp("space")
        time.sleep(0.03)
        pydirectinput.keyUp("right")
        #time.sleep(1)
    

    def fit_curve(self):
        pass


tc = TrajectoryCalculations()
tc.jump(37)