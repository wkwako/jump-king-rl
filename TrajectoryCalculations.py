import pydirectinput
pydirectinput.PAUSE=0
import time


class TrajectoryCalculations:
    def __init__(self):
        #action: [duration_left_arrow, duration_right_arrow, duration_spacebar]

        #normal adjacency list: #1: [3,5,6]
        #dict{pair of platform ids: action}
        self.graph = {}

        #dict{platform_id: (y, x0, x1)} OR dict{platform_id: (x0, y, x1, y)}
        self.platforms = {}

        #dict{platform_id: }
        self.slopes = {}

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

    def simulate_jump(self):
        #returns x,y location if jump would land on a platform; returns None otherwise
        pass

    def slope_to_platform(self):
        #returns a platform if character would land on slope that carries them to a platform; returns False otherwise
        pass

    def BFS(self):
        #outputs platform id pairs, location to perform jump, and actions
        #(1, 2): (x, y, action1)
        #(2, 4): (x, y, action4)
        #(4, 5): (x, y, action2)
        pass

    def create_graph(self):
        #first pass: find all platform id pairs and actions between them
        #1, 3 --> (x, y, action1)
        #1, 3 --> (x, y, action2)
        #1, 3 --> (x, y, action3)
        
        #second pass: clean graph, select best key value pairs. platform id pairs are unique
        pass

    def execute_plan(self):
        pass

    def move_to_location(self, location):
        #walks to location
        x,y = location
        pass
    


tc = TrajectoryCalculations()
tc.jump(35)