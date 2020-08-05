# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

# Basic imports
import cv2
import math
import numpy as np

# Local imports
from depth_distance import Measure
from skimage import draw
from process_depth import *


def paint_arrow(image,Measure,road_val_range=(70,100)):
    height, width = int(image.shape[0]), int(image.shape[1]-1)
    bot_point = height-1,width/2
    checked_row = int(height-height*0.08)
    reclassifying_val = 90, 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    def process_image(image,cond):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[cond(gray)] = reclassifying_val[0]
        gray[~cond(gray)] = reclassifying_val[1]

        return gray


    def cond(image):
        return ((image > road_val_range[0]) &  # road_val_range[1] and road_val_range[2] is the range of 
                (image < road_val_range[1]))   # brightness of road surface (the part of the picture we need).
        # Return False if its not the road

    def find_goal(processed_img):


        def elim_blocked(lineup,row):
            target = None
            while lineup.size != 0 and target is None:
                index = lineup.size//2
                col = lineup[index]
                if not Measure.blocked((row,col)):
                    target = row,col
                    break
                lineup = np.delete(lineup,index)

            return target


        all_road = np.where(processed_img[:checked_row,:] == reclassifying_val[0])

        if all_road[0].size != 0:
            lineup = np.where(processed_img[checked_row] == reclassifying_val[0])[0]
            goal = elim_blocked(lineup,checked_row)
            return goal
        else:
            return None

    def get_turning_angle(current_pos,destination):
        bot_right = (height-height/10,width/2)
        ang = math.degrees(math.atan2(destination[1]-current_pos[1], destination[0]-current_pos[0]) - \
            math.atan2(bot_right[1]-current_pos[1], bot_right[0]-current_pos[0]))
        # return str(ang + 360) if ang < 0 else str(ang)
        return 360+ang if ang < -180 else ang


    def extend(point):
        slope = (point[1]-bot_point[1])/(point[0] - bot_point[0])
        b = point[1] - (slope*point[0])
        new_x = max(0,min(height,point[1]-int(height*0.1)))
        new_y = max(0,min(width,slope * new_x + b))
        print(new_x,new_y)

        return new_x,new_y

    processed_img = process_image(image, cond)
    goal = find_goal(processed_img)

    if goal is not None:
        ''' Let's go!!! '''
        # goal_extended = extend(goal)
        turning_angle = get_turning_angle(bot_point,goal)
        text_pos = int(height - height/10),int(width/10)
        cv2.putText(image,str(turning_angle),text_pos,font,1, \
            (255,255,255),1,cv2.LINE_AA)

        rr,cc = draw.line(int(goal[0]),int(goal[1]), \
                        int(bot_point[0]),int(bot_point[1]))
        image[rr,cc] = 255

    else:
        cv2.putText(image,'No path found!',(height//2,width//2), \
        font,1,(255,255,255),2, cv2.LINE_AA)

    return image

