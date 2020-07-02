# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

import cv2
import math
import numpy as np
from path_state import PathState
from distance import Heuristic
from astar import astar
from timeit import default_timer as timer
from skimage import draw

def paint_path(image, road_val_range):

    """

    This is a small file containing a function that helps paint a trajectory for the robot.

    Prerequisites
    -------------
    Opencv: Works with images.
    Numpy: Matrices manipulations and calculations.
    Skikit-image: Paints the image.

    Parameters
    ----------
    image : Image as numpy array
    road_val_range : The brightness of the road.

    Returns
    -------
    image : The painted version of the image.

    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = int(image.shape[0]), int(image.shape[1]-1)
    current_pos = height-1,width/2
    # This is the substitute for the road_val_range.
    # When the pixels where the roads appear is determined,
    # they are assigned to the first value of the tuple.
    # Every other pixels are assigned to the latter value.
    reclassifying_val = 90, 0

    def process_image(image,cond):
        reclassifying_val = 90, 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[cond(gray)] = reclassifying_val[0]
        gray[~cond(gray)] = reclassifying_val[1]     # Boolean Indexing
        return gray

    # Private function which return a boolean-indexed version of the image.
    # It selects only the pixel contains the value that fits the given brightness
    # of the road.
    def cond(image):
        return ((image > road_val_range[0]) &  # road_val_range[1] and road_val_range[2] is the range of 
                (image < road_val_range[1]))   # brightness of road surface (the part of the picture we need).
        # Return False if its not the road

    def find_road_top_bot(image):
        try:
            # Remove the bottom part of the image from the search by slicing until (height-height/10)
            all_road = np.where(image[0:int(height-height/10),:] == reclassifying_val[0])
            top = all_road[0][0], all_road[1][0]
            
            bot_row = all_road[0][::-1][0]
            list_bot_col = np.where(image[bot_row] == reclassifying_val[0])[0]
            mid_bot_col = list_bot_col[len(list_bot_col)//2]
            bot = bot_row,mid_bot_col
            return top,bot
        except:
            return None

    def get_turning_angle(current_pos,destination):
        bot_right = (height-height/10,width/2)
        ang = math.degrees(math.atan2(destination[1]-current_pos[1], destination[0]-current_pos[0]) - \
            math.atan2(bot_right[1]-current_pos[1], bot_right[0]-current_pos[0]))
        # return str(ang + 360) if ang < 0 else str(ang)
        return 360+ang if ang < -180 else ang

    processed_img = process_image(image, cond)

    if find_road_top_bot(processed_img) != None:
        goal,start = find_road_top_bot(processed_img)
        ''' Let's go!!! '''
        grid_S = PathState(start,processed_img)
        grid_G = PathState(goal, processed_img)
        heuristic = Heuristic(grid_G)

        plan1 = astar(grid_S,
                        lambda state: heuristic(state) < 21,
                        heuristic)

        condition = False
        # print(list(plan))

        def valid_plan(nice):
            if nice != None:
                wow = list(nice)
                if len(wow) != 0:
                    return wow

        plan = valid_plan(plan1)
        if plan != None:
            # print(plan)
            # # # print(list(plan1)) 
            # # # print(plan)
            # # plan = plan1
            first = plan[0][0]
            # print(len(plan))
            first_target = (current_pos,first)
            turning_angle = get_turning_angle(current_pos,first)
            text_position = int(height/10),int(width/10)
            cv2.putText(image,str(turning_angle),text_position,font,1,(255,255,255),1,cv2.LINE_AA)
            plan.insert(0,first_target)

            for x, y in plan:
                rr, cc = draw.line(
                    int(x[0]), int(x[1]), int(y[0]), int(y[1]))
                image[rr, cc] = 255
        else:
        # except:
            cv2.putText(image,'No path found!',(height//2,width//2), \
            font,1,(255,255,255),2, cv2.LINE_AA)
    else:
        cv2.putText(image,'No pavement detected!',(height//2,width//2), \
        font,1,(255,255,255),2, cv2.LINE_AA)

    return image

# ''' Let's have some tests! '''
# img = cv2.imread(
#     'Test Data\\00e9be89-00001005_train_color.png', 1) # Read image.
# start = timer()
# processed = paint_path(img, (89,92))
# end = timer()
# print(end - start)
# # The time it takes to run this function should be around 12ms

# ''' Display the image '''
# cv2.imshow('ngon', processed)
# cv2.waitKey(0)
    














