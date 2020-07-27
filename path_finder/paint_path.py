# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

import cv2
import math
import numpy as np
from path_state import PathState
from distance import Heuristic
from depth_distance import Measure
from astar import astar
from timeit import default_timer as timer
from skimage import draw

def paint_path(image, road_val_range, Measure):

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

    # def find_road_top_bot(image):
    #     # Remove the bottom part of the image from the search by slicing until (height-height/10)
    #     all_road = np.where(np.logical_and(image[0:int(height-height/10),:] == reclassifying_val[0] 
    #                 ,Measure.blocked(image[0:int(height-height/10),:])))
    #     print(all)
    #     if all_road[0].size != 0:
    #         top_row = all_road[0][0]
    #         top_row_filtered = list(filter(lambda x: Measure > 500, daCon))
    #         list_top_col = np.where(image[top_row] == reclassifying_val[0])[0]
    #         mid_top_col = list_top_col[len(list_top_col)//2]
    #         top = top_row, mid_top_col
            
    #         bot_row = all_road[0][::-1][0]
    #         list_bot_col = np.where(image[bot_row] == reclassifying_val[0])[0]

    #         mid_bot_col = list_bot_col[len(list_bot_col)//2]
    #         bot = bot_row,mid_bot_col
    #         print(top,bot)
    #         return top,bot
    #     else:
    #         return None




    def find_goal_start(image):

        def elim_blocked(lineup,row):
            target = None

            while lineup.size != 0 and target is None:
                index = lineup.size//2
                col = lineup[index]
                if not Measure.blocked((row,col)):
                    target = row,col
                lineup = np.delete(lineup,index)

            return target

        def find_goal(all_road):
            goal = None
            i = 0
            all_rows = all_road[0]
            while goal is None and i < all_rows.size:
                row = all_rows[i]
                lineup = np.where(image[row] == reclassifying_val[0])[0]
                goal = elim_blocked(lineup,row)

            return goal

        def find_start(all_road):
            # start = None
            # row = all_road[0][::-1][0]
            # print(row)
            # lineup = image[row]
            # lineup = lineup[lineup == reclassifying_val[0]]

            # start = elim_blocked(lineup,row)
            start = None
            i = 0
            all_rows = all_road[0][::-1]
            while start is None and i < all_rows.size:
                row = all_rows[i]
                lineup = np.where(image[row] == reclassifying_val[0])[0]
                start = elim_blocked(lineup,row)

            return start

        all_road = np.where(image[0:int(height-height/10),:] == reclassifying_val[0])

        if all_road[0].size != 0:
            goal = find_goal(all_road)
            start = find_start(all_road)
            # print(goal)
            # print(start)
            return goal,start
            
        else:
            return None

    def get_turning_angle(current_pos,destination):
        bot_right = (height-height/10,width/2)
        ang = math.degrees(math.atan2(destination[1]-current_pos[1], destination[0]-current_pos[0]) - \
            math.atan2(bot_right[1]-current_pos[1], bot_right[0]-current_pos[0]))
        # return str(ang + 360) if ang < 0 else str(ang)
        return 360+ang if ang < -180 else ang

    processed_img = process_image(image, cond)
    goal_start = find_goal_start(processed_img)

    if goal_start is not None:
        ''' Let's go!!! '''
        goal,start = goal_start
        grid_S = PathState(start,processed_img)
        grid_G = PathState(goal, processed_img)
        heuristic = Heuristic(grid_G)

        plan1 = astar(grid_S,
                        lambda state: heuristic(state) < 100,
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

if __name__ == "__main__":
    def nice(num):
        return 1 + num

    