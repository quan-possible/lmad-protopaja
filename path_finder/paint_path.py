# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

# Basic imports
import cv2
import math
import numpy as np

# Local imports
from path_state import *
from distance import Heuristic
from depth_distance import Measure
from astar import astar
from skimage import draw
from process_depth import *

def paint_path(depth_image,image,Measure, \
                depth_scale=0.001,thres=(70,80)):

    """
    Paint the trajectory for the robot. It uses the A-star algorithm to find the shortest
    path to the upper most pixel which contains the segmented pavement.

    Prerequisites
    -------------

    Realsense SDK and Pyrealsense: Realsense camera interface.
    Opencv: Works with images.
    Numpy: Matrices manipulations and calculations.
    Skikit-image: Paints the image.
    
    Parameters
    ----------
    depth_image: numpy.ndarray
        Depth image coming from the Realsense camera
    image : numpy.ndarray
        RGB image of the street
    Measure : Measure object
        See 'depth_distance.py'
    depth_scale : float
        The scale of the stream of depth coming from the Realsense camera.
        (For example, depth_scale=0.001 means a pixel value of 1000 equals
         1 meter in real life)
    thres : pair of (int,int)
        Brightness value (B&W) of the part of the pavement the robot needs
        to follow
    
    Returns
    -------
    image : numpy.ndarray
        The painted version of the image.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = int(image.shape[0]), int(image.shape[1])
    mid_bottom = height-1,int(width/2)

    # Clipping distance used to remove background. This function only acts on
    # target within a 4-meter radius.
    clipping_distance_in_meters = 4
    clipping_distance = 4 / depth_scale

    # thres only the needed part of the image (i.e. the pavement we are travelling on).
    def process_image(image,cond):
        image = remove_background(depth_image,image,clipping_distance_in_meters,depth_scale)
        mask = cv2.inRange(image, (0, 0, 255), (0, 0, 255))
        """gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[cond(gray)] = new_val
        gray[~cond(gray)] = 0    # Boolean Indexing"""
        mask[mask == 255] = new_val
        return mask

    # Function which return a boolean-indexed version of the image.
    # It selects only the pixel contains the value that fits the given brightness
    # of the pavement.
    def cond(image):
        return np.logical_and(image>thres[0], image<thres[1])   # brightness of pavement surface (the part of the picture we need).
        # return image == thres[0]
        
        # Return False if its not the pavement


    # Find the coordinate of the starting point and the goal for the A-star algorithm.
    # Note: This requires the image to be already processed (using the function processed_image above)
    def find_goal_start(processed_img):
        bot_row = int(height-height*0.04)

        # Find the point in the middle of the row that pass the 'blocked' condition from object Measure.
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

        # Find the pixel highest (x is smallest) in the middle (y = width/2) which contains the pavement
        # and pass the 'blocked' condition.
        def find_goal(all_pavement):
            goal = None
            i = 40 # not 0 as to avoid the fluctuating edge.
            all_rows = set(all_pavement[0])
            all_rows = list(all_rows)
            while goal is None and i < len(all_rows):
                row = all_rows[i]
                lineup = np.where(processed_img[row] == new_val)[0]
                goal = elim_blocked(lineup,row)
                i += 1

            return goal

        # Find the bottom middle pixel which has the brightness value of 'new_val' and pass the 
        # 'blocked' condition.
        def find_start(all_pavement):
            lineup = np.where(processed_img[bot_row] == new_val)[0]
            start = elim_blocked(lineup,bot_row)
            return start

        # Find all the indices of the pixel which has 'new_val' brightness value.
        all_pavement = np.where(processed_img[:bot_row,:] == new_val)

        # Start finding goal and start
        if all_pavement[0].size != 0:
            goal = find_goal(all_pavement)
            start = find_start(all_pavement)

            return goal,start
            
        else:
            return None,None

    # Given the first point on the path, calculate the angle the robot has to make to get to that
    def get_turning_angle(target):
        bot_right = (height-height/10,width/2)
        ang = math.degrees(math.atan2(target[1]-mid_bottom[1], target[0]-mid_bottom[0]) - \
            math.atan2(bot_right[1]-mid_bottom[1], bot_right[0]-mid_bottom[0]))
        # return str(ang + 360) if ang < 0 else str(ang)
        return 360+ang if ang < -180 else ang

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    processed_img = process_image(image, cond)
    
    goal,start = find_goal_start(processed_img)
    print(goal)

    if goal is not None and start is not None:
        ''' Let's go!!! '''
        grid_S = PathState(start,processed_img,Measure)
        grid_G = PathState(goal, processed_img,Measure)
        heuristic = Heuristic(grid_G, Measure.measure)

        plan1 = astar(grid_S,
                        lambda state: heuristic(state) < 1,
                        heuristic)

        def valid_plan(nice):
            if nice != None:
                wow = list(nice)
                if len(wow) != 0:
                    return wow

        plan = valid_plan(plan1)
        if plan != None:


            # Print the turning angle on image
            first = plan[0][0]
            first_target = (mid_bottom,first)
            turning_angle = get_turning_angle(first)
            text_position = int(height - height/10),int(width/10)
            cv2.putText(image,str(turning_angle),text_position,font,1,(255,255,255),1,cv2.LINE_AA)
            plan.insert(0,first_target)

            # Draw the path
            for x, y in plan:
                # rr, cc = draw.line(
                #     int(x[0]), int(x[1]), int(y[0]), int(y[1]))
                # image[rr, cc] = 255
                line_thickness = 2
                cv2.line(image, (int(x[1]), int(x[0])), (int(y[1]), int(y[0])), \
                 (255, 255, 255), thickness=line_thickness)

        else:
            cv2.putText(image,'No path found!',(height//2,width//2), \
            font,1,(255,255,255),2, cv2.LINE_AA)
    else:
        cv2.putText(image,'No pavement detected!',(height//2,width//2), \
        font,1,(255,255,255),2, cv2.LINE_AA)

    return image



if __name__ == "__main__":
    nice = list(range(1000))
    start_time = time.time()
    
    wow = set(nice)
    new = list(wow)

    end_time = time.time()
    print(new)

    


    