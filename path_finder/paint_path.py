# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

# Basic imports
import cv2
import math
import numpy as np

# Local imports
from path_state import PathState
from distance import Heuristic
from depth_distance import Measure
from astar import astar
from timeit import default_timer as timer
from skimage import draw
from process_depth import *

def paint_path(depth_image,image,Measure, \
                depth_scale=0.001,threshold=(70,100)):

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
        The scale of the stream of depth coming from the Realsense camera 
        (for example, 0.001 means a value of 1000 for a pixel given by the 
        camera equals 1 meter in real life)
    threshold : pair of (int,int)
        Brightness value (B&W) of the part of the pavement the robot needs
        to follow
    
    Returns
    -------
    image : numpy.ndarray
        The painted version of the image.
    """

    clipping_distance_in_meters = 4
    clipping_distance = 4 / depth_scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = int(image.shape[0]), int(image.shape[1]-1)
    mid_bottom = height-1,width/2
    # This is the new values for the thresholded pixels.
    # When the pixels where the pavements appear is determined,
    # they are assigned to the first value of the tuple.
    # Every other pixels are assigned to the latter value.
    new_val = 90, 0

    # Threshold only the needed part of the image.
    def process_image(image,cond):
        image = remove_background(depth_image,image,clipping_distance_in_meters,depth_scale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[cond(gray)] = new_val[0]
        gray[~cond(gray)] = new_val[1]     # Boolean Indexing

        return gray

    # Function which return a boolean-indexed version of the image.
    # It selects only the pixel contains the value that fits the given brightness
    # of the pavement.
    def cond(image):
        return ((image > threshold[0]) &  # threshold[1] and threshold[2] is the range of 
                (image < threshold[1]))   # brightness of pavement surface (the part of the picture we need).
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
            # i = int(height*0.1)
            i = 40
            all_rows = set(all_pavement[0])
            all_rows = list(all_rows)
            while goal is None and i < len(all_rows):
                row = all_rows[i]
                lineup = np.where(processed_img[row] == new_val[0])[0]
                goal = elim_blocked(lineup,row)
                i += 1

            return goal

        # Find the bottom middle pixel which has the brightness value of 'new_val' and pass the 
        # 'blocked' condition.
        def find_start(all_pavement):
            lineup = np.where(processed_img[bot_row] == new_val[0])[0]
            start = elim_blocked(lineup,bot_row)
            return start

        # Find all the indices of the pixel which has 'new_val' brightness value.
        all_pavement = np.where(processed_img[:bot_row,:] == new_val[0])

        if all_pavement[0].size != 0:
            goal = find_goal(all_pavement)
            start = find_start(all_pavement)
            # print(goal)
            # print(start)
            return goal,start
            
        else:
            return None,None

    def get_turning_angle(mid_bottom,destination):
        bot_right = (height-height/10,width/2)
        ang = math.degrees(math.atan2(destination[1]-mid_bottom[1], destination[0]-mid_bottom[0]) - \
            math.atan2(bot_right[1]-mid_bottom[1], bot_right[0]-mid_bottom[0]))
        # return str(ang + 360) if ang < 0 else str(ang)
        return 360+ang if ang < -180 else ang


    processed_img = process_image(image, cond)

    goal,start = find_goal_start(processed_img)
    print(goal)

    if goal is not None and start is not None:
        ''' Let's go!!! '''
        grid_S = PathState(start,processed_img,Measure)
        grid_G = PathState(goal, processed_img,Measure)
        heuristic = Heuristic(grid_G, Measure.measure)

        plan1 = astar(grid_S,
                        lambda state: heuristic(state) < 0.3,
                        heuristic)

        def valid_plan(cac):
            if nice != None:
                wow = list(nice)
                if len(wow) != 0:
                    return wow

        plan = valid_plan(plan1)
        if plan != None:
            first = plan[0][0]
            first_target = (mid_bottom,first)

            # Print the turning angle on image
            turning_angle = get_turning_angle(mid_bottom,first)
            text_position = int(height/10),int(width/10)
            cv2.putText(image,str(turning_angle),text_position,font,1,(255,255,255),1,cv2.LINE_AA)
            plan.insert(0,first_target)

            # Draw the path
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
    nice = list(range(1000))
    start_time = time.time()
    
    wow = set(nice)
    new = list(wow)

    end_time = time.time()
    print(new)

    


    