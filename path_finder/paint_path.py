# Append 'path_finder' folder to the directory
import sys
sys.path.insert(1, 'path_finder')
# Basic imports
import cv2
import math
import numpy as np
# Local imports
from path_state import *
from distance import euclidean
from astar import astar
from depth_distance import Measure
# External imports
from timeit import default_timer as timer
from skimage import draw
import pyrealsense2

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
        blocked = Measure.blocked
        measure = Measure.measure
        ''' Let's go!!! '''
        grid_S = PathState(start,processed_img,blocked)
        grid_G = PathState(goal, processed_img,blocked)
        heuristic = Heuristic(grid_G,measure)

        plan1 = astar(grid_S,
                        lambda state: heuristic(state) < 21,
                        heuristic)

        condition = False

        def valid_plan(nice):
            if nice != None:
                wow = list(nice)
                if len(wow) != 0:
                    return wow

        plan = valid_plan(plan1)
        if plan != None:
            first = plan[0][0]
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





''' Let's have some tests! '''
if __name__ == "__main__":
    img = cv2.imread(
        'Test Data\\00e9be89-00001005_train_color.png', 1) # Read image.
    start = timer()
    processed = paint_path(img, (89,92))
    end = timer()
    print(end - start)
    # The time it takes to run this function should be around 12ms

    ''' Display the image '''
    cv2.imshow('ngon', processed)
    cv2.waitKey(0)


if __name__ == "__main__":
    # Create a pipeline
    pipeline = rs.pipeline()
    bag = '20200716_170459.bag'

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_device_from_file(bag, False)
    config.enable_all_streams()
    width,height = 640,480

    profile = self.pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(True)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2 #2 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame() # depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            daMeasure = Measure(depth_frame,color_frame,depth_scale)
            output,contours = detect_obstacle(depth_image, color_image, depth_scale)

            cv2.imshow('ngon', output)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
    














