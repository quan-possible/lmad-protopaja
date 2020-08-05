# Local imports
from process_depth import *

# Basic imports
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import random as rng
from statistics import mean

rng.seed(12345)

def detect_obstacle(depth_image, color_image,depth_colormap,depth_scale = 0.001):

    """
    Detect obstacles by deploying Canny filter on the depth_colormap.

    Prerequisites
    -------------
    Realsense SDK and Pyrealsense: Realsense camera interface.
    Opencv: Works with images.
    Numpy: Matrices manipulations and calculations.
    
    Parameters
    ----------
    depth_image: numpy.ndarray
        Depth image coming from the Realsense camera
    color_image : numpy.ndarray
        RGB image
    depth_colormap : numpy.ndarray
        Depth image converted to colormap
    depth_scale : float
        The scale of the stream of depth coming from the Realsense camera.
        (For example, depth_scale=0.001 means a pixel value of 1000 equals
         1 meter in real life)
    
    Returns
    -------
    image : numpy.ndarray
        The painted version of the image.
    filtered_contours : list of list of (int,int)
        list of list of pixels that form the contours of the obstacles.

    """

    width,height = 640,480
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = int(height/10),int(width/10)
    # We will be removing the background of objects more than
    # clipping_distance_in_meters meters away
    clipping_distance_in_meters = 5
    clipping_distance = 5 / depth_scale
    threshold = 0.1 / depth_scale

    # First, remove the background
    bg_removed = remove_background(depth_image,depth_colormap,clipping_distance_in_meters)

    # Deploy the Canny filter
    cannied = cv2.Canny(bg_removed,20,100)

    # Cut off the line at the border
    cannied_bool = np.logical_and(cannied == 255, depth_image < (clipping_distance-threshold))
    new_cannied = np.zeros((height,width),dtype=np.uint8)
    new_cannied[cannied_bool] = 255

    # Perform morphological operations
    kernel = np.ones((21,21), np.uint8)
    img = cv2.dilate(new_cannied,kernel,1)
    img = cv2.erode(img, kernel, 1)

    # Find Contours
    contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    # Filter out contours that have 0 depth-value  
    filtered_contours = []
    for c in contours:
        daCon = list(tuple(reversed(item)) for sublist in c for item in sublist)
        filtering = list(filter(lambda x: depth_image[x] > 0, daCon))
        filtered_contours.append(filtering)

    # Blank image for drawing contours.
    drawing = np.zeros((new_cannied.shape[0], new_cannied.shape[1], 3), dtype=np.uint8)

    # Determine the closest obstacle and its distance to the robot
    distance = []
    for c in filtered_contours:
        # Determine the most extreme points along the contour
        all_dist = list(map(lambda x: depth_image[x],c))
        if all_dist:
            distance.append(mean(all_dist))

    if filtered_contours and distance:
        cv2.putText(drawing,str(min(distance)),text_position,font,1,(255,255,255),1,cv2.LINE_AA)

    # Approximate contours to polygons
    contours_poly = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)

    # Draw polygonal contours
    for i in range(len(contours_poly)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)

    # nice = np.hstack((color_image, depth_colormap, drawing))

    # Return the contours image and the list of contours
    return drawing,filtered_contours

if __name__ == "__main__":
    bag = r'20200722_150121.bag'
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_device_from_file(bag, False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    depth_sensor = profile.get_device()

    depth_scale = 0.001
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 5 
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
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = process_depth(depth_image)

            output,contours = detect_obstacle(depth_image,color_image,depth_colormap, depth_scale)

            cv2.imshow('co', output)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


        