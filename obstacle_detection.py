## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
from statistics import mean
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import argparse
import random as rng
rng.seed(12345)


# Streaming loop


def detect_obstacle(depth_image, color_image, Measure,\
                        clipping_distance_in_meters=2):

    width,height = 640,480
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = int(height/10),int(width/10)
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    depth_scale = Measure.depth_scale
    clipping_distance = clipping_distance_in_meters / depth_scale
    threshold = 0.1 / depth_scale
    
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 0
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_JET)

    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, depth_colormap)

    smoothened = cv2.bilateralFilter(bg_removed,15,80,80)
    cannied = cv2.Canny(smoothened,20,60)
    # cannied_bool = np.where(np.logical_and(cannied == 255, depth_image < 1900))
    cannied_bool = np.logical_and(cannied == 255, depth_image < (clipping_distance-threshold))

    new_cannied = np.zeros((height,width),dtype=np.uint8)

    new_cannied[cannied_bool] = 255
    # gray_color_image[cannied_bool] = 255

    kernel = np.ones((21,21), np.uint8)
    img = cv2.dilate(new_cannied,kernel,1)
    img = cv2.erode(img, kernel, 1)

    contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    # Filter out contours that have 0 depth-value  
    filtered_contours = []
    for c in contours:
        daCon = list(tuple(reversed(item)) for sublist in c for item in sublist)
        filtering = list(filter(lambda x: depth_image[x] > 0, daCon))
        filtered_contours.append(filtering)

    distance = []
    for c in filtered_contours:
        # determine the most extreme points along the contour
        # resized = c[::2]
        all_dist = list(map(lambda x: depth_image[x],c))
        if all_dist:
            distance.append(mean(all_dist))
                
    if filtered_contours and distance:
        cv2.putText(color_image,str(min(distance)),text_position,font,1,(255,255,255),1,cv2.LINE_AA)

    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours[i])

    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(color_image, contours_poly, i, color)

    return color_image,filtered_contours

if __name__ == "__main__":
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    width,height = 640,480

    # Start streaming
    profile = pipeline.start(config)

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
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            output,contours = detect_obstacle(depth_image, color_image, depth_scale,\
                            clipping_distance_in_meters)

            cv2.imshow('ngon', output)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


        