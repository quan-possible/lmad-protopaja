# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2



def process_depth(depth_image):

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.12), cv2.COLORMAP_JET)
    smoothened = cv2.bilateralFilter(depth_colormap,15,80,80)

    return smoothened


def remove_background(depth_image,target_image,clipping_distance_in_meters,depth_scale=0.001,fillin = 0):

    clipping_distance = clipping_distance_in_meters / depth_scale
    # Remove background - Set pixels further than clipping_distance to grey
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) # depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), fillin, target_image)

    return bg_removed







