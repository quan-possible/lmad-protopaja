# Interfacing Realsense camera
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

'''
File contains helper functions for processing the depth image.
'''

def process_depth(depth_image):

    """
    Given depth_image, return smoothened depth_colormap.

    Helper function

    Parameters
    ----------
    depth_image : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Smoothened depth_colormap.
    """
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.12), cv2.COLORMAP_JET)
    smoothened = cv2.bilateralFilter(depth_colormap,15,80,80)

    return smoothened


def remove_background(depth_image,target_image,clipping_distance_in_meters,depth_scale=0.001,fillin = 0):

    """
    Remove background of image.

    Helper function

    Parameters
    ----------
    depth_image : numpy.ndarray
        Depth image.
    target_image : numpy.ndarray
        Image to be removed background.
    clipping_distance_in_meters : distance
        Distance to background
    depth_scale : float
        The scale of the stream of depth coming from the Realsense camera.
        (For example, depth_scale=0.001 means a pixel value of 1000 equals
         1 meter in real life)
    fillin : int
        Value of pixel to fill in the removed background.

    Returns
    -------
    numpy.ndarray
        Smoothened depth_colormap.
    """

    clipping_distance = clipping_distance_in_meters / depth_scale
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) # depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), fillin, target_image)
    
    return bg_removed







