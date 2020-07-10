# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

# Basic import:
import cv2
import sys
import numpy as np
import pyrealsense2 as rs

# Pytorch import:
import torch
import torch.nn.functional as F

import paint_path
from torch.utils.data import Dataset, DataLoader
from dataset import *
from models import *
from obstacle_detection import *


def fragment(img, n, channel_first=False):
    """ Fragment image into smaller pieces.
        Height & width of the images is divided
        by a specific number.

        Args:
            img (2d-array like): Input image for fragmentation.
            n (int): The number the height & width be divided into.
            channel_first (bool): If True, then color channel is set
                    as first channel (suitable for torch tensor).

        Return:
            fragments (list of 2d-array like): List of fragmented frame.
    """
    h, w, c = img.shape
    step_h = h // n
    step_w = w // n
    fragments = []
    for i in range(0, h-step_h+1, step_h):
        for j in range(0, w-step_w+1, step_w):
            if channel_first:
                fragments.append(img[i:i+step_h, j:j+step_w].permute(2, 0, 1))
            else:
                fragments.append(img[i:i+step_h, j:j+step_w])
    return fragments

def glue_fragments(img, n):
    """ Combine fragmented framed into a single frame.

        Args:
            img (2d-array like): Input image for combining.
            n (int): The number of fragments per side.

        Return:
            frame (2d-array like): Glued frame.
    """
    h, w = img[0].shape
    frame = np.zeros((h*n, w*n))
    for i in range(n):
        for j in range(n):
            h_pos, w_pos = i*h, j*w
            frame[h_pos:h_pos+h, w_pos:w_pos+w] = img[i*n+j]
    return frame

if __name__ == "__main__":

    #########################################
    ## INITIALIZE MODELS / ALGORITHMS      ##
    #########################################

    # Auxiliary dataset to use image processing functions:
    classes = ['road', 'sidewalk', 'terrain', 'person', 'car']
    fn = BCG(classes=classes, new_size=(320,160))

    # Get available device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize a model:
    unet = nn.DataParallel(
        UNet(in_channels=3,
             out_channels=len(classes)+1,
             init_features=16,
             bias=True)
    )
    # Load pretrained parameters:
    unet.load_state_dict(torch.load('./saved_models/25.06.20_unet_61_val_nll=-0.20364859596888224.pth', map_location='cpu'))
    # Load model to current device:
    unet.to(device)
    # Toggle evaluation mode:
    unet.eval()

    #########################################
    ## READ CAMERA FEED & APPLY ALGORITHMS ##
    #########################################

    # Initialize a camera stream:
    # Configure depth and color streams
    pipeline = rs.pipeline()
    
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

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    n = 1

    while(True):
        # Capture frame-by-frame
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Our operations on the frame come here
        # Resize camera frame:
        seg = fn.resize(color_image)
        dep = fn.resize(depth_image)

        # Normalize image frame:
        seg = fn.image_transform(seg)
        # Convert np.array to torch tensor and push to device:
        seg = torch.from_numpy(seg).to(device)
        # Divide image frame into fragments to utilize GPUs:
        seg = fragment(seg, n, True)
        # Stack frame fragments into batch of tensors:
        seg = torch.stack(seg)
        # Segmentation model:
        with torch.no_grad():
            seg = unet(seg)
        # Get the predict label (with highest probability):
        seg = seg.argmax(dim=1).cpu().numpy()
        # Compress fragments to a single frame:
        seg = glue_fragments(seg, n)
        # Convert to colored frame:
        seg = fn.convert_label(seg, True)
        seg = fn.convert_color(seg, False)
        seg = seg[:,:,::-1]
        # Draw possible path:
        seg = cv2.resize(seg, (640, 480), interpolation=cv2.INTER_AREA)
        # seg = paint_path.paint_path(seg, (89, 92))
        output = detect_obstacle(depth_image, seg, depth_scale)
        # ngon = np.hstack((output,color_image))
        # Display the resulting frame
        cv2.imshow('frame',output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()



