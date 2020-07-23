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

from paint_path import paint_path
from torch.utils.data import Dataset, DataLoader
from dataset import *
from models import *
from new_models import *
from obstacle_detection import *
from depth_distance import Measure


if __name__ == "__main__":

    #########################################
    ## INITIALIZE MODELS / ALGORITHMS      ##
    #########################################

    # Initialize dataset instances for some processing utilities:
    drivable_fn = Drivable(new_size=(448, 224))
    classes = ['road', 'sidewalk', 'terrain', 'person', 'car']
    segment_fn = BCG(classes=classes, new_size=(448, 224))
    # Get available device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize a model:
    drivable = ResAttnGateUNet(ch_in=3, ch_out=3, ch_init=16, bias=False)
    segment = nn.DataParallel(OrigUNet(in_channels=3, out_channels=len(classes)+1, init_features=16, bias=True))
    #unet = nn.DataParallel(unet)
    # Load pretrained parameters:
    drivable.load_state_dict(torch.load('./saved_models/21.07.20_ResAttnGateUNet_23_val_cel=-0.1606.pt', map_location='cpu'))
    segment.load_state_dict(torch.load('./saved_models/25.06.20_unet_61_val_nll=-0.20364859596888224.pth', map_location='cpu'))
    # Load model to current device:
    drivable.to(device)
    segment.to(device)
    # Toggle evaluation mode:
    drivable.eval()
    segment.eval()

    #########################################
    ## READ CAMERA FEED & APPLY ALGORITHMS ##
    #########################################
    bag = r'20200722_160359.bag'
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_device_from_file(bag, False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    depth_sensor = profile.get_device()

    # # Initialize a camera stream:
    # # Configure depth and color streams
    # pipeline = rs.pipeline()
    
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # width,height = 640,480

    # # Start streaming
    # profile = pipeline.start(config)

    # # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = 0.001
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
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Our operations on the frame come here

        # Resize camera frame:
        frame = drivable_fn.resize(color_image)
        # Normalize image frame:
        tmp = drivable_fn.image_transform(frame).transpose((2, 0, 1))
        # Convert np.array to torch tensor and push to device:
        tmp = torch.from_numpy(tmp).to(device)
        tmp = tmp.unsqueeze(0)
        # Segmentation model:
        with torch.no_grad():
            seg = segment(tmp)
            drive = drivable(tmp)
        # Get the predict label (with highest probability):
        seg = seg.argmax(dim=1).cpu().numpy()[0]
        drive = drive.argmax(dim=1).cpu().numpy()[0]
        # Convert to colored frame:
        seg = segment_fn.convert_label(seg, True)
        seg = segment_fn.convert_color(seg, False)
        seg = seg[:,:,::-1]
        seg = cv2.resize(seg, (640, 480), interpolation=cv2.INTER_AREA)
        drive = drivable_fn.convert_color(drive, False)[:, :, ::-1]
        drive = cv2.resize(drive, (640, 480), interpolation=cv2.INTER_AREA)
        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)


        # # Resize camera frame:
        # seg = fn.resize(color_image)
        # dep = fn.resize(depth_image)

        # # Normalize image frame:
        # seg = fn.image_transform(seg)
        # # Convert np.array to torch tensor and push to device:
        # seg = torch.from_numpy(seg).to(device)
        # # Divide image frame into fragments to utilize GPUs:
        # seg = fragment(seg, n, True)
        # # Stack frame fragments into batch of tensors:
        # seg = torch.stack(seg)
        # # Segmentation model:
        # with torch.no_grad():
        #     seg = unet(seg)
        # # Get the predict label (with highest probability):
        # seg = seg.argmax(dim=1).cpu().numpy()
        # # Compress fragments to a single frame:
        # seg = glue_fragments(seg, n)
        # # Convert to colored frame:
        # seg = fn.convert_label(seg, True)
        # seg = fn.convert_color(seg, False)
        # seg = seg[:,:,::-1]
        # Draw possible path:

        drive,obstacles = detect_obstacle(depth_image, drive, depth_scale)
        daMeasure = Measure(depth_frame,color_frame,depth_scale, obstacles)
        output = paint_path(seg, (89, 92), daMeasure)
        ngon = np.hstack((seg,drive,color_image))
        # Display the resulting frame
        cv2.imshow('frame',ngon)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()



