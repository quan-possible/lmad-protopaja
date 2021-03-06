# Import path_finder folder
import sys
sys.path.insert(1, 'path_finder')

# Basic import:
import cv2
import sys
import numpy as np
import pyrealsense2 as rs
import time

# Pytorch import:
import torch
import torch.nn.functional as F

# Local import:
from paint_path import paint_path
from torch.utils.data import Dataset, DataLoader
from dataset import *
from models import *
from obstacle_detection import *
from depth_distance import Measure
from process_depth import process_depth,remove_background


if __name__ == "__main__":

    #########################################
    ##   INITIALIZE MODELS / ALGORITHMS    ##
    #########################################

    # Initialize dataset instances for some processing utilities:
    drivable_fn = Drivable(new_size=(448, 224))
    classes = ['road', 'sidewalk', 'terrain', 'person', 'car']
    segment_fn = BCG(classes=classes, new_size=(448, 224))
    # Get available device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize a model:
    drivable = ResAttnGateUNet(ch_in=3, ch_out=3, ch_init=16, bias=False)
    segment = nn.DataParallel(
        UNet(in_channels=3, out_channels=len(classes)+1, init_features=16, bias=True))
    #unet = nn.DataParallel(unet)
    # Load pretrained parameters:
    drivable.load_state_dict(torch.load(
        './saved_models/21.07.20_ResAttnGateUNet_23_val_cel=-0.1606.pt', map_location='cpu'))
    segment.load_state_dict(torch.load(
        './saved_models/25.06.20_unet_61_val_nll=-0.20364859596888224.pth', map_location='cpu'))
    # Load model to current device:
    drivable.to(device)
    segment.to(device)
    # Toggle evaluation mode:
    drivable.eval()
    segment.eval()

    #########################################
    ## READ CAMERA FEED & APPLY ALGORITHMS ##
    #########################################
    bag = r'20200812_162029.bag'
    pipeline = rs.pipeline()
    width, height = 640, 480

    config = rs.config()

    ''' If uncommented, the program will use video stream from Realsense camera directly '''
    # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    ''' If uncommented, the program will use .bag file '''
    config.enable_device_from_file(bag, False)
    config.enable_all_streams()

    profile = pipeline.start(config)
    depth_sensor = profile.get_device()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = 0.001
    print("Depth Scale is: ", depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    n = 1
    out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (width*2, height))

    while(True):
        start_time = time.time()
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
        depth_colormap = process_depth(depth_image)
        # Our operations on the frame come here

        # Resize camera frame:
        frame = drivable_fn.resize(color_image[:, :, ::-1])
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
        drive = drivable_fn.convert_color(drive, False)[:, :, ::-1]
        drive[seg == 1] = (232, 35, 244)
        drive[seg == 2] = (152, 251, 152)
        drive[seg == 3] = (60, 20, 220)
        drive[seg == 4] = (142, 0, 0)
        drive = cv2.resize(drive, (640, 480), interpolation=cv2.INTER_AREA)

        # Detect obstacles
        obs_image, obstacles = detect_obstacle(
            depth_image, drive, depth_colormap, depth_scale)
        # Form Measure object from the obstacles
        measure_obj = Measure(depth_frame, color_frame, depth_scale, obstacles)
        # Paint the path.
        output = paint_path(depth_image, drive, measure_obj, depth_scale)
        blended = cv2.add(obs_image, output)
        stacked = np.hstack((blended, color_image))
        end_time = time.time()
        print(str(1/(end_time-start_time)) + ' FPS')
        # Write to video
        out.write(stacked)
        # Display the resulting frame
        cv2.imshow('frame', stacked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed

    # When everything done, release the capture
    cv2.destroyAllWindows()
