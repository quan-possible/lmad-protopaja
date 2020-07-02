# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

# Basic import:
import cv2
import sys
import numpy as np
# Pytorch import:
import torch
import torch.nn.functional as F

import paint_path
from torch.utils.data import Dataset, DataLoader
from dataset import *
from models import *


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
    fn = BCG(classes=classes, new_size=(320, 160))

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
    n = 1
    cap = cv2.VideoCapture(2)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # Resize camera frame:
        seg = fn.resize(frame)
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
        seg = cv2.resize(seg, (720, 360), interpolation=cv2.INTER_AREA)
        seg = paint_path.paint_path(seg, (89, 92))


        # Display the resulting frame
        cv2.imshow('frame',seg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



