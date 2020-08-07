# Basic import:
import cv2
import sys
import numpy as np
# Pytorch import:
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import *
from models import *
from path_finding import *

if __name__ == "__main__":

    #########################################
    ## INITIALIZE MODELS / ALGORITHMS      ##
    #########################################

    # Initialize dataset instances for some processing utilities:
    drivable_fn = Drivable(new_size=(320, 160))
    classes = ['road', 'sidewalk', 'terrain', 'person', 'car']
    segment_fn = BCG(classes=classes, new_size=(320, 160))
    # Get available device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize a model:
    drivable = ResAttnGateUNet(ch_in=3, ch_out=3, ch_init=16, bias=False)
    segment = nn.DataParallel(UNet(in_channels=3, out_channels=len(classes)+1, init_features=16, bias=True))
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

    # Initialize a camera stream:

    #cap = cv2.VideoCapture(2)
    import pafy
    url = "https://www.youtube.com/watch?v=B8Yyf6WKgaM"
    vpafy = pafy.new(url)
    play = vpafy.getbest(preftype="mp4")
    cap = cv2.VideoCapture(play.url)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Resize camera frame:
        frame = drivable_fn.resize(frame)
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
        #seg = cv2.resize(seg, (720, 360), interpolation=cv2.INTER_AREA)
        drive = drivable_fn.convert_color(drive, False)[:, :, ::-1]
        #drive = cv2.resize(drive, (720, 360), interpolation=cv2.INTER_AREA)
        #frame = cv2.resize(frame, (720, 360), interpolation=cv2.INTER_LINEAR)
        #seg = paint_path(seg, (89, 92))


        # Display the resulting frame
        cv2.imshow('frame',np.vstack((seg, drive, frame)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


