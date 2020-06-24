import cv2
from matplotlib import pyplot as plt 
import numpy as np

img = cv2.imread('Test Data\color_labels\\train\\0a0c3694-f3444902_train_color.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

'''Edge Detection'''
edges = cv2.Canny(gray, 20, 30)
edges_high_thresh = cv2.Canny(gray, 60, 120)
images= np.hstack((gray, edges, edges_high_thresh))

''' HoughLinesP Transformation'''

# rho = 1  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid
# threshold = 15  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 50  # minimum number of pixels making up a line
# max_line_gap = 20  # maximum gap in pixels between connectable line segments
# line_image = np.copy(img) * 0  # creating a blank to draw lines on

# # Run Hough on edge detected image
# # Output "lines" is an array containing endpoints of detected line segments
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                     min_line_length, max_line_gap)

# for line in lines:
#     for x1,y1,x2,y2 in line:
#     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)


'''
    PRINT THE PICTURE
'''

# f,ax = plt.subplots(1,2)
# ax[0].imshow(gray)
# ax[1].imshow(edges_high_thresh)
# plt.show()

cv2.imshow('ngon', gray)
cv2.waitKey(0)
plt.hist(gray.ravel(),256,[0,256])
plt.show()

