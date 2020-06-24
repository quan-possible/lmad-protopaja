import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import statistics

# Remove all the unecessary cars, skies,... and make it a 0 value (trash_val)
def process_image(image, cond, det_val, trash_val):
    image[cond(image)] = det_val
    image[~cond(image)] = trash_val     # Boolean Indexing
    return image


def cond(check_what):
    return ((check_what > lum_range1) &  # lum_range1 and lum_range2 is the range of luminance
            (check_what < lum_range2))           # of road surface (the part of the picture we need).
    # Return False if its not the road

# Find the first occurence of road in the image. Return that pixel's coordinate.
def find_road_y(processed_image):
    itemindex = np.where(processed_img == 90)
    return itemindex[0][0], itemindex[0][::-1][0]

# Find the first occurence of road in the row ,ie. find the edge of the road.
# Admittedly, there are better ways to do this but i did this in the beginning 
# for tests, and i saw no harm.
def check_cond(daRow):
    i = 0
    while daRow[i] != 90:
        i += 1
    else:
        return i

# Take the average of 2 edges point
def find_average_row(row):
    reverse_row = row[::-1]
    found_right = width-check_cond(reverse_row)
    found_left = check_cond(row)
    average_pos = (found_left + found_right)/2
    return int(average_pos)

# OBSOLETE. Find target for robot. 
def find_dot(image, cali):
    avg_list = []
    for row in image[cali:cali+20]:
        avg_list.append(find_average_row(row))
    return int(statistics.mean(avg_list)), int(cali)

# Calculate magnitude of vector
def mag(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Most important function. It paints the path 
def paint_path(image):
    avg_list = []
    i = cali_top
    for row in image[cali_top:cali_bot]:
        avg_list.append((i, find_average_row(row)))
        i += 1
    return avg_list


# List of constants
img = cv2.imread(
    'Test Data\color_labels\\val\\7d6c1119-00000000_train_color.png', 0) # Read image.
height, width = int(img.shape[0]), int(img.shape[1])
lum_range1, lum_range2 = 89, 92  # CHANGE THIS FOR DIFFERENT TYPE OF LABELS.
det_val, trash_val = 90, 0

''' Process the image. Refer to the process_image function for more details. '''
processed_img = process_image(img, cond, det_val, trash_val)
cali_top, cali_bot = find_road_y(processed_img) # The rows of the image which have roads in it

''' This single loop do the whole job '''
for x, y in paint_path(processed_img):
    processed_img[x, y] = 255

''' The commented lines below are obsolete '''
# # Let's work!
# cali_distance = height-height/4  # By default, use height-height/4
# target_x, target_y = find_dot(processed_img, int(cali_distance))
# # DRAW THE CIRCLE!!!
# cv2.circle(img, (target_x, target_y), 20, (255, 255, 255), -1)
# # Now the arrow part...
# bot_arrow_x, bot_arrow_y = int(width/2), height
# target_proj = target_x, height
# theta = math.pi/2
# processed_img[paint_path(processed_img)]=255
# if target_x != bot_arrow_x:
#     print((target_proj[0], bot_arrow_x, target_proj[1], bot_arrow_y))
#     print((target_x, bot_arrow_x, target_y, bot_arrow_y))
#     # print(math.acos(mag(target_x, bot_arrow_x, target_y, bot_arrow_y)))
#     theta = math.pi - math.acos(mag(target_proj[0], bot_arrow_x, target_proj[1], bot_arrow_y) /
#                                 mag(target_x, bot_arrow_x, target_y, bot_arrow_y))
# theta = math.acos(mag(mid,mid,cali_top,height) / mag(target[0],mid,target[1],height))
# print(theta)
# arrow_center = int(width-width/5), int(height/5)
# arrow_length = 50
# arrow_tip = int(arrow_center[0] + arrow_length*math.cos(theta)), \
#     int(arrow_center[1] - arrow_length*math.sin(theta))
# real_arrow_x = cv2.line(img, arrow_center, arrow_tip, (255, 255, 255), 2)
# cv2.circle(img, arrow_center, 70, (255, 255, 255), 2)


''' Display the image '''
# cv2.imshow('ngon', img)
# cv2.waitKey(0)
imgplot = plt.imshow(processed_img)
plt.show()

# print((processed_img > lum_range1) & (processed_img < lum_range2))
