import cv2
import math
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import statistics
from itertools import zip_longest
from timeit import default_timer as timer
from skimage import draw


det_val, trash_val = 90, 0
lum_range = (89,92)

# Calculate magnitude of vector
def mag(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Remove all the unecessary cars, skies,... and make it a 0 value (trash_val)
def process_image(image, cond):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[cond(gray)] = det_val
    gray[~cond(gray)] = trash_val     # Boolean Indexing
    return gray

def cond(image):
    return ((image > lum_range[0]) &  # lum_range[1] and lum_range[2] is the range of luminance
            (image < lum_range[1]))           # of road surface (the part of the picture we need).
    # Return False if its not the road

# def find_top_road(image):
#     first_row = np.where(image == 90)[0]
#     all_in_first_row = np.where(image[first_row] == 90)[0]
#     return first_row,all_in_first_row[len(all_in_first_row)//2]

def check_cond(daRow):
    i = 0
    while daRow[i] != 90:
        i += 1
    else:
        return i

def find_average_row(row,width,check_cond):
    reversed_row = row[::-1]
    found_right = width-np.where(reversed_row == 90)[0][0]-1
    found_left = np.where(row == 90)[0][0]
    # found_right = width-check_cond(reversed_row)
    # found_left = check_cond(row)
    average_pos = (found_left + found_right)/2
    return int(average_pos)

# Find the first occurence of road in the row ,ie. find the edge of the road.
# Admittedly, there are better ways to do this but i did this in the beginning
# for tests, and i saw no harm.
def find_cali_area(image):
    height, width = int(image.shape[0]), int(image.shape[1])
    try:
        itemindex = np.where(image[0:int(height-height/10),:] == 90)
        return itemindex[0][0], itemindex[0][::-1][0]
    except:
        return None

def paint_path(image):
    height, width = int(image.shape[0]), int(image.shape[1]-1)

    ''' Process the image. Refer to the process_image function for more details. '''
    processed_img = process_image(image, cond)
    # cv2.imshow('ngon', processed_img)
    # cv2.waitKey(0)
    # The rows of the image which have roads in it
    # try:
    cali_area = find_cali_area(processed_img)

    ''' This single loop do the whole job '''
    if cali_area != None:
        avg_list = []
        i = cali_area[0]
        
        for row in processed_img[cali_area[0]:cali_area[1]]:
            try:
                avg_list.append((i, find_average_row(row,width,check_cond)))
                i += 1
            except:
                pass
        avg_list = avg_list[::10]
        avg_list.append((height-1,width/2))
        nice = zip(avg_list,avg_list[1:])
        # print(list(nice))
        for x, y in nice:
            rr,cc = draw.line(int(x[0]),int(x[1]),int(y[0]),int(y[1]))
            image[rr, cc] =  255
    else:
        rr,cc = draw.line(int(height-1),int(width/2),0,int(width/2))
        image[rr, cc] =  255
# except:
        # pass

    return image

# # List of constants

img = cv2.imread(
    'Test Data\\00e9be89-00001005_train_color.png', 1) # Read image.

processed = paint_path(img)

# your code execution
# e2 = cv.getTickCount()
# time = (e2 - e1)/ cv.getTickFrequency()

# start = timer()
# img = cv2.imread(
#     'Test Data\\00e9be89-00001005_train_color.png', 1) # Read image.
# end = timer()
# print(end - start)
# height, width = int(img.shape[0]), int(img.shape[1])
# lum_range[1], lum_range[2] = 89, 92  # CHANGE THIS FOR DIFFERENT TYPE OF LABELS.
# det_val, trash_val = 90, 0

# ''' Process the image. Refer to the process_image function for more details. '''
# processed_img = process_image(img, cond, det_val, trash_val)
# cali_top, cali_bot = find_cali_area(processed_img) # The rows of the image which have roads in it
# ''' This single loop do the whole job '''
# for x, y in paint_path(processed_img):
#     processed_img[x, y] = 255
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
# # OBSOLETE. Find target for robot.
# def __find_dot(image, cali):
#     avg_list = []
#     for row in image[cali:cali+20]:
#         avg_list.append(find_average_row(row))
#     return int(statistics.mean(avg_list)), int(cali)

''' Display the image '''
cv2.imshow('ngon', processed)
cv2.waitKey(0)
# imgplot = plt.imshow(processed_img)
# plt.show()

# print((processed_img > lum_range[1]) & (processed_img < lum_range[2]))
