import cv2
import math
import numpy as np
import statistics
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from itertools import zip_longest
from timeit import default_timer as timer
from skimage import draw
from scipy.interpolate import BSpline



def paint_path(image, road_val_range):

    """

    This is a small file containing a function that helps paint a trajectory for the robot.

    Prerequisites
    -------------
    Opencv: Works with images.
    Numpy: Matrices manipulations and calculations.
    Skikit-image: Paints the image.

    Parameters
    ----------
    image : Image as numpy array
    road_val_range : The brightness of the road

    Returns
    -------
    image : The painted version of the image.

    """

    height, width = int(image.shape[0]), int(image.shape[1]-1)

    # This is the substitute for the road_val_range.
    # When the pixels where the roads appear is determined,
    # they are assigned to the first value of the tuple.
    # Every other pixels are assigned to the latter value.
    reclassifying_val = 90, 0

    def mag(x1, x2, y1, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def find_road_top_bot(image):
        height, width = int(image.shape[0]), int(image.shape[1])
        try:
            itemindex = np.where(image[0:int(height-height/10),:] == reclassifying_val[0])
            return itemindex[0][0], itemindex[0][::-1][0]
        except:
            return None

    # Find the middle point of a slice of road contained in a row of the image.
    # It does so by finding the first and last element in the row that has the same
    # brightness value of the road and calculate their average.
    def find_row_average(row):
        reversed_row = row[::-1]
        found_right = width-np.where(reversed_row == reclassifying_val[0])[0][0]-1
        found_left = np.where(row == reclassifying_val[0])[0][0]
        average_pos = (found_left + found_right)/2
        return int(average_pos)

    # Private function which return a boolean-indexed version of the image.
    # It selects only the pixel contains the value that fits the given brightness
    # of the road.
    def cond(image):
        return ((image > road_val_range[0]) &  # road_val_range[1] and road_val_range[2] is the range of 
                (image < road_val_range[1]))   # brightness of road surface (the part of the picture we need).
        # Return False if its not the road

    # Turn the image into a grayscale version and remove every thing but the road,
    # using boolean-indexing function.
    def process_image(image, cond):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray[cond(gray)] = reclassifying_val[0]
        gray[~cond(gray)] = reclassifying_val[1]     # Boolean Indexing
        return gray

    # Confer https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.line
    def draw_line(x, y) -> np.ndarray:
        rr, cc = draw.line(
            int(x[0]), int(x[1]), int(y[0]), int(y[1]))
        image[rr, cc] = 255


    # Process the image. Refer to the process_image function for more details.
    processed_img = process_image(image, cond)
    road_top_bot = find_road_top_bot(processed_img)

    ''' This single loop do the job '''
    if road_top_bot != None:
        avg_list = []

        # Keep count of the row number (y-coordinate).
        i = road_top_bot[0]
        
        # Loop through all the rows that contain the road, append the averaged coordinates
        # to avg_list (cf. find_row_average).        
        for row in processed_img[road_top_bot[0]:road_top_bot[1]]:

            # We need to use some exception catching as there could be rows 
            # that don't have pixels of the road (broken segmentation).
            try:
                avg_list.append((i, find_row_average(row)))
                i += 1
            except:
                pass

        # Reduce the number of coordinates to smoothen the path.    
        avg_list = avg_list[::10]
        # We also have to add our place in to the list .ie
        # the bottom middle point.
        avg_list.append((height-1,width/2))
        nice = zip(avg_list,avg_list[1:]) # Zip these to make pairs.

        for x, y in nice:
            draw_line(x,y)

    return image

''' Let's have some tests! '''
img = cv2.imread(
    'Test Data\\00e9be89-00001005_train_color.png', 1) # Read image.
start = timer()
processed = paint_path(img, (89,92))
end = timer()
print(end - start)
# The time it takes to run this function should be around 12ms

''' Display the image '''
cv2.imshow('ngon', processed)
cv2.waitKey(0)
# imgplot = plt.imshow(processed_img)
# plt.show()