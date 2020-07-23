import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Untitled.png',1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.hist(gray.ravel(),256,[0,256]); plt.show()


