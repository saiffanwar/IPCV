import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt
#   edge detection opencv function

img = cv.imread('images/positives/'+sys.argv[1])
img = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
cv.imwrite('edge.jpg',img)
