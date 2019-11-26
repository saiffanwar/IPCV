import numpy as np
import cv2 as cv
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px

cv.circle(img,(255,255), 200, (255,255,255), 1)
cv.imwrite('circle.jpg', img)
