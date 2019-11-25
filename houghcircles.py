import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt

# create edges
img = cv.imread(sys.argv[1])
edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
cv.imwrite('edges.jpg', edges)


def hough(image):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(0.0, 360.0))
    width, height = edges.shape
    # max_r = np.amin(width, height)

    # preprocess trig values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    # Hough accumulator array of theta vs rho
    # votes = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    # Vote in the hough accumulator for each (x,y)
    for x in range(width):
        for y in range(height):
            # Only consider non-zero edges
            if edges[x, y] > 0:
                # for each possible line around (x, y)
                cv.circle(image, (x, y), 10, (0, 255, 0))
    cv.imwrite('circles.jpg', image)

hough(img)
