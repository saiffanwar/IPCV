import cv2 as cv, sys, numpy as np
import pprint as pp
np.set_printoptions(threshold=sys.maxsize)

#   edge detection opencv function

#first i will read the image and create the binary version using Canny

img = cv.imread('images/positives/'+sys.argv[1])
edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
cv.imwrite('edges.jpg', edges)

# thetas is an array of all the angles i will be calculating lines for in image space
thetas = np.deg2rad(np.arange(-90.0, 90.0))
width = edges.shape[0]
height = edges.shape[1]
# max is the maximum possible rho value for a line in image space
max = int(round(np.sqrt((width**2) + (height**2))))
# rhos is a list of all possible rho values to the nearest integer up to the max
rhos = np.arange(0, max, 1)

def hough_line():
    # Rho and Theta ranges

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)


    # (row, col) indexes to edges
    votes = np.zeros((len(rhos), len(thetas)))

    # Vote in the hough accumulator

    for i in range(width):
        for j in range(height):
            if(edges[i][j] != 0):
                x = i+1
                y = j+1
                for t in range(0, len(thetas)):
                    rho = int(round(abs((x * cos_theta[t]) + (y * sin_theta[t]))))
                    votes[rho][t] = votes[rho][t]+1
    return votes, thetas, rhos


def hough2line(votes, thetas, rhos):
    t = np.amax(votes)/2
    print(t)
    lines = np.argwhere(votes > 100)
    for l1 in range(len(lines)):
        x = int(lines[l1][0] + 1)
        y = round(np.deg2rad(int(lines[l1][1] + 1)), 2)
        # print(x, y)
        m = ((-np.cos(y))/(np.sin(y)))
        c = (x/(np.sin(y)))
        # print(m, c)
        pt1 = (0, int(c))
        pt2 = ((int((c)/m)), 0)
        # print(pt1, pt2)
        cv.line(edges, pt1, pt2, (255, 255, 255), 1)
    cv.imwrite('detectedline.jpg', edges )

votes, thetas, rhos = hough_line()
hough2line(votes, thetas, rhos)
