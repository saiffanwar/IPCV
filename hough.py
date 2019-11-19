import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt

# img = cv.imread('images/positives/'+sys.argv[1])
# edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
# cv.imwrite('edges.jpg', edges )

# cv.imwrite('edge.jpg',img)

def hough(image):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = image.shape
    max_rho = int(round(np.sqrt((width**2) + (height**2))))
    rhos = np.linspace(-max_rho, max_rho, max_rho*2)

    # preprocess trig values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    # Hough accumulator array of theta vs rho
    votes = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    # Vote in the hough accumulator for each (x,y)
    for x in range(width):
        for y in range(height):
            # Only consider non-zero edges
            if edges[x, y] > 0:
                # for each possible line around (x, y)
                for j, theta in enumerate(thetas):
                    # Calculate corresponding rho
                    rho = x * cos_theta[j] + y * sin_theta[j]
                    # Map rho calculated to index of nearest rho value available
                    i = np.argmin(np.abs(rhos - rho))
                    # increment bin in accumulator
                    votes[i , j] += 1
    return votes, thetas, rhos

# maps vote accumulator to a set of line endings [(x1, y1), (x1, y2)]
def votes2lines(image, votes, thetas, rhos, threshold=100):
    lines = []
    width, height = image.shape
    # for each line (rho, theta) 

    for i, rho in enumerate(rhos):
        for j, theta in enumerate(thetas):
            # if sufficient votes
            if votes[i,j] >= threshold and theta != 0:
                # determine parametric form y = mx + c
                m = -np.cos(theta)/np.sin(theta)
                c = rho/np.sin(theta)

                # axis intersections to plot
                p1 = (int(c),0)
                p2 = (int((m*width)+c),int(width))
                lines.append([p1, p2])
                # cv.line(image, p1, p2, (255, 0, 0), 1)
    cv.imwrite('lines.jpg', image )
    
    return np.asarray(lines)

# votes, thetas, rhos = hough(image=edges)
# hough2line(edges, votes, thetas, rhos)

