import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt
#   edge detection opencv function


img = cv.imread('images/positives/'+sys.argv[1])
edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
cv.imwrite('edges.jpg', edges )

# cv.imwrite('edge.jpg',img)

def hough(edges, edge_threshold=1, vote_threshold=1):

    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = edges.shape
    max_rho = int(round(np.sqrt((width**2) + (height**2))))
    rhos = np.linspace(-max_rho, max_rho, max_rho*2)

    # preprocess trig values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    # Hough accumulator array of theta vs rho
    votes = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    # map all edges below threshold to 0
    edges[edges < edge_threshold] = 0

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
def hough2line(votes, thetas, rhos, threshold=100):
    width, height = edges.shape
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
                cv.line(edges, p1, p2, (255, 0, 0), 1)
    cv.imwrite('lines.jpg', edges )





votes, thetas, rhos = hough(edges)
hough2line(votes, thetas, rhos)
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(votes, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
# ax.set_title('Hough transform')
# ax.set_xlabel('Angles (degrees)')
# ax.set_ylabel('Distance (pixels)')
# ax.axis('image')

# plt.show()
