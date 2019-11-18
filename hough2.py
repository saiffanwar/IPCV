import cv2 as cv, sys, numpy as np
#   edge detection opencv function


img = cv.imread('images/positives/'+sys.argv[1])
edges = cv.Canny(image=img, threshold1 = 300, threshold2 = 500 )
cv.imwrite('edges.jpg', edges)
# cv.imwrite('edge.jpg',img)
thetas = (np.arange(-np.pi/2, np.pi/2, np.pi/180 ))
width = edges.shape[0]
height = edges.shape[1]
max = int(round(np.sqrt((width**2) + (height**2))))
rhos = np.arange(0, max, 1)

def hough_line(edge_threshold=5, vote_threshold=1):
    # Rho and Theta ranges

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)


    # (row, col) indexes to edges
    edges[edges < edge_threshold] = 0
    Ys, Xs = np.nonzero(edges)
    votes = np.zeros((len(rhos), len(thetas)))
    print(votes[692][179])
    print(votes.shape)

    # Vote in the hough accumulator

    for i in range(len(Xs)):
        x = Xs[i]
        y = Ys[i]
        for j in range(len(thetas)):
            rho = int(round(abs((x * cos_theta[j]) + (y * sin_theta[j]))))
            votes[rho][j] = votes[rho][j]+1

    return votes, thetas, rhos


def hough2line(votes, thetas, rhos):
    t = np.amax(votes)/2
    lines = np.argwhere(votes > t)
    print(len(lines))
    for l1 in range(len(lines)):
        x = int(lines[l1][0] + 1)
        y = round(np.deg2rad(int(lines[l1][1] + 1)), 2)
        print(x, y)
        m = ((-np.cos(y))/(np.sin(y)))
        c = (x/(np.sin(y)))
        print(m, c)
        pt1 = (0, int(c))
        pt2 = ((int((c)/m)), 0)
        print(pt1, pt2)
        cv.line(edges, pt1, pt2, (255, 255, 255), 1)
    cv.imwrite('detectedline.jpg', edges )

votes, thetas, rhos = hough_line()
hough2line(votes, thetas, rhos)
