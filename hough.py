import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt
#   edge detection opencv function


img = cv.imread('images/positives/'+sys.argv[1])
edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
# cv.imwrite('edge.jpg',img)

def hough_line(edge_threshold=5, vote_threshold=1):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width = edges.shape[0]
    height = edges.shape[1]
    max = int(round(np.sqrt((width**2) + (height**2))))
    rhos = np.linspace(-max, max, max * 2)

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    # Hough accumulator array of theta vs rho
    votes = np.zeros((len(thetas), 2 * max), dtype=np.uint8)


    # (row, col) indexes to edges
    edges[edges < edge_threshold] = 0
    Ys, Xs = np.nonzero(edges)
    voters = [[[(max,max), (0, 0)] for t in range(len(thetas))] for r in range(len(rhos))]
    # Vote in the hough accumulator
    for i in range(len(Xs)):
        x = Xs[i]
        y = Ys[i]

        for j, theta in enumerate(thetas):
                # Calculate rho. diag_len is added for a positive index
                rho = max + int(round(x * cos_theta[j] + y * sin_theta[j]))
                votes[j, rho] += 1
                try:
                    voters[theta][rho].append((x,y))
                except:

    votes[votes < vote_threshold] = 0
    return votes, voters, thetas, rhos

def hough2line(votes, voters, thetas, rhos):

    Ts, Rs =  np.nonzero(votes)
    for i in range(len(Rs)):
        rho = Rs[i]
        theta = Ts[i]
        print(rho, theta)
        min = np.min(voters[theta][rho])
        max = np.max(voters[theta][rho])

    img = cv.line(img, min, max, (0, 255, 0), 2)
    cv.imwrite('detectedline.jpg', img)

votes, voters, thetas, rhos = hough_line()
hough2line(votes, voters, thetas, rhos)
