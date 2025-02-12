import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt

def hough_lines(image, threshold=70):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    height, width = image.shape
    max_rho = int(round(np.sqrt((width**2) + (height**2))))
    rhos = np.linspace(-max_rho, max_rho, max_rho*2)
    # preprocess trig values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    _, nz = np.nonzero(image)
    if(len(nz)<3500):
        threshold = 50


    # Hough accumulator array of theta vs rho
    votes = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    last_voter = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    # Vote in the hough accumulator for each (x,y)
    for y in range(height):
        for x in range(width):

            # Only consider non-zero edges
            if image[y, x] > 0:
                # for each possible line around (x, y)
                for j, theta in enumerate(thetas):
                    # Calculate corresponding rho
                    rho = x * cos_theta[j] + y * sin_theta[j]
                    # Map rho calculated to index of nearest rho value available
                    i = np.argmin(np.abs(rhos - rho))
                    # increment bin in accumulator
                    votes[i , j] += 1
                    # remember most recent y value that votes for (rho, theta)
                    last_voter[i, j] = y
    # print(votes, thetas, rhos)
    # show_hough_line(image, votes, thetas, rhos)

    lines = []
    percentile = np.percentile(votes, 99.99)
    if percentile> 70:
        threshold = percentile
    # for each line (rho, theta)
    for i, rho in enumerate(rhos):
        for j, theta in enumerate(thetas):
            # if sufficient votes
            if votes[i,j] >= threshold:
                if theta != 0:
                    # determine parametric form y = mx + c
                    m = -np.cos(theta)/np.sin(theta)
                    c = rho/np.sin(theta)

                else:
                    m = 0
                    c = last_voter[i,j]

                # default p1 and p2 are the y-intercept and where x=width resp.
                p1 = (0, int(c))
                p2 = (int(width), int((m*width)+c))

                # if y-intercept is negative, update p1 to where y=0
                if c < 0:
                    p1 = (int(-c/m),0)
                # if y-intercept is bigger than height, update p1 to where y=height
                elif c > height:
                    p1 = (int((height-c)/m), height)
                # if y<0 for p2, update p2 to where y=0
                if p2[1] < 0:
                    p2 = (int(-c/m),0)
                # if y>height for p2, update p2 to where y=height
                elif p2[1] > height:
                    p2 = (int((height-c)/m), height)
                lines.append([p1, p2])
    return np.asarray(lines)

def hough_ellipse(edges, leastVotes = 50, leastDistance = 20, min_b = 40, min_a = 40, centre=None, min_c=20):
    detected = []
    detectedvotes=[]
    height = edges.shape[0]
    width = edges.shape[1]
    # 1 store all edge pixels in 1d array
    ys, xs = np.nonzero(edges)
    nonzeros = np.array(list(zip(ys, xs)))
    # print(len(nonzeros))
    if(len(nonzeros)<1500):
        if centre != None:
            leastVotes=20
        else:
            leastVotes=50
    # accumulator size of maximum minor axis length which is half of the width or height of the image
    # 2 clear accumulator
    accumulator = np.zeros(int(np.minimum(height , width)/2))
    # 3 for each pixel
    for p1 in nonzeros:
        y1, x1 = p1
    # 4 for each other pixel
        for p2 in nonzeros:
            y2, x2 = p2
            if x2 != x1 and y2 != y1:
            # 4 if distance between p1 and p2 is greater than leastDistance
                d12 = np.sqrt((x2-x1)**2 + ((y2-y1)**2))
                if d12 > leastDistance:
                # 5 calculate centre, orientation and half length of major axis
                    x0 = (x1 + x2)/2
                    y0 = (y1 + y2)/2
                    a = np.sqrt((x2-x1)**2 + (y2 - y1)**2)/2
                    alpha = np.arctan((y2 - y1)/(x2 - x1))
                    # theshold a, x0 and y0 vlues
                    valid = False
                    if a > min_a:
                        if np.all(centre == None):
                            if x0 > 0.45*width and x0 < 0.55*width and y0 > 0.45*height and y0 < 0.55*height:
                                valid = True
                        else:
                            xc, yc = centre
                            if x0 > xc-min_c  and x0 < xc+min_c and y0 > yc-min_c and y0 < yc+min_c:
                                valid =True
                    if valid:
                    # 6 for each third pixel
                        for p3 in nonzeros:
                            y3, x3 = p3
                            # 6 if distance between p3 and p0 is greater than leastDistance
                            d01 = np.sqrt((x0-x1)**2 + ((y0-y1)**2))
                            d02 = np.sqrt((x0-x2)**2 + ((y0-y2)**2))
                            d03 = np.sqrt((x0-x3)**2 + ((y0-y3)**2))
                            if d03 > leastDistance and d03 < d01 and d03 < d02:
                                # 7 calculate the length of the minor axis
                                f = np.sqrt((x2-x3)**2 + ((y2-y3)**2))
                                cos2_tau = ((a**2 + d03**2 - f**2)/(2*a*d03))**2
                                if cos2_tau > 1.0:
                                    cos2_tau = 1.0
                                sin2_tau = 1-(cos2_tau)
                                b = np.sqrt((a**2 * d03**2 * sin2_tau)/(a**2 - (d03**2 * cos2_tau)))
                                # 8 update accumulator for length b (also threshold b)
                                if b >= 0.66*a and b < len(accumulator):
                                    accumulator[int(b)] += 1
                        # 10 find maximum element in accumulator array
                        b = np.argmax(accumulator)
                        votes = accumulator[b]
                        # 10 if votes is greater than leastVotes
                        if votes > leastVotes:
                            # print(votes)
                            detectedvotes.append(a*b)
                        # 11 ellipse detected
                            # print(x0, y0, b, a, alpha)
                            detected.append((x0, y0, b, a, alpha))
                        # 12 remove pixels of detetcted ellipse from edges
                            for i, pixel in enumerate(nonzeros):
                                y, x = pixel
                                X = x - x0
                                Y = y - y0
                                if np.round((((X*np.cos(alpha) + Y*np.sin(alpha))**2)/a**2) + (((X*np.sin(alpha) + Y*np.cos(alpha))**2)/b**2)) == 1:
                                    edges[y, x] = 0
                            ys, xs = np.nonzero(edges)
                            nonzeros = np.array(list(zip(ys, xs)))

                    # 13 clear accumulator
                    accumulator = np.zeros(int(np.maximum(height , width)/2))
        # print(detected)
        detected = np.asarray(detected)
        if len(detected) > 0:
            finalDetected = detected[np.argwhere(detectedvotes == np.amax(detectedvotes)).flatten()]
            # print(finalDetected)
            return finalDetected
        else:
            return []
        return detected


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# img = cv.imread('images/positives/'+sys.argv[1])
# # img = cv.imread(sys.argv[1])
# edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 1000)
# cv.imwrite('edges.jpg', edges)
# x0, y0, a, b, alpha = hough_ellipse(edges)
# cv.ellipse(img, (int(x0), int(y0)), (int(a), int(b)),  alpha , 0, 360, (0,0,255), 1)
# cv.imwrite('ellipse.jpg', img)
