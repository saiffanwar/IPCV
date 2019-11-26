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

def hough_ellipse(edges, minvotes = 10, min2a = 10):
    width = edges.shape[0]
    height = edges.shape[1]
    #  a list of all coordinates in edges that are not zero
    ys, xs = np.nonzero(edges)
    nonzeros = np.array(list(zip(ys, xs)))
    print(nonzeros)
    # accumulator size of maximum minor axis length which is half of the width or height of the image
    accumulator = np.zeros(int(np.maximum(height , width)/2))
    for i1 in range(0, len(nonzeros)):
        for i2 in (len(nonzeros)-1, i1, -1):   
            x1, y1 = nonzeros[i1]
            x2, y2 = nonzeros[i2]
            d12 = np.sqrt(((x1-x2)**2) + ((y1-y2)**2))
            accumulator = np.zeros(int(np.maximum(height , width)/2))
            if  x1-x2 and d12 > min2a:
                x0 = (x1 + x2)/2
                y0 = (y1 + y2)/2
                a = d12/2;
                alpha = np.arctan((y2 - y1)/(x2 - x1))
                for i3 in range(0, len(nonzeros)):
                    if np.all(nonzeros[i3] == nonzeros[i1]) or np.all(nonzeros[i3] == nonzeros[i2]):
                        continue
                    x3, y3 = nonzeros[i3]
                    d03 = np.sqrt(((x3-x0)**2) + ((y3-y0)**2))
                    if (d03 >= a):
                        continue
                    f = np.sqrt(((x3-x2)**2) + ((y3-y2)**2))
                    cos2_tau = ((a**2 + d03**2 - f**2)/(2*a*d03))**2
                    sin2_tau = (1-cos2_tau)**2
                    b = np.sqrt( ((a**2) * (d03**2) * sin2_tau) / ((a**2) - (d03**2) * cos2_tau) )
                    b= int(np.round(b))
                    if (0 <= b and b < len(accumulator)):
                        accumulator[b] += 1
                    maxi = np.argmax(accumulator)

                    if accumulator[maxi] > minvotes:
                        print((x0, y0, a, maxi, alpha))
                        return (x0, y0, a, maxi, alpha)
                    else:
                        continue
    print("none")
    return None