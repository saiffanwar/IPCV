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

def get_line_endpoints(line, width, height):
    min_x = width
    min_y = height
    max_x = 0
    max_y = 0

    for coord in line:
        x,y = coord
        if min_x > x and min_y > y:
            min_x = x
            min_y = y
        elif max_x < x and max_y < y:
            max_x = x
            max_y = y

    return [(min_x, min_y), (max_x, max_y)]

def PPHT(image, vote_threshold=25, gap_threshold=6, length_threshold=4):
    buffer = np.copy(image)
    lines = []

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
    # while image not empty
    while(np.sum(buffer) > 0):

        # randomly select a non-zero pixel
        X, Y = np.nonzero(buffer)

        random = np.random.choice(range(len(X)), 1)
        x = X[random][0]
        y = Y[random][0]
        
        updated = []

        # Update accumulator with random pixel
        for j, theta in enumerate(thetas):
                    # Calculate corresponding rho
                    rho = x * cos_theta[j] + y * sin_theta[j]
                    # Map rho calculated to index of nearest rho value available
                    i = np.argmin(np.abs(rhos - rho))
                    # increment bin in accumulator
                    votes[i , j] += 1
                    # remember bins updated
                    updated.append((i,j))

        # remove pixel from buffer
        buffer[x,y]=0

        # if highest peak in accumulator was a bin just updated and meets the threshold
        peak = np.unravel_index(np.argmax(votes), votes.shape)
        rho_index, theta_index = peak

        if votes[rho_index, theta_index] <= vote_threshold:
            continue

        theta = thetas[theta_index]
        rho = rhos[rho_index]
        if peak in updated:
            print("Line found at random location", x, y)
            # search orginal image along the line specified from (x,y) to find the
            # longest segment of pixels that does not exceed the gap threshold and
            # is at least the length threshold

            # all points of infinite line y = mx+c that lie in image space 
            Y = []
            # if angled line
            if np.sin(theta) != 0:
                m = -np.cos(theta)/np.sin(theta)
                c = rho/np.sin(theta)
            # if horizontal line
            else:
                m = 0
                c = y
            for w in range(width):
                z = int((m*w)+c)
                if z >= height:
                    break
                Y.append(z)
            print(Y)
            line = [(x,y)]
            gap_count = 0
            length_count = 0
            
            # find longest segment in postive direction
            for j in range (int(x+1), len(Y)):
                if image[j, Y[j]] != 0:
                    line.append((j, Y[j]))
                    length_count+=1
                else:
                    gap_count+=1
                if gap_count > gap_threshold:
                    gap_count = 0
                    break
            
            # find longest segment in negative direction
            for j in range (0, int(x)):
                if image[j, Y[j]] != 0:
                    line.append((j, Y[j]))
                    length_count +=1
                else:
                    gap_count += 1
                if gap_count > gap_threshold:
                    break
            
            # remove points in line segment from buffer
            for coord in line:
                i,j = coord
                buffer[i,j] = 0

            # remove votes these points placed for this line
            votes[rho_index, theta_index] -= len(line)

            # if line segment meets length threshold, add to output list
            if len(line) >= length_threshold:
                lines.append(get_line_endpoints(line, width, height))
            
            
        # if not, check next random pixel
        else:
            continue

    return lines




# votes, thetas, rhos = hough(image=edges)
# hough2line(edges, votes, thetas, rhos)
# img = cv.imread('images/positives/'+sys.argv[1])
# edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
# lines = PPHT(edges)
# print(lines)
# for line in lines:
#     p1, p2 = line
#     cv.line(img, p1, p2, (255, 0, 0), 1)
# cv.imwrite('lines.jpg',img)
