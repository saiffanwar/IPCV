import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt

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
            if image[x, y] > 0:
                # for each possible line around (x, y)
                for j, theta in enumerate(thetas):
                    # Calculate corresponding rho
                    rho = x * cos_theta[j] + y * sin_theta[j]
                    # Map rho calculated to index of nearest rho value available
                    i = np.argmin(np.abs(rhos - rho))
                    # increment bin in accumulator
                    votes[i , j] += 1
    return votes, thetas, rhos

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

def PPHT(image, gap_threshold=6, length_threshold=4):
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
        x = X[random]
        y = Y[random]
        
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

        # if highest peak in accumulator was a bin just updated
        peak = np.unravel_index(np.argmax(votes), votes.shape)
        rho_index, theta_index = peak
        theta = thetas[theta_index]
        rho = rhos[rho_index]
        if peak in updated:
            # search orginal image along the line specified from (x,y) to find the
            # longest segment of pixels that does not exceed the gap threshold and
            # is at least the length threshold

            # all points of infinite line y = mx+c that lie in image space 
            Y = []
            # if angled line
            if np.sin(theta) != 0:
                m = int(-np.cos(theta)/np.sin(theta))
                c = int(rho/np.sin(theta))
            # if horizontal line
            else:
                m = 0
                c = int(y)
            for w in range(width):
                y = ((m*w)+c)
                if y > height:
                    break
                Y.append(y)

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
img = cv.imread('images/positives/'+sys.argv[1])
edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 500 )
lines = PPHT(edges)
print(lines)
for line in lines:
    p1, p2 = line
    cv.line(img, p1, p2, (255, 0, 0), 1)
cv.imwrite('lines.jpg',img)
