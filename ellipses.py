import cv2 as cv, sys, numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/positives/'+sys.argv[1])
# img = cv.imread(sys.argv[1])
edges = cv.Canny(image=img, threshold1 = 100, threshold2 = 1000)
cv.imwrite('edges.jpg', edges)
width = img.shape[0]
height = img.shape[1]


def hough_ellipse(threshold = 60, min2a = 60):
    #  a list of all coordinates in edges that are not zero
    ys, xs = np.nonzero(edges)
    nz = np.array(list(zip(ys, xs)))
    nz2 = np.flip(np.array(list(zip(xs, ys))))
    # accumulator size of maximum minor axis length which is half of the width or height of the image
    accumulator = np.zeros(int(np.maximum(height , width)/2))
    for i1 in nz:
        for i2 in nz2:
            print(i2)
            x1, y1 = i1
            x2, y2 = i2
            d12 = np.sqrt(((x1-x2)**2) + ((y1-y2)**2))
            accumulator = accumulator * 0
            if  np.absolute(x1-x2) > min2a and d12 > min2a:
                x0 = (x1 + x2)/2
                y0 = (y1 + y2)/2
                print(x0, y0)
                a = d12/2;
                alpha = np.arctan((y2 - y1)/(x2 - x1))
                for i3 in nz:
                    if np.all(i3 == i1) or np.all(i3 == i2):
                        continue
                    x3, y3 = i3
                    d03 = np.sqrt(((x3-x0)**2) + ((y3-y0)**2))
                    if (d03 >= a):
                        continue
                    f = np.sqrt(((x3-x2)**2) + ((y3-y2)**2))
                    cos_tau = (a**2 + d03**2 - f**2)/(2*a*d03)
                    sin_tau = np.sqrt(1-cos_tau**2)
                    b = np.sqrt((a**2 * d03**2 * sin_tau**2)/(a**2 - d03**2 * cos_tau**2))
                    if (0 < b < len(accumulator)):
                        accumulator[int(b)] += 1
                    maxi = np.argmax(accumulator)
                    if maxi > threshold:
                        cv.ellipse(img, (int(x0), int(y0)), (int(a), int(maxi)),  0, 0, 360, (0,0,255), 1)
                        cv.imwrite('ellipse.jpg', img)
                    else:
                        continue
    return (x0, y0, a, maxi, alpha)

hough_ellipse()
