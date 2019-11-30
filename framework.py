import numpy as np
import cv2 as cv
import argparse
import operator
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import unravel_index

from eval import eval, iou, iouThreshold
from hough import hough_lines, hough_ellipse

path = 'images/positives/'

classifier2groundTruths = {"frontalface" : {'dart4.jpg': [[342, 99, 135, 176]],
                'dart5.jpg': [[65, 141, 55, 66], [56, 243, 59, 80], [191, 206, 59, 79], [252, 169, 54, 65], [290, 236, 57, 74], [373, 189, 68, 66], [428, 228, 58, 80], [518, 172, 54, 69], [560, 237, 59, 78], [648, 180, 57, 69], [683, 240, 52, 72]],
                'dart13.jpg': [[418, 118, 118, 143]],
                'dart14.jpg': [[463, 200, 91, 126], [727, 174, 102, 124]],
                'dart15.jpg': []},
                "dartboard": {'dart0.jpg': [[442, 16, 155, 177]],
                'dart1.jpg': [[198, 133, 191, 191]],
                'dart2.jpg': [[102,97,89,86]],
                'dart3.jpg': [[325, 148, 65, 71]],
                'dart4.jpg': [[184, 95, 169, 194]],
                'dart5.jpg': [[433, 141, 92, 104]],
                'dart6.jpg': [[213, 117, 59, 61]],
                'dart7.jpg': [[256, 171, 121, 142]],
                'dart8.jpg': [[69, 254, 58, 84], [844, 219, 112, 118]],
                'dart9.jpg': [[202, 48, 232, 232]],
                'dart10.jpg': [[91, 106, 97, 108], [585,125,55,85],[919,148,33,63]],
                'dart11.jpg': [[178, 105, 55, 49]],
                'dart12.jpg': [[157, 78, 58, 135]],
                'dart13.jpg': [[277, 120, 125, 129]],
                'dart14.jpg': [[120, 101, 125, 123], [990, 95, 120, 124]],
                'dart15.jpg': [[155, 57, 125, 136]]}
}

parser = argparse.ArgumentParser(description = 'Framework for running, testing and optimising object detectors')
parser.add_argument('--image', help='File name of image in /images/positives to use', default = 'dart5.jpg')
parser.add_argument('--classifier', help='Name of trained classifier to use', choices=['frontalface', 'dartboard'], default = 'frontalface')
parser.add_argument('--job', help='Job to perform', choices = ['detect', 'benchmark', 'optimise'], default = 'detect')
parser.add_argument('--sf', help='Scale Factor hyper-parameter (>=1.1)', type=float, default = 1.1)
parser.add_argument('--mn', help='Minimum Neighbours hyper-paramter (>=1)', type=int, default = 1)

args = parser.parse_args()

if args.sf < 1.1:
    parser.error("Minimum SF is 1.1")
if args.mn < 1:
    parser.error("Minimum MN is 1")

def line_intersection(line1, line2):
    xdiff = (line1[0,0] - line1[1,0], line2[0,0] - line2[1,0])
    ydiff = (line1[0,1] - line1[1,1], line2[0,1] - line2[1,1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (int(x), int(y))

#detects image and returns array called 'faces' with subarrays containg x, y, width and height of all boxes around detected faces.
def detect(image, scaleFactor=1.2, minNeighbors=1.7):
    if args.classifier == "dartboard":
        scaleFactor = 1.1
        minNeighbors = 1
    img = cv.imread('images/positives/'+image)
    output = img.copy()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    faces = np.asarray(face_cascade.detectMultiScale(image=gray, scaleFactor=scaleFactor,
    minNeighbors=minNeighbors, minSize=(50, 50), maxSize=(500,500)))

    if args.classifier == "dartboard":
        # highlights regions of interest and draws them onto the image.
        ROIs = []
        tmp_faces = []
        final_faces = []
        dropped = []
        for i, f1 in enumerate(faces):
            if  i not in dropped:
                for j, f2 in enumerate(faces):
                    if i != j and j not in dropped:
                        z = iou(f1, f2)
                        z = iouThreshold(z, 0.1)
                        if z == 1:
                            _, _, w1, h1 = f1
                            _, _, w2, h2 = f2
                            if (w1*h1) <= (w2*h2):
                                dropped.append(j)
                            else:
                                dropped.append(i)

        tmp_faces = [face for i, face in enumerate(faces) if i not in dropped]

        for (x,y,w,h) in tmp_faces:

            roi = img[y:y+h, x:x+w]
            edges = cv.Canny(image=roi, threshold1 = 100, threshold2 = 200 )
            lines = hough_lines(image=edges)
            intersections = []
            if(len(lines) > 0):
                for i, line in enumerate(lines):
                    x1, y1 = line[0]
                    x2, y2 = line[1]
                    p1, p2 = (x1+x, y1+y), (x2+x, y2+y)
                    cv.line(output, p1, p2, (255, 0, 0), 1)
                    for j, line2 in enumerate(lines):
                        if i != j:
                            p = line_intersection(line, line2)
                            if p != None:
                                intersections.append(p)

            centre = None
            if(len(intersections)>0):
                intersections = np.asarray(intersections)
                meanx = np.mean(intersections[:, 0])
                meany = np.mean(intersections[:, 1])
                centre = (int(np.round(meanx)), int(np.round(meany)))
                cv.circle(output,  (int(np.round(meanx))+x, int(np.round(meany))+y), 5, (255, 255, 0), 2)

            edges = cv.Canny(image=roi, threshold1=100, threshold2=250)
            cv.imwrite('edges.jpg', edges)
            ellipses = hough_ellipse(edges, centre = centre)
            if np.all(ellipses) != None:
                for ellipse in ellipses:
                    x0, y0, a, b, alpha = ellipse
                    x0 += x
                    y0 += y
                    cv.ellipse(output, (int(x0), int(y0)), (int(a), int(b)), alpha, 0, 360, (255,0,0), 2)
            else:
                ellipses = []

            if len(ellipses) > 0 :
                x0,  y0,  a, b,  alpha = ellipses[0]
                x0 += x
                y0 += y
                # if 4*a*b < 0.* w * h:
                x = int(x0-a)
                y=int(y0-b)
                w=int(2*a)
                h=int(2*b)

                output = cv.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
                final_faces.append((x,y,w,h))
    else:
        final_faces = []
        for (x,y,w,h) in faces:
            output = cv.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
            final_faces.append((x,y,w,h))

    for(x,y,w,h) in groundTruths[image]:
        output = cv.rectangle(output,(x,y),(x+w,y+h),(0,0,255),2)


#displays the image with roi
    cv.imwrite('images/detected/'+image,output)
    return eval(groundTruths[image], final_faces)



def benchmark (scaleFactor=1.2, minNeighbors=7):
    P, R = [], []
    for k in groundTruths.keys():
        tp, fp, fn, p, r, f1 =  detect(k, scaleFactor, minNeighbors)
        P.append(p)
        R.append(r)
        print(k, ": TPs=", tp, ", FPs=", fp, ", FNs=", fn, ", Precision=", p, ", Recall=", r, "and F1=", f1)

    AP = np.mean(P)
    AR = np.mean(R)
    AF1 = 2 * ((AP*AR)/ (AP + AR))
    # print("average", AP, AR, AF1)
    return AP, AR, AF1

def gridsearch ():
    Z = []
    X = np.asarray([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    Y = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for sf in X:
        for mn in Y:
            _, _, f = benchmark(scaleFactor=sf, minNeighbors=mn)
            Z.append(f)
    Z = np.asarray(Z).reshape((10,10))
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, Z.T, cmap='coolwarm', vmin=0, vmax=1)
    ax.set_title('Macro-Average F1-Scores for different configurations of the \nhyper-parameters of the Frontal Face detector on the Test set', size=11)
    # set the limits of the plot to the limits of the data
    ax.axis([1.1, 2.0, 1, 10])
    key = plt.colorbar(c)
    plt.xlabel("Scale Factor", size=11)
    plt.ylabel("Minimum Neighbours", size=11)
    plt.show()

    optimal = unravel_index(Z.argmax(), Z.shape)
    return X[optimal[0]], Y[optimal[1]]

np.seterr(all='raise')
    # Load Classifier
face_cascade = cv.CascadeClassifier(args.classifier+'.xml')
groundTruths = classifier2groundTruths[args.classifier]
if args.job == "detect":
    tp, fp, fn, p, r, f1 = detect(args.image, scaleFactor=args.sf, minNeighbors=args.mn)
    print("Scores for", args.classifier, "on", args.image, ": TPs=", tp, ", FPs=", fp, ", FNs=", fn, ", Precision=", p, ", Recall=", r, "and F1=", f1)
elif args.job == "benchmark":
    AP, AR, F1 = benchmark(scaleFactor=args.sf, minNeighbors=args.mn)
    print("Benchmarks for", args.classifier, ": AP=", AP, ", AR=", AR, "and Macro-Average F1=", F1)

elif args.job == "optimise":
    sf, mn = gridsearch()
    print("Optimal hyper-paramters for", args.classifier, ": SF=", sf, ", and MN =", mn)
