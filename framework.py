import numpy as np
import cv2 as cv
import argparse
import operator
import matplotlib.pyplot as plt 
from matplotlib import cm
from numpy import unravel_index

from eval import eval
from hough import hough_lines

path = 'images/positives/'

classifier2groundTruths = {"frontalface" : {'dart4.jpg': [[342, 99, 135, 176]],
                'dart5.jpg': [[65, 141, 55, 66], [56, 243, 59, 80], [191, 206, 59, 79], [252, 169, 54, 65], [290, 236, 57, 74], [373, 189, 68, 66], [428, 228, 58, 80], [518, 172, 54, 69], [560, 237, 59, 78], [648, 180, 57, 69], [683, 240, 52, 72]],
                'dart13.jpg': [[418, 118, 118, 143]],
                'dart14.jpg': [[463, 200, 91, 126], [727, 174, 102, 124]],
                'dart15.jpg': []},
                "dartboard": {'dart0.jpg': [[444, 14, 152, 175]],
                'dart1.jpg': [[196, 131, 195, 192]],
                'dart2.jpg': [[103, 97, 88, 88]],
                'dart3.jpg': [[325, 149, 64, 69]],
                'dart4.jpg': [[185, 96, 168, 194]],
                'dart5.jpg': [[434, 142, 90, 104]],
                'dart6.jpg': [[213, 118, 59, 61]],
                'dart7.jpg': [[256, 171, 114, 143]],
                'dart8.jpg': [[844, 219, 113, 118], [68, 254, 58, 85]],
                'dart9.jpg': [[205, 49, 228, 230]],
                'dart10.jpg': [[586, 130, 53, 81], [918, 150, 33, 63], [93, 104, 93, 109]],
                'dart11.jpg': [[176, 105, 56, 49]],
                'dart12.jpg': [[158, 79, 57, 133]],
                'dart13.jpg': [[275, 123, 126, 127]],
                'dart14.jpg': [[123, 103, 122, 123]],
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


#detects image and returns array called 'faces' with subarrays containg x, y, width and height of all boxes around detected faces.
def detect(image, scaleFactor=1.1, minNeighbors=1.1):
    # ori
    img = cv.imread('images/positives/'+image)
    output = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    faces = np.asarray(face_cascade.detectMultiScale(image=gray, scaleFactor=scaleFactor,
    minNeighbors=minNeighbors, minSize=(50, 50), maxSize=(500,500)))

    # highlights regions of interest and draws them onto the image.
    ROIs = []
    for (x,y,w,h) in faces:
        output = cv.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h, x:x+w]
        edges = cv.Canny(image=roi, threshold1 = 100, threshold2 = 500 )
        lines = hough_lines(image=edges)
        if(len(lines) > 0):
            for line in lines:
                x1, y1 = line[0]
                x2, y2 = line[1]
                p1, p2 = (x1+x, y1+y), (x2+x, y2+y)
                cv.line(output, p1, p2, (255, 0, 0), 1)

    for(x,y,w,h) in groundTruths[image]:
        output = cv.rectangle(output,(x,y),(x+w,y+h),(0,0,255),2)

    faces_count = faces.shape[0]


#displays the image with roi
    cv.imwrite('detected.jpg',output)
    return eval(groundTruths[image], faces)



def benchmark (scaleFactor=1.1, minNeighbors=1):
    P, R = [], []
    for k in groundTruths.keys():
        _, _, _, p, r, f1 =  detect(k, scaleFactor, minNeighbors)
        P.append(p)
        R.append(r)
        # print(k, p, r, f1)
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
