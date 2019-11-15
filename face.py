import numpy as np
import cv2 as cv
import argparse
import operator
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from eval import eval

groundTruths = {'dart4.jpg': [[342, 99, 135, 176]],
                'dart5.jpg': [[65, 141, 55, 66], [56, 243, 59, 80], [191, 206, 59, 79], [252, 169, 54, 65], [290, 236, 57, 74], [373, 189, 68, 66], [428, 228, 58, 80], [518, 172, 54, 69], [560, 237, 59, 78], [648, 180, 57, 69], [683, 240, 52, 72]],
                'dart13.jpg': [[418, 118, 118, 143]],
                'dart14.jpg': [[463, 200, 91, 126], [727, 174, 102, 124]],
                'dart15.jpg': [[64, 128, 59, 83], [372, 103, 53, 85], [539, 130, 78, 84]]
}

parser = argparse.ArgumentParser(description = 'Face Detector')
parser.add_argument('--image', help='Path to image', default = 'dart5.jpg')
args = parser.parse_args()

# Uses classifier frontface.xml
face_cascade = cv.CascadeClassifier('frontalface.xml')


#detects image and returns array called 'faces' with subarrays containg x, y, width and height of all boxes around detected faces.
def detect(path, image, scaleFactor=1.1, minNeighbors=1.1):
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    faces = np.asarray(face_cascade.detectMultiScale(image=gray, scaleFactor=scaleFactor,
     minNeighbors=minNeighbors, minSize=(50, 50), maxSize=(500,500)))

# highlights regions of interest and draws them onto the image.
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    for(x,y,w,h) in groundTruths[image]:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    faces_count = faces.shape[0]
#displays the image with roi
    cv.imwrite('detected.jpg',img)
    return groundTruths[image], faces



def benchmark (scaleFactor=1.1, minNeighbors=1):
    P, R = [], []
    for k in groundTruths.keys():
        targets, predicts = detect('images/positives/'+k, k, scaleFactor, minNeighbors)
        p, r, f1 = eval(targets, predicts)
        P.append(p)
        R.append(r)
        # print(k, p, r, f1)
    AP = np.mean(P)
    AR = np.mean(R)
    AF1 = 2 * ((AP*AR)/ (AP + AR))
    # print("average", AP, AR, AF1)
    return AP, AR, AF1

def gridsearch ():
    SF, MN = [], []
    for sf in np.linspace(1.1, 10, 11):
        _ , _ , f = benchmark(scaleFactor=sf)
        SF.append(f)
    for mn in range (1, 10):
        _ , _ , f = benchmark(scaleFactor=1.1, minNeighbors=mn)
        MN.append(f)
    print(max(SF), np.argmax(SF),1+(0.1*np.argmax(SF)))
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(np.linspace(1.1,10,11), np.asarray(SF))
    ax2.plot(np.arange(1, 10, 1), np.asarray(MN))
    plt.show()

    return opt, config[opt]

            

#print(detect('images/positives/'+args.image, args.image))
print(benchmark(4.7, 3))
# print(gridsearch())
