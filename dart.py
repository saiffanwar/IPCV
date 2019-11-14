import numpy as np
import cv2 as cv
import argparse

from eval import eval

groundTruths = {'dart0.jpg': [[444, 14, 152, 175]],
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
                'dart15.jpg': [[155, 57, 125, 136]]
}

parser = argparse.ArgumentParser(description = 'Dartboard Detector')
parser.add_argument('--image', help='Path to image', default = 'dart5.jpg')
args = parser.parse_args()

# Uses classifier frontface.xml
dart_cascade = cv.CascadeClassifier('dartcascade/cascade.xml')


#detects image and returns array called 'faces' with subarrays containg x, y, width and height of all boxes around detected faces.
def detect(image):
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    dartboards = np.asarray(dart_cascade.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=1, minSize=(50, 50), maxSize=(500,500)))

# highlights regions of interest and draws them onto the image.
    for (x,y,w,h) in dartboards:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    for(x,y,w,h) in groundTruths[args.image]:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    dartboards_count = dartboards.shape[0]
#displays the image with roi
    print(eval(groundTruths[args.image], dartboards))
    cv.imwrite('detected.jpg',img)


# images=('dart0.jpg','dart1.jpg','dart2.jpg','dart3.jpg','dart4.jpg','dart5.jpg','dart6.jpg','dart7.jpg','dart8.jpg','dart9.jpg','dart10.jpg','dart11.jpg','dart12.jpg','dart13.jpg','dart14.jpg', 'dart15.jpg')
# for i in images:
#     detect('images/positives/'+i, i)

detect('images/positives/'+args.image)
