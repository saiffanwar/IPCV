import numpy as np
import cv2 as cv
import argparse

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
def detect(image):

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# highlights regions of interest and draws them onto the image.
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    for(x,y,w,h) in groundTruths[args.image]:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    faces_count = faces.size/4
    print(faces_count)
    print(faces)
#displays the image with roi
    print(eval(groundTruths[args.image], faces))
    cv.imshow('detected.jpg',img)
    cv.waitKey(0)
    cv.destroyAllWindows()


detect('images/positives/'+args.image)
