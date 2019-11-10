import numpy as np
import cv2 as cv

# Uses classifier frontface.xml
face_cascade = cv.CascadeClassifier('frontalface.xml')


#detects image and returns array called 'faces' with subarrays containg x, y, width and height of all boxes around detected faces.
def detect(image):

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# highlights regions of interest and draws them onto the image.
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    faces_count = faces.size/4
    print(faces_count)
    print(faces)
#displays the image with roi
    cv.imshow('detected.jpg',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
detect('images/positives/dart5.jpg')

