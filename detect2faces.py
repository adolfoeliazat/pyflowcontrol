import io
import cv2
import numpy
import time

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
grayscales = 1.7
#cap.set(4,800)
#cap.set(5,600)
#cap.set(6,24)

ret,frame = cap.read()
ret,frame2 = cap2.read()

ttl = 10
timestart = time.time()
timeend = timestart+ttl

cv2.startWindowThread()
cv2.namedWindow("LiveDetect")
cv2.imshow("LiveDetect",frame)
cv2.startWindowThread()
cv2.namedWindow("LiveDetect2")
cv2.imshow("LiveDetect2",frame2)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while timeend>time.time():
    ret,frame = cap.read()
    ret,frame2 = cap2.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, grayscales, 5)
    faces2 = face_cascade.detectMultiScale(gray2, grayscales, 5)
    #print("Found "+str(len(faces))+" face(s)")
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in faces2:
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("LiveDetect",frame)
    cv2.imshow("LiveDetect2",frame2)
cv2.imwrite('result.jpg',frame)
cv2.imwrite('result2.jpg',frame2)
